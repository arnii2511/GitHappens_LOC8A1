from __future__ import annotations

from datetime import timezone
from typing import Optional

import numpy as np
import pandas as pd

from ..pipeline.checklist import verification_checklist
from ..pipeline.helpers import safe_float
from ..retrieval import TextEmbeddingService, TwoTowerRetriever
from .collaborative import CollaborativeModel
from .common import as_text
from .feature_builder import PairFeatureBuilder
from .ltr import LearningToRankModel
from .supervised import OnlineSupervisedModel


class HybridRanker:
    def __init__(
        self,
        buyers: pd.DataFrame,
        exporters: pd.DataFrame,
        news: pd.DataFrame,
        random_state: int = 42,
        embedding_dim: int = 24,
        collab_min_interactions: int = 10,
        collab_refresh_every: int = 100,
        supervised_refresh_every: int = 80,
        ltr_refresh_every: int = 200,
        max_interactions: int = 250_000,
        base_collab_weight: float = 0.20,
        min_interactions_for_full_collab: int = 20,
        ltr_weight: float = 0.30,
        exploration_rate: float = 0.15,
        exploration_probability: float = 0.45,
        half_life_days: float = 45.0,
        prefer_gpu: bool = True,
    ):
        self.builder = PairFeatureBuilder(buyers, exporters, news)
        self.supervised = OnlineSupervisedModel(
            random_state=random_state,
            half_life_days=half_life_days,
            prefer_gpu=prefer_gpu,
        )
        self.collaborative = CollaborativeModel(
            buyer_ids=self.builder.buyers["Buyer_ID"].astype(str).tolist(),
            exporter_ids=self.builder.exporters["Exporter_ID"].astype(str).tolist(),
            embedding_dim=embedding_dim,
            random_state=random_state,
            half_life_days=half_life_days,
            prefer_gpu=prefer_gpu,
            min_interactions=collab_min_interactions,
        )
        self.ltr = LearningToRankModel(
            random_state=random_state,
            half_life_days=half_life_days,
            prefer_gpu=prefer_gpu,
        )

        self.collab_refresh_every = int(max(20, collab_refresh_every))
        self.supervised_refresh_every = int(max(20, supervised_refresh_every))
        self.ltr_refresh_every = int(max(50, ltr_refresh_every))
        self.max_interactions = int(max(10_000, max_interactions))
        self.base_collab_weight = float(np.clip(base_collab_weight, 0.0, 0.5))
        self.min_interactions_for_full_collab = int(max(5, min_interactions_for_full_collab))
        self.ltr_weight = float(np.clip(ltr_weight, 0.0, 0.7))
        self.exploration_rate = float(np.clip(exploration_rate, 0.0, 0.5))
        self.exploration_probability = float(np.clip(exploration_probability, 0.0, 1.0))

        self.text_encoder = TextEmbeddingService(prefer_gpu=prefer_gpu)
        self.text_encoder.fit(self.builder.buyers, self.builder.exporters)
        self.builder.set_text_encoder(self.text_encoder)
        self.retriever = TwoTowerRetriever(
            self.builder.buyers,
            self.builder.exporters,
            text_encoder=self.text_encoder,
            prefer_gpu=prefer_gpu,
        )

        self.interactions = pd.DataFrame(columns=["buyer_id", "exporter_id", "action", "ts"])
        self._updates_since_collab = 0
        self._updates_since_supervised = 0
        self._updates_since_ltr = 0
        self._updates_since_retriever = 0
        self.retrieval_refresh_every = 300
        self._buyer_swipe_counts: dict[str, int] = {}
        self._buyer_seen_exporters: dict[str, set[str]] = {}
        self.rng = np.random.default_rng(random_state)
        self.is_trained = False

    def fit(self, interactions: Optional[pd.DataFrame] = None):
        if interactions is None:
            interactions = pd.DataFrame(columns=["buyer_id", "exporter_id", "action", "ts"])
        self.interactions = self._sanitize_interactions(interactions).tail(self.max_interactions).reset_index(drop=True)
        self._rebuild_behavior_memory()
        self.builder.update_interaction_stats(self.interactions)
        self.retriever.fit(self.interactions)
        self.builder.set_retrieval_scorer(self.retriever.score_pairs if self.retriever.ready else None)
        self.builder.set_industry_assoc_lookup(self.retriever.industry_assoc_for_pair if self.retriever.ready else None)
        self.collaborative.fit(self.interactions)
        self.supervised.fit(self.interactions, self.builder)
        self.ltr.fit(self.interactions, self.builder)
        self._updates_since_supervised = 0
        self._updates_since_collab = 0
        self._updates_since_ltr = 0
        self._updates_since_retriever = 0
        self.is_trained = bool(self.supervised.ready or self.ltr.ready or self.collaborative.ready or self.retriever.ready)

    def refresh_news(self, news: pd.DataFrame):
        self.builder.refresh_news(news)

    def rank_for_buyer(self, buyer_row: pd.Series, top_k: int = 10):
        if not self.is_trained:
            raise RuntimeError("Ranker is not trained. Train the model before requesting suggestions.")
        top_k = int(max(1, top_k))
        buyer_id = as_text(buyer_row.get("Buyer_ID"))
        retrieval_top = max(200, top_k * 25)
        retrieval_candidates = self.retriever.retrieve_for_buyer(buyer_id, top_k=retrieval_top) if self.retriever.ready else None
        feature_df, warning = self.builder.candidate_features_for_buyer(buyer_row, retrieval_candidates=retrieval_candidates)
        if feature_df.empty:
            return []

        seen_exporters = self._buyer_seen_exporters.get(buyer_id, set())

        model_p = self.supervised.predict_proba(feature_df)
        collab_p = self.collaborative.score(buyer_id, feature_df["exporter_id"].to_numpy())
        collab_weight = self._adaptive_collab_weight(buyer_id)
        blend = np.clip((1.0 - collab_weight) * model_p + collab_weight * collab_p, 0.0, 1.0)

        ltr_p = self.ltr.score(feature_df)
        effective_ltr_weight = self.ltr_weight if self.ltr.ready else 0.0
        final_p = np.clip((1.0 - effective_ltr_weight) * blend + effective_ltr_weight * ltr_p, 0.0, 1.0)
        ltr_signal = ltr_p if self.ltr.ready else np.full_like(model_p, 0.5, dtype=np.float64)

        uncertainty = np.clip(1.0 - (np.abs(model_p - 0.5) * 2.0), 0.0, 1.0)
        confidence = np.clip(
            0.70 * (1.0 - uncertainty)
            + 0.20 * np.abs(collab_p - 0.5) * 2.0
            + 0.10 * np.abs(ltr_signal - 0.5) * 2.0,
            0.0,
            1.0,
        )

        feature_df["ml_score"] = model_p * 100.0
        feature_df["collab_score"] = collab_p * 100.0
        feature_df["ltr_score"] = ltr_p * 100.0
        feature_df["adaptive_collab_weight"] = collab_weight * 100.0
        feature_df["final_rank"] = final_p * 100.0
        feature_df["confidence"] = confidence * 100.0
        feature_df["uncertainty"] = uncertainty
        feature_df["pair_seen"] = feature_df["exporter_id"].astype(str).isin(seen_exporters)
        feature_df["is_exploration"] = False
        feature_df["exploration_bonus"] = 0.0

        feature_df = self._apply_contextual_exploration(feature_df, top_k=top_k)

        cards = []
        for _, row in feature_df.iterrows():
            reasons = [
                f"Industry fit: {round(safe_float(row.get('industry_similarity', row.get('industry_match', 0.0)), 0.0) * 100.0, 1)}",
                f"Industry assoc: {round(safe_float(row.get('industry_assoc_score', 0.0), 0.0) * 100.0, 1)}",
                f"Learned intent fit: {round(safe_float(row['intent_fit'], 0.0), 1)}",
                f"Capacity fit: {round(safe_float(row['cap_fit'], 0.0), 1)}",
                f"Trust pairing: {round(safe_float(row['pair_trust'], 0.0), 1)}",
                f"Retriever score: {round(safe_float(row.get('retrieval_score', 0.5), 0.5) * 100.0, 1)}",
                f"Text similarity: {round(safe_float(row.get('text_similarity', 0.5), 0.5) * 100.0, 1)}",
            ]
            if bool(row.get("is_exploration", False)):
                reasons.append("Exploration pick: high-potential unseen exporter")
            if as_text(row.get("candidate_source")) == "explore":
                reasons.append("Diverse candidate from related industry cluster")

            card = {
                "buyer_id": buyer_id,
                "exporter_id": row["exporter_id"],
                "exporter_state": row["exporter_state"],
                "exporter_cert": row["exporter_cert"],
                "match_score": round(safe_float(row["match_after_risk"], 0.0), 2),
                "trust_score": round(safe_float(row["pair_trust"], 0.0), 2),
                "intent_score": round(safe_float(row["min_intent"], 0.0), 2),
                "risk_penalty": round(safe_float(row["total_risk_penalty"], 0.0), 2),
                "news_risk_penalty": round(safe_float(row["news_risk_penalty"], 0.0), 2),
                "exporter_risk_penalty": round(safe_float(row["exporter_risk_penalty"], 0.0), 2),
                "shock_score": round(safe_float(row["shock_score"], 0.0), 4),
                "industry_similarity": round(
                    safe_float(row.get("industry_similarity", row.get("industry_match", 0.0)), 0.0) * 100.0, 2
                ),
                "industry_assoc_score": round(safe_float(row.get("industry_assoc_score", 0.0), 0.0) * 100.0, 2),
                "industry_assoc_hit": bool(safe_float(row.get("industry_assoc_hit", 0.0), 0.0) > 0.0),
                "ml_score": round(safe_float(row["ml_score"], 0.0), 2),
                "collab_score": round(safe_float(row["collab_score"], 0.0), 2),
                "ltr_score": round(safe_float(row["ltr_score"], 0.0), 2),
                "retrieval_score": round(safe_float(row.get("retrieval_score", 0.5), 0.5) * 100.0, 2),
                "retrieval_rank_norm": round(safe_float(row.get("retrieval_rank_norm", 0.5), 0.5) * 100.0, 2),
                "text_similarity": round(safe_float(row.get("text_similarity", 0.5), 0.5) * 100.0, 2),
                "adaptive_collab_weight": round(safe_float(row["adaptive_collab_weight"], 0.0), 2),
                "dynamic_match_weights": {
                    "cap_fit": round(safe_float(row.get("w_cap_fit", 0.0), 0.0), 4),
                    "intent": round(safe_float(row.get("w_intent", 0.0), 0.0), 4),
                    "comm": round(safe_float(row.get("w_comm", 0.0), 0.0), 4),
                },
                "behavioral_features": {
                    "buyer_swipe_count_norm": round(safe_float(row.get("buyer_swipe_count_norm", 0.0), 0.0), 4),
                    "buyer_right_rate": round(safe_float(row.get("buyer_right_rate", 0.5), 0.5), 4),
                    "exporter_swipe_count_norm": round(safe_float(row.get("exporter_swipe_count_norm", 0.0), 0.0), 4),
                    "exporter_right_rate": round(safe_float(row.get("exporter_right_rate", 0.5), 0.5), 4),
                    "pair_interaction_count_norm": round(safe_float(row.get("pair_interaction_count_norm", 0.0), 0.0), 4),
                    "pair_right_rate": round(safe_float(row.get("pair_right_rate", 0.5), 0.5), 4),
                },
                "candidate_source": as_text(row.get("candidate_source")) or "core",
                "confidence": round(safe_float(row["confidence"], 0.0), 2),
                "is_exploration": bool(row.get("is_exploration", False)),
                "final_rank": round(safe_float(row["final_rank"], 0.0), 2),
                "reasons": reasons,
                "warning": warning,
            }
            card["verification_checklist"] = verification_checklist(card, buyer_row)
            cards.append(card)
        return cards

    def ingest_swipe(self, buyer_id: str, exporter_id: str, action: str):
        action = as_text(action).lower()
        if action not in {"left", "right"}:
            return

        buyer_id = as_text(buyer_id)
        exporter_id = as_text(exporter_id)

        row = pd.DataFrame(
            [
                {
                    "buyer_id": buyer_id,
                    "exporter_id": exporter_id,
                    "action": action,
                    "ts": pd.Timestamp.now(tz=timezone.utc),
                }
            ]
        )
        self.interactions = pd.concat([self.interactions, row], ignore_index=True).tail(self.max_interactions)
        self._updates_since_supervised += 1
        self._updates_since_collab += 1
        self._updates_since_ltr += 1
        self._updates_since_retriever += 1

        self._buyer_swipe_counts[buyer_id] = int(self._buyer_swipe_counts.get(buyer_id, 0) + 1)
        if buyer_id not in self._buyer_seen_exporters:
            self._buyer_seen_exporters[buyer_id] = set()
        self._buyer_seen_exporters[buyer_id].add(exporter_id)
        self.builder.ingest_interaction(buyer_id, exporter_id, action, row.iloc[0]["ts"])

        single_feature = self.builder.single_pair_features(buyer_id, exporter_id)
        did_online = self.supervised.update_single(single_feature, 1 if action == "right" else 0, sample_weight=1.0)
        if (not did_online) and self._updates_since_supervised >= self.supervised_refresh_every:
            self.supervised.fit(self.interactions, self.builder)
            self._updates_since_supervised = 0

        if self._updates_since_collab >= self.collab_refresh_every:
            self.collaborative.fit(self.interactions)
            self._updates_since_collab = 0

        if self._updates_since_ltr >= self.ltr_refresh_every:
            self.ltr.fit(self.interactions, self.builder)
            self._updates_since_ltr = 0

        if self._updates_since_retriever >= self.retrieval_refresh_every:
            self.retriever.fit(self.interactions)
            self.builder.set_retrieval_scorer(self.retriever.score_pairs if self.retriever.ready else None)
            self.builder.set_industry_assoc_lookup(self.retriever.industry_assoc_for_pair if self.retriever.ready else None)
            self._updates_since_retriever = 0

    def _sanitize_interactions(self, interactions: pd.DataFrame) -> pd.DataFrame:
        if interactions is None or interactions.empty:
            return pd.DataFrame(columns=["buyer_id", "exporter_id", "action", "ts"])
        df = interactions.copy()
        for col in ("buyer_id", "exporter_id", "action"):
            if col not in df.columns:
                df[col] = None
        if "ts" not in df.columns:
            df["ts"] = pd.Timestamp.now(tz=timezone.utc)
        df["buyer_id"] = df["buyer_id"].astype(str).str.strip()
        df["exporter_id"] = df["exporter_id"].astype(str).str.strip()
        df["action"] = df["action"].astype(str).str.strip().str.lower()
        df = df[df["action"].isin(["left", "right"])]
        df = df[(df["buyer_id"] != "") & (df["exporter_id"] != "")]
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
        df = df.dropna(subset=["ts"]).sort_values("ts")
        return df[["buyer_id", "exporter_id", "action", "ts"]]

    def _rebuild_behavior_memory(self):
        self._buyer_swipe_counts = {}
        self._buyer_seen_exporters = {}
        if self.interactions.empty:
            return

        grouped = self.interactions.groupby("buyer_id", sort=False)["exporter_id"].agg(list)
        for buyer_id, ex_list in grouped.items():
            b = as_text(buyer_id)
            exporters = [as_text(x) for x in ex_list]
            self._buyer_swipe_counts[b] = len(exporters)
            self._buyer_seen_exporters[b] = set(exporters)

    def _adaptive_collab_weight(self, buyer_id: str) -> float:
        if not self.collaborative.ready:
            return 0.0
        count = int(self._buyer_swipe_counts.get(as_text(buyer_id), 0))
        if count <= 0:
            return 0.0
        ramp = min(1.0, np.log1p(float(count)) / np.log1p(float(self.min_interactions_for_full_collab)))
        return float(np.clip(self.base_collab_weight * ramp, 0.0, self.base_collab_weight))

    def _apply_contextual_exploration(self, feature_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
        ranked = feature_df.sort_values("final_rank", ascending=False).copy()
        if top_k <= 1 or self.exploration_rate <= 0.0:
            return ranked.head(top_k)

        unseen_pool = ranked[~ranked["pair_seen"]].copy()
        if unseen_pool.empty or self.rng.random() > self.exploration_probability:
            return ranked.head(top_k)

        explore_slots = int(max(1, round(top_k * self.exploration_rate)))
        if explore_slots >= top_k:
            explore_slots = top_k - 1
        if explore_slots <= 0:
            return ranked.head(top_k)

        exploit_slots = top_k - explore_slots
        exploit_df = ranked.head(exploit_slots).copy()
        remaining = ranked[~ranked["exporter_id"].isin(exploit_df["exporter_id"])].copy()

        unseen_pool = remaining[~remaining["pair_seen"]].copy()
        if unseen_pool.empty:
            return ranked.head(top_k)

        unseen_pool["bandit_score"] = 0.70 * (unseen_pool["ml_score"] / 100.0) + 0.30 * unseen_pool["uncertainty"]
        source_col = unseen_pool.get("candidate_source")
        if source_col is None:
            diverse_pool = pd.DataFrame(columns=unseen_pool.columns)
        else:
            diverse_pool = unseen_pool[source_col.astype(str) == "explore"].copy()

        if not diverse_pool.empty:
            diverse_take = int(max(1, round(explore_slots * 0.6)))
            diverse_take = min(diverse_take, explore_slots, len(diverse_pool))
            explore_diverse = diverse_pool.nlargest(diverse_take, "bandit_score").copy()
            remaining_slots = explore_slots - len(explore_diverse)
            if remaining_slots > 0:
                fillers = unseen_pool[~unseen_pool["exporter_id"].isin(explore_diverse["exporter_id"])]
                explore_df = pd.concat([explore_diverse, fillers.nlargest(remaining_slots, "bandit_score")], ignore_index=True)
            else:
                explore_df = explore_diverse
        else:
            explore_df = unseen_pool.nlargest(explore_slots, "bandit_score").copy()
        explore_df["is_exploration"] = True
        explore_df["exploration_bonus"] = np.clip(
            explore_df["bandit_score"] * 100.0 - explore_df["final_rank"],
            0.0,
            100.0,
        )

        selected = pd.concat([exploit_df, explore_df], ignore_index=True)
        if len(selected) < top_k:
            fillers = ranked[~ranked["exporter_id"].isin(selected["exporter_id"])].head(top_k - len(selected))
            selected = pd.concat([selected, fillers], ignore_index=True)

        return selected.head(top_k)
