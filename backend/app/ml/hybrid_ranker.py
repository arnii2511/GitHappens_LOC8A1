from __future__ import annotations

from datetime import timezone
from typing import Optional

import numpy as np
import pandas as pd

from ..pipeline.checklist import verification_checklist
from ..pipeline.helpers import safe_float
from .collaborative import CollaborativeModel
from .common import as_text
from .feature_builder import PairFeatureBuilder
from .supervised import OnlineSupervisedModel


class HybridRanker:
    def __init__(
        self,
        buyers: pd.DataFrame,
        exporters: pd.DataFrame,
        news: pd.DataFrame,
        random_state: int = 42,
        embedding_dim: int = 24,
        collab_refresh_every: int = 100,
        max_interactions: int = 250_000,
    ):
        self.builder = PairFeatureBuilder(buyers, exporters, news)
        self.supervised = OnlineSupervisedModel(random_state=random_state)
        self.collaborative = CollaborativeModel(
            buyer_ids=self.builder.buyers["Buyer_ID"].astype(str).tolist(),
            exporter_ids=self.builder.exporters["Exporter_ID"].astype(str).tolist(),
            embedding_dim=embedding_dim,
            random_state=random_state,
        )

        self.collab_refresh_every = int(max(20, collab_refresh_every))
        self.max_interactions = int(max(10_000, max_interactions))
        self.interactions = pd.DataFrame(columns=["buyer_id", "exporter_id", "action", "ts"])
        self._updates_since_collab = 0

    def fit(self, interactions: Optional[pd.DataFrame] = None):
        if interactions is None:
            interactions = pd.DataFrame(columns=["buyer_id", "exporter_id", "action", "ts"])
        self.interactions = self._sanitize_interactions(interactions).tail(self.max_interactions).reset_index(drop=True)
        self.collaborative.fit(self.interactions)
        self.supervised.fit(self.interactions, self.builder)

    def refresh_news(self, news: pd.DataFrame):
        self.builder.refresh_news(news)

    def rank_for_buyer(self, buyer_row: pd.Series, top_k: int = 10):
        feature_df, warning = self.builder.candidate_features_for_buyer(buyer_row)
        if feature_df.empty:
            return []

        buyer_id = as_text(buyer_row.get("Buyer_ID"))
        model_p = self.supervised.predict_proba(feature_df)
        collab_p = self.collaborative.score(buyer_id, feature_df["exporter_id"].to_numpy())

        if self.collaborative.ready:
            blend = np.clip(0.8 * model_p + 0.2 * collab_p, 0.0, 1.0)
        else:
            blend = model_p

        confidence = np.clip(np.abs(model_p - 0.5) * 2.0 + 0.15 * np.abs(collab_p - 0.5) * 2.0, 0.0, 1.0)
        feature_df["ml_score"] = model_p * 100.0
        feature_df["collab_score"] = collab_p * 100.0
        feature_df["final_rank"] = blend * 100.0
        feature_df["confidence"] = confidence * 100.0
        feature_df = feature_df.sort_values("final_rank", ascending=False).head(int(max(1, top_k)))

        cards = []
        for _, row in feature_df.iterrows():
            reasons = [
                f"Industry match: {row['industry']}",
                f"Learned intent fit: {round(safe_float(row['intent_fit'], 0.0), 1)}",
                f"Capacity fit: {round(safe_float(row['cap_fit'], 0.0), 1)}",
                f"Trust pairing: {round(safe_float(row['pair_trust'], 0.0), 1)}",
            ]
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
                "ml_score": round(safe_float(row["ml_score"], 0.0), 2),
                "collab_score": round(safe_float(row["collab_score"], 0.0), 2),
                "confidence": round(safe_float(row["confidence"], 0.0), 2),
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

        row = pd.DataFrame(
            [
                {
                    "buyer_id": as_text(buyer_id),
                    "exporter_id": as_text(exporter_id),
                    "action": action,
                    "ts": pd.Timestamp.now(tz=timezone.utc),
                }
            ]
        )
        self.interactions = pd.concat([self.interactions, row], ignore_index=True).tail(self.max_interactions)
        self._updates_since_collab += 1

        single_feature = self.builder.single_pair_features(as_text(buyer_id), as_text(exporter_id))
        self.supervised.update_single(single_feature, 1 if action == "right" else 0)

        if self._updates_since_collab >= self.collab_refresh_every:
            self.collaborative.fit(self.interactions)
            self._updates_since_collab = 0

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
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        df = df.dropna(subset=["ts"]).sort_values("ts")
        return df[["buyer_id", "exporter_id", "action", "ts"]]
