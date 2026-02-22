from __future__ import annotations

from datetime import timezone
from typing import Optional

import numpy as np
import pandas as pd

from ..industry_map import canonicalize, industry_similarity
from ..pipeline.checklist import verification_checklist
from ..pipeline.helpers import safe_float
from ..retrieval import TextEmbeddingService, TwoTowerRetriever
from .collaborative import CollaborativeModel
from .common import as_text
from .feature_builder import PairFeatureBuilder
from .ltr import LearningToRankModel
from .ncf import NeuralCollaborativeFilteringModel
from .supervised import OnlineSupervisedModel


class _DisabledComponent:
    def __init__(self, backend: str = "none", device: str = "none"):
        self.backend = backend
        self.device = device
        self.ready = False


class HybridRanker:
    def __init__(
        self,
        buyers: pd.DataFrame,
        exporters: pd.DataFrame,
        news: pd.DataFrame,
        random_state: int = 42,
        prefer_gpu: bool = True,
        max_interactions: int = 250_000,
        auto_tune_weights: bool = True,
        tune_eval_buyers: int = 120,
        online_full_refresh_every: int = 120,
        recommendation_version: str = "hybrid-v1",
    ):
        self.builder = PairFeatureBuilder(buyers, exporters, news)
        self.supervised = OnlineSupervisedModel(random_state=random_state, half_life_days=45.0, prefer_gpu=prefer_gpu)
        self.ltr = LearningToRankModel(random_state=random_state, half_life_days=45.0, prefer_gpu=prefer_gpu)
        self.collaborative = CollaborativeModel(
            buyer_ids=self.builder.buyers["Buyer_ID"].astype(str).tolist(),
            exporter_ids=self.builder.exporters["Exporter_ID"].astype(str).tolist(),
            embedding_dim=24,
            random_state=random_state,
            half_life_days=45.0,
            prefer_gpu=prefer_gpu,
            min_interactions=10,
        )
        self.ncf = NeuralCollaborativeFilteringModel(
            buyer_ids=self.builder.buyers["Buyer_ID"].astype(str).tolist(),
            exporter_ids=self.builder.exporters["Exporter_ID"].astype(str).tolist(),
            embedding_dim=24,
            hidden_dim=64,
            epochs=4,
            batch_size=4096,
            lr=1e-3,
            half_life_days=45.0,
            prefer_gpu=prefer_gpu,
            min_interactions=300,
        )
        self.graph = _DisabledComponent(backend="none", device="none")

        self.random_state = int(random_state)
        self.max_interactions = int(max(10_000, max_interactions))
        self.auto_tune_weights = bool(auto_tune_weights)
        self.tune_eval_buyers = int(max(30, tune_eval_buyers))
        self.online_full_refresh_every = int(max(20, online_full_refresh_every))
        self._updates_since_full_refresh = 0
        self.recommendation_version = as_text(recommendation_version) or "hybrid-v1"

        self.source_weights = {"two_tower": 0.70, "svd": 0.15, "ncf": 0.10, "popularity": 0.05}
        self.blend_weights = {"deep": 0.62, "wide": 0.23, "seq": 0.10, "text": 0.05}
        self._normalize_weight_dict(self.source_weights)
        self._normalize_weight_dict(self.blend_weights)
        self.tuning_info = {"ran": False, "n_eval_buyers": 0, "best_precision_at_10": None}

        self.text_encoder = TextEmbeddingService(prefer_gpu=prefer_gpu, enable_teacher=False)
        self.text_encoder.fit(self.builder.buyers, self.builder.exporters)
        self.builder.set_text_encoder(self.text_encoder)
        self.builder.set_teacher_enabled(False)

        self.retriever = TwoTowerRetriever(
            self.builder.buyers,
            self.builder.exporters,
            text_encoder=self.text_encoder,
            prefer_gpu=prefer_gpu,
            enable_hard_negatives=False,
            enable_logq_correction=False,
            enable_distillation=False,
        )

        self.interactions = pd.DataFrame(
            columns=[
                "buyer_id",
                "exporter_id",
                "action",
                "ts",
                "session_id",
                "shown_rank",
                "source",
                "dwell_ms",
                "device",
                "region",
                "recommendation_version",
            ]
        )
        self._buyer_swipe_counts: dict[str, int] = {}
        self._buyer_seen_exporters: dict[str, set[str]] = {}
        self._candidate_cache: dict[tuple[str, int, str], pd.DataFrame] = {}
        self.is_trained = False

    def _normalize_weight_dict(self, d: dict[str, float]) -> None:
        s = float(sum(max(0.0, float(v)) for v in d.values()))
        if s <= 0.0:
            n = max(1, len(d))
            for k in d:
                d[k] = 1.0 / float(n)
            return
        for k in d:
            d[k] = float(max(0.0, d[k]) / s)

    def _source_signature(self, source_weights: dict[str, float]) -> str:
        return "|".join([f"{k}:{round(float(source_weights[k]), 4)}" for k in sorted(source_weights.keys())])

    def _rank_norm(self, values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64)
        n = int(v.size)
        if n <= 0:
            return v
        if n < 5:
            return np.clip(v, 0.0, 1.0)
        order = np.argsort(v, kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64)
        ranks /= float(max(1, n - 1))
        return np.clip(ranks, 0.0, 1.0)

    def _sanitize_interactions(self, interactions: pd.DataFrame) -> pd.DataFrame:
        if interactions is None or interactions.empty:
            return pd.DataFrame(
                columns=[
                    "buyer_id",
                    "exporter_id",
                    "action",
                    "ts",
                    "session_id",
                    "shown_rank",
                    "source",
                    "dwell_ms",
                    "device",
                    "region",
                    "recommendation_version",
                ]
            )
        df = interactions.copy()
        for col in ("buyer_id", "exporter_id", "action"):
            if col not in df.columns:
                df[col] = None
        if "ts" not in df.columns:
            df["ts"] = pd.Timestamp.now(tz=timezone.utc)
        if "session_id" not in df.columns:
            df["session_id"] = ""
        if "shown_rank" not in df.columns:
            df["shown_rank"] = np.nan
        if "source" not in df.columns:
            df["source"] = "unknown"
        if "dwell_ms" not in df.columns:
            df["dwell_ms"] = np.nan
        if "device" not in df.columns:
            df["device"] = ""
        if "region" not in df.columns:
            df["region"] = ""
        if "recommendation_version" not in df.columns:
            df["recommendation_version"] = self.recommendation_version
        df["buyer_id"] = df["buyer_id"].astype(str).str.strip()
        df["exporter_id"] = df["exporter_id"].astype(str).str.strip()
        df["action"] = df["action"].astype(str).str.strip().str.lower()
        df = df[df["action"].isin(["left", "right"])]
        df = df[(df["buyer_id"] != "") & (df["exporter_id"] != "")]
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
        df["session_id"] = df["session_id"].astype(str).str.strip()
        df["shown_rank"] = pd.to_numeric(df["shown_rank"], errors="coerce")
        df["source"] = df["source"].astype(str).str.strip().str.lower()
        df["dwell_ms"] = pd.to_numeric(df["dwell_ms"], errors="coerce")
        df["device"] = df["device"].astype(str).str.strip().str.lower()
        df["region"] = df["region"].astype(str).str.strip().str.upper()
        df["recommendation_version"] = df["recommendation_version"].astype(str).str.strip()
        return df.dropna(subset=["ts"]).sort_values("ts")[
            [
                "buyer_id",
                "exporter_id",
                "action",
                "ts",
                "session_id",
                "shown_rank",
                "source",
                "dwell_ms",
                "device",
                "region",
                "recommendation_version",
            ]
        ]

    def _rebuild_behavior_memory(self):
        self._buyer_swipe_counts = {}
        self._buyer_seen_exporters = {}
        if self.interactions.empty:
            return
        grouped = self.interactions.groupby("buyer_id", sort=False)["exporter_id"].agg(list)
        for buyer_id, ex_list in grouped.items():
            b = as_text(buyer_id)
            xs = [as_text(x) for x in ex_list]
            self._buyer_swipe_counts[b] = len(xs)
            self._buyer_seen_exporters[b] = set(xs)

    def _adaptive_collab_weight(self, _buyer_id: str) -> float:
        return 0.0

    def fit(self, interactions: Optional[pd.DataFrame] = None, crossed_features: Optional[pd.DataFrame] = None):
        if interactions is None:
            interactions = pd.DataFrame(columns=["buyer_id", "exporter_id", "action", "ts"])
        self.interactions = self._sanitize_interactions(interactions).tail(self.max_interactions).reset_index(drop=True)
        self._rebuild_behavior_memory()

        self.builder.update_interaction_stats(self.interactions)
        self.retriever.fit(self.interactions)
        self.builder.set_retrieval_scorer(self.retriever.score_pairs if self.retriever.ready else None)
        self.builder.set_industry_assoc_lookup(None)
        self.builder.set_graph_scorer(None)

        self.collaborative.fit(self.interactions)
        self.ncf.fit(self.interactions)
        self.supervised.fit(self.interactions, self.builder, crossed_features=crossed_features)
        self.ltr.fit(self.interactions, self.builder, crossed_features=crossed_features)

        self._candidate_cache = {}
        self._updates_since_full_refresh = 0
        self.is_trained = bool(
            self.supervised.ready or self.ltr.ready or self.retriever.ready or self.collaborative.ready or self.ncf.ready
        )
        if self.auto_tune_weights:
            self._tune_weights_from_history(max_buyers=self.tune_eval_buyers)

    def refresh_news(self, news: pd.DataFrame):
        self.builder.refresh_news(news)

    def _collab_scores_for_df(self, feature_df: pd.DataFrame) -> np.ndarray:
        out = np.zeros(len(feature_df), dtype=np.float64)
        if not self.collaborative.ready or feature_df.empty:
            return out
        b = feature_df["buyer_id"].astype(str).to_numpy(dtype=object)
        e = feature_df["exporter_id"].astype(str).to_numpy(dtype=object)
        for bid in np.unique(b):
            m = b == bid
            out[m] = self.collaborative.score(str(bid), e[m])
        return np.clip(out, 0.0, 1.0)

    def _ncf_scores_for_df(self, feature_df: pd.DataFrame) -> np.ndarray:
        out = np.zeros(len(feature_df), dtype=np.float64)
        if not self.ncf.ready or feature_df.empty:
            return out
        b = feature_df["buyer_id"].astype(str).to_numpy(dtype=object)
        e = feature_df["exporter_id"].astype(str).to_numpy(dtype=object)
        for bid in np.unique(b):
            m = b == bid
            out[m] = self.ncf.score(str(bid), e[m])
        return np.clip(out, 0.0, 1.0)

    def _multi_source_candidates_for_buyer_row(
        self,
        buyer_row: pd.Series,
        top_k: int,
        source_weights: Optional[dict[str, float]] = None,
    ) -> pd.DataFrame | None:
        buyer_id = as_text(buyer_row.get("Buyer_ID"))
        if buyer_id == "":
            return None
        if source_weights is None:
            source_weights = self.source_weights
        ids_all = self.builder.exporters["Exporter_ID"].astype(str).to_numpy(dtype=object)
        if ids_all.size == 0:
            return None
        base_k = int(max(300, top_k))
        source_tables: dict[str, pd.DataFrame] = {}

        if self.retriever.ready:
            tt = self.retriever.retrieve_for_buyer(buyer_id, top_k=base_k)
            if tt is not None and not tt.empty:
                x = tt[["exporter_id", "retrieval_score"]].copy()
                x = x.rename(columns={"retrieval_score": "source_score"})
                x["source"] = "two_tower"
                source_tables["two_tower"] = x

        if self.collaborative.ready:
            svd = np.asarray(self.collaborative.score(buyer_id, ids_all), dtype=np.float64)
            take = min(350, len(svd))
            if take > 0:
                idx = np.argsort(-svd)[:take]
                source_tables["svd"] = pd.DataFrame({"exporter_id": ids_all[idx].astype(str), "source_score": svd[idx], "source": "svd"})

        if self.ncf.ready:
            ncf = np.asarray(self.ncf.score(buyer_id, ids_all), dtype=np.float64)
            take = min(350, len(ncf))
            if take > 0:
                idx = np.argsort(-ncf)[:take]
                source_tables["ncf"] = pd.DataFrame({"exporter_id": ids_all[idx].astype(str), "source_score": ncf[idx], "source": "ncf"})

        buyer_ind = canonicalize(as_text(buyer_row.get("Industry", "")))
        if buyer_ind != "unknown":
            exp_df = self.builder.exporters[["Exporter_ID", "Industry"]].copy()
            exp_df["exporter_id"] = exp_df["Exporter_ID"].astype(str)
            counts = np.array([float(self.builder._exporter_hist_raw.get(x, {}).get("count", 0.0)) for x in exp_df["exporter_id"].tolist()], dtype=np.float64)
            maxc = float(max(1.0, np.max(counts) if counts.size else 1.0))
            c_norm = np.clip(np.log1p(counts) / np.log1p(maxc), 0.0, 1.0)
            i_sim = exp_df["Industry"].apply(lambda x: industry_similarity(buyer_ind, x)).to_numpy(dtype=np.float64)
            pop = np.clip(0.70 * c_norm + 0.30 * i_sim, 0.0, 1.0)
            take = min(180, len(pop))
            if take > 0:
                idx = np.argsort(-pop)[:take]
                source_tables["popularity"] = pd.DataFrame({"exporter_id": exp_df["exporter_id"].to_numpy(dtype=object)[idx].astype(str), "source_score": pop[idx], "source": "popularity"})

        if not source_tables:
            return None

        agg: dict[str, float] = {}
        best_src: dict[str, tuple[str, float]] = {}
        for src, df in source_tables.items():
            w = float(source_weights.get(src, 0.0))
            if w <= 0.0 or df.empty:
                continue
            work = df.copy()
            work["source_score"] = np.clip(pd.to_numeric(work["source_score"], errors="coerce").fillna(0.0), 0.0, 1.0)
            work = work.sort_values("source_score", ascending=False).reset_index(drop=True)
            n = len(work)
            work["rank_norm"] = 1.0 if n <= 1 else 1.0 - (work.index.to_numpy(dtype=np.float64) / float(n - 1))
            work["signal"] = np.clip(0.70 * work["source_score"] + 0.30 * work["rank_norm"], 0.0, 1.0)
            for _, r in work.iterrows():
                ex = str(r["exporter_id"])
                c = w * float(r["signal"])
                agg[ex] = float(agg.get(ex, 0.0) + c)
                p = best_src.get(ex)
                if p is None or c > p[1]:
                    best_src[ex] = (src, c)

        if not agg:
            return None
        out = pd.DataFrame([{"exporter_id": ex, "retrieval_score": np.clip(sc, 0.0, 1.0)} for ex, sc in agg.items()])
        out = out.sort_values("retrieval_score", ascending=False).head(int(max(120, top_k))).reset_index(drop=True)
        if len(out) <= 1:
            out["retrieval_rank_norm"] = 1.0
        else:
            out["retrieval_rank_norm"] = 1.0 - (out.index.to_numpy(dtype=np.float64) / float(len(out) - 1))
        out["industry_assoc_score"] = 0.0
        out["industry_assoc_hit"] = 0.0
        out["candidate_source"] = out["exporter_id"].astype(str).apply(lambda x: best_src.get(x, ("two_tower", 0.0))[0])
        return out[["exporter_id", "retrieval_score", "retrieval_rank_norm", "industry_assoc_score", "industry_assoc_hit", "candidate_source"]]

    def multi_source_candidates_for_buyer(self, buyer_id: str, top_k: int = 400) -> pd.DataFrame:
        if buyer_id not in self.builder.buyers_idx.index:
            return pd.DataFrame()
        k = int(max(50, top_k))
        sig = self._source_signature(self.source_weights)
        key = (str(buyer_id), k, sig)
        if key in self._candidate_cache:
            return self._candidate_cache[key].copy()
        row = self.builder.buyers_idx.loc[buyer_id]
        out = self._multi_source_candidates_for_buyer_row(row, top_k=k, source_weights=self.source_weights)
        if out is None:
            out = pd.DataFrame()
        self._candidate_cache[key] = out.copy()
        return out

    def _score_components(self, feature_df: pd.DataFrame, blend_weights: Optional[dict[str, float]] = None) -> dict[str, np.ndarray]:
        bw = dict(self.blend_weights if blend_weights is None else blend_weights)
        self._normalize_weight_dict(bw)

        model_p = np.clip(self.supervised.predict_proba(feature_df), 0.0, 1.0)
        ltr_p = np.clip(self.ltr.score(feature_df), 0.0, 1.0)
        collab_p = self._collab_scores_for_df(feature_df)
        ncf_p = self._ncf_scores_for_df(feature_df)
        retrieval_p = np.clip(pd.to_numeric(feature_df.get("retrieval_score", 0.5), errors="coerce").fillna(0.5).to_numpy(dtype=np.float64), 0.0, 1.0)
        text_p = np.clip(pd.to_numeric(feature_df.get("text_similarity", 0.5), errors="coerce").fillna(0.5).to_numpy(dtype=np.float64), 0.0, 1.0)
        seq_p = np.clip(pd.to_numeric(feature_df.get("sequence_score", 0.5), errors="coerce").fillna(0.5).to_numpy(dtype=np.float64), 0.0, 1.0)
        risk_p = np.clip(pd.to_numeric(feature_df.get("total_risk_penalty", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64) / 10.0, 0.0, 1.0)
        match_p = np.clip(pd.to_numeric(feature_df.get("match_after_risk", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64) / 100.0, 0.0, 1.0)
        trust_p = np.clip(pd.to_numeric(feature_df.get("pair_trust", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64) / 100.0, 0.0, 1.0)
        intent_p = np.clip(pd.to_numeric(feature_df.get("intent_fit", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64) / 100.0, 0.0, 1.0)
        industry_p = np.clip(pd.to_numeric(feature_df.get("industry_similarity", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64), 0.0, 1.0)
        hs_p = np.clip(pd.to_numeric(feature_df.get("hs_match_score", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64), 0.0, 1.0)
        country_comp_p = np.clip(
            pd.to_numeric(feature_df.get("country_complement_score", 0.5), errors="coerce").fillna(0.5).to_numpy(dtype=np.float64),
            0.0,
            1.0,
        )
        comm_p = np.clip(pd.to_numeric(feature_df.get("comm_score", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64) / 100.0, 0.0, 1.0)
        exporter_pop = np.clip(
            pd.to_numeric(feature_df.get("exporter_swipe_count_norm", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64),
            0.0,
            1.0,
        )
        buyer_cold = (
            np.clip(pd.to_numeric(feature_df.get("buyer_swipe_count_norm", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64), 0.0, 1.0)
            < 0.08
        ).astype(np.float64)
        pair_right = np.clip(pd.to_numeric(feature_df.get("pair_right_rate", 0.5), errors="coerce").fillna(0.5).to_numpy(dtype=np.float64), 0.0, 1.0)

        model_n = self._rank_norm(model_p)
        ltr_n = self._rank_norm(ltr_p)
        retrieval_n = self._rank_norm(retrieval_p)
        collab_n = self._rank_norm(collab_p)
        ncf_n = self._rank_norm(ncf_p)
        seq_n = self._rank_norm(seq_p)
        text_n = self._rank_norm(text_p)

        wide_n = self._rank_norm(np.clip(
            0.25 * match_p
            + 0.16 * trust_p
            + 0.11 * intent_p
            + 0.11 * industry_p
            + 0.10 * hs_p
            + 0.10 * (1.0 - risk_p)
            + 0.06 * collab_p
            + 0.06 * comm_p
            + 0.05 * ncf_p,
            0.0,
            1.0,
        ))
        deep_n = self._rank_norm(np.clip(
            0.34 * model_p + 0.34 * ltr_p + 0.18 * retrieval_p + 0.08 * collab_p + 0.05 * ncf_p + 0.01 * text_p,
            0.0,
            1.0,
        ))

        final = np.clip(bw["deep"] * deep_n + bw["wide"] * wide_n + bw["seq"] * seq_n + bw["text"] * text_n, 0.0, 1.0)
        # Popularity bias control: penalize globally popular exporters unless personalized fit is also high.
        personalized_fit = np.clip(0.50 * pair_right + 0.30 * industry_p + 0.20 * intent_p, 0.0, 1.0)
        pop_penalty = np.clip(exporter_pop - personalized_fit, 0.0, 1.0)
        final = np.clip(final - 0.08 * pop_penalty, 0.0, 1.0)
        # Cold start hybrid boost: when buyer has little history, rely more on HS+country compatibility.
        hybrid_rule = np.clip(0.70 * hs_p + 0.30 * country_comp_p, 0.0, 1.0)
        final = np.clip((1.0 - 0.18 * buyer_cold) * final + (0.18 * buyer_cold) * hybrid_rule, 0.0, 1.0)
        comp_stack = np.column_stack([model_n, ltr_n, retrieval_n, collab_n, ncf_n, seq_n, text_n])
        disagreement = np.std(comp_stack, axis=1)
        agreement = 1.0 - np.clip(disagreement / 0.35, 0.0, 1.0)
        margin = np.abs(final - 0.5) * 2.0
        confidence = np.clip(0.65 * agreement + 0.35 * margin, 0.0, 1.0)
        return {
            "final": final,
            "deep": deep_n,
            "wide": wide_n,
            "model": model_p,
            "ltr": ltr_p,
            "collab": collab_p,
            "ncf": ncf_p,
            "retrieval": retrieval_p,
            "sequence": seq_p,
            "text": text_p,
            "confidence": confidence,
            "hs": hs_p,
            "country_comp": country_comp_p,
            "pop_penalty": pop_penalty,
        }

    def score_feature_df(self, feature_df: pd.DataFrame) -> np.ndarray:
        if feature_df is None or feature_df.empty:
            return np.zeros(0, dtype=np.float64)
        return self._score_components(feature_df)["final"]

    def _precision_for_weights(self, buyers_eval: list[str], positives_by_buyer: dict[str, set[str]], source_w: dict[str, float], blend_w: dict[str, float], top_k: int = 10) -> tuple[float, int]:
        vals = []
        used = 0
        for buyer_id in buyers_eval:
            pos = positives_by_buyer.get(str(buyer_id), set())
            if not pos or buyer_id not in self.builder.buyers_idx.index:
                continue
            row = self.builder.buyers_idx.loc[buyer_id]
            cands = self._multi_source_candidates_for_buyer_row(row, top_k=max(500, top_k * 40), source_weights=source_w)
            fdf, _ = self.builder.candidate_features_for_buyer(row, retrieval_candidates=cands)
            if fdf is None or fdf.empty:
                continue
            final = self._score_components(fdf, blend_weights=blend_w)["final"]
            top = fdf.iloc[np.argsort(-final)].head(top_k)["exporter_id"].astype(str).tolist()
            vals.append(float(len(set(top) & set(pos)) / float(top_k)))
            used += 1
        if used <= 0:
            return 0.0, 0
        return float(np.mean(vals)), int(used)

    def _tune_weights_from_history(self, max_buyers: int = 120) -> None:
        self.tuning_info = {"ran": False, "n_eval_buyers": 0, "best_precision_at_10": None}
        if self.interactions is None or self.interactions.empty or len(self.interactions) < 1200:
            return

        val_parts = []
        for _, g in self.interactions.groupby("buyer_id", sort=False):
            g = g.sort_values("ts")
            n = len(g)
            if n < 5:
                continue
            n_val = max(1, int(round(n * 0.2)))
            val_parts.append(g.iloc[-n_val:])
        if not val_parts:
            return
        val_df = pd.concat(val_parts, ignore_index=True)
        pos_df = val_df[val_df["action"] == "right"]
        if pos_df.empty:
            return

        positives_by_buyer = pos_df.groupby("buyer_id")["exporter_id"].agg(lambda x: set(x.astype(str).tolist())).to_dict()
        buyers = list(positives_by_buyer.keys())
        if not buyers:
            return
        rng = np.random.default_rng(self.random_state)
        if len(buyers) > max_buyers:
            buyers = rng.choice(np.array(buyers, dtype=object), size=max_buyers, replace=False).tolist()

        source_opts = [
            {"two_tower": 0.70, "svd": 0.15, "ncf": 0.10, "popularity": 0.05},
            {"two_tower": 0.74, "svd": 0.14, "ncf": 0.08, "popularity": 0.04},
            {"two_tower": 0.66, "svd": 0.18, "ncf": 0.11, "popularity": 0.05},
            {"two_tower": 0.62, "svd": 0.22, "ncf": 0.11, "popularity": 0.05},
        ]
        blend_opts = [
            {"deep": 0.62, "wide": 0.23, "seq": 0.10, "text": 0.05},
            {"deep": 0.66, "wide": 0.20, "seq": 0.10, "text": 0.04},
            {"deep": 0.58, "wide": 0.27, "seq": 0.10, "text": 0.05},
            {"deep": 0.64, "wide": 0.21, "seq": 0.11, "text": 0.04},
            {"deep": 0.60, "wide": 0.24, "seq": 0.11, "text": 0.05},
        ]

        best_p = -1.0
        best_src = dict(self.source_weights)
        best_blend = dict(self.blend_weights)
        best_n = 0
        for sw in source_opts:
            s = dict(sw)
            self._normalize_weight_dict(s)
            for bw in blend_opts:
                b = dict(bw)
                self._normalize_weight_dict(b)
                p, n_eval = self._precision_for_weights(buyers, positives_by_buyer, source_w=s, blend_w=b, top_k=10)
                if n_eval > 0 and p > best_p:
                    best_p = p
                    best_src = s
                    best_blend = b
                    best_n = n_eval

        if best_p >= 0.0 and best_n >= 20:
            self.source_weights = dict(best_src)
            self.blend_weights = dict(best_blend)
            self._candidate_cache = {}
            self.tuning_info = {"ran": True, "n_eval_buyers": int(best_n), "best_precision_at_10": float(best_p)}

    def rank_for_buyer(self, buyer_row: pd.Series, top_k: int = 10):
        if not self.is_trained:
            raise RuntimeError("Ranker is not trained.")
        top_k = int(max(1, top_k))
        buyer_id = as_text(buyer_row.get("Buyer_ID"))

        retrieval_candidates = self._multi_source_candidates_for_buyer_row(
            buyer_row,
            top_k=max(500, top_k * 40),
            source_weights=self.source_weights,
        )
        feature_df, warning = self.builder.candidate_features_for_buyer(buyer_row, retrieval_candidates=retrieval_candidates)
        if feature_df.empty:
            return []

        comp = self._score_components(feature_df, blend_weights=self.blend_weights)
        pair_n = np.clip(pd.to_numeric(feature_df.get("pair_interaction_count_norm", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64), 0.0, 1.0)

        feature_df = feature_df.copy()
        feature_df["ml_score"] = comp["model"] * 100.0
        feature_df["collab_score"] = comp["collab"] * 100.0
        feature_df["ncf_score"] = comp["ncf"] * 100.0
        feature_df["ltr_score"] = comp["ltr"] * 100.0
        feature_df["final_rank"] = comp["final"] * 100.0
        feature_df["confidence"] = comp["confidence"] * 100.0
        feature_df["history_signal"] = comp["sequence"] * 100.0
        feature_df["history_weight"] = np.clip(5.0 + 25.0 * pair_n, 5.0, 30.0)
        feature_df["hs_score"] = comp["hs"] * 100.0
        feature_df["country_comp_score"] = comp["country_comp"] * 100.0
        feature_df["popularity_penalty"] = comp["pop_penalty"] * 100.0
        feature_df["is_exploration"] = False
        feature_df = feature_df.sort_values("final_rank", ascending=False).head(top_k)

        cards = []
        for idx, row in enumerate(feature_df.iterrows(), start=1):
            _, row = row
            card = {
                "buyer_id": buyer_id,
                "exporter_id": row["exporter_id"],
                "exporter_state": row.get("exporter_state"),
                "exporter_cert": row.get("exporter_cert"),
                "match_score": round(safe_float(row.get("match_after_risk", 0.0), 0.0), 2),
                "trust_score": round(safe_float(row.get("pair_trust", 0.0), 0.0), 2),
                "intent_score": round(safe_float(row.get("min_intent", 0.0), 0.0), 2),
                "risk_penalty": round(safe_float(row.get("total_risk_penalty", 0.0), 0.0), 2),
                "news_risk_penalty": round(safe_float(row.get("news_risk_penalty", 0.0), 0.0), 2),
                "exporter_risk_penalty": round(safe_float(row.get("exporter_risk_penalty", 0.0), 0.0), 2),
                "shock_score": round(safe_float(row.get("shock_score", 0.0), 0.0), 4),
                "industry_similarity": round(safe_float(row.get("industry_similarity", 0.0), 0.0) * 100.0, 2),
                "hs_match_score": round(safe_float(row.get("hs_score", 0.0), 0.0), 2),
                "country_complement_score": round(safe_float(row.get("country_comp_score", 0.0), 0.0), 2),
                "popularity_penalty": round(safe_float(row.get("popularity_penalty", 0.0), 0.0), 2),
                "industry_assoc_score": 0.0,
                "industry_assoc_hit": False,
                "ml_score": round(safe_float(row.get("ml_score", 0.0), 0.0), 2),
                "collab_score": round(safe_float(row.get("collab_score", 0.0), 0.0), 2),
                "ncf_score": round(safe_float(row.get("ncf_score", 0.0), 0.0), 2),
                "ltr_score": round(safe_float(row.get("ltr_score", 0.0), 0.0), 2),
                "retrieval_score": round(safe_float(row.get("retrieval_score", 0.5), 0.5) * 100.0, 2),
                "retrieval_rank_norm": round(safe_float(row.get("retrieval_rank_norm", 0.5), 0.5) * 100.0, 2),
                "text_similarity": round(safe_float(row.get("text_similarity", 0.5), 0.5) * 100.0, 2),
                "teacher_score": round(safe_float(row.get("text_similarity", 0.5), 0.5) * 100.0, 2),
                "graph_sim": round(safe_float(row.get("retrieval_score", 0.5), 0.5) * 100.0, 2),
                "adaptive_collab_weight": 0.0,
                "history_weight": round(safe_float(row.get("history_weight", 10.0), 10.0), 2),
                "history_signal": round(safe_float(row.get("history_signal", 50.0), 50.0), 2),
                "dynamic_match_weights": {
                    "cap_fit": round(safe_float(row.get("w_cap_fit", 0.0), 0.0), 4),
                    "intent": round(safe_float(row.get("w_intent", 0.0), 0.0), 4),
                    "comm": round(safe_float(row.get("w_comm", 0.0), 0.0), 4),
                },
                "behavioral_features": {
                    "buyer_swipe_count_norm": round(safe_float(row.get("buyer_swipe_count_norm", 0.0), 0.0), 4),
                    "buyer_right_rate": round(safe_float(row.get("buyer_right_rate", 0.5), 0.5), 4),
                    "buyer_recent_right_rate": round(safe_float(row.get("buyer_recent_right_rate", 0.5), 0.5), 4),
                    "buyer_avg_dwell_norm": round(safe_float(row.get("buyer_avg_dwell_norm", 0.0), 0.0), 4),
                    "exporter_swipe_count_norm": round(safe_float(row.get("exporter_swipe_count_norm", 0.0), 0.0), 4),
                    "exporter_right_rate": round(safe_float(row.get("exporter_right_rate", 0.5), 0.5), 4),
                    "exporter_avg_dwell_norm": round(safe_float(row.get("exporter_avg_dwell_norm", 0.0), 0.0), 4),
                    "pair_interaction_count_norm": round(safe_float(row.get("pair_interaction_count_norm", 0.0), 0.0), 4),
                    "pair_right_rate": round(safe_float(row.get("pair_right_rate", 0.5), 0.5), 4),
                    "pair_avg_dwell_norm": round(safe_float(row.get("pair_avg_dwell_norm", 0.0), 0.0), 4),
                    "pair_last_shown_rank_norm": round(safe_float(row.get("pair_last_shown_rank_norm", 0.5), 0.5), 4),
                },
                "candidate_source": as_text(row.get("candidate_source")) or "two_tower",
                "shown_rank": idx,
                "recommendation_version": self.recommendation_version,
                "confidence": round(safe_float(row.get("confidence", 0.0), 0.0), 2),
                "is_exploration": False,
                "final_rank": round(safe_float(row.get("final_rank", 0.0), 0.0), 2),
                "reasons": [
                    f"SVD collaborative: {round(safe_float(row.get('collab_score', 0.0), 0.0), 1)}",
                    f"Neural CF: {round(safe_float(row.get('ncf_score', 0.0), 0.0), 1)}",
                    f"Retriever score: {round(safe_float(row.get('retrieval_score', 0.5), 0.5) * 100.0, 1)}",
                    f"Text similarity: {round(safe_float(row.get('text_similarity', 0.5), 0.5) * 100.0, 1)}",
                ],
                "warning": warning,
            }
            card["verification_checklist"] = verification_checklist(card, buyer_row)
            cards.append(card)
        return cards

    def ingest_swipe(
        self,
        buyer_id: str,
        exporter_id: str,
        action: str,
        *,
        ts=None,
        session_id: str | None = None,
        shown_rank: int | None = None,
        source: str | None = None,
        dwell_ms: int | None = None,
        device: str | None = None,
        region: str | None = None,
        recommendation_version: str | None = None,
    ):
        action = as_text(action).lower()
        if action not in {"left", "right"}:
            return
        event_ts = pd.to_datetime(ts, errors="coerce", utc=True)
        if pd.isna(event_ts):
            event_ts = pd.Timestamp.now(tz=timezone.utc)
        row = pd.DataFrame(
            [
                {
                    "buyer_id": as_text(buyer_id),
                    "exporter_id": as_text(exporter_id),
                    "action": action,
                    "ts": event_ts,
                    "session_id": as_text(session_id),
                    "shown_rank": shown_rank,
                    "source": as_text(source).lower() or "unknown",
                    "dwell_ms": dwell_ms,
                    "device": as_text(device).lower(),
                    "region": as_text(region).upper(),
                    "recommendation_version": as_text(recommendation_version) or self.recommendation_version,
                }
            ]
        )
        self.interactions = pd.concat([self.interactions, row], ignore_index=True).tail(self.max_interactions)
        self._rebuild_behavior_memory()
        self.builder.ingest_interaction(
            as_text(buyer_id),
            as_text(exporter_id),
            action,
            row.iloc[0]["ts"],
            shown_rank=row.iloc[0].get("shown_rank"),
            source=row.iloc[0].get("source"),
            dwell_ms=row.iloc[0].get("dwell_ms"),
            session_id=row.iloc[0].get("session_id"),
        )
        self.supervised.update_single(self.builder.single_pair_features(as_text(buyer_id), as_text(exporter_id)), 1 if action == "right" else 0, sample_weight=1.0)

        self._updates_since_full_refresh += 1
        if self._updates_since_full_refresh >= self.online_full_refresh_every:
            self.builder.update_interaction_stats(self.interactions)
            self.retriever.fit(self.interactions)
            self.builder.set_retrieval_scorer(self.retriever.score_pairs if self.retriever.ready else None)
            self.collaborative.fit(self.interactions)
            self.ncf.fit(self.interactions)
            self.supervised.fit(self.interactions, self.builder, crossed_features=None)
            self.ltr.fit(self.interactions, self.builder, crossed_features=None)
            self._candidate_cache = {}
            self._updates_since_full_refresh = 0
            if self.auto_tune_weights and len(self.interactions) >= 1500:
                self._tune_weights_from_history(max_buyers=max(40, self.tune_eval_buyers // 2))
