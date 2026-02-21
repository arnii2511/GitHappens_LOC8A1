from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..industry_map import canonicalize, industry_similarity
from ..pipeline.dynamic_weights import get_dynamic_weights
from ..pipeline.helpers import capacity_fit, safe_float
from ..pipeline.risk import news_risk_penalty
from .common import as_float_series, as_text


class PairFeatureBuilder:
    def __init__(
        self,
        buyers: pd.DataFrame,
        exporters: pd.DataFrame,
        news: pd.DataFrame,
        candidate_pool_size: int = 900,
        industry_threshold: float = 0.5,
        exploration_pool_ratio: float = 0.22,
        min_core_candidates: int = 140,
    ):
        self.buyers = self._prepare_buyers(buyers)
        self.exporters = self._prepare_exporters(exporters)
        self.news = news.copy()

        self.buyers_idx = self.buyers.set_index("Buyer_ID", drop=False)
        self.exporters_idx = self.exporters.set_index("Exporter_ID", drop=False)
        self._news_cache: Dict[Tuple[str, str], Tuple[float, Optional[str], float]] = {}

        self.candidate_pool_size = int(max(100, candidate_pool_size))
        self.industry_threshold = float(np.clip(industry_threshold, 0.0, 1.0))
        self.exploration_pool_ratio = float(np.clip(exploration_pool_ratio, 0.0, 0.8))
        self.min_core_candidates = int(max(20, min_core_candidates))

        self._history_ref_ts = pd.Timestamp.now(tz="UTC")
        self._buyer_hist_raw: dict[str, dict[str, object]] = {}
        self._exporter_hist_raw: dict[str, dict[str, object]] = {}
        self._pair_hist_raw: dict[tuple[str, str], dict[str, object]] = {}
        self._max_buyer_count = 1.0
        self._max_exporter_count = 1.0
        self._max_pair_count = 1.0
        self.text_encoder = None
        self.retrieval_scorer = None
        self.industry_assoc_lookup = None

    def refresh_news(self, news: pd.DataFrame):
        self.news = news.copy()
        self._news_cache.clear()

    def set_text_encoder(self, text_encoder) -> None:
        self.text_encoder = text_encoder

    def set_retrieval_scorer(self, retrieval_scorer) -> None:
        self.retrieval_scorer = retrieval_scorer

    def set_industry_assoc_lookup(self, industry_assoc_lookup) -> None:
        self.industry_assoc_lookup = industry_assoc_lookup

    def update_interaction_stats(self, interactions: pd.DataFrame):
        self._buyer_hist_raw = {}
        self._exporter_hist_raw = {}
        self._pair_hist_raw = {}
        self._max_buyer_count = 1.0
        self._max_exporter_count = 1.0
        self._max_pair_count = 1.0
        self._history_ref_ts = pd.Timestamp.now(tz="UTC")

        if interactions is None or interactions.empty:
            return

        df = interactions.copy()
        for col in ("buyer_id", "exporter_id", "action"):
            if col not in df.columns:
                return
        if "ts" not in df.columns:
            df["ts"] = pd.Timestamp.now(tz="UTC")

        df["buyer_id"] = df["buyer_id"].astype(str).str.strip()
        df["exporter_id"] = df["exporter_id"].astype(str).str.strip()
        df["action"] = df["action"].astype(str).str.strip().str.lower()
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
        df = df[df["action"].isin(["left", "right"])]
        df = df[(df["buyer_id"] != "") & (df["exporter_id"] != "")]
        df = df.dropna(subset=["ts"])
        if df.empty:
            return

        self._history_ref_ts = pd.to_datetime(df["ts"].max(), utc=True)
        df["right"] = (df["action"] == "right").astype(np.float64)

        buyer_grp = (
            df.groupby("buyer_id", sort=False)
            .agg(count=("action", "size"), right_sum=("right", "sum"), last_ts=("ts", "max"))
            .reset_index()
        )
        for _, row in buyer_grp.iterrows():
            self._buyer_hist_raw[str(row["buyer_id"])] = {
                "count": int(row["count"]),
                "right_sum": float(row["right_sum"]),
                "last_ts": pd.to_datetime(row["last_ts"], utc=True),
            }
        if not buyer_grp.empty:
            self._max_buyer_count = float(max(1.0, buyer_grp["count"].max()))

        exporter_grp = (
            df.groupby("exporter_id", sort=False)
            .agg(count=("action", "size"), right_sum=("right", "sum"), last_ts=("ts", "max"))
            .reset_index()
        )
        for _, row in exporter_grp.iterrows():
            self._exporter_hist_raw[str(row["exporter_id"])] = {
                "count": int(row["count"]),
                "right_sum": float(row["right_sum"]),
                "last_ts": pd.to_datetime(row["last_ts"], utc=True),
            }
        if not exporter_grp.empty:
            self._max_exporter_count = float(max(1.0, exporter_grp["count"].max()))

        pair_grp = (
            df.groupby(["buyer_id", "exporter_id"], sort=False)
            .agg(count=("action", "size"), right_sum=("right", "sum"), last_ts=("ts", "max"))
            .reset_index()
        )
        for _, row in pair_grp.iterrows():
            self._pair_hist_raw[(str(row["buyer_id"]), str(row["exporter_id"]))] = {
                "count": int(row["count"]),
                "right_sum": float(row["right_sum"]),
                "last_ts": pd.to_datetime(row["last_ts"], utc=True),
            }
        if not pair_grp.empty:
            self._max_pair_count = float(max(1.0, pair_grp["count"].max()))

    def ingest_interaction(self, buyer_id: str, exporter_id: str, action: str, ts) -> None:
        buyer_id = as_text(buyer_id)
        exporter_id = as_text(exporter_id)
        action = as_text(action).lower()
        if buyer_id == "" or exporter_id == "" or action not in {"left", "right"}:
            return

        ts = pd.to_datetime(ts, errors="coerce", utc=True)
        if pd.isna(ts):
            ts = pd.Timestamp.now(tz="UTC")
        if ts > self._history_ref_ts:
            self._history_ref_ts = ts
        right_inc = 1.0 if action == "right" else 0.0

        b = self._buyer_hist_raw.get(buyer_id, {"count": 0, "right_sum": 0.0, "last_ts": ts})
        b["count"] = int(b["count"]) + 1
        b["right_sum"] = float(b["right_sum"]) + right_inc
        b["last_ts"] = max(pd.to_datetime(b["last_ts"], utc=True), ts)
        self._buyer_hist_raw[buyer_id] = b
        self._max_buyer_count = float(max(self._max_buyer_count, float(b["count"])))

        e = self._exporter_hist_raw.get(exporter_id, {"count": 0, "right_sum": 0.0, "last_ts": ts})
        e["count"] = int(e["count"]) + 1
        e["right_sum"] = float(e["right_sum"]) + right_inc
        e["last_ts"] = max(pd.to_datetime(e["last_ts"], utc=True), ts)
        self._exporter_hist_raw[exporter_id] = e
        self._max_exporter_count = float(max(self._max_exporter_count, float(e["count"])))

        pair_key = (buyer_id, exporter_id)
        p = self._pair_hist_raw.get(pair_key, {"count": 0, "right_sum": 0.0, "last_ts": ts})
        p["count"] = int(p["count"]) + 1
        p["right_sum"] = float(p["right_sum"]) + right_inc
        p["last_ts"] = max(pd.to_datetime(p["last_ts"], utc=True), ts)
        self._pair_hist_raw[pair_key] = p
        self._max_pair_count = float(max(self._max_pair_count, float(p["count"])))

    def _prepare_buyers(self, buyers: pd.DataFrame) -> pd.DataFrame:
        df = buyers.copy()
        if "Buyer_ID" not in df.columns:
            raise ValueError("buyers data must include Buyer_ID")
        df["Buyer_ID"] = df["Buyer_ID"].astype(str).str.strip()
        df = df[df["Buyer_ID"] != ""].drop_duplicates("Buyer_ID", keep="last").reset_index(drop=True)
        if "Industry" in df.columns:
            df["Industry"] = df["Industry"].astype(str).str.strip().str.lower()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df

    def _prepare_exporters(self, exporters: pd.DataFrame) -> pd.DataFrame:
        df = exporters.copy()
        if "Exporter_ID" not in df.columns:
            raise ValueError("exporters data must include Exporter_ID")
        df["Exporter_ID"] = df["Exporter_ID"].astype(str).str.strip()
        df = df[df["Exporter_ID"] != ""].drop_duplicates("Exporter_ID", keep="last").reset_index(drop=True)
        if "Industry" in df.columns:
            df["Industry"] = df["Industry"].astype(str).str.strip().str.lower()
        return df

    def _bucket_exporters(self, exporters: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        buckets: Dict[str, pd.DataFrame] = {}
        if "Industry" not in exporters.columns:
            return buckets
        for industry, part in exporters.groupby("Industry", sort=False):
            key = as_text(industry).lower()
            if not key:
                continue
            buckets[key] = part.reset_index(drop=True)
        return buckets

    def _comm_score(self, preferred_channel: str) -> Tuple[float, Optional[str]]:
        p = as_text(preferred_channel).lower()
        if p in {"email", "linkedin"}:
            return 100.0, None
        if p == "whatsapp":
            return 70.0, "WhatsApp preferred (higher miscommunication risk)"
        return 55.0, "Preferred channel missing (confidence reduced)"

    def _cached_news_penalty(self, industry: str, ref_date) -> Tuple[float, Optional[str], float]:
        d = pd.to_datetime(ref_date, errors="coerce")
        d_key = "na" if pd.isna(d) else d.normalize().strftime("%Y-%m-%d")
        key = (as_text(industry).lower(), d_key)
        if key in self._news_cache:
            return self._news_cache[key]
        result = news_risk_penalty(self.news, key[0], d)
        self._news_cache[key] = result
        return result

    def _vector_capacity_fit(self, exporter_qty: np.ndarray, buyer_avg: float) -> np.ndarray:
        out = np.full(exporter_qty.shape[0], 60.0, dtype=np.float64)
        buyer_avg = safe_float(buyer_avg, np.nan)
        if np.isnan(buyer_avg) or buyer_avg <= 0:
            return out
        q = np.asarray(exporter_qty, dtype=np.float64)
        mask = np.isfinite(q) & (q > 0)
        if not np.any(mask):
            return out
        ratio = np.clip(q[mask] / buyer_avg, 1e-12, None)
        out[mask] = np.clip(100.0 * np.exp(-np.abs(np.log(ratio))), 0.0, 100.0)
        return out

    def _norm_count(self, count: float, max_count: float) -> float:
        count = float(max(0.0, count))
        denom = np.log1p(max(1.0, float(max_count)))
        if denom <= 0.0:
            return 0.0
        return float(np.clip(np.log1p(count) / denom, 0.0, 1.0))

    def _norm_days(self, days: float) -> float:
        if not np.isfinite(days):
            return 1.0
        return float(np.clip(days / 180.0, 0.0, 1.0))

    def _stat_to_features(self, stat: Optional[dict], max_count: float, default_right_rate: float = 0.5) -> tuple[float, float, float]:
        if not stat:
            return 0.0, float(default_right_rate), 1.0
        count = float(stat.get("count", 0.0))
        right_sum = float(stat.get("right_sum", 0.0))
        last_ts = pd.to_datetime(stat.get("last_ts"), errors="coerce", utc=True)
        right_rate = right_sum / max(1.0, count)
        if pd.isna(last_ts):
            days = 365.0
        else:
            days = float(max(0.0, (self._history_ref_ts - last_ts).total_seconds() / 86400.0))
        return self._norm_count(count, max_count), float(np.clip(right_rate, 0.0, 1.0)), self._norm_days(days)

    def _candidate_pool_for_buyer(self, buyer_industry: str, buyer_avg: float) -> pd.DataFrame:
        exporters = self.exporters.copy()
        if exporters.empty:
            return exporters

        if "Industry" in exporters.columns:
            exporters["industry_sim"] = exporters["Industry"].apply(lambda x: industry_similarity(buyer_industry, x))
        else:
            exporters["industry_sim"] = 0.0

        qty = as_float_series(exporters, "Quantity_Tons", np.nan)
        cap_fit_quick = self._vector_capacity_fit(qty, buyer_avg)
        exporter_trust = as_float_series(exporters, "exporter_trust", 0.0)
        exporter_intent = as_float_series(exporters, "exporter_intent", 0.0)
        pre_score = (
            0.42 * (exporters["industry_sim"].to_numpy(dtype=np.float64) * 100.0)
            + 0.23 * cap_fit_quick
            + 0.20 * exporter_intent
            + 0.15 * exporter_trust
        )
        exporters["pre_score"] = pre_score

        core = exporters[exporters["industry_sim"] >= self.industry_threshold].copy()
        core["candidate_source"] = "core"
        core_target = int(round(self.candidate_pool_size * (1.0 - self.exploration_pool_ratio)))
        core_target = max(self.min_core_candidates, core_target)
        core_pick = core.nlargest(min(len(core), core_target), "pre_score")

        remaining_quota = max(0, self.candidate_pool_size - len(core_pick))
        if remaining_quota <= 0:
            return core_pick.reset_index(drop=True)

        remainder = exporters[~exporters["Exporter_ID"].isin(core_pick["Exporter_ID"])].copy()
        if remainder.empty:
            return core_pick.reset_index(drop=True)

        remainder["candidate_source"] = "explore"
        remainder["pre_score"] = 0.65 * remainder["pre_score"] + 35.0 * (
            1.0 - remainder["industry_sim"].to_numpy(dtype=np.float64)
        )
        explore_pick = remainder.nlargest(min(len(remainder), remaining_quota), "pre_score")

        selected = pd.concat([core_pick, explore_pick], ignore_index=True)
        if selected.empty:
            selected = exporters.nlargest(min(len(exporters), self.candidate_pool_size), "pre_score").copy()
            selected["candidate_source"] = "fallback"
        return selected.drop_duplicates("Exporter_ID", keep="first").reset_index(drop=True)

    def candidate_features_for_buyer(
        self,
        buyer_row: pd.Series,
        retrieval_candidates: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        industry = canonicalize(as_text(buyer_row.get("Industry", "")))
        if industry == "unknown":
            return pd.DataFrame(), None

        buyer_id = as_text(buyer_row.get("Buyer_ID"))
        buyer_avg = safe_float(buyer_row.get("Avg_Order_Tons", np.nan), np.nan)
        buyer_trust = safe_float(buyer_row.get("buyer_trust", 0.0), 0.0)
        buyer_intent = safe_float(buyer_row.get("buyer_intent", 0.0), 0.0)
        ref_date = buyer_row.get("Date", pd.NaT)
        comm_score, comm_warn = self._comm_score(buyer_row.get("Preferred_Channel", ""))
        news_penalty, risk_warn, shock = self._cached_news_penalty(industry, ref_date)
        w_match = get_dynamic_weights(buyer_row, news_penalty)

        retrieval_map = {}
        if retrieval_candidates is not None and not retrieval_candidates.empty:
            rc = retrieval_candidates.copy()
            if "exporter_id" in rc.columns:
                rc["exporter_id"] = rc["exporter_id"].astype(str)
                if "retrieval_score" not in rc.columns:
                    rc["retrieval_score"] = 0.5
                if "retrieval_rank_norm" not in rc.columns:
                    rc["retrieval_rank_norm"] = 0.5
                if "industry_assoc_score" not in rc.columns:
                    rc["industry_assoc_score"] = 0.0
                if "industry_assoc_hit" not in rc.columns:
                    rc["industry_assoc_hit"] = 0.0
                if "candidate_source" not in rc.columns:
                    rc["candidate_source"] = "retrieval"
                retrieval_map = {
                    str(r["exporter_id"]): (
                        float(r.get("retrieval_score", 0.5)),
                        float(r.get("retrieval_rank_norm", 0.5)),
                        float(r.get("industry_assoc_score", 0.0)),
                        float(r.get("industry_assoc_hit", 0.0)),
                        str(r.get("candidate_source", "retrieval")),
                    )
                    for _, r in rc.iterrows()
                }
        if retrieval_map:
            exporters = self.exporters[self.exporters["Exporter_ID"].astype(str).isin(list(retrieval_map.keys()))].copy()
            exporters["candidate_source"] = "retrieval"
            if "Industry" in exporters.columns:
                exporters["industry_sim"] = exporters["Industry"].apply(lambda x: industry_similarity(industry, x))
            else:
                exporters["industry_sim"] = 0.0
        else:
            exporters = self._candidate_pool_for_buyer(industry, buyer_avg)
        if exporters.empty:
            return pd.DataFrame(), None

        exporter_qty = as_float_series(exporters, "Quantity_Tons", np.nan)
        exporter_trust = as_float_series(exporters, "exporter_trust", 0.0)
        exporter_intent = as_float_series(exporters, "exporter_intent", 0.0)
        exporter_state = exporters["State"].astype(str).to_numpy() if "State" in exporters.columns else np.array([""] * len(exporters))
        exporter_cert = exporters["Certification"].astype(str).to_numpy() if "Certification" in exporters.columns else np.array([""] * len(exporters))
        industry_sim = as_float_series(exporters, "industry_sim", 0.0)

        cap_fit = self._vector_capacity_fit(exporter_qty, buyer_avg)
        intent_fit = (buyer_intent / 100.0) * (exporter_intent / 100.0) * 100.0
        pair_trust = 0.5 * (buyer_trust + exporter_trust)

        ex_risk = (
            0.30 * np.abs(as_float_series(exporters, "Tariff_Impact", 0.0))
            + 0.25 * np.abs(as_float_series(exporters, "StockMarket_Impact", 0.0))
            + 0.20 * as_float_series(exporters, "War_Risk", 0.0)
            + 0.15 * as_float_series(exporters, "Natural_Calamity_Risk", 0.0)
            + 0.10 * np.abs(as_float_series(exporters, "Currency_Shift", 0.0))
        )
        ex_risk_penalty = np.minimum(20.0, 20.0 * ex_risk)
        total_penalty = np.minimum(30.0, news_penalty + ex_risk_penalty)

        non_industry_match = (
            w_match["cap_fit"] * cap_fit
            + w_match["intent"] * intent_fit
            + w_match["comm"] * comm_score
        )
        base_match = 0.80 * non_industry_match + 0.20 * (industry_sim * 100.0)
        match_after_risk = np.clip(base_match - total_penalty, 0.0, 100.0)

        buyer_count_n, buyer_right_rate, buyer_days_n = self._stat_to_features(
            self._buyer_hist_raw.get(buyer_id),
            self._max_buyer_count,
            default_right_rate=0.5,
        )
        exporter_count_n = np.zeros(len(exporters), dtype=np.float64)
        exporter_right_rate = np.full(len(exporters), 0.5, dtype=np.float64)
        exporter_days_n = np.ones(len(exporters), dtype=np.float64)
        pair_count_n = np.zeros(len(exporters), dtype=np.float64)
        pair_right_rate = np.full(len(exporters), buyer_right_rate, dtype=np.float64)
        pair_days_n = np.ones(len(exporters), dtype=np.float64)

        exporter_ids = exporters["Exporter_ID"].astype(str).tolist()
        for i, ex_id in enumerate(exporter_ids):
            e_count_n, e_right_rate, e_days_n = self._stat_to_features(
                self._exporter_hist_raw.get(ex_id),
                self._max_exporter_count,
                default_right_rate=0.5,
            )
            exporter_count_n[i] = e_count_n
            exporter_right_rate[i] = e_right_rate
            exporter_days_n[i] = e_days_n

            p_count_n, p_right_rate, p_days_n = self._stat_to_features(
                self._pair_hist_raw.get((buyer_id, ex_id)),
                self._max_pair_count,
                default_right_rate=buyer_right_rate,
            )
            pair_count_n[i] = p_count_n
            pair_right_rate[i] = p_right_rate
            pair_days_n[i] = p_days_n

        if retrieval_map:
            retrieval_score = np.array([retrieval_map.get(x, (0.5, 0.5, 0.0, 0.0, "retrieval"))[0] for x in exporter_ids], dtype=np.float64)
            retrieval_rank_norm = np.array([retrieval_map.get(x, (0.5, 0.5, 0.0, 0.0, "retrieval"))[1] for x in exporter_ids], dtype=np.float64)
            industry_assoc_score = np.array([retrieval_map.get(x, (0.5, 0.5, 0.0, 0.0, "retrieval"))[2] for x in exporter_ids], dtype=np.float64)
            industry_assoc_hit = np.array([retrieval_map.get(x, (0.5, 0.5, 0.0, 0.0, "retrieval"))[3] for x in exporter_ids], dtype=np.float64)
            candidate_source = np.array([retrieval_map.get(x, (0.5, 0.5, 0.0, 0.0, "retrieval"))[4] for x in exporter_ids], dtype=object)
        elif self.retrieval_scorer is not None:
            retrieval_score = np.asarray(self.retrieval_scorer(buyer_id, exporter_ids), dtype=np.float64)
            retrieval_rank_norm = np.linspace(1.0, 0.2, len(exporter_ids), dtype=np.float64)
            industry_assoc_score = np.zeros(len(exporter_ids), dtype=np.float64)
            industry_assoc_hit = np.zeros(len(exporter_ids), dtype=np.float64)
            candidate_source = np.full(len(exporter_ids), "retrieval", dtype=object)
            if self.industry_assoc_lookup is not None:
                for i, ex_id in enumerate(exporter_ids):
                    s, h, src = self.industry_assoc_lookup(buyer_id, ex_id)
                    industry_assoc_score[i] = float(s)
                    industry_assoc_hit[i] = float(h)
                    candidate_source[i] = str(src)
        else:
            retrieval_score = np.full(len(exporters), 0.5, dtype=np.float64)
            retrieval_rank_norm = np.full(len(exporters), 0.5, dtype=np.float64)
            industry_assoc_score = np.zeros(len(exporters), dtype=np.float64)
            industry_assoc_hit = np.zeros(len(exporters), dtype=np.float64)
            candidate_source = exporters.get("candidate_source", "core").astype(str).to_numpy()

        if self.text_encoder is not None:
            text_similarity = np.asarray(self.text_encoder.similarity_scores(buyer_id, exporter_ids), dtype=np.float64)
        else:
            text_similarity = np.asarray(industry_sim, dtype=np.float64)

        warning = risk_warn or comm_warn
        out = pd.DataFrame(
            {
                "buyer_id": buyer_id,
                "exporter_id": np.array(exporter_ids, dtype=object),
                "exporter_state": exporter_state,
                "exporter_cert": exporter_cert,
                "industry": industry,
                "candidate_source": candidate_source,
                "industry_match": industry_sim,
                "industry_similarity": industry_sim,
                "industry_assoc_score": industry_assoc_score,
                "industry_assoc_hit": industry_assoc_hit,
                "cap_fit": cap_fit,
                "intent_fit": intent_fit,
                "pair_trust": pair_trust,
                "min_intent": np.minimum(buyer_intent, exporter_intent),
                "buyer_intent": np.full(len(exporters), buyer_intent, dtype=np.float64),
                "exporter_intent": exporter_intent,
                "comm_score": np.full(len(exporters), comm_score, dtype=np.float64),
                "news_risk_penalty": np.full(len(exporters), news_penalty, dtype=np.float64),
                "exporter_risk_penalty": ex_risk_penalty,
                "total_risk_penalty": total_penalty,
                "shock_score": np.full(len(exporters), shock, dtype=np.float64),
                "match_after_risk": match_after_risk,
                "buyer_trust": np.full(len(exporters), buyer_trust, dtype=np.float64),
                "exporter_trust": exporter_trust,
                "w_cap_fit": np.full(len(exporters), w_match["cap_fit"], dtype=np.float64),
                "w_intent": np.full(len(exporters), w_match["intent"], dtype=np.float64),
                "w_comm": np.full(len(exporters), w_match["comm"], dtype=np.float64),
                "buyer_swipe_count_norm": np.full(len(exporters), buyer_count_n, dtype=np.float64),
                "buyer_right_rate": np.full(len(exporters), buyer_right_rate, dtype=np.float64),
                "buyer_days_since_last": np.full(len(exporters), buyer_days_n, dtype=np.float64),
                "exporter_swipe_count_norm": exporter_count_n,
                "exporter_right_rate": exporter_right_rate,
                "exporter_days_since_last": exporter_days_n,
                "pair_interaction_count_norm": pair_count_n,
                "pair_right_rate": pair_right_rate,
                "pair_days_since_last": pair_days_n,
                "retrieval_score": retrieval_score,
                "retrieval_rank_norm": retrieval_rank_norm,
                "text_similarity": text_similarity,
            }
        )
        return out, warning

    def single_pair_features(self, buyer_id: str, exporter_id: str) -> Optional[pd.DataFrame]:
        if buyer_id not in self.buyers_idx.index or exporter_id not in self.exporters_idx.index:
            return None

        buyer = self.buyers_idx.loc[buyer_id]
        exporter = self.exporters_idx.loc[exporter_id]

        buyer_industry = canonicalize(as_text(buyer.get("Industry", "")))
        exporter_industry = canonicalize(as_text(exporter.get("Industry", "")))
        industry_match = industry_similarity(buyer_industry, exporter_industry)

        buyer_trust = safe_float(buyer.get("buyer_trust", 0.0), 0.0)
        buyer_intent = safe_float(buyer.get("buyer_intent", 0.0), 0.0)
        exporter_trust = safe_float(exporter.get("exporter_trust", 0.0), 0.0)
        exporter_intent = safe_float(exporter.get("exporter_intent", 0.0), 0.0)

        cap = capacity_fit(exporter.get("Quantity_Tons", np.nan), buyer.get("Avg_Order_Tons", np.nan))
        intent_fit = (buyer_intent / 100.0) * (exporter_intent / 100.0) * 100.0
        pair_trust = 0.5 * (buyer_trust + exporter_trust)
        comm_score, _ = self._comm_score(buyer.get("Preferred_Channel", ""))

        n_penalty, _, shock = self._cached_news_penalty(buyer_industry, buyer.get("Date", pd.NaT))
        w_match = get_dynamic_weights(buyer, n_penalty)
        ex_risk = (
            0.30 * abs(safe_float(exporter.get("Tariff_Impact", 0), 0))
            + 0.25 * abs(safe_float(exporter.get("StockMarket_Impact", 0), 0))
            + 0.20 * safe_float(exporter.get("War_Risk", 0), 0)
            + 0.15 * safe_float(exporter.get("Natural_Calamity_Risk", 0), 0)
            + 0.10 * abs(safe_float(exporter.get("Currency_Shift", 0), 0))
        )
        ex_penalty = float(min(20.0, 20.0 * ex_risk))
        total_penalty = float(min(30.0, n_penalty + ex_penalty))
        non_industry_match = (
            w_match["cap_fit"] * cap
            + w_match["intent"] * intent_fit
            + w_match["comm"] * comm_score
        )
        base_match = 0.80 * non_industry_match + 0.20 * (industry_match * 100.0)
        match_after_risk = float(np.clip(base_match - total_penalty, 0.0, 100.0))

        buyer_count_n, buyer_right_rate, buyer_days_n = self._stat_to_features(
            self._buyer_hist_raw.get(as_text(buyer_id)),
            self._max_buyer_count,
            default_right_rate=0.5,
        )
        exporter_count_n, exporter_right_rate, exporter_days_n = self._stat_to_features(
            self._exporter_hist_raw.get(as_text(exporter_id)),
            self._max_exporter_count,
            default_right_rate=0.5,
        )
        pair_count_n, pair_right_rate, pair_days_n = self._stat_to_features(
            self._pair_hist_raw.get((as_text(buyer_id), as_text(exporter_id))),
            self._max_pair_count,
            default_right_rate=buyer_right_rate,
        )
        if self.retrieval_scorer is not None:
            retrieval_score = float(np.asarray(self.retrieval_scorer(as_text(buyer_id), [as_text(exporter_id)]), dtype=np.float64)[0])
        else:
            retrieval_score = 0.5
        retrieval_rank_norm = 0.5
        industry_assoc_score = 0.0
        industry_assoc_hit = 0.0
        candidate_source = "industry_other"
        if self.industry_assoc_lookup is not None:
            industry_assoc_score, industry_assoc_hit, candidate_source = self.industry_assoc_lookup(
                as_text(buyer_id), as_text(exporter_id)
            )
        if self.text_encoder is not None:
            text_similarity = float(self.text_encoder.similarity_pair(as_text(buyer_id), as_text(exporter_id)))
        else:
            text_similarity = float(np.clip(industry_match, 0.0, 1.0))

        return pd.DataFrame(
            [
                {
                    "industry_match": industry_match,
                    "industry_similarity": industry_match,
                    "cap_fit": cap,
                    "intent_fit": intent_fit,
                    "pair_trust": pair_trust,
                    "min_intent": min(buyer_intent, exporter_intent),
                    "buyer_intent": buyer_intent,
                    "exporter_intent": exporter_intent,
                    "comm_score": comm_score,
                    "news_risk_penalty": n_penalty,
                    "exporter_risk_penalty": ex_penalty,
                    "total_risk_penalty": total_penalty,
                    "shock_score": shock,
                    "match_after_risk": match_after_risk,
                    "buyer_trust": buyer_trust,
                    "exporter_trust": exporter_trust,
                    "w_cap_fit": w_match["cap_fit"],
                    "w_intent": w_match["intent"],
                    "w_comm": w_match["comm"],
                    "buyer_swipe_count_norm": buyer_count_n,
                    "buyer_right_rate": buyer_right_rate,
                    "buyer_days_since_last": buyer_days_n,
                    "exporter_swipe_count_norm": exporter_count_n,
                    "exporter_right_rate": exporter_right_rate,
                    "exporter_days_since_last": exporter_days_n,
                    "pair_interaction_count_norm": pair_count_n,
                    "pair_right_rate": pair_right_rate,
                    "pair_days_since_last": pair_days_n,
                    "retrieval_score": retrieval_score,
                    "retrieval_rank_norm": retrieval_rank_norm,
                    "text_similarity": text_similarity,
                    "industry_assoc_score": float(industry_assoc_score),
                    "industry_assoc_hit": float(industry_assoc_hit),
                    "candidate_source": str(candidate_source),
                }
            ]
        )
