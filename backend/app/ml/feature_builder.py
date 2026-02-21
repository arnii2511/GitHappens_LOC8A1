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
    def __init__(self, buyers: pd.DataFrame, exporters: pd.DataFrame, news: pd.DataFrame):
        self.buyers = self._prepare_buyers(buyers)
        self.exporters = self._prepare_exporters(exporters)
        self.news = news.copy()

        self.buyers_idx = self.buyers.set_index("Buyer_ID", drop=False)
        self.exporters_idx = self.exporters.set_index("Exporter_ID", drop=False)
        self.exporters_by_industry = self._bucket_exporters(self.exporters)
        self._news_cache: Dict[Tuple[str, str], Tuple[float, Optional[str], float]] = {}

    def refresh_news(self, news: pd.DataFrame):
        self.news = news.copy()
        self._news_cache.clear()

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

    def candidate_features_for_buyer(self, buyer_row: pd.Series) -> Tuple[pd.DataFrame, Optional[str]]:
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

        exporters = self.exporters.reset_index(drop=True).copy()
        if "Industry" in exporters.columns:
            exporters["industry_sim"] = exporters["Industry"].apply(lambda x: industry_similarity(industry, x))
        else:
            exporters["industry_sim"] = 0.0
        exporters = exporters[exporters["industry_sim"] >= 0.5].reset_index(drop=True)
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

        warning = risk_warn or comm_warn
        out = pd.DataFrame(
            {
                "buyer_id": buyer_id,
                "exporter_id": exporters["Exporter_ID"].astype(str).to_numpy(),
                "exporter_state": exporter_state,
                "exporter_cert": exporter_cert,
                "industry": industry,
                "industry_match": industry_sim,
                "industry_similarity": industry_sim,
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
                }
            ]
        )
