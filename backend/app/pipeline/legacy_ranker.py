import numpy as np
import pandas as pd

from .checklist import verification_checklist
from .helpers import capacity_fit, safe_float
from .risk import news_risk_penalty


def build_feed_for_buyer(
    buyer_row: pd.Series,
    exporters_feat: pd.DataFrame,
    news: pd.DataFrame,
    top_k: int = 10,
):
    industry = str(buyer_row.get("Industry", "")).strip().lower()
    ref_date = buyer_row.get("Date", pd.NaT)
    buyer_avg = pd.to_numeric(buyer_row.get("Avg_Order_Tons", np.nan), errors="coerce")

    cands = exporters_feat[exporters_feat["Industry"] == industry].copy()
    if cands.empty:
        return []

    cands["cap_fit"] = cands.get("Quantity_Tons", np.nan).apply(lambda q: capacity_fit(q, buyer_avg))
    risk_penalty, risk_warn, shock = news_risk_penalty(news, industry, ref_date)

    preferred = str(buyer_row.get("Preferred_Channel", "")).strip().lower()
    if preferred in {"email", "linkedin"}:
        comm_score, comm_warn = 100.0, None
    elif preferred == "whatsapp":
        comm_score, comm_warn = 70.0, "WhatsApp preferred (higher miscommunication risk)"
    else:
        comm_score, comm_warn = 55.0, "Preferred channel missing (confidence reduced)"

    buyer_trust = safe_float(buyer_row.get("buyer_trust", 0), 0)
    buyer_intent = safe_float(buyer_row.get("buyer_intent", 0), 0)

    cards = []
    for _, ex in cands.iterrows():
        exporter_trust = safe_float(ex.get("exporter_trust", 0), 0)
        exporter_intent = safe_float(ex.get("exporter_intent", 0), 0)

        intent_fit = (buyer_intent / 100.0) * (exporter_intent / 100.0) * 100.0
        pair_trust = 0.5 * buyer_trust + 0.5 * exporter_trust

        match_score = (
            0.45 * safe_float(ex.get("cap_fit", 60), 60) +
            0.30 * intent_fit +
            0.25 * comm_score
        )

        ex_risk = (
            0.30 * abs(safe_float(ex.get("Tariff_Impact", 0), 0)) +
            0.25 * abs(safe_float(ex.get("StockMarket_Impact", 0), 0)) +
            0.20 * safe_float(ex.get("War_Risk", 0), 0) +
            0.15 * safe_float(ex.get("Natural_Calamity_Risk", 0), 0) +
            0.10 * abs(safe_float(ex.get("Currency_Shift", 0), 0))
        )
        ex_risk_penalty = float(min(20.0, 20.0 * ex_risk))
        total_penalty = float(min(30.0, risk_penalty + ex_risk_penalty))
        match_after_risk = float(np.clip(match_score - total_penalty, 0, 100))

        final_rank = float(np.clip(
            0.7 * match_after_risk +
            0.2 * pair_trust +
            0.1 * min(buyer_intent, exporter_intent),
            0, 100,
        ))

        reasons = [
            f"Industry match: {industry}",
            f"Capacity fit: {round(safe_float(ex.get('cap_fit', 0)), 1)}",
            f"Trust: buyer {round(buyer_trust,1)} & exporter {round(exporter_trust,1)}",
            f"Intent: buyer {round(buyer_intent,1)} & exporter {round(exporter_intent,1)}",
        ]
        warning = risk_warn or comm_warn
        card = {
            "buyer_id": buyer_row.get("Buyer_ID"),
            "exporter_id": ex.get("Exporter_ID"),
            "exporter_state": ex.get("State"),
            "exporter_cert": ex.get("Certification"),
            "match_score": round(match_after_risk, 2),
            "trust_score": round(pair_trust, 2),
            "intent_score": round(min(buyer_intent, exporter_intent), 2),
            "risk_penalty": round(total_penalty, 2),
            "news_risk_penalty": round(risk_penalty, 2),
            "exporter_risk_penalty": round(ex_risk_penalty, 2),
            "shock_score": round(shock, 4),
            "final_rank": round(final_rank, 2),
            "reasons": reasons,
            "warning": warning,
        }
        card["verification_checklist"] = verification_checklist(card, buyer_row)
        cards.append(card)

    cards.sort(key=lambda x: x["final_rank"], reverse=True)
    return cards[:top_k]
