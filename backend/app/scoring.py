import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from datetime import timedelta

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR", "../data")

def _cert_score(x: str) -> float:
    if not isinstance(x, str) or x.strip() == "" or x.strip().lower() == "none":
        return 0.0
    x = x.strip().upper()
    if x in {"EU-GMP", "SOC2"}: return 1.0
    if x in {"ISO9001", "ISO14001", "ISO27001"}: return 0.7
    if x in {"IEC", "GDPR", "UL"}: return 0.4
    return 0.3

def _impact_weight(level: str) -> float:
    if not isinstance(level, str): return 0.8
    level = level.strip().lower()
    if level == "high": return 1.0
    if level == "medium": return 0.8
    return 0.6

def load_data():
    buyers = pd.read_csv(os.path.join(DATA_DIR, "EXIM_DatasetAlgo_Hackathon(Importer_LiveSignals_v5_Updated).csv"), encoding="utf-8", engine="python")
    exporters = pd.read_csv(os.path.join(DATA_DIR, "EXIM_DatasetAlgo_Hackathon(Exporter_LiveSignals_v5_Updated).csv"), encoding="utf-8", engine="python")
    news = pd.read_csv(os.path.join(DATA_DIR, "EXIM_DatasetAlgo_Hackathon(Global_News_LiveSignals_Updated).csv"), encoding="utf-8", engine="python")

    for df in (buyers, exporters, news):
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    buyers["Industry"] = buyers["Industry"].astype(str).str.strip()
    exporters["Industry"] = exporters["Industry"].astype(str).str.strip()
    news["Affected_Industry"] = news["Affected_Industry"].astype(str).str.strip()

    # numeric coercion
    buyers["Avg_Order_Tons"] = pd.to_numeric(buyers.get("Avg_Order_Tons"), errors="coerce")
    buyers["Response_Probability"] = pd.to_numeric(buyers.get("Response_Probability"), errors="coerce")

    exporters["Manufacturing_Capacity_Tons"] = pd.to_numeric(exporters.get("Manufacturing_Capacity_Tons"), errors="coerce")
    exporters["Shipment_Value_USD"] = pd.to_numeric(exporters.get("Shipment_Value_USD"), errors="coerce")
    exporters["Quantity_Tons"] = pd.to_numeric(exporters.get("Quantity_Tons"), errors="coerce")

    return buyers, exporters, news

def compute_buyer_scores(buyers: pd.DataFrame) -> pd.DataFrame:
    df = buyers.copy()
    df["cert_score"] = df["Certification"].apply(_cert_score)
    df["rp_filled"] = df["Response_Probability"].fillna(0.35)

    df["buyer_trust"] = 100 * (
        0.40 * df["Good_Payment_History"].fillna(0).astype(float)
        + 0.25 * df["Prompt_Response"].fillna(0).astype(float)
        + 0.20 * df["rp_filled"].astype(float)
        + 0.15 * df["cert_score"].astype(float)
    )

    sn = pd.to_numeric(df["SalesNav_ProfileVisits"], errors="coerce").fillna(0)
    sn_norm = (sn - sn.min()) / (sn.max() - sn.min() + 1e-9)

    def _funding(x):
        if isinstance(x, str) and x.strip().lower() == "unknown":
            return 0.0
        try:
            return float(x)
        except:
            return 0.0

    df["funding_val"] = df["Funding_Event"].apply(_funding)

    df["buyer_intent"] = 100 * (
        0.50 * df["Intent_Score"].fillna(0).astype(float)
        + 0.15 * df["funding_val"].fillna(0).astype(float)
        + 0.10 * df["Engagement_Spike"].fillna(0).astype(float)
        + 0.10 * df["DecisionMaker_Change"].fillna(0).astype(float)
        + 0.10 * df["Hiring_Growth"].fillna(0).astype(float)
        + 0.05 * sn_norm.astype(float)
    )

    df["buyer_trust"] = df["buyer_trust"].clip(0, 100)
    df["buyer_intent"] = df["buyer_intent"].clip(0, 100)
    return df

def compute_exporter_scores(exporters: pd.DataFrame) -> pd.DataFrame:
    df = exporters.copy()
    df["cert_score"] = df["Certification"].apply(_cert_score)

    df["exporter_trust"] = 100 * (
        0.35 * df["Good_Payment_Terms"].fillna(0).astype(float)
        + 0.35 * df["Prompt_Response_Score"].fillna(0).astype(float)
        + 0.30 * df["cert_score"].astype(float)
    )

    li = pd.to_numeric(df["LinkedIn_Activity"], errors="coerce").fillna(0)
    pv = pd.to_numeric(df["SalesNav_ProfileViews"], errors="coerce").fillna(0)
    li_norm = (li - li.min()) / (li.max() - li.min() + 1e-9)
    pv_norm = (pv - pv.min()) / (pv.max() - pv.min() + 1e-9)

    df["exporter_intent"] = 100 * (
        0.55 * df["Intent_Score"].fillna(0).astype(float)
        + 0.15 * df["Hiring_Signal"].fillna(0).astype(float)
        + 0.10 * li_norm.astype(float)
        + 0.10 * pv_norm.astype(float)
        + 0.10 * df["SalesNav_JobChange"].fillna(0).astype(float)
    )

    df["exporter_trust"] = df["exporter_trust"].clip(0, 100)
    df["exporter_intent"] = df["exporter_intent"].clip(0, 100)
    return df

def _news_penalty(news: pd.DataFrame, industry: str, ref_date: pd.Timestamp, lookback_days: int = 30):
    if pd.isna(ref_date):
        ref_date = news["Date"].max()

    start = ref_date - timedelta(days=lookback_days)
    n = news[(news["Affected_Industry"] == industry) & (news["Date"] >= start) & (news["Date"] <= ref_date)].copy()
    if n.empty:
        return 0.0, None

    n["w"] = n["Impact_Level"].apply(_impact_weight)
    n["shock"] = n["w"] * (
        0.30 * np.abs(pd.to_numeric(n["Tariff_Change"], errors="coerce").fillna(0))
        + 0.25 * np.abs(pd.to_numeric(n["StockMarket_Shock"], errors="coerce").fillna(0))
        + 0.20 * pd.to_numeric(n["War_Flag"], errors="coerce").fillna(0)
        + 0.15 * pd.to_numeric(n["Natural_Calamity_Flag"], errors="coerce").fillna(0)
        + 0.10 * np.abs(pd.to_numeric(n["Currency_Shift"], errors="coerce").fillna(0))
    )

    row = n.sort_values("shock", ascending=False).iloc[0]
    penalty = float(min(30.0, 30.0 * row["shock"]))
    warn = f"Market risk: {row['Event_Type']} ({row['Impact_Level']})"
    return penalty, warn

def _capacity_fit(exporter_qty: float, buyer_avg: float) -> float:
    if np.isnan(exporter_qty) or np.isnan(buyer_avg) or buyer_avg <= 0 or exporter_qty <= 0:
        return 60.0
    ratio = exporter_qty / buyer_avg
    return float(np.clip(100.0 * np.exp(-abs(np.log(ratio))), 0, 100))

def build_feed_for_buyer(buyer_row: pd.Series, exporters_scored: pd.DataFrame, news: pd.DataFrame, top_k: int = 10):
    industry = str(buyer_row["Industry"]).strip()
    ref_date = buyer_row["Date"]
    buyer_avg = pd.to_numeric(buyer_row.get("Avg_Order_Tons", np.nan), errors="coerce")

    cands = exporters_scored[exporters_scored["Industry"] == industry].copy()
    if cands.empty:
        return []

    cands["cap_fit"] = cands["Quantity_Tons"].apply(lambda q: _capacity_fit(q, buyer_avg))
    penalty, news_warn = _news_penalty(news, industry, ref_date)

    preferred = str(buyer_row.get("Preferred_Channel", "")).strip().lower()
    if preferred in {"email", "linkedin"}:
        comm_score, comm_warn = 100.0, None
    elif preferred == "whatsapp":
        comm_score, comm_warn = 70.0, "WhatsApp preferred (higher miscommunication risk)"
    else:
        comm_score, comm_warn = 50.0, "Preferred channel missing (confidence reduced)"

    out = []
    for _, ex in cands.iterrows():
        intent_pair = (buyer_row["buyer_intent"]/100.0) * (ex["exporter_intent"]/100.0) * 100.0
        base_match = 0.45 * ex["cap_fit"] + 0.25 * comm_score + 0.30 * intent_pair
        match_score = float(np.clip(base_match - penalty, 0, 100))

        pair_trust = 0.5 * float(buyer_row["buyer_trust"]) + 0.5 * float(ex["exporter_trust"])
        final_rank = 0.70 * match_score + 0.20 * pair_trust + 0.10 * min(float(buyer_row["buyer_intent"]), float(ex["exporter_intent"]))

        reasons = [
            f"Industry match: {industry}",
            f"Capacity fit: {round(float(ex['cap_fit']), 1)}",
            f"Intent fit: buyer {round(float(buyer_row['buyer_intent']),1)} & exporter {round(float(ex['exporter_intent']),1)}"
        ]
        warning = news_warn or comm_warn

        out.append({
            "buyer_id": buyer_row["Buyer_ID"],
            "exporter_id": ex["Exporter_ID"],
            "exporter_state": ex.get("State"),
            "exporter_cert": ex.get("Certification"),
            "match_score": round(match_score, 2),
            "trust_score": round(pair_trust, 2),
            "intent_score": round(min(float(buyer_row["buyer_intent"]), float(ex["exporter_intent"])), 2),
            "reasons": reasons,
            "warning": warning,
            "debug": {"final_rank": round(float(final_rank), 2), "news_penalty": round(float(penalty), 2)}
        })

    out.sort(key=lambda x: x["debug"]["final_rank"], reverse=True)
    return out[:top_k]