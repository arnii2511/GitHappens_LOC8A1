from datetime import timedelta

import numpy as np
import pandas as pd

from ..industry_map import canonicalize, industry_similarity
from .helpers import impact_weight


def news_risk_penalty(news: pd.DataFrame, industry: str, ref_date: pd.Timestamp, lookback_days: int = 30):
    if pd.isna(ref_date):
        ref_date = news["Date"].max()

    start = ref_date - timedelta(days=lookback_days)
    buyer_industry = canonicalize(industry)
    if buyer_industry == "unknown":
        return 0.0, None, 0.0

    n = news[(news["Date"] >= start) & (news["Date"] <= ref_date)].copy()
    if n.empty:
        return 0.0, None, 0.0
    if "Affected_Industry" in n.columns:
        affected = n["Affected_Industry"]
    else:
        affected = pd.Series([""] * len(n), index=n.index)
    n["industry_sim"] = affected.apply(lambda x: industry_similarity(buyer_industry, x))
    n = n[n["industry_sim"] > 0.0].copy()
    if n.empty:
        return 0.0, None, 0.0

    n["w"] = n["Impact_Level"].apply(impact_weight)

    tariff = np.abs(pd.to_numeric(n.get("Tariff_Change"), errors="coerce").fillna(0))
    stock = np.abs(pd.to_numeric(n.get("StockMarket_Shock"), errors="coerce").fillna(0))
    war = pd.to_numeric(n.get("War_Flag"), errors="coerce").fillna(0)
    calam = pd.to_numeric(n.get("Natural_Calamity_Flag"), errors="coerce").fillna(0)
    fx = np.abs(pd.to_numeric(n.get("Currency_Shift"), errors="coerce").fillna(0))

    n["shock"] = n["industry_sim"] * n["w"] * (
        0.30 * tariff +
        0.25 * stock +
        0.20 * war +
        0.15 * calam +
        0.10 * fx
    )

    row = n.sort_values("shock", ascending=False).iloc[0]
    shock_score = float(row["shock"])
    penalty = float(min(30.0, 30.0 * shock_score))
    warn = (
        f"Market risk: {row['Event_Type']} ({row['Impact_Level']}) "
        f"[industry_sim={round(float(row['industry_sim']), 2)}]"
    )
    return penalty, warn, shock_score
