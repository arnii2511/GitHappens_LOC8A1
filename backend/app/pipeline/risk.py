from datetime import timedelta

import numpy as np
import pandas as pd

from .helpers import impact_weight


def news_risk_penalty(news: pd.DataFrame, industry: str, ref_date: pd.Timestamp, lookback_days: int = 30):
    if pd.isna(ref_date):
        ref_date = news["Date"].max()

    start = ref_date - timedelta(days=lookback_days)
    n = news[(news["Affected_Industry"] == industry) & (news["Date"] >= start) & (news["Date"] <= ref_date)].copy()
    if n.empty:
        return 0.0, None, 0.0

    n["w"] = n["Impact_Level"].apply(impact_weight)

    tariff = np.abs(pd.to_numeric(n.get("Tariff_Change"), errors="coerce").fillna(0))
    stock = np.abs(pd.to_numeric(n.get("StockMarket_Shock"), errors="coerce").fillna(0))
    war = pd.to_numeric(n.get("War_Flag"), errors="coerce").fillna(0)
    calam = pd.to_numeric(n.get("Natural_Calamity_Flag"), errors="coerce").fillna(0)
    fx = np.abs(pd.to_numeric(n.get("Currency_Shift"), errors="coerce").fillna(0))

    n["shock"] = n["w"] * (
        0.30 * tariff +
        0.25 * stock +
        0.20 * war +
        0.15 * calam +
        0.10 * fx
    )

    row = n.sort_values("shock", ascending=False).iloc[0]
    shock_score = float(row["shock"])
    penalty = float(min(30.0, 30.0 * shock_score))
    warn = f"Market risk: {row['Event_Type']} ({row['Impact_Level']})"
    return penalty, warn, shock_score
