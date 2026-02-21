from __future__ import annotations

import numpy as np
import pandas as pd


def get_dynamic_weights(buyer_row: pd.Series, risk_penalty: float) -> dict[str, float]:
    w = {
        "cap_fit": 0.45,
        "intent": 0.30,
        "comm": 0.25,
    }

    buyer_avg = pd.to_numeric(pd.Series([buyer_row.get("Avg_Order_Tons", np.nan)]), errors="coerce").iloc[0]
    if pd.isna(buyer_avg):
        w["cap_fit"] -= 0.15
        w["intent"] += 0.10
        w["comm"] += 0.05

    if float(risk_penalty) >= 10.0:
        w["intent"] -= 0.10
        w["cap_fit"] += 0.05
        w["comm"] += 0.05

    total = float(max(1e-9, sum(w.values())))
    return {k: float(v / total) for k, v in w.items()}
