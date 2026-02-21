import numpy as np
import pandas as pd


def safe_float(x, default=0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mn, mx = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx) or mx - mn == 0:
        return pd.Series([0.0] * len(series), index=series.index)
    return (s - mn) / (mx - mn)


def cert_score(x: str) -> float:
    if not isinstance(x, str) or x.strip() == "" or x.strip().lower() == "none":
        return 0.0
    x = x.strip().upper()
    if x in {"EU-GMP", "SOC2"}:
        return 1.0
    if x in {"ISO9001", "ISO14001", "ISO27001"}:
        return 0.7
    if x in {"IEC", "GDPR", "UL"}:
        return 0.4
    return 0.3


def impact_weight(level: str) -> float:
    if not isinstance(level, str):
        return 0.8
    level = level.strip().lower()
    if level == "high":
        return 1.0
    if level == "medium":
        return 0.8
    return 0.6


def capacity_fit(exporter_qty: float, buyer_avg: float) -> float:
    exporter_qty = safe_float(exporter_qty, default=np.nan)
    buyer_avg = safe_float(buyer_avg, default=np.nan)
    if np.isnan(exporter_qty) or np.isnan(buyer_avg) or buyer_avg <= 0 or exporter_qty <= 0:
        return 60.0
    ratio = exporter_qty / buyer_avg
    return float(np.clip(100.0 * np.exp(-abs(np.log(ratio))), 0, 100))
