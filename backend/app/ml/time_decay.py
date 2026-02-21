import numpy as np
import pandas as pd


def compute_time_decay_weights(
    df: pd.DataFrame,
    ts_col: str = "ts",
    half_life_days: float = 45.0,
    min_weight: float = 0.05,
) -> np.ndarray:
    if df is None or df.empty:
        return np.zeros(0, dtype=np.float64)

    if ts_col not in df.columns:
        return np.ones(len(df), dtype=np.float64)

    ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    now = pd.Timestamp.utcnow()
    age_days = ((now - ts).dt.total_seconds() / 86400.0).to_numpy(dtype=np.float64)
    age_days = np.where(np.isfinite(age_days) & (age_days > 0.0), age_days, 0.0)

    decay = np.exp(-np.log(2.0) * age_days / float(max(1e-6, half_life_days)))
    return np.clip(decay, float(min_weight), 1.0).astype(np.float64)
