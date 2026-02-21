import numpy as np
import pandas as pd

try:
    from scipy import sparse
    from sklearn.decomposition import TruncatedSVD
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:
    sparse = None
    TruncatedSVD = None
    SGDClassifier = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False


def as_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def as_float_series(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), default, dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).to_numpy(dtype=np.float64)


def normalize_rows(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    return arr / norm
