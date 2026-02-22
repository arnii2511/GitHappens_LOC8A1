import os
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

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

try:
    from lightgbm import LGBMRanker

    LIGHTGBM_AVAILABLE = True
except Exception:
    LGBMRanker = None
    LIGHTGBM_AVAILABLE = False

try:
    from xgboost import XGBClassifier, XGBRanker

    XGBOOST_AVAILABLE = True
except Exception:
    XGBClassifier = None
    XGBRanker = None
    XGBOOST_AVAILABLE = False

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except Exception:
    cp = None
    CUPY_AVAILABLE = False

try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.feature_extraction.text import TfidfVectorizer

    SKLEARN_TEXT_AVAILABLE = True
except Exception:
    NearestNeighbors = None
    TfidfVectorizer = None
    SKLEARN_TEXT_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
    CROSS_ENCODER_AVAILABLE = True
except Exception:
    SentenceTransformer = None
    CrossEncoder = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    CROSS_ENCODER_AVAILABLE = False


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
