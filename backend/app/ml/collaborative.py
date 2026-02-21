import numpy as np
import pandas as pd

from .common import CUPY_AVAILABLE, SKLEARN_AVAILABLE, TruncatedSVD, as_text, cp, normalize_rows, sparse
from .time_decay import compute_time_decay_weights


class CollaborativeModel:
    def __init__(
        self,
        buyer_ids,
        exporter_ids,
        embedding_dim: int = 24,
        random_state: int = 42,
        half_life_days: float = 45.0,
        prefer_gpu: bool = True,
        min_interactions: int = 10,
    ):
        self.embedding_dim = int(max(2, embedding_dim))
        self.random_state = random_state
        self.half_life_days = float(max(1.0, half_life_days))
        self.prefer_gpu = bool(prefer_gpu)
        self.min_interactions = int(max(2, min_interactions))

        self._buyer_pos = {str(b): i for i, b in enumerate(list(buyer_ids))}
        self._exporter_pos = {str(e): i for i, e in enumerate(list(exporter_ids))}
        self.buyer_embeddings = np.zeros((len(self._buyer_pos), 1), dtype=np.float64)
        self.exporter_embeddings = np.zeros((len(self._exporter_pos), 1), dtype=np.float64)
        self.ready = False
        self.device = "cpu"

    def fit(self, interactions: pd.DataFrame):
        self.ready = False
        self.device = "cpu"
        if interactions.empty:
            return

        data = interactions.copy()
        data = data[data["action"].isin(["left", "right"])]
        if len(data) < self.min_interactions:
            return

        b = data["buyer_id"].astype(str).map(self._buyer_pos)
        e = data["exporter_id"].astype(str).map(self._exporter_pos)
        m = b.notna() & e.notna()
        data = data[m].copy()
        if data.empty:
            return

        rows = b[m].astype(np.int64).to_numpy()
        cols = e[m].astype(np.int64).to_numpy()
        base_vals = np.where(data["action"].to_numpy() == "right", 1.0, -0.25).astype(np.float64)
        decay = compute_time_decay_weights(data, ts_col="ts", half_life_days=self.half_life_days)
        vals = base_vals * decay

        n_b, n_e = len(self._buyer_pos), len(self._exporter_pos)
        if n_b < 2 or n_e < 2:
            return

        if self.prefer_gpu and self._fit_gpu(rows, cols, vals, n_b, n_e):
            return

        self._fit_cpu(rows, cols, vals, n_b, n_e)

    def _fit_gpu(self, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, n_b: int, n_e: int) -> bool:
        if not CUPY_AVAILABLE:
            return False
        try:
            k = min(self.embedding_dim, min(n_b, n_e) - 1)
            if k < 2:
                return False

            mat = cp.zeros((n_b, n_e), dtype=cp.float32)
            cp_rows = cp.asarray(rows, dtype=cp.int64)
            cp_cols = cp.asarray(cols, dtype=cp.int64)
            cp_vals = cp.asarray(vals, dtype=cp.float32)
            cp.add.at(mat, (cp_rows, cp_cols), cp_vals)

            u, s, vt = cp.linalg.svd(mat, full_matrices=False)
            buyer_latent = u[:, :k] * s[:k]
            exporter_latent = vt[:k, :].T

            self.buyer_embeddings = normalize_rows(cp.asnumpy(buyer_latent))
            self.exporter_embeddings = normalize_rows(cp.asnumpy(exporter_latent))
            self.ready = True
            self.device = "gpu"
            return True
        except Exception:
            self.ready = False
            self.device = "cpu"
            return False

    def _fit_cpu(self, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, n_b: int, n_e: int):
        if not SKLEARN_AVAILABLE or sparse is None:
            self.ready = False
            self.device = "cpu"
            return

        mat = sparse.coo_matrix((vals, (rows, cols)), shape=(n_b, n_e), dtype=np.float64).tocsr()
        k = min(self.embedding_dim, min(n_b, n_e) - 1)
        if k < 2:
            self.ready = False
            self.device = "cpu"
            return

        svd = TruncatedSVD(n_components=k, random_state=self.random_state)
        buyer_latent = svd.fit_transform(mat)
        exporter_latent = svd.components_.T

        self.buyer_embeddings = normalize_rows(buyer_latent)
        self.exporter_embeddings = normalize_rows(exporter_latent)
        self.ready = True
        self.device = "cpu"

    def score(self, buyer_id: str, exporter_ids: np.ndarray) -> np.ndarray:
        out = np.zeros(len(exporter_ids), dtype=np.float64)
        if not self.ready:
            return out

        b_pos = self._buyer_pos.get(as_text(buyer_id))
        if b_pos is None:
            return out

        b_vec = self.buyer_embeddings[b_pos]
        idx = np.array([self._exporter_pos.get(as_text(x), -1) for x in exporter_ids], dtype=np.int64)
        valid = idx >= 0
        if not np.any(valid):
            return out

        dots = self.exporter_embeddings[idx[valid]] @ b_vec
        out[valid] = np.clip((dots + 1.0) * 0.5, 0.0, 1.0)
        return out
