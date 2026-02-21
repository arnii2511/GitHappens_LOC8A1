from __future__ import annotations

import numpy as np

from ..ml.common import NearestNeighbors, SKLEARN_TEXT_AVAILABLE, normalize_rows


class ANNIndex:
    def __init__(self):
        self.ids: np.ndarray = np.array([], dtype=object)
        self.embeddings = np.zeros((0, 1), dtype=np.float64)
        self.index = None
        self.ready = False
        self.backend = "bruteforce"

    def fit(self, exporter_ids: list[str], exporter_embeddings: np.ndarray):
        if exporter_embeddings is None or len(exporter_ids) == 0:
            self.ids = np.array([], dtype=object)
            self.embeddings = np.zeros((0, 1), dtype=np.float64)
            self.index = None
            self.ready = False
            self.backend = "bruteforce"
            return

        emb = np.asarray(exporter_embeddings, dtype=np.float64)
        if emb.ndim != 2 or emb.shape[0] == 0:
            self.ids = np.array([], dtype=object)
            self.embeddings = np.zeros((0, 1), dtype=np.float64)
            self.index = None
            self.ready = False
            self.backend = "bruteforce"
            return

        self.ids = np.asarray([str(x) for x in exporter_ids], dtype=object)
        self.embeddings = normalize_rows(emb)

        if SKLEARN_TEXT_AVAILABLE and NearestNeighbors is not None and len(self.ids) >= 2:
            try:
                self.index = NearestNeighbors(metric="cosine", algorithm="auto")
                self.index.fit(self.embeddings)
                self.ready = True
                self.backend = "sklearn_nn"
                return
            except Exception:
                self.index = None

        self.index = None
        self.ready = True
        self.backend = "bruteforce"

    def search(self, query_embedding: np.ndarray, top_k: int = 200) -> tuple[np.ndarray, np.ndarray]:
        if not self.ready or self.ids.size == 0:
            return np.array([], dtype=object), np.array([], dtype=np.float64)

        q = np.asarray(query_embedding, dtype=np.float64).reshape(1, -1)
        q = normalize_rows(q)
        k = int(max(1, min(top_k, len(self.ids))))

        if self.index is not None:
            distances, indices = self.index.kneighbors(q, n_neighbors=k)
            idx = indices[0]
            ids = self.ids[idx]
            scores = 1.0 - np.asarray(distances[0], dtype=np.float64)
            return ids, np.clip(scores, 0.0, 1.0)

        sims = (self.embeddings @ q[0]).astype(np.float64)
        order = np.argsort(-sims)[:k]
        return self.ids[order], np.clip(sims[order], 0.0, 1.0)
