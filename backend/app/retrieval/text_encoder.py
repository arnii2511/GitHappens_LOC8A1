from __future__ import annotations

import numpy as np
import pandas as pd

from ..ml.common import (
    SENTENCE_TRANSFORMERS_AVAILABLE,
    SKLEARN_TEXT_AVAILABLE,
    SentenceTransformer,
    TfidfVectorizer,
    normalize_rows,
)


class TextEmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", prefer_gpu: bool = True):
        self.model_name = model_name
        self.prefer_gpu = bool(prefer_gpu)
        self.backend = "none"
        self.ready = False

        self._st_model = None
        self._tfidf = None
        self._buyers = {}
        self._exporters = {}
        self._dim = 0

    def _buyer_text(self, row: pd.Series) -> str:
        parts = [
            str(row.get("Industry", "")),
            str(row.get("Country", "")),
            str(row.get("Certification", "")),
            str(row.get("Preferred_Channel", "")),
        ]
        return " | ".join(p.strip() for p in parts if str(p).strip())

    def _exporter_text(self, row: pd.Series) -> str:
        parts = [
            str(row.get("Industry", "")),
            str(row.get("State", "")),
            str(row.get("Certification", "")),
        ]
        return " | ".join(p.strip() for p in parts if str(p).strip())

    def fit(self, buyers: pd.DataFrame, exporters: pd.DataFrame):
        self.ready = False
        self._buyers = {}
        self._exporters = {}
        self._dim = 0

        b = buyers.dropna(subset=["Buyer_ID"]).copy()
        e = exporters.dropna(subset=["Exporter_ID"]).copy()
        if b.empty or e.empty:
            return

        b["Buyer_ID"] = b["Buyer_ID"].astype(str).str.strip()
        e["Exporter_ID"] = e["Exporter_ID"].astype(str).str.strip()
        b = b[b["Buyer_ID"] != ""]
        e = e[e["Exporter_ID"] != ""]
        if b.empty or e.empty:
            return

        buyer_texts = b.apply(self._buyer_text, axis=1).tolist()
        exporter_texts = e.apply(self._exporter_text, axis=1).tolist()
        all_texts = buyer_texts + exporter_texts
        if len(all_texts) == 0:
            return

        emb = self._encode(all_texts)
        if emb is None or emb.size == 0:
            return

        emb = normalize_rows(emb)
        b_mat = emb[: len(buyer_texts)]
        e_mat = emb[len(buyer_texts) :]

        self._buyers = {str(i): b_mat[idx] for idx, i in enumerate(b["Buyer_ID"].astype(str).tolist())}
        self._exporters = {str(i): e_mat[idx] for idx, i in enumerate(e["Exporter_ID"].astype(str).tolist())}
        self._dim = int(emb.shape[1])
        self.ready = True

    def _encode(self, texts: list[str]) -> np.ndarray | None:
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                if self._st_model is None:
                    device = "cuda" if self.prefer_gpu else "cpu"
                    self._st_model = SentenceTransformer(self.model_name, device=device)
                mat = self._st_model.encode(
                    texts,
                    batch_size=128,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                )
                self.backend = "sentence_transformer"
                return np.asarray(mat, dtype=np.float64)
            except Exception:
                self._st_model = None

        if SKLEARN_TEXT_AVAILABLE and TfidfVectorizer is not None:
            try:
                self._tfidf = TfidfVectorizer(max_features=512, ngram_range=(1, 2), min_df=1)
                mat = self._tfidf.fit_transform(texts).toarray()
                self.backend = "tfidf"
                return np.asarray(mat, dtype=np.float64)
            except Exception:
                self._tfidf = None
                return None
        return None

    def similarity_scores(self, buyer_id: str, exporter_ids: list[str]) -> np.ndarray:
        if not self.ready or len(exporter_ids) == 0:
            return np.full(len(exporter_ids), 0.5, dtype=np.float64)
        b = self._buyers.get(str(buyer_id))
        if b is None:
            return np.full(len(exporter_ids), 0.5, dtype=np.float64)

        out = np.full(len(exporter_ids), 0.5, dtype=np.float64)
        for i, ex_id in enumerate(exporter_ids):
            ev = self._exporters.get(str(ex_id))
            if ev is None:
                continue
            out[i] = float(np.clip(np.dot(b, ev), -1.0, 1.0))
        return np.clip((out + 1.0) * 0.5, 0.0, 1.0)

    def similarity_pair(self, buyer_id: str, exporter_id: str) -> float:
        return float(self.similarity_scores(str(buyer_id), [str(exporter_id)])[0])

    @property
    def dim(self) -> int:
        return int(self._dim)
