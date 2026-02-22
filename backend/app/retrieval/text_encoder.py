from __future__ import annotations

import atexit
import hashlib
import os
import pickle

import numpy as np
import pandas as pd

from ..ml.common import (
    CROSS_ENCODER_AVAILABLE,
    SENTENCE_TRANSFORMERS_AVAILABLE,
    SKLEARN_TEXT_AVAILABLE,
    CrossEncoder,
    SentenceTransformer,
    TfidfVectorizer,
    normalize_rows,
)


class TextEmbeddingService:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        teacher_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        prefer_gpu: bool = True,
        enable_teacher: bool = True,
        teacher_cache_path: str | None = None,
        teacher_cache_flush_every: int = 1000,
        teacher_cache_max_entries: int = 500_000,
        teacher_cache_only: bool | None = None,
    ):
        self.model_name = model_name
        self.teacher_model_name = teacher_model_name
        self.prefer_gpu = bool(prefer_gpu)
        self.enable_teacher = bool(enable_teacher)
        self.backend = "none"
        self.ready = False
        self.teacher_backend = "none"
        self.teacher_ready = False
        self.teacher_cache_path = (
            teacher_cache_path
            if teacher_cache_path
            else os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache", "teacher_scores_v1.pkl")
            )
        )
        self.teacher_cache_flush_every = int(max(50, teacher_cache_flush_every))
        self.teacher_cache_max_entries = int(max(10_000, teacher_cache_max_entries))
        self.teacher_cache_loaded_entries = 0
        if teacher_cache_only is None:
            teacher_cache_only = str(os.getenv("TEACHER_CACHE_ONLY", "0")).strip().lower() in {"1", "true", "yes", "y"}
        self.teacher_cache_only = bool(teacher_cache_only)

        self._st_model = None
        self._cross_encoder = None
        self._tfidf = None
        self._buyers = {}
        self._exporters = {}
        self._buyer_text_map = {}
        self._exporter_text_map = {}
        self._teacher_cache = {}
        self._teacher_cache_dirty = 0
        self._dim = 0
        atexit.register(self._save_teacher_cache, True)

    def _text_hash(self, text: str) -> str:
        if not text:
            return "na"
        return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]

    def _teacher_key(self, buyer_id: str, exporter_id: str) -> str:
        bt = self._buyer_text_map.get(str(buyer_id), "")
        et = self._exporter_text_map.get(str(exporter_id), "")
        return (
            f"{self.teacher_model_name}|{str(buyer_id)}|{str(exporter_id)}|"
            f"{self._text_hash(bt)}|{self._text_hash(et)}"
        )

    def _load_teacher_cache(self) -> None:
        self.teacher_cache_loaded_entries = 0
        if not self.enable_teacher or not self.teacher_cache_path:
            self._teacher_cache = {}
            self._teacher_cache_dirty = 0
            return
        path = self.teacher_cache_path
        if not os.path.exists(path):
            self._teacher_cache = {}
            self._teacher_cache_dirty = 0
            return
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            if isinstance(payload, dict) and "scores" in payload and isinstance(payload.get("scores"), dict):
                scores = payload.get("scores", {})
            elif isinstance(payload, dict):
                scores = payload
            else:
                scores = {}
            self._teacher_cache = {str(k): float(v) for k, v in scores.items()}
            self.teacher_cache_loaded_entries = int(len(self._teacher_cache))
            self._teacher_cache_dirty = 0
        except Exception:
            self._teacher_cache = {}
            self._teacher_cache_dirty = 0

    def _save_teacher_cache(self, force: bool = False) -> None:
        if not self.enable_teacher or not self.teacher_cache_path:
            return
        if (not force) and self._teacher_cache_dirty < self.teacher_cache_flush_every:
            return
        try:
            if len(self._teacher_cache) > self.teacher_cache_max_entries:
                # Keep most recent entries (dict preserves insertion order).
                trimmed = list(self._teacher_cache.items())[-self.teacher_cache_max_entries :]
                self._teacher_cache = dict(trimmed)
            out_dir = os.path.dirname(self.teacher_cache_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            tmp_path = f"{self.teacher_cache_path}.tmp"
            payload = {
                "version": 1,
                "teacher_model_name": self.teacher_model_name,
                "scores": self._teacher_cache,
            }
            with open(tmp_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, self.teacher_cache_path)
            self._teacher_cache_dirty = 0
        except Exception:
            # Best-effort cache; ignore IO failures.
            pass

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
        self.teacher_ready = False
        self._buyers = {}
        self._exporters = {}
        self._buyer_text_map = {}
        self._exporter_text_map = {}
        self._teacher_cache = {}
        self._teacher_cache_dirty = 0
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
        self._buyer_text_map = {str(i): buyer_texts[idx] for idx, i in enumerate(b["Buyer_ID"].astype(str).tolist())}
        self._exporter_text_map = {str(i): exporter_texts[idx] for idx, i in enumerate(e["Exporter_ID"].astype(str).tolist())}
        self._load_teacher_cache()
        self._dim = int(emb.shape[1])
        self.ready = True
        self.teacher_ready = bool(self.enable_teacher and len(self._buyer_text_map) > 0 and len(self._exporter_text_map) > 0)

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

    def _load_cross_encoder(self) -> bool:
        if not self.enable_teacher:
            return False
        if self._cross_encoder is not None:
            return True
        if not CROSS_ENCODER_AVAILABLE or CrossEncoder is None:
            return False
        try:
            device = "cuda" if self.prefer_gpu else "cpu"
            self._cross_encoder = CrossEncoder(self.teacher_model_name, device=device)
            self.teacher_backend = "cross_encoder"
            return True
        except Exception:
            self._cross_encoder = None
            return False

    def teacher_scores_for_pairs(self, buyer_ids: list[str], exporter_ids: list[str]) -> np.ndarray:
        n = int(min(len(buyer_ids), len(exporter_ids)))
        if n <= 0:
            return np.zeros(0, dtype=np.float64)

        out = np.full(n, np.nan, dtype=np.float64)
        unresolved_idx = []
        unresolved_pairs = []
        for i in range(n):
            b = str(buyer_ids[i])
            e = str(exporter_ids[i])
            key = self._teacher_key(b, e)
            if key in self._teacher_cache:
                out[i] = float(self._teacher_cache[key])
                continue
            unresolved_idx.append(i)
            unresolved_pairs.append((b, e))

        if unresolved_pairs and (not self.teacher_cache_only) and self._load_cross_encoder():
            try:
                texts = []
                valid_pos = []
                for j, (b, e) in enumerate(unresolved_pairs):
                    bt = self._buyer_text_map.get(b, "")
                    et = self._exporter_text_map.get(e, "")
                    if not bt or not et:
                        continue
                    texts.append([bt, et])
                    valid_pos.append(j)
                if texts:
                    raw = np.asarray(self._cross_encoder.predict(texts), dtype=np.float64).reshape(-1)
                    if raw.size > 0:
                        # Convert logits to probabilities.
                        scores = 1.0 / (1.0 + np.exp(-np.clip(raw, -30.0, 30.0)))
                        for local_idx, score in zip(valid_pos, scores):
                            i = unresolved_idx[local_idx]
                            b, e = unresolved_pairs[local_idx]
                            s = float(np.clip(score, 0.0, 1.0))
                            out[i] = s
                            k = self._teacher_key(b, e)
                            self._teacher_cache[k] = s
                            self._teacher_cache_dirty += 1
            except Exception:
                pass

        # Fallback for unresolved pairs: use bi-encoder similarity.
        unresolved = np.where(~np.isfinite(out))[0]
        if unresolved.size > 0:
            for i in unresolved.tolist():
                b = str(buyer_ids[i])
                e = str(exporter_ids[i])
                s = float(self.similarity_pair(b, e))
                out[i] = s
                k = self._teacher_key(b, e)
                self._teacher_cache[k] = s
                self._teacher_cache_dirty += 1
            if self.teacher_backend == "none":
                self.teacher_backend = self.backend or "bi_encoder_fallback"

        self._save_teacher_cache(False)
        self.teacher_ready = True
        return np.clip(out, 0.0, 1.0).astype(np.float64)

    def teacher_pair_score(self, buyer_id: str, exporter_id: str) -> float:
        scores = self.teacher_scores_for_pairs([str(buyer_id)], [str(exporter_id)])
        if scores.size <= 0:
            return 0.5
        return float(scores[0])

    @property
    def dim(self) -> int:
        return int(self._dim)
