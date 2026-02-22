from __future__ import annotations

import numpy as np
import pandas as pd

from ..industry_map import RELATED, canonicalize
from .common import SKLEARN_AVAILABLE, TruncatedSVD, normalize_rows, sparse
from .time_decay import compute_time_decay_weights


class GraphFeatureService:
    def __init__(self, buyers: pd.DataFrame, exporters: pd.DataFrame, embedding_dim: int = 24):
        self.embedding_dim = int(max(8, embedding_dim))

        b = buyers.dropna(subset=["Buyer_ID"]).copy()
        e = exporters.dropna(subset=["Exporter_ID"]).copy()
        b["Buyer_ID"] = b["Buyer_ID"].astype(str).str.strip()
        e["Exporter_ID"] = e["Exporter_ID"].astype(str).str.strip()
        b = b[b["Buyer_ID"] != ""].drop_duplicates("Buyer_ID", keep="last")
        e = e[e["Exporter_ID"] != ""].drop_duplicates("Exporter_ID", keep="last")

        self._buyer_ids = b["Buyer_ID"].astype(str).tolist()
        self._exporter_ids = e["Exporter_ID"].astype(str).tolist()
        self._buyer_pos = {x: i for i, x in enumerate(self._buyer_ids)}
        self._exporter_pos = {x: i for i, x in enumerate(self._exporter_ids)}
        self._exporter_industry = {
            str(r["Exporter_ID"]): canonicalize(str(r.get("Industry", "")))
            for _, r in e.iterrows()
        }

        self._buyer_emb = np.zeros((len(self._buyer_ids), self.embedding_dim), dtype=np.float64)
        self._exporter_emb = np.zeros((len(self._exporter_ids), self.embedding_dim), dtype=np.float64)
        self.ready = False
        self.backend = "none"

    def _sanitize_interactions(self, interactions: pd.DataFrame) -> pd.DataFrame:
        if interactions is None or interactions.empty:
            return pd.DataFrame(columns=["buyer_id", "exporter_id", "action", "ts"])
        df = interactions.copy()
        for col in ("buyer_id", "exporter_id", "action"):
            if col not in df.columns:
                return pd.DataFrame(columns=["buyer_id", "exporter_id", "action", "ts"])
        if "ts" not in df.columns:
            df["ts"] = pd.Timestamp.now(tz="UTC")
        df["buyer_id"] = df["buyer_id"].astype(str).str.strip()
        df["exporter_id"] = df["exporter_id"].astype(str).str.strip()
        df["action"] = df["action"].astype(str).str.strip().str.lower()
        df = df[df["action"].isin(["left", "right"])]
        df = df[(df["buyer_id"] != "") & (df["exporter_id"] != "")]
        df = df[df["buyer_id"].isin(self._buyer_pos) & df["exporter_id"].isin(self._exporter_pos)]
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
        return df.dropna(subset=["ts"])[["buyer_id", "exporter_id", "action", "ts"]]

    def fit(self, interactions: pd.DataFrame):
        self.ready = False
        clean = self._sanitize_interactions(interactions)
        if clean.empty:
            return

        b_idx = clean["buyer_id"].map(self._buyer_pos).astype(np.int64).to_numpy()
        e_idx = clean["exporter_id"].map(self._exporter_pos).astype(np.int64).to_numpy()
        decay = compute_time_decay_weights(clean, ts_col="ts", half_life_days=60.0).astype(np.float64)
        signal = np.where(clean["action"].to_numpy(dtype=object) == "right", 1.0, 0.15).astype(np.float64)
        w = np.clip(decay * signal, 1e-4, None)

        n_b, n_e = len(self._buyer_ids), len(self._exporter_ids)
        if n_b <= 1 or n_e <= 1:
            return

        if SKLEARN_AVAILABLE and sparse is not None and TruncatedSVD is not None:
            try:
                mat = sparse.coo_matrix((w, (b_idx, e_idx)), shape=(n_b, n_e)).tocsr()
                k = int(max(2, min(self.embedding_dim, min(n_b, n_e) - 1)))
                svd = TruncatedSVD(n_components=k, random_state=42)
                b_lat = svd.fit_transform(mat)
                e_lat = svd.components_.T
                self._buyer_emb = normalize_rows(np.asarray(b_lat, dtype=np.float64))
                self._exporter_emb = normalize_rows(np.asarray(e_lat, dtype=np.float64))
                self._industry_smoothing()
                self.backend = "svd_graph"
                self.ready = True
                return
            except Exception:
                pass

        # Fallback: weighted interaction profiles.
        b_prof = np.zeros((n_b, n_e), dtype=np.float64)
        b_prof[b_idx, e_idx] += w
        e_prof = b_prof.T
        self._buyer_emb = normalize_rows(b_prof[:, : min(n_e, self.embedding_dim)])
        self._exporter_emb = normalize_rows(e_prof[:, : min(n_b, self.embedding_dim)])
        self._industry_smoothing()
        self.backend = "profile_graph"
        self.ready = True

    def _industry_smoothing(self):
        if self._exporter_emb.size == 0:
            return
        ind_to_idxs: dict[str, list[int]] = {}
        for ex_id, idx in self._exporter_pos.items():
            ind = self._exporter_industry.get(str(ex_id), "unknown")
            ind_to_idxs.setdefault(ind, []).append(int(idx))
        ind_centroid = {}
        for ind, idxs in ind_to_idxs.items():
            emb = self._exporter_emb[np.asarray(idxs, dtype=np.int64)]
            if emb.size > 0:
                ind_centroid[ind] = np.mean(emb, axis=0)
        if not ind_centroid:
            return
        out = self._exporter_emb.copy()
        for ex_id, idx in self._exporter_pos.items():
            ind = self._exporter_industry.get(str(ex_id), "unknown")
            base = out[int(idx)]
            mix = np.zeros_like(base)
            mix_w = 0.0
            for rel_ind, rel_w in RELATED.get(ind, {}).items():
                if rel_ind == ind:
                    continue
                c = ind_centroid.get(rel_ind)
                if c is None:
                    continue
                w = float(np.clip(rel_w, 0.0, 1.0))
                mix += c * w
                mix_w += w
            if mix_w > 0:
                out[int(idx)] = 0.90 * base + 0.10 * (mix / mix_w)
        self._exporter_emb = normalize_rows(out)

    def score_pairs(self, buyer_id: str, exporter_ids: list[str]) -> np.ndarray:
        if not self.ready or len(exporter_ids) == 0:
            return np.full(len(exporter_ids), 0.5, dtype=np.float64)
        b_pos = self._buyer_pos.get(str(buyer_id))
        if b_pos is None:
            return np.full(len(exporter_ids), 0.5, dtype=np.float64)
        b = self._buyer_emb[int(b_pos)]
        out = np.full(len(exporter_ids), 0.5, dtype=np.float64)
        for i, ex_id in enumerate(exporter_ids):
            e_pos = self._exporter_pos.get(str(ex_id))
            if e_pos is None:
                continue
            out[i] = float(np.clip(np.dot(b, self._exporter_emb[int(e_pos)]), -1.0, 1.0))
        return np.clip((out + 1.0) * 0.5, 0.0, 1.0)

    def score_pair(self, buyer_id: str, exporter_id: str) -> float:
        return float(self.score_pairs(str(buyer_id), [str(exporter_id)])[0])
