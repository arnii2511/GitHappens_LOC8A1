from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..industry_map import canonicalize
from ..ml.common import F, SKLEARN_AVAILABLE, StandardScaler, TORCH_AVAILABLE, nn, normalize_rows, torch
from ..ml.time_decay import compute_time_decay_weights
from .ann_index import ANNIndex
from .industry_rules import IndustryAssociationMiner
from .text_encoder import TextEmbeddingService


@dataclass
class _TwoTowerConfig:
    embedding_dim: int = 32
    hidden_dim: int = 96
    epochs: int = 6
    batch_size: int = 2048
    lr: float = 1e-3
    neg_per_pos: int = 1


class _Tower(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class TwoTowerRetriever:
    def __init__(
        self,
        buyers: pd.DataFrame,
        exporters: pd.DataFrame,
        text_encoder: TextEmbeddingService | None = None,
        prefer_gpu: bool = True,
        embedding_dim: int = 32,
        hidden_dim: int = 96,
        epochs: int = 6,
        batch_size: int = 2048,
        lr: float = 1e-3,
        neg_per_pos: int = 1,
    ):
        self.buyers = buyers.dropna(subset=["Buyer_ID"]).copy()
        self.exporters = exporters.dropna(subset=["Exporter_ID"]).copy()
        self.buyers["Buyer_ID"] = self.buyers["Buyer_ID"].astype(str).str.strip()
        self.exporters["Exporter_ID"] = self.exporters["Exporter_ID"].astype(str).str.strip()
        self.buyers = self.buyers[self.buyers["Buyer_ID"] != ""].drop_duplicates("Buyer_ID", keep="last")
        self.exporters = self.exporters[self.exporters["Exporter_ID"] != ""].drop_duplicates("Exporter_ID", keep="last")

        self.text_encoder = text_encoder
        self.prefer_gpu = bool(prefer_gpu)
        self.cfg = _TwoTowerConfig(
            embedding_dim=int(max(8, embedding_dim)),
            hidden_dim=int(max(16, hidden_dim)),
            epochs=int(max(2, epochs)),
            batch_size=int(max(128, batch_size)),
            lr=float(max(1e-5, lr)),
            neg_per_pos=int(max(0, neg_per_pos)),
        )

        self._buyer_ids = self.buyers["Buyer_ID"].astype(str).tolist()
        self._exporter_ids = self.exporters["Exporter_ID"].astype(str).tolist()
        self._buyer_pos = {x: i for i, x in enumerate(self._buyer_ids)}
        self._exporter_pos = {x: i for i, x in enumerate(self._exporter_ids)}
        self._buyer_industry = {
            str(r["Buyer_ID"]): canonicalize(str(r.get("Industry", "")))
            for _, r in self.buyers.iterrows()
        }
        self._exporter_industry = {
            str(r["Exporter_ID"]): canonicalize(str(r.get("Industry", "")))
            for _, r in self.exporters.iterrows()
        }
        self._industry_to_exporters: dict[str, list[str]] = {}
        for ex_id, ind in self._exporter_industry.items():
            self._industry_to_exporters.setdefault(ind, []).append(ex_id)
        self._assoc = IndustryAssociationMiner()

        self._buyer_scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self._exporter_scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self._buyer_matrix = np.zeros((len(self._buyer_ids), 1), dtype=np.float64)
        self._exporter_matrix = np.zeros((len(self._exporter_ids), 1), dtype=np.float64)
        self._buyer_emb = np.zeros((len(self._buyer_ids), self.cfg.embedding_dim), dtype=np.float64)
        self._exporter_emb = np.zeros((len(self._exporter_ids), self.cfg.embedding_dim), dtype=np.float64)

        self._ann = ANNIndex()
        self.ready = False
        self.backend = "none"
        self.device = "cpu"
        self.model = None

    def _to_num(self, df: pd.DataFrame, cols: list[str], default: float = 0.0) -> np.ndarray:
        arr = []
        for c in cols:
            if c in df.columns:
                s = pd.to_numeric(df[c], errors="coerce").fillna(default).to_numpy(dtype=np.float64)
            else:
                s = np.full(len(df), default, dtype=np.float64)
            arr.append(s)
        if not arr:
            return np.zeros((len(df), 1), dtype=np.float64)
        return np.column_stack(arr)

    def _build_feature_matrices(self):
        buyer_cols = [
            "buyer_trust",
            "buyer_intent",
            "Avg_Order_Tons",
            "Revenue_Size_USD",
            "Team_Size",
            "Response_Probability",
            "Prompt_Response",
            "Hiring_Growth",
            "Engagement_Spike",
            "Tariff_News",
            "StockMarket_Shock",
            "War_Event",
            "Natural_Calamity",
            "Currency_Fluctuation",
        ]
        exporter_cols = [
            "exporter_trust",
            "exporter_intent",
            "Quantity_Tons",
            "Manufacturing_Capacity_Tons",
            "Revenue_Size_USD",
            "Team_Size",
            "Prompt_Response_Score",
            "Hiring_Signal",
            "Tariff_Impact",
            "StockMarket_Impact",
            "War_Risk",
            "Natural_Calamity_Risk",
            "Currency_Shift",
        ]
        b = self._to_num(self.buyers, buyer_cols, default=0.0)
        e = self._to_num(self.exporters, exporter_cols, default=0.0)

        if self._buyer_scaler is not None:
            b = self._buyer_scaler.fit_transform(b)
        if self._exporter_scaler is not None:
            e = self._exporter_scaler.fit_transform(e)

        if self.text_encoder is not None and self.text_encoder.ready and self.text_encoder.dim > 0:
            b_txt = np.zeros((len(self._buyer_ids), self.text_encoder.dim), dtype=np.float64)
            e_txt = np.zeros((len(self._exporter_ids), self.text_encoder.dim), dtype=np.float64)
            for i, bid in enumerate(self._buyer_ids):
                b_txt[i] = self.text_encoder._buyers.get(bid, np.zeros(self.text_encoder.dim, dtype=np.float64))
            for i, eid in enumerate(self._exporter_ids):
                e_txt[i] = self.text_encoder._exporters.get(eid, np.zeros(self.text_encoder.dim, dtype=np.float64))
            b = np.hstack([b, b_txt])
            e = np.hstack([e, e_txt])

        self._buyer_matrix = np.asarray(b, dtype=np.float64)
        self._exporter_matrix = np.asarray(e, dtype=np.float64)

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
        df["action"] = df["action"].astype(str).str.lower().str.strip()
        df = df[df["action"].isin(["left", "right"])]
        df = df[(df["buyer_id"] != "") & (df["exporter_id"] != "")]
        df = df[df["buyer_id"].isin(self._buyer_pos) & df["exporter_id"].isin(self._exporter_pos)]
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
        df = df.dropna(subset=["ts"])
        return df[["buyer_id", "exporter_id", "action", "ts"]]

    def _build_pairs(self, interactions: pd.DataFrame):
        if interactions.empty:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        df = interactions.copy()
        b_idx = df["buyer_id"].map(self._buyer_pos).astype(np.int64).to_numpy()
        e_idx = df["exporter_id"].map(self._exporter_pos).astype(np.int64).to_numpy()
        y = (df["action"] == "right").astype(np.float64).to_numpy()
        w = compute_time_decay_weights(df, ts_col="ts", half_life_days=45.0).astype(np.float64)

        if self.cfg.neg_per_pos > 0:
            pos_rows = df[df["action"] == "right"]
            if not pos_rows.empty:
                rng = np.random.default_rng(42)
                neg_b = []
                neg_e = []
                for _, r in pos_rows.iterrows():
                    b = self._buyer_pos.get(str(r["buyer_id"]))
                    if b is None:
                        continue
                    for _ in range(self.cfg.neg_per_pos):
                        neg_ex = rng.integers(0, len(self._exporter_ids))
                        neg_b.append(b)
                        neg_e.append(int(neg_ex))
                if neg_b:
                    b_idx = np.concatenate([b_idx, np.asarray(neg_b, dtype=np.int64)])
                    e_idx = np.concatenate([e_idx, np.asarray(neg_e, dtype=np.int64)])
                    y = np.concatenate([y, np.zeros(len(neg_b), dtype=np.float64)])
                    w = np.concatenate([w, np.full(len(neg_b), 0.6, dtype=np.float64)])
        return b_idx, e_idx, y, w

    def _fit_torch(self, b_idx: np.ndarray, e_idx: np.ndarray, y: np.ndarray, w: np.ndarray) -> bool:
        if not TORCH_AVAILABLE or torch is None or nn is None or F is None:
            return False
        if b_idx.size < 200:
            return False
        try:
            use_gpu = self.prefer_gpu and torch.cuda.is_available()
            device = torch.device("cuda" if use_gpu else "cpu")

            b_x = torch.tensor(self._buyer_matrix, dtype=torch.float32, device=device)
            e_x = torch.tensor(self._exporter_matrix, dtype=torch.float32, device=device)
            b_i = torch.tensor(b_idx, dtype=torch.long, device=device)
            e_i = torch.tensor(e_idx, dtype=torch.long, device=device)
            y_t = torch.tensor(y, dtype=torch.float32, device=device)
            w_t = torch.tensor(w, dtype=torch.float32, device=device)

            b_tower = _Tower(self._buyer_matrix.shape[1], self.cfg.hidden_dim, self.cfg.embedding_dim).to(device)
            e_tower = _Tower(self._exporter_matrix.shape[1], self.cfg.hidden_dim, self.cfg.embedding_dim).to(device)
            params = list(b_tower.parameters()) + list(e_tower.parameters())
            opt = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=1e-4)

            n = b_i.shape[0]
            batch_size = int(min(self.cfg.batch_size, n))
            for _ in range(self.cfg.epochs):
                perm = torch.randperm(n, device=device)
                for start in range(0, n, batch_size):
                    idx = perm[start : start + batch_size]
                    bb = F.normalize(b_tower(b_x[b_i[idx]]), dim=1)
                    ee = F.normalize(e_tower(e_x[e_i[idx]]), dim=1)
                    logits = torch.sum(bb * ee, dim=1) * 10.0
                    loss_raw = F.binary_cross_entropy_with_logits(logits, y_t[idx], reduction="none")
                    loss = torch.mean(loss_raw * w_t[idx])

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            with torch.no_grad():
                b_emb = F.normalize(b_tower(b_x), dim=1).detach().cpu().numpy().astype(np.float64)
                e_emb = F.normalize(e_tower(e_x), dim=1).detach().cpu().numpy().astype(np.float64)

            self._buyer_emb = b_emb
            self._exporter_emb = e_emb
            self.model = {"buyer": b_tower, "exporter": e_tower}
            self.device = "gpu" if use_gpu else "cpu"
            self.backend = "two_tower_torch"
            return True
        except Exception:
            self.model = None
            return False

    def _fit_fallback(self):
        # Feature-cosine fallback when deep training is not possible.
        self._buyer_emb = normalize_rows(self._buyer_matrix)
        self._exporter_emb = normalize_rows(self._exporter_matrix)
        self.backend = "feature_cosine"
        self.device = "cpu"

    def fit(self, interactions: pd.DataFrame):
        self.ready = False
        self._build_feature_matrices()
        clean = self._sanitize_interactions(interactions)
        b_idx, e_idx, y, w = self._build_pairs(clean)

        trained = self._fit_torch(b_idx, e_idx, y, w)
        if not trained:
            self._fit_fallback()

        # Fit cross-industry association rules on positive interactions.
        pos = clean[clean["action"] == "right"]
        pairs = []
        if not pos.empty:
            for _, r in pos.iterrows():
                b = self._buyer_industry.get(str(r["buyer_id"]), "unknown")
                e = self._exporter_industry.get(str(r["exporter_id"]), "unknown")
                pairs.append((b, e))
        self._assoc.fit(pairs)

        self._ann.fit(self._exporter_ids, self._exporter_emb)
        self.ready = bool(self._ann.ready)

    def score_pairs(self, buyer_id: str, exporter_ids: list[str]) -> np.ndarray:
        if not self.ready or len(exporter_ids) == 0:
            return np.full(len(exporter_ids), 0.5, dtype=np.float64)
        b_pos = self._buyer_pos.get(str(buyer_id))
        if b_pos is None:
            return np.full(len(exporter_ids), 0.5, dtype=np.float64)

        b = self._buyer_emb[b_pos]
        out = np.full(len(exporter_ids), 0.5, dtype=np.float64)
        for i, ex_id in enumerate(exporter_ids):
            e_pos = self._exporter_pos.get(str(ex_id))
            if e_pos is None:
                continue
            out[i] = float(np.clip(np.dot(b, self._exporter_emb[e_pos]), -1.0, 1.0))
        return np.clip((out + 1.0) * 0.5, 0.0, 1.0)

    def retrieve_for_buyer(self, buyer_id: str, top_k: int = 200) -> pd.DataFrame:
        if not self.ready:
            return pd.DataFrame(
                columns=[
                    "exporter_id",
                    "retrieval_score",
                    "retrieval_rank_norm",
                    "industry_assoc_score",
                    "industry_assoc_hit",
                    "candidate_source",
                ]
            )
        b_pos = self._buyer_pos.get(str(buyer_id))
        if b_pos is None:
            return pd.DataFrame(
                columns=[
                    "exporter_id",
                    "retrieval_score",
                    "retrieval_rank_norm",
                    "industry_assoc_score",
                    "industry_assoc_hit",
                    "candidate_source",
                ]
            )

        buyer_cluster = self._buyer_industry.get(str(buyer_id), "unknown")
        assoc_scores = self._assoc.associated_exporter_clusters(buyer_cluster) if self._assoc.ready else {}

        candidate_ids: list[str] = []
        if buyer_cluster in self._industry_to_exporters:
            candidate_ids.extend(self._industry_to_exporters.get(buyer_cluster, []))
        for cluster in assoc_scores.keys():
            candidate_ids.extend(self._industry_to_exporters.get(cluster, []))

        candidate_ids = list(dict.fromkeys([str(x) for x in candidate_ids if str(x) in self._exporter_pos]))

        # Fallback: if association expansion empty, use ANN global retrieval.
        if not candidate_ids:
            ids, scores = self._ann.search(self._buyer_emb[b_pos], top_k=top_k)
            if len(ids) == 0:
                return pd.DataFrame(
                    columns=[
                        "exporter_id",
                        "retrieval_score",
                        "retrieval_rank_norm",
                        "industry_assoc_score",
                        "industry_assoc_hit",
                        "candidate_source",
                    ]
                )
            ranks = np.arange(1, len(ids) + 1, dtype=np.float64)
            rank_norm = 1.0 - (ranks - 1.0) / max(1.0, float(len(ids) - 1))
            return pd.DataFrame(
                {
                    "exporter_id": ids.astype(str),
                    "retrieval_score": np.asarray(scores, dtype=np.float64),
                    "retrieval_rank_norm": np.asarray(rank_norm, dtype=np.float64),
                    "industry_assoc_score": np.zeros(len(ids), dtype=np.float64),
                    "industry_assoc_hit": np.zeros(len(ids), dtype=np.float64),
                    "candidate_source": np.full(len(ids), "ann_fallback", dtype=object),
                }
            )

        b_vec = self._buyer_emb[b_pos]
        idx = np.asarray([self._exporter_pos[x] for x in candidate_ids], dtype=np.int64)
        sims = np.clip(self._exporter_emb[idx] @ b_vec, -1.0, 1.0)
        scores = np.clip((sims + 1.0) * 0.5, 0.0, 1.0)
        order = np.argsort(-scores)[: int(max(1, min(top_k, len(candidate_ids))))]
        ids = np.asarray(candidate_ids, dtype=object)[order]
        scores = scores[order]

        assoc_score_arr = np.zeros(len(ids), dtype=np.float64)
        assoc_hit_arr = np.zeros(len(ids), dtype=np.float64)
        source_arr = np.full(len(ids), "industry_direct", dtype=object)
        for i, ex_id in enumerate(ids.astype(str)):
            ex_cluster = self._exporter_industry.get(str(ex_id), "unknown")
            if ex_cluster == buyer_cluster:
                assoc_score_arr[i] = 1.0
                assoc_hit_arr[i] = 0.0
                source_arr[i] = "industry_direct"
            elif ex_cluster in assoc_scores:
                assoc_score_arr[i] = float(np.clip(assoc_scores.get(ex_cluster, 0.0), 0.0, 3.0) / 3.0)
                assoc_hit_arr[i] = 1.0
                source_arr[i] = "industry_assoc"
            else:
                assoc_score_arr[i] = 0.0
                assoc_hit_arr[i] = 0.0
                source_arr[i] = "industry_other"

        ranks = np.arange(1, len(ids) + 1, dtype=np.float64)
        rank_norm = 1.0 - (ranks - 1.0) / max(1.0, float(len(ids) - 1))
        return pd.DataFrame(
            {
                "exporter_id": ids.astype(str),
                "retrieval_score": np.asarray(scores, dtype=np.float64),
                "retrieval_rank_norm": np.asarray(rank_norm, dtype=np.float64),
                "industry_assoc_score": assoc_score_arr,
                "industry_assoc_hit": assoc_hit_arr,
                "candidate_source": source_arr,
            }
        )

    def industry_assoc_for_pair(self, buyer_id: str, exporter_id: str) -> tuple[float, float, str]:
        b = self._buyer_industry.get(str(buyer_id), "unknown")
        e = self._exporter_industry.get(str(exporter_id), "unknown")
        if b == "unknown" or e == "unknown":
            return 0.0, 0.0, "unknown"
        if b == e:
            return 1.0, 0.0, "industry_direct"
        assoc = self._assoc.associated_exporter_clusters(b) if self._assoc.ready else {}
        if e in assoc:
            s = float(np.clip(assoc.get(e, 0.0), 0.0, 3.0) / 3.0)
            return s, 1.0, "industry_assoc"
        return 0.0, 0.0, "industry_other"
