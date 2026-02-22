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
    hard_negative_ratio: float = 0.70
    logq_alpha: float = 0.75
    neg_cache_topk: int = 2000
    distill_weight: float = 0.20


_TowerBase = nn.Module if nn is not None else object


class _Tower(_TowerBase):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        if nn is None:
            raise RuntimeError("PyTorch nn is not available.")
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        if nn is None:
            raise RuntimeError("PyTorch nn is not available.")
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
        hard_negative_ratio: float = 0.70,
        logq_alpha: float = 0.75,
        neg_cache_topk: int = 2000,
        distill_weight: float = 0.20,
        enable_hard_negatives: bool = True,
        enable_logq_correction: bool = True,
        enable_distillation: bool = True,
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
            hard_negative_ratio=float(np.clip(hard_negative_ratio, 0.0, 1.0)),
            logq_alpha=float(np.clip(logq_alpha, 0.1, 1.0)),
            neg_cache_topk=int(max(64, neg_cache_topk)),
            distill_weight=float(np.clip(distill_weight, 0.0, 1.0)),
        )
        self.enable_hard_negatives = bool(enable_hard_negatives)
        self.enable_logq_correction = bool(enable_logq_correction)
        self.enable_distillation = bool(enable_distillation)

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
        n_exporters = max(1, len(self._exporter_ids))
        self._exporter_sampling_prob = np.full(n_exporters, 1.0 / n_exporters, dtype=np.float64)
        self.ready = False
        self.backend = "none"
        self.device = "cpu"
        self.model = None

    def _build_exporter_sampling_prob(self, interactions: pd.DataFrame) -> np.ndarray:
        n = len(self._exporter_ids)
        if n <= 0:
            return np.array([], dtype=np.float64)
        counts = np.ones(n, dtype=np.float64)
        if interactions is not None and not interactions.empty and "exporter_id" in interactions.columns:
            vc = interactions["exporter_id"].astype(str).value_counts()
            for ex_id, cnt in vc.items():
                idx = self._exporter_pos.get(str(ex_id))
                if idx is not None:
                    counts[int(idx)] += float(cnt)
        probs = np.power(counts, self.cfg.logq_alpha)
        s = float(np.sum(probs))
        if not np.isfinite(s) or s <= 0.0:
            return np.full(n, 1.0 / n, dtype=np.float64)
        return (probs / s).astype(np.float64)

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
        self._exporter_sampling_prob = self._build_exporter_sampling_prob(df)
        b_idx = df["buyer_id"].map(self._buyer_pos).astype(np.int64).to_numpy()
        e_idx = df["exporter_id"].map(self._exporter_pos).astype(np.int64).to_numpy()
        y = (df["action"] == "right").astype(np.float64).to_numpy()
        w = compute_time_decay_weights(df, ts_col="ts", half_life_days=45.0).astype(np.float64)

        if self.cfg.neg_per_pos > 0:
            pos_rows = df[df["action"] == "right"]
            if not pos_rows.empty:
                rng = np.random.default_rng(42)
                buyer_pos_sets: dict[int, set[int]] = {}
                for _, r in pos_rows.iterrows():
                    b = self._buyer_pos.get(str(r["buyer_id"]))
                    e = self._exporter_pos.get(str(r["exporter_id"]))
                    if b is None or e is None:
                        continue
                    buyer_pos_sets.setdefault(int(b), set()).add(int(e))

                buyer_hard_pool: dict[int, np.ndarray] = {}
                for b, pos_set in buyer_pos_sets.items():
                    buyer_id = self._buyer_ids[int(b)] if 0 <= int(b) < len(self._buyer_ids) else ""
                    b_cluster = self._buyer_industry.get(str(buyer_id), "unknown")
                    assoc = self._assoc.associated_exporter_clusters(b_cluster) if self._assoc.ready else {}
                    pool_ids: list[int] = []
                    for ex_id in self._industry_to_exporters.get(b_cluster, []):
                        e = self._exporter_pos.get(str(ex_id))
                        if e is not None and int(e) not in pos_set:
                            pool_ids.append(int(e))
                    for c in assoc.keys():
                        for ex_id in self._industry_to_exporters.get(c, []):
                            e = self._exporter_pos.get(str(ex_id))
                            if e is not None and int(e) not in pos_set:
                                pool_ids.append(int(e))
                    if pool_ids:
                        buyer_hard_pool[int(b)] = np.asarray(sorted(set(pool_ids)), dtype=np.int64)
                    else:
                        buyer_hard_pool[int(b)] = np.array([], dtype=np.int64)

                cache_top = int(min(self.cfg.neg_cache_topk, len(self._exporter_ids)))
                if cache_top > 0:
                    pop_cache = np.argsort(-self._exporter_sampling_prob)[:cache_top].astype(np.int64)
                else:
                    pop_cache = np.array([], dtype=np.int64)
                neg_b = []
                neg_e = []
                neg_w = []
                for _, r in pos_rows.iterrows():
                    b = self._buyer_pos.get(str(r["buyer_id"]))
                    if b is None:
                        continue
                    b = int(b)
                    pos_set = buyer_pos_sets.get(b, set())
                    hard_pool = buyer_hard_pool.get(b, np.array([], dtype=np.int64))
                    for _ in range(self.cfg.neg_per_pos):
                        if not self.enable_hard_negatives:
                            neg_ex = int(rng.integers(0, len(self._exporter_ids)))
                            retry = 0
                            while neg_ex in pos_set and retry < 8:
                                neg_ex = int(rng.integers(0, len(self._exporter_ids)))
                                retry += 1
                            if neg_ex in pos_set:
                                continue
                            neg_b.append(b)
                            neg_e.append(int(neg_ex))
                            neg_w.append(0.60)
                            continue

                        neg_ex = None
                        use_hard = (
                            self.enable_hard_negatives
                            and hard_pool.size > 0
                            and float(rng.random()) < self.cfg.hard_negative_ratio
                        )
                        if use_hard:
                            neg_ex = int(hard_pool[rng.integers(0, hard_pool.size)])
                            w_hint = 0.75
                        elif pop_cache.size > 0 and float(rng.random()) < 0.5:
                            neg_ex = int(pop_cache[rng.integers(0, pop_cache.size)])
                            w_hint = 0.65
                        else:
                            neg_ex = int(rng.choice(len(self._exporter_ids), p=self._exporter_sampling_prob))
                            w_hint = 0.60

                        retry = 0
                        while neg_ex in pos_set and retry < 8:
                            neg_ex = int(rng.choice(len(self._exporter_ids), p=self._exporter_sampling_prob))
                            retry += 1
                        if neg_ex in pos_set:
                            continue

                        neg_b.append(b)
                        neg_e.append(int(neg_ex))
                        neg_w.append(float(w_hint))
                if neg_b:
                    b_idx = np.concatenate([b_idx, np.asarray(neg_b, dtype=np.int64)])
                    e_idx = np.concatenate([e_idx, np.asarray(neg_e, dtype=np.int64)])
                    y = np.concatenate([y, np.zeros(len(neg_b), dtype=np.float64)])
                    w = np.concatenate([w, np.asarray(neg_w, dtype=np.float64)])
        return b_idx, e_idx, y, w

    def _fit_torch(
        self,
        b_idx: np.ndarray,
        e_idx: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        teacher_scores: np.ndarray | None = None,
    ) -> bool:
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
            q = np.clip(self._exporter_sampling_prob, 1e-12, 1.0)
            log_q = torch.tensor(np.log(q), dtype=torch.float32, device=device)
            t_t = None
            if teacher_scores is not None and teacher_scores.size == y.size:
                t_t = torch.tensor(np.asarray(teacher_scores, dtype=np.float32), dtype=torch.float32, device=device)

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
                    # logQ-style correction reduces popularity/sample-bias in sampled negatives.
                    if self.enable_logq_correction:
                        logits = logits - log_q[e_i[idx]]
                    loss_raw = F.binary_cross_entropy_with_logits(logits, y_t[idx], reduction="none")
                    if self.enable_distillation and t_t is not None:
                        pred_p = torch.sigmoid(logits)
                        distill_raw = F.mse_loss(pred_p, t_t[idx], reduction="none")
                        loss_raw = loss_raw + self.cfg.distill_weight * distill_raw
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
        b = np.asarray(self._buyer_matrix, dtype=np.float64)
        e = np.asarray(self._exporter_matrix, dtype=np.float64)
        if b.ndim != 2 or e.ndim != 2 or b.shape[0] == 0 or e.shape[0] == 0:
            self._buyer_emb = np.zeros((len(self._buyer_ids), 1), dtype=np.float64)
            self._exporter_emb = np.zeros((len(self._exporter_ids), 1), dtype=np.float64)
        else:
            # Buyer/exporter raw feature spaces may differ in width; align for cosine scoring.
            if b.shape[1] != e.shape[1]:
                d = int(max(1, min(b.shape[1], e.shape[1])))
                b = b[:, :d]
                e = e[:, :d]
            self._buyer_emb = normalize_rows(b)
            self._exporter_emb = normalize_rows(e)
        self.backend = "feature_cosine"
        self.device = "cpu"

    def fit(self, interactions: pd.DataFrame):
        self.ready = False
        self._build_feature_matrices()
        clean = self._sanitize_interactions(interactions)
        # Fit association rules first so hard-negative sampler can use cross-industry confusables.
        pos = clean[clean["action"] == "right"]
        pairs = []
        if not pos.empty:
            for _, r in pos.iterrows():
                b = self._buyer_industry.get(str(r["buyer_id"]), "unknown")
                e = self._exporter_industry.get(str(r["exporter_id"]), "unknown")
                pairs.append((b, e))
        self._assoc.fit(pairs)

        b_idx, e_idx, y, w = self._build_pairs(clean)

        teacher_scores = None
        if self.enable_distillation and self.text_encoder is not None and b_idx.size > 0:
            b_ids = [self._buyer_ids[int(i)] for i in b_idx.tolist()]
            e_ids = [self._exporter_ids[int(i)] for i in e_idx.tolist()]
            try:
                teacher_scores = np.asarray(self.text_encoder.teacher_scores_for_pairs(b_ids, e_ids), dtype=np.float64)
            except Exception:
                teacher_scores = None

        trained = self._fit_torch(b_idx, e_idx, y, w, teacher_scores=teacher_scores)
        if not trained:
            self._fit_fallback()

        self._ann.fit(self._exporter_ids, self._exporter_emb)
        self.ready = bool(self._ann.ready)

    def score_pairs(self, buyer_id: str, exporter_ids: list[str]) -> np.ndarray:
        if not self.ready or len(exporter_ids) == 0:
            return np.full(len(exporter_ids), 0.5, dtype=np.float64)
        b_pos = self._buyer_pos.get(str(buyer_id))
        if b_pos is None:
            return np.full(len(exporter_ids), 0.5, dtype=np.float64)

        b = self._buyer_emb[b_pos]
        d_b = int(b.shape[0]) if b.ndim == 1 else int(b.shape[-1])
        out = np.full(len(exporter_ids), 0.5, dtype=np.float64)
        for i, ex_id in enumerate(exporter_ids):
            e_pos = self._exporter_pos.get(str(ex_id))
            if e_pos is None:
                continue
            ev = self._exporter_emb[e_pos]
            d_e = int(ev.shape[0]) if ev.ndim == 1 else int(ev.shape[-1])
            if d_b != d_e:
                d = int(max(1, min(d_b, d_e)))
                out[i] = float(np.clip(np.dot(b[:d], ev[:d]), -1.0, 1.0))
            else:
                out[i] = float(np.clip(np.dot(b, ev), -1.0, 1.0))
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
                "candidate_source": np.full(len(ids), "two_tower", dtype=object),
            }
        )

    def industry_assoc_for_pair(self, buyer_id: str, exporter_id: str) -> tuple[float, float, str]:
        _ = buyer_id
        _ = exporter_id
        return 0.0, 0.0, "two_tower"
