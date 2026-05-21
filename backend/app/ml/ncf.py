from __future__ import annotations

import numpy as np
import pandas as pd

from .common import F, TORCH_AVAILABLE, as_text, nn, torch
from .time_decay import compute_time_decay_weights


if TORCH_AVAILABLE and nn is not None and torch is not None:

    class _NCFNet(nn.Module):
        def __init__(self, n_buyers: int, n_exporters: int, emb_dim: int, hidden_dim: int):
            super().__init__()
            self.buyer_emb = nn.Embedding(n_buyers, emb_dim)
            self.exporter_emb = nn.Embedding(n_exporters, emb_dim)
            self.mlp = nn.Sequential(
                nn.Linear(emb_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, buyer_idx, exporter_idx):
            b = self.buyer_emb(buyer_idx)
            e = self.exporter_emb(exporter_idx)
            x = torch.cat([b, e], dim=1)
            return self.mlp(x).squeeze(1)

else:

    class _NCFNet:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is not available for NCF.")


class NeuralCollaborativeFilteringModel:
    def __init__(
        self,
        buyer_ids,
        exporter_ids,
        embedding_dim: int = 24,
        hidden_dim: int = 64,
        epochs: int = 4,
        batch_size: int = 4096,
        lr: float = 1e-3,
        half_life_days: float = 45.0,
        prefer_gpu: bool = True,
        min_interactions: int = 300,
    ):
        self.embedding_dim = int(max(8, embedding_dim))
        self.hidden_dim = int(max(16, hidden_dim))
        self.epochs = int(max(1, epochs))
        self.batch_size = int(max(256, batch_size))
        self.lr = float(max(1e-5, lr))
        self.half_life_days = float(max(1.0, half_life_days))
        self.prefer_gpu = bool(prefer_gpu)
        self.min_interactions = int(max(100, min_interactions))

        self._buyer_pos = {str(b): i for i, b in enumerate(list(buyer_ids))}
        self._exporter_pos = {str(e): i for i, e in enumerate(list(exporter_ids))}
        self.model = None
        self.ready = False
        self.device = "cpu"
        self.backend = "none"

    def fit(self, interactions: pd.DataFrame):
        self.ready = False
        self.device = "cpu"
        self.backend = "none"
        self.model = None

        if not TORCH_AVAILABLE or torch is None or nn is None or F is None:
            return
        if interactions is None or interactions.empty:
            return
        if len(interactions) < self.min_interactions:
            return

        df = interactions.copy()
        for col in ("buyer_id", "exporter_id", "action"):
            if col not in df.columns:
                return
        if "ts" not in df.columns:
            df["ts"] = pd.Timestamp.utcnow()
        df["buyer_id"] = df["buyer_id"].astype(str).str.strip()
        df["exporter_id"] = df["exporter_id"].astype(str).str.strip()
        df["action"] = df["action"].astype(str).str.strip().str.lower()
        df = df[df["action"].isin(["left", "right"])]
        df = df[(df["buyer_id"] != "") & (df["exporter_id"] != "")]
        df = df[df["buyer_id"].isin(self._buyer_pos) & df["exporter_id"].isin(self._exporter_pos)]
        if len(df) < self.min_interactions:
            return

        b_idx = df["buyer_id"].map(self._buyer_pos).astype(np.int64).to_numpy()
        e_idx = df["exporter_id"].map(self._exporter_pos).astype(np.int64).to_numpy()
        y = (df["action"] == "right").astype(np.float32).to_numpy()
        w = compute_time_decay_weights(df, ts_col="ts", half_life_days=self.half_life_days).astype(np.float32)

        try:
            use_gpu = bool(self.prefer_gpu and torch.cuda.is_available())
            device = torch.device("cuda" if use_gpu else "cpu")
            model = _NCFNet(
                n_buyers=len(self._buyer_pos),
                n_exporters=len(self._exporter_pos),
                emb_dim=self.embedding_dim,
                hidden_dim=self.hidden_dim,
            ).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)

            bt = torch.tensor(b_idx, dtype=torch.long, device=device)
            et = torch.tensor(e_idx, dtype=torch.long, device=device)
            yt = torch.tensor(y, dtype=torch.float32, device=device)
            wt = torch.tensor(w, dtype=torch.float32, device=device)

            n = int(len(df))
            bs = int(min(self.batch_size, n))
            for _ in range(self.epochs):
                perm = torch.randperm(n, device=device)
                for start in range(0, n, bs):
                    idx = perm[start : start + bs]
                    logits = model(bt[idx], et[idx])
                    loss = F.binary_cross_entropy_with_logits(logits, yt[idx], weight=wt[idx], reduction="mean")
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            self.model = model
            self.ready = True
            self.device = "gpu" if use_gpu else "cpu"
            self.backend = "ncf_torch"
        except Exception:
            self.model = None
            self.ready = False
            self.device = "cpu"
            self.backend = "none"

    def score(self, buyer_id: str, exporter_ids: np.ndarray) -> np.ndarray:
        out = np.zeros(len(exporter_ids), dtype=np.float64)
        if not self.ready or self.model is None:
            return out
        b_pos = self._buyer_pos.get(as_text(buyer_id))
        if b_pos is None or len(exporter_ids) == 0:
            return out

        ex_idx = np.array([self._exporter_pos.get(as_text(x), -1) for x in exporter_ids], dtype=np.int64)
        valid = ex_idx >= 0
        if not np.any(valid):
            return out
        try:
            device = next(self.model.parameters()).device
            bt = torch.full((int(np.sum(valid)),), int(b_pos), dtype=torch.long, device=device)
            et = torch.tensor(ex_idx[valid], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = self.model(bt, et)
                p = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)
            out[valid] = np.clip(p, 0.0, 1.0)
        except Exception:
            pass
        return out
