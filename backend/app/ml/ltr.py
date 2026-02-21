from __future__ import annotations

from typing import TYPE_CHECKING, Tuple
import warnings

import numpy as np
import pandas as pd

from .common import CUPY_AVAILABLE, LIGHTGBM_AVAILABLE, LGBMRanker, XGBOOST_AVAILABLE, XGBRanker, cp
from .constants import FEATURE_COLUMNS
from .time_decay import compute_time_decay_weights

if TYPE_CHECKING:
    from .feature_builder import PairFeatureBuilder


class LearningToRankModel:
    def __init__(self, random_state: int = 42, half_life_days: float = 45.0, prefer_gpu: bool = True):
        self.random_state = random_state
        self.half_life_days = float(max(1.0, half_life_days))
        self.prefer_gpu = bool(prefer_gpu)
        self.model = None
        self.backend = None
        self.device = "cpu"
        self.ready = False

    def fit(self, interactions: pd.DataFrame, builder: "PairFeatureBuilder"):
        X, y, group, w = self._build_training_dataset(interactions, builder)
        if X.size == 0:
            X, y, group, w = self._build_bootstrap_dataset(builder)
            if X.size == 0:
                self.ready = False
                return

        train_plan = []
        if self.prefer_gpu:
            train_plan = [
                ("xgboost", "gpu"),
                ("lightgbm", "gpu"),
                ("xgboost", "cpu"),
                ("lightgbm", "cpu"),
            ]
        else:
            train_plan = [
                ("xgboost", "cpu"),
                ("lightgbm", "cpu"),
            ]

        for backend, device in train_plan:
            if backend == "xgboost" and self._fit_xgboost(X, y, group=group, sample_weight=w, device=device):
                return
            if backend == "lightgbm" and self._fit_lightgbm(X, y, group=group, sample_weight=w, device=device):
                return

        self.model = None
        self.backend = None
        self.device = "cpu"
        self.ready = False

    def _fit_xgboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group: list[int],
        sample_weight: np.ndarray,
        device: str,
    ) -> bool:
        if not XGBOOST_AVAILABLE:
            return False
        param_grid = [
            {"n_estimators": 240, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 2, "gamma": 0.05},
            {"n_estimators": 320, "learning_rate": 0.03, "max_depth": 8, "min_child_weight": 3, "gamma": 0.1},
            {"n_estimators": 180, "learning_rate": 0.08, "max_depth": 4, "min_child_weight": 5, "gamma": 0.2},
        ]
        best_model = None
        best_score = -1.0

        for trial in param_grid:
            try:
                params = {
                    "objective": "rank:ndcg",
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "reg_lambda": 1.0,
                    "random_state": self.random_state,
                    "tree_method": "hist",
                    **trial,
                }
                if device == "gpu":
                    params["device"] = "cuda"

                model = XGBRanker(**params)
                try:
                    model.fit(X, y, group=group, sample_weight=sample_weight, verbose=False)
                except Exception:
                    model.fit(X, y, group=group, verbose=False)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*Falling back to prediction using DMatrix due to mismatched devices.*",
                        category=UserWarning,
                    )
                    if device == "gpu" and CUPY_AVAILABLE:
                        pred = cp.asnumpy(model.predict(cp.asarray(X))).astype(np.float64)
                    else:
                        pred = np.asarray(model.predict(X), dtype=np.float64)
                score = self._group_ndcg(y, pred, group, k=10)
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception:
                continue

        if best_model is None:
            self.model = None
            self.backend = None
            self.device = "cpu"
            self.ready = False
            return False

        self.model = best_model
        self.backend = "xgboost"
        self.device = device
        self.ready = True
        return True

    def _fit_lightgbm(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group: list[int],
        sample_weight: np.ndarray,
        device: str,
    ) -> bool:
        if not LIGHTGBM_AVAILABLE:
            return False
        param_grid = [
            {"n_estimators": 180, "learning_rate": 0.06, "num_leaves": 31, "min_data_in_leaf": 20},
            {"n_estimators": 260, "learning_rate": 0.04, "num_leaves": 63, "min_data_in_leaf": 30},
            {"n_estimators": 140, "learning_rate": 0.09, "num_leaves": 24, "min_data_in_leaf": 15},
        ]
        best_model = None
        best_score = -1.0

        for trial in param_grid:
            try:
                params = {
                    "objective": "lambdarank",
                    "metric": "ndcg",
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "random_state": self.random_state,
                    **trial,
                }
                if device == "gpu":
                    params["device"] = "gpu"

                model = LGBMRanker(**params)
                model.fit(X, y, group=group, sample_weight=sample_weight)
                pred = np.asarray(model.predict(X), dtype=np.float64)
                score = self._group_ndcg(y, pred, group, k=10)
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception:
                continue

        if best_model is None:
            self.model = None
            self.backend = None
            self.device = "cpu"
            self.ready = False
            return False

        self.model = best_model
        self.backend = "lightgbm"
        self.device = device
        self.ready = True
        return True

    def score(self, feature_df: pd.DataFrame) -> np.ndarray:
        if feature_df.empty:
            return np.zeros(0, dtype=np.float64)
        if not self.ready or self.model is None:
            return np.zeros(len(feature_df), dtype=np.float64)

        x = feature_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*Falling back to prediction using DMatrix due to mismatched devices.*",
                category=UserWarning,
            )
            if self.backend == "xgboost" and self.device == "gpu" and CUPY_AVAILABLE:
                x_gpu = cp.asarray(x)
                raw = cp.asnumpy(self.model.predict(x_gpu)).astype(np.float64)
            else:
                raw = np.asarray(self.model.predict(x), dtype=np.float64)
        if raw.size == 0:
            return np.zeros(len(feature_df), dtype=np.float64)

        mn, mx = np.nanmin(raw), np.nanmax(raw)
        if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < 1e-12:
            return np.full(len(raw), 0.5, dtype=np.float64)
        return np.clip((raw - mn) / (mx - mn), 0.0, 1.0)

    def _build_training_dataset(
        self, interactions: pd.DataFrame, builder: "PairFeatureBuilder"
    ) -> Tuple[np.ndarray, np.ndarray, list[int], np.ndarray]:
        if interactions is None or interactions.empty:
            return np.zeros((0, len(FEATURE_COLUMNS))), np.array([]), [], np.array([])

        data = interactions.copy()
        if len(data) > 80_000:
            data = data.sample(80_000, random_state=self.random_state)

        data["buyer_id"] = data["buyer_id"].astype(str)
        data["exporter_id"] = data["exporter_id"].astype(str)
        valid = data["buyer_id"].isin(builder.buyers_idx.index) & data["exporter_id"].isin(builder.exporters_idx.index)
        data = data[valid]
        if data.empty:
            return np.zeros((0, len(FEATURE_COLUMNS))), np.array([]), [], np.array([])

        data = data.copy()
        data["label"] = (data["action"].astype(str).str.lower() == "right").astype(np.int64)
        if data["label"].nunique() < 2:
            return np.zeros((0, len(FEATURE_COLUMNS))), np.array([]), [], np.array([])
        data["weight"] = compute_time_decay_weights(data, ts_col="ts", half_life_days=self.half_life_days)

        rows = []
        labels = []
        qids = []
        weights = []
        for _, r in data.iterrows():
            f = builder.single_pair_features(r["buyer_id"], r["exporter_id"])
            if f is None or f.empty:
                continue
            rows.append(f.iloc[0][FEATURE_COLUMNS].to_dict())
            labels.append(int(r["label"]))
            qids.append(str(r["buyer_id"]))
            weights.append(float(r["weight"]))

        if not rows:
            return np.zeros((0, len(FEATURE_COLUMNS))), np.array([]), [], np.array([])

        train_df = pd.DataFrame(rows)
        train_df["label"] = np.asarray(labels, dtype=np.int64)
        train_df["qid"] = np.asarray(qids, dtype=object)
        train_df["weight"] = np.asarray(weights, dtype=np.float64)

        group_sizes = train_df.groupby("qid", sort=False).size()
        valid_qids = group_sizes[group_sizes >= 2].index
        train_df = train_df[train_df["qid"].isin(valid_qids)]
        if train_df.empty or train_df["label"].nunique() < 2:
            return np.zeros((0, len(FEATURE_COLUMNS))), np.array([]), [], np.array([])

        train_df = train_df.sort_values("qid", kind="stable")
        group = train_df.groupby("qid", sort=False).size().astype(int).tolist()

        if len(group) < 10 or len(train_df) < 60:
            return np.zeros((0, len(FEATURE_COLUMNS))), np.array([]), [], np.array([])

        x = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        y = train_df["label"].to_numpy(dtype=np.int64)
        w = train_df["weight"].to_numpy(dtype=np.float64) * self._class_balance_weights(y)
        return x, y, group, w

    def _build_bootstrap_dataset(self, builder: "PairFeatureBuilder") -> Tuple[np.ndarray, np.ndarray, list[int], np.ndarray]:
        if builder.buyers.empty:
            return np.zeros((0, len(FEATURE_COLUMNS))), np.array([]), [], np.array([])

        sample_n = min(200, len(builder.buyers))
        sample_buyers = builder.buyers.sample(sample_n, random_state=self.random_state)
        frames = []
        for _, buyer in sample_buyers.iterrows():
            cands, _ = builder.candidate_features_for_buyer(buyer)
            if len(cands) < 2:
                continue
            keep_n = min(30, len(cands))
            cands = cands.nlargest(keep_n, "match_after_risk")
            threshold = float(cands["match_after_risk"].median())
            cands["label"] = (cands["match_after_risk"] >= threshold).astype(np.int64)
            cands["qid"] = str(buyer.get("Buyer_ID"))
            cands["weight"] = 1.0
            frames.append(cands[FEATURE_COLUMNS + ["label", "qid", "weight"]])

        if not frames:
            return np.zeros((0, len(FEATURE_COLUMNS))), np.array([]), [], np.array([])

        train_df = pd.concat(frames, ignore_index=True)
        train_df = train_df.sort_values("qid", kind="stable")
        group = train_df.groupby("qid", sort=False).size().astype(int).tolist()
        x = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        y = train_df["label"].to_numpy(dtype=np.int64)
        w = train_df["weight"].to_numpy(dtype=np.float64) * self._class_balance_weights(y)
        return x, y, group, w

    def _class_balance_weights(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.int64)
        if y.size == 0:
            return np.array([], dtype=np.float64)
        n_pos = float(np.sum(y == 1))
        n_neg = float(np.sum(y == 0))
        if n_pos <= 0 or n_neg <= 0:
            return np.ones(y.shape[0], dtype=np.float64)
        pos_w = y.size / (2.0 * n_pos)
        neg_w = y.size / (2.0 * n_neg)
        return np.where(y == 1, pos_w, neg_w).astype(np.float64)

    def _group_ndcg(self, y_true: np.ndarray, y_pred: np.ndarray, group: list[int], k: int = 10) -> float:
        if y_true.size == 0 or y_pred.size == 0 or not group:
            return 0.0
        offset = 0
        scores = []
        for g in group:
            size = int(g)
            if size <= 1:
                offset += size
                continue
            yt = y_true[offset:offset + size]
            yp = y_pred[offset:offset + size]
            offset += size

            order = np.argsort(-yp)
            top = yt[order][:k]
            gains = top.astype(np.float64)
            discounts = np.log2(np.arange(2, len(top) + 2, dtype=np.float64))
            dcg = float(np.sum(gains / discounts))

            ideal = np.sort(yt)[::-1][:k].astype(np.float64)
            idcg = float(np.sum(ideal / discounts[: len(ideal)]))
            if idcg > 0.0:
                scores.append(dcg / idcg)
        if not scores:
            return 0.0
        return float(np.mean(scores))
