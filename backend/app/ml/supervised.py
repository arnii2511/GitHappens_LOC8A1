from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple
import warnings

import numpy as np
import pandas as pd

from .common import (
    CUPY_AVAILABLE,
    SKLEARN_AVAILABLE,
    SGDClassifier,
    StandardScaler,
    XGBClassifier,
    XGBOOST_AVAILABLE,
    cp,
)
from .constants import FEATURE_COLUMNS
from .time_decay import compute_time_decay_weights

if TYPE_CHECKING:
    from .feature_builder import PairFeatureBuilder


class OnlineSupervisedModel:
    def __init__(self, random_state: int = 42, half_life_days: float = 45.0, prefer_gpu: bool = True):
        self.random_state = random_state
        self.half_life_days = float(max(1.0, half_life_days))
        self.prefer_gpu = bool(prefer_gpu)
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.model = None
        self.ready = False
        self.backend = None
        self.device = "cpu"
        self.supports_online = True

    def fit(
        self,
        interactions: pd.DataFrame,
        builder: "PairFeatureBuilder",
        crossed_features: Optional[pd.DataFrame] = None,
    ):
        X, y, w = self._supervised_training_set(interactions, builder, crossed_features=crossed_features)
        if X.empty or y.size < 30 or np.unique(y).size < 2:
            X, y, w = self._bootstrap_content_training_set(builder)
            if X.empty or y.size < 30 or np.unique(y).size < 2:
                self.ready = False
                self.backend = None
                self.device = "cpu"
                self.supports_online = True
                return

        x_values = X[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        sample_weight = np.asarray(w, dtype=np.float64)

        if self.prefer_gpu and self._fit_xgboost_gpu(x_values, y, sample_weight):
            return

        if self._fit_sgd_cpu(x_values, y, sample_weight):
            return

        self.model = None
        self.ready = False
        self.backend = None
        self.device = "cpu"
        self.supports_online = True

    def _fit_xgboost_gpu(self, x_values: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> bool:
        if not XGBOOST_AVAILABLE:
            return False
        try:
            n_pos = float(np.sum(y == 1))
            n_neg = float(np.sum(y == 0))
            scale_pos_weight = float(n_neg / max(1.0, n_pos))
            params = {
                "objective": "binary:logistic",
                "n_estimators": 220,
                "learning_rate": 0.05,
                "max_depth": 6,
                "min_child_weight": 3,
                "gamma": 0.1,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 1.0,
                "max_delta_step": 1,
                "scale_pos_weight": scale_pos_weight,
                "random_state": self.random_state,
                "tree_method": "hist",
                "device": "cuda",
            }
            self.model = XGBClassifier(**params)
            w = sample_weight * self._class_balance_weights(y)
            self.model.fit(x_values, y.astype(np.int64), sample_weight=w, verbose=False)
            self.ready = True
            self.backend = "xgboost"
            self.device = "gpu"
            self.supports_online = False
            self.scaler = None
            return True
        except Exception:
            self.model = None
            self.ready = False
            self.backend = None
            self.device = "cpu"
            self.supports_online = True
            return False

    def _fit_sgd_cpu(self, x_values: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> bool:
        if not SKLEARN_AVAILABLE:
            return False
        try:
            self.scaler = StandardScaler()
            self.model = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=5e-4,
                random_state=self.random_state,
            )
            w = sample_weight * self._class_balance_weights(y)
            self.scaler.partial_fit(x_values)
            x_scaled = self.scaler.transform(x_values)
            self.model.partial_fit(
                x_scaled,
                y.astype(np.int64),
                classes=np.array([0, 1], dtype=np.int64),
                sample_weight=w,
            )
            self.ready = True
            self.backend = "sgd"
            self.device = "cpu"
            self.supports_online = True
            return True
        except Exception:
            self.model = None
            self.ready = False
            self.backend = None
            self.device = "cpu"
            self.supports_online = True
            return False

    def predict_proba(self, feature_df: pd.DataFrame) -> np.ndarray:
        if feature_df.empty:
            return np.zeros(0, dtype=np.float64)

        if self.ready and self.model is not None:
            x = feature_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)

            if self.backend == "xgboost":
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*Falling back to prediction using DMatrix due to mismatched devices.*",
                        category=UserWarning,
                    )
                    if self.device == "gpu" and CUPY_AVAILABLE:
                        x_gpu = cp.asarray(x)
                        p = cp.asnumpy(self.model.predict_proba(x_gpu)[:, 1]).astype(np.float64)
                    else:
                        p = np.asarray(self.model.predict_proba(x)[:, 1], dtype=np.float64)
                return np.clip(p, 0.0, 1.0)

            if self.backend == "sgd" and self.scaler is not None:
                xs = self.scaler.transform(x)
                p = self.model.predict_proba(xs)[:, 1]
                return np.clip(p, 0.0, 1.0)

        base = feature_df["match_after_risk"].to_numpy(dtype=np.float64) / 100.0
        return np.clip(base, 0.0, 1.0)

    def update_single(self, single_pair_feature_df: pd.DataFrame, label: int, sample_weight: float = 1.0) -> bool:
        if single_pair_feature_df is None or single_pair_feature_df.empty:
            return False
        if not self.supports_online or self.backend != "sgd":
            return False
        if not SKLEARN_AVAILABLE:
            return False

        x = single_pair_feature_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        y = np.array([int(label)], dtype=np.int64)
        w = np.array([float(max(0.01, sample_weight))], dtype=np.float64)

        if self.scaler is None:
            self.scaler = StandardScaler()
        self.scaler.partial_fit(x)
        xs = self.scaler.transform(x)

        if self.model is None:
            self.model = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=5e-4,
                random_state=self.random_state,
            )
            self.model.partial_fit(xs, y, classes=np.array([0, 1], dtype=np.int64), sample_weight=w)
        else:
            self.model.partial_fit(xs, y, sample_weight=w)

        self.ready = True
        self.backend = "sgd"
        self.device = "cpu"
        self.supports_online = True
        return True

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

    def _supervised_training_set(
        self,
        interactions: pd.DataFrame,
        builder: "PairFeatureBuilder",
        crossed_features: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        parts_X: list[pd.DataFrame] = []
        parts_y: list[np.ndarray] = []
        parts_w: list[np.ndarray] = []

        if crossed_features is not None and (not crossed_features.empty):
            Xc, yc, wc = self._supervised_training_from_crossed(crossed_features)
            if (not Xc.empty) and yc.size > 0:
                parts_X.append(Xc)
                parts_y.append(yc)
                parts_w.append(wc)

        Xi, yi, wi = self._supervised_training_from_interactions(interactions, builder)
        if (not Xi.empty) and yi.size > 0:
            parts_X.append(Xi)
            parts_y.append(yi)
            parts_w.append(wi)

        if not parts_X:
            return pd.DataFrame(), np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        X = pd.concat(parts_X, ignore_index=True)
        y = np.concatenate(parts_y).astype(np.int64, copy=False)
        w = np.concatenate(parts_w).astype(np.float64, copy=False)
        return X, y, w

    def _supervised_training_from_interactions(
        self, interactions: pd.DataFrame, builder: "PairFeatureBuilder"
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        if interactions.empty:
            return pd.DataFrame(), np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        data = interactions.copy()
        if len(data) > 60_000:
            data = data.sample(60_000, random_state=self.random_state)

        data["buyer_id"] = data["buyer_id"].astype(str)
        data["exporter_id"] = data["exporter_id"].astype(str)
        valid = data["buyer_id"].isin(builder.buyers_idx.index) & data["exporter_id"].isin(builder.exporters_idx.index)
        data = data[valid]
        if data.empty:
            return pd.DataFrame(), np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        data = data.copy()
        data["decay_weight"] = compute_time_decay_weights(data, ts_col="ts", half_life_days=self.half_life_days)

        rows = []
        labels = []
        weights = []
        for _, r in data.iterrows():
            f = builder.single_pair_features(r["buyer_id"], r["exporter_id"])
            if f is None or f.empty:
                continue
            row = f.iloc[0]
            label = 1 if r["action"] == "right" else 0
            plaus = self._plausible_positive_score(row)
            sample_w = float(r["decay_weight"])
            # PU-style denoising: down-weight likely-positive unlabeled negatives.
            if label == 0:
                sample_w *= float(np.clip(1.0 - 0.65 * plaus, 0.15, 1.0))
            else:
                sample_w *= float(np.clip(1.0 + 0.10 * plaus, 1.0, 1.25))
            rows.append(row[FEATURE_COLUMNS].to_dict())
            labels.append(label)
            weights.append(sample_w)

        if not rows:
            return pd.DataFrame(), np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        return (
            pd.DataFrame(rows),
            np.asarray(labels, dtype=np.int64),
            np.asarray(weights, dtype=np.float64),
        )

    def _supervised_training_from_crossed(self, crossed: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        if crossed is None or crossed.empty:
            return pd.DataFrame(), np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        data = crossed.copy()
        if len(data) > 120_000:
            data = data.sample(120_000, random_state=self.random_state)

        if "label" not in data.columns:
            if "action" in data.columns:
                data["label"] = (data["action"].astype(str).str.lower() == "right").astype(np.int64)
            else:
                return pd.DataFrame(), np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        else:
            data["label"] = pd.to_numeric(data["label"], errors="coerce").fillna(0).astype(np.int64)
            data["label"] = (data["label"] > 0).astype(np.int64)

        if data["label"].nunique() < 2:
            return pd.DataFrame(), np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        if "ts" in data.columns:
            data["ts"] = pd.to_datetime(data["ts"], errors="coerce", utc=True)
            data["decay_weight"] = compute_time_decay_weights(data, ts_col="ts", half_life_days=self.half_life_days)
        else:
            data["decay_weight"] = 1.0

        for col in FEATURE_COLUMNS:
            if col not in data.columns:
                data[col] = 0.0

        X = data[FEATURE_COLUMNS].copy()
        X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y = data["label"].to_numpy(dtype=np.int64)

        industry = np.clip(pd.to_numeric(X.get("industry_similarity", 0.0), errors="coerce").to_numpy(dtype=np.float64), 0.0, 1.0)
        text_sim = np.clip(pd.to_numeric(X.get("text_similarity", 0.0), errors="coerce").to_numpy(dtype=np.float64), 0.0, 1.0)
        retrieval = np.clip(pd.to_numeric(X.get("retrieval_score", 0.0), errors="coerce").to_numpy(dtype=np.float64), 0.0, 1.0)
        match = np.clip(pd.to_numeric(X.get("match_after_risk", 0.0), errors="coerce").to_numpy(dtype=np.float64) / 100.0, 0.0, 1.0)
        plaus = np.clip(0.35 * industry + 0.30 * text_sim + 0.20 * retrieval + 0.15 * match, 0.0, 1.0)

        w = pd.to_numeric(data.get("decay_weight", 1.0), errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
        neg_mask = y == 0
        pos_mask = y == 1
        w[neg_mask] *= np.clip(1.0 - 0.65 * plaus[neg_mask], 0.15, 1.0)
        w[pos_mask] *= np.clip(1.0 + 0.10 * plaus[pos_mask], 1.0, 1.25)

        valid = np.isfinite(w)
        if not np.any(valid):
            return pd.DataFrame(), np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        if not np.all(valid):
            X = X.loc[valid].reset_index(drop=True)
            y = y[valid]
            w = w[valid]
        return X, y, w

    def _bootstrap_content_training_set(self, builder: "PairFeatureBuilder") -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        if builder.buyers.empty:
            return pd.DataFrame(), np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        sample_n = min(180, len(builder.buyers))
        sample_buyers = builder.buyers.sample(sample_n, random_state=self.random_state)
        frames = []
        for _, buyer in sample_buyers.iterrows():
            cands, _ = builder.candidate_features_for_buyer(buyer)
            if cands.empty:
                continue
            keep_n = min(25, len(cands))
            cands = cands.nlargest(keep_n, "match_after_risk")
            threshold = float(cands["match_after_risk"].median())
            cands["label"] = (cands["match_after_risk"] >= threshold).astype(np.int64)
            cands["weight"] = 1.0
            frames.append(cands[FEATURE_COLUMNS + ["label", "weight"]])

        if not frames:
            return pd.DataFrame(), np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        all_df = pd.concat(frames, ignore_index=True)
        y = all_df["label"].to_numpy(dtype=np.int64)
        X = all_df[FEATURE_COLUMNS]
        w = all_df["weight"].to_numpy(dtype=np.float64)
        return X, y, w

    def _plausible_positive_score(self, row: pd.Series) -> float:
        industry = float(np.clip(pd.to_numeric(row.get("industry_similarity", 0.0), errors="coerce"), 0.0, 1.0))
        text_sim = float(np.clip(pd.to_numeric(row.get("text_similarity", 0.0), errors="coerce"), 0.0, 1.0))
        retrieval = float(np.clip(pd.to_numeric(row.get("retrieval_score", 0.0), errors="coerce"), 0.0, 1.0))
        match = float(np.clip(pd.to_numeric(row.get("match_after_risk", 0.0), errors="coerce") / 100.0, 0.0, 1.0))
        score = 0.35 * industry + 0.30 * text_sim + 0.20 * retrieval + 0.15 * match
        if not np.isfinite(score):
            return 0.0
        return float(np.clip(score, 0.0, 1.0))
