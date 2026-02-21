from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np
import pandas as pd

from .common import SKLEARN_AVAILABLE, SGDClassifier, StandardScaler
from .constants import FEATURE_COLUMNS

if TYPE_CHECKING:
    from .feature_builder import PairFeatureBuilder


class OnlineSupervisedModel:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.model = None
        self.ready = False

    def fit(self, interactions: pd.DataFrame, builder: "PairFeatureBuilder"):
        if not SKLEARN_AVAILABLE:
            self.ready = False
            return

        X, y = self._supervised_training_set(interactions, builder)
        if X.empty or y.size < 30 or np.unique(y).size < 2:
            X, y = self._bootstrap_content_training_set(builder)
            if X.empty or y.size < 30 or np.unique(y).size < 2:
                self.ready = False
                return

        self.scaler = StandardScaler()
        self.model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=5e-4,
            class_weight="balanced",
            random_state=self.random_state,
        )
        x_values = X[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        self.scaler.partial_fit(x_values)
        x_scaled = self.scaler.transform(x_values)
        self.model.partial_fit(x_scaled, y.astype(np.int64), classes=np.array([0, 1], dtype=np.int64))
        self.ready = True

    def predict_proba(self, feature_df: pd.DataFrame) -> np.ndarray:
        if feature_df.empty:
            return np.zeros(0, dtype=np.float64)

        if self.ready and self.model is not None and self.scaler is not None:
            x = feature_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
            xs = self.scaler.transform(x)
            p = self.model.predict_proba(xs)[:, 1]
            return np.clip(p, 0.0, 1.0)

        base = feature_df["match_after_risk"].to_numpy(dtype=np.float64) / 100.0
        return np.clip(base, 0.0, 1.0)

    def update_single(self, single_pair_feature_df: pd.DataFrame, label: int):
        if not SKLEARN_AVAILABLE or single_pair_feature_df is None or single_pair_feature_df.empty:
            return
        x = single_pair_feature_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        y = np.array([int(label)], dtype=np.int64)

        if self.scaler is None:
            self.scaler = StandardScaler()
        self.scaler.partial_fit(x)
        xs = self.scaler.transform(x)

        if self.model is None:
            self.model = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=5e-4,
                class_weight="balanced",
                random_state=self.random_state,
            )
            self.model.partial_fit(xs, y, classes=np.array([0, 1], dtype=np.int64))
        else:
            self.model.partial_fit(xs, y)
        self.ready = True

    def _supervised_training_set(self, interactions: pd.DataFrame, builder: "PairFeatureBuilder") -> Tuple[pd.DataFrame, np.ndarray]:
        if interactions.empty:
            return pd.DataFrame(), np.array([], dtype=np.int64)

        data = interactions.copy()
        if len(data) > 60_000:
            data = data.sample(60_000, random_state=self.random_state)

        data["buyer_id"] = data["buyer_id"].astype(str)
        data["exporter_id"] = data["exporter_id"].astype(str)
        valid = data["buyer_id"].isin(builder.buyers_idx.index) & data["exporter_id"].isin(builder.exporters_idx.index)
        data = data[valid]
        if data.empty:
            return pd.DataFrame(), np.array([], dtype=np.int64)

        rows = []
        labels = []
        for _, r in data.iterrows():
            f = builder.single_pair_features(r["buyer_id"], r["exporter_id"])
            if f is None or f.empty:
                continue
            rows.append(f.iloc[0][FEATURE_COLUMNS].to_dict())
            labels.append(1 if r["action"] == "right" else 0)

        if not rows:
            return pd.DataFrame(), np.array([], dtype=np.int64)
        return pd.DataFrame(rows), np.asarray(labels, dtype=np.int64)

    def _bootstrap_content_training_set(self, builder: "PairFeatureBuilder") -> Tuple[pd.DataFrame, np.ndarray]:
        if builder.buyers.empty:
            return pd.DataFrame(), np.array([], dtype=np.int64)

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
            frames.append(cands[FEATURE_COLUMNS + ["label"]])

        if not frames:
            return pd.DataFrame(), np.array([], dtype=np.int64)
        all_df = pd.concat(frames, ignore_index=True)
        y = all_df["label"].to_numpy(dtype=np.int64)
        X = all_df[FEATURE_COLUMNS]
        return X, y
