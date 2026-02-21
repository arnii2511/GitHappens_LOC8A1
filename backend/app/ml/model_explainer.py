from __future__ import annotations

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from .constants import FEATURE_COLUMNS


class ModelExplainer:
    """
    True model-level explainability layer.

    Works with:
    - SGDClassifier (linear -> coefficient-based contribution)
    - XGBoost (feature importance based explanation)
    - Fallback baseline (match_after_risk based)

    Does NOT modify any existing pipeline.
    """

    def __init__(self, supervised_model):
        self.model_wrapper = supervised_model

    # ================================================================
    # PUBLIC METHOD
    # ================================================================

    def explain_prediction(
        self,
        feature_row: pd.Series,
        predicted_proba: float,
    ) -> Dict[str, Any]:

        backend = self.model_wrapper.backend

        if backend == "sgd":
            return self._explain_sgd(feature_row, predicted_proba)

        if backend == "xgboost":
            return self._explain_xgb(feature_row, predicted_proba)

        return self._explain_fallback(feature_row, predicted_proba)

    # ================================================================
    # SGD (LINEAR MODEL) - TRUE FEATURE CONTRIBUTION
    # ================================================================

    def _explain_sgd(self, row: pd.Series, proba: float) -> Dict[str, Any]:

        model = self.model_wrapper.model
        scaler = self.model_wrapper.scaler

        if model is None or scaler is None:
            return self._explain_fallback(row, proba)

        x = row[FEATURE_COLUMNS].to_numpy(dtype=np.float64).reshape(1, -1)
        x_scaled = scaler.transform(x)[0]

        coefs = model.coef_[0]
        contributions = coefs * x_scaled

        feature_contrib = list(zip(FEATURE_COLUMNS, contributions))
        feature_contrib.sort(key=lambda x: abs(x[1]), reverse=True)

        top_positive = [(f, float(c)) for f, c in feature_contrib if c > 0][:5]
        top_negative = [(f, float(c)) for f, c in feature_contrib if c < 0][:5]

        direction = "High" if proba >= 0.5 else "Low"

        return {
            "model_type": "SGD (linear)",
            "prediction_probability": float(proba),
            "prediction_direction": direction,
            "top_positive_drivers": top_positive,
            "top_negative_drivers": top_negative,
            "summary": self._generate_summary(direction, top_positive, top_negative),
        }

    # ================================================================
    # XGBOOST - FEATURE IMPORTANCE BASED
    # ================================================================

    def _explain_xgb(self, row: pd.Series, proba: float) -> Dict[str, Any]:

        model = self.model_wrapper.model

        if model is None:
            return self._explain_fallback(row, proba)

        importance = model.feature_importances_
        feature_contrib = list(zip(FEATURE_COLUMNS, importance))
        feature_contrib.sort(key=lambda x: abs(x[1]), reverse=True)

        top_features = [(f, float(c)) for f, c in feature_contrib[:5]]

        direction = "High" if proba >= 0.5 else "Low"

        return {
            "model_type": "XGBoost (tree)",
            "prediction_probability": float(proba),
            "prediction_direction": direction,
            "top_important_features": top_features,
            "summary": f"Prediction driven primarily by: {', '.join([f for f, _ in top_features])}.",
        }

    # ================================================================
    # FALLBACK (NO MODEL)
    # ================================================================

    def _explain_fallback(self, row: pd.Series, proba: float) -> Dict[str, Any]:

        match_score = float(row.get("match_after_risk", 0.0))
        industry = float(row.get("industry_similarity", 0.0))
        intent = float(row.get("intent_fit", 0.0))

        direction = "High" if proba >= 0.5 else "Low"

        return {
            "model_type": "Baseline",
            "prediction_probability": float(proba),
            "prediction_direction": direction,
            "drivers": {
                "match_after_risk": match_score,
                "industry_similarity": industry,
                "intent_fit": intent,
            },
            "summary": "Prediction derived from rule-based match_after_risk scoring.",
        }

    # ================================================================
    # HUMAN SUMMARY GENERATOR
    # ================================================================

    def _generate_summary(
        self,
        direction: str,
        positives: List[Tuple[str, float]],
        negatives: List[Tuple[str, float]],
    ) -> str:

        summary_parts = []

        if direction == "High":
            if positives:
                summary_parts.append(
                    f"Score increased mainly due to {positives[0][0]}."
                )
            if negatives:
                summary_parts.append(
                    f"Score slightly reduced by {negatives[0][0]}."
                )
        else:
            if negatives:
                summary_parts.append(
                    f"Score reduced mainly due to {negatives[0][0]}."
                )
            if positives:
                summary_parts.append(
                    f"Some positive signal from {positives[0][0]}."
                )

        return " ".join(summary_parts)
