"""
Compatibility facade.

This module keeps the previous public API while delegating implementation to
the new pipeline package.
"""

from .pipeline.checklist import verification_checklist
from .pipeline.data_loader import load_data_clean
from .pipeline.feature_engineering import engineer_buyer_features, engineer_exporter_features
from .pipeline.helpers import capacity_fit as _capacity_fit
from .pipeline.helpers import cert_score as _cert_score
from .pipeline.helpers import impact_weight as _impact_weight
from .pipeline.helpers import minmax as _minmax
from .pipeline.helpers import safe_float as _safe_float
from .pipeline.legacy_ranker import build_feed_for_buyer
from .pipeline.risk import news_risk_penalty

__all__ = [
    "_safe_float",
    "_minmax",
    "_cert_score",
    "_impact_weight",
    "_capacity_fit",
    "verification_checklist",
    "load_data_clean",
    "engineer_buyer_features",
    "engineer_exporter_features",
    "news_risk_penalty",
    "build_feed_for_buyer",
]
