"""
Explainable AI for BXMA Data Analytics Platform.

Implements transparent decision-making systems:
- SHAP-based risk attribution
- Decision audit trails

Provides explainability for all portfolio decisions with
27+ recorded decision factors per recommendation.

References:
- "A Unified Approach to Interpreting Model Predictions" (Lundberg & Lee, 2017)
"""

from bxma.explainability.shap_analysis import (
    SHAPExplainer,
    SHAPResult,
    RiskAttributionExplainer,
    DecisionAuditTrail,
)

__all__ = [
    "SHAPExplainer",
    "SHAPResult",
    "RiskAttributionExplainer",
    "DecisionAuditTrail",
]
