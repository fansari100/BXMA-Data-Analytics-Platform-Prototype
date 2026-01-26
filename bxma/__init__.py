"""
BXMA Risk/Quant Analytics Platform
===================================

Blackstone Multi-Asset Investing's bleeding-edge quantitative analytics system
for portfolio analytics, risk modeling, optimization, and performance attribution.

Architecture Overview:
- data/: High-performance data infrastructure (Polars, DuckDB, Arrow)
- risk/: Risk models (VaR, CVaR, Factor Models, DCC-GARCH, Copulas)
- optimization/: Portfolio optimization (Convex, Differentiable, RL-based)
- attribution/: Performance attribution (Brinson-Fachler, Geometric, Multi-period)
- construction/: Portfolio construction (HRP, Black-Litterman, Risk Parity)
- ml/: Machine learning models (LSTM, Transformers, Bayesian NN)
- reporting/: Risk reporting stack (RiskMetrics integration, dashboards)
- streaming/: Real-time analytics pipeline
- explainability/: SHAP, Bayesian networks for transparent decisions

January 2026 - Built on cutting-edge academic research and production-grade systems.
"""

__version__ = "1.0.0"
__author__ = "BXMA Risk/Quant Team"

from bxma.core.config import BXMAConfig
from bxma.core.types import (
    AssetClass,
    RiskMeasure,
    OptimizationObjective,
    AttributionMethod,
)

__all__ = [
    "BXMAConfig",
    "AssetClass",
    "RiskMeasure",
    "OptimizationObjective",
    "AttributionMethod",
    "__version__",
]
