"""
Advanced Risk Modeling for BXMA Risk/Quant Platform.

Implements cutting-edge risk analytics:
- Value-at-Risk (VaR): Parametric, Historical, Monte Carlo, Cornish-Fisher
- Expected Shortfall (CVaR): Tail risk measures
- Factor Models: Statistical (PCA), Fundamental, DCC-GARCH
- Copula Models: Tail dependence and extreme value theory
- Regime Detection: HMM, Neural regime switching
"""

from bxma.risk.var import (
    VaREngine,
    ParametricVaR,
    HistoricalVaR,
    MonteCarloVaR,
    CornishFisherVaR,
)
from bxma.risk.factor_models import (
    FactorModel,
    StatisticalFactorModel,
    FundamentalFactorModel,
    DynamicFactorModel,
)
from bxma.risk.covariance import (
    CovarianceEstimator,
    SampleCovariance,
    LedoitWolfCovariance,
    DCCGARCHCovariance,
    ExponentialCovariance,
)
from bxma.risk.regime import (
    RegimeDetector,
    HMMRegimeDetector,
    NeuralRegimeDetector,
)

__all__ = [
    "VaREngine",
    "ParametricVaR",
    "HistoricalVaR",
    "MonteCarloVaR",
    "CornishFisherVaR",
    "FactorModel",
    "StatisticalFactorModel",
    "FundamentalFactorModel",
    "DynamicFactorModel",
    "CovarianceEstimator",
    "SampleCovariance",
    "LedoitWolfCovariance",
    "DCCGARCHCovariance",
    "ExponentialCovariance",
    "RegimeDetector",
    "HMMRegimeDetector",
    "NeuralRegimeDetector",
]
