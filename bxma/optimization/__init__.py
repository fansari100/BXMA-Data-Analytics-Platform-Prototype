"""
Portfolio Optimization Engine for BXMA Risk/Quant Platform.

Implements cutting-edge optimization techniques:
- Classical: Mean-Variance, Min Variance, Max Sharpe
- Risk Parity: Equal Risk Contribution, Hierarchical Risk Parity
- Robust: Worst-case, Black-Litterman, Entropy Pooling
- ML-Based: Deep RL, Differentiable Optimization, Neural Networks
- CVaR: Mean-CVaR, Min-CVaR optimization

References:
- "Portfolio Selection" (Markowitz, 1952)
- "Global Portfolio Optimization" (Black & Litterman, 1992)
- "A Unified Framework for Portfolio Optimization" (Roncalli, 2010)
- "Hierarchical Risk Parity" (Lopez de Prado, 2016)
"""

from bxma.optimization.classical import (
    MeanVarianceOptimizer,
    MinVarianceOptimizer,
    MaxSharpeOptimizer,
    MaxDiversificationOptimizer,
)
from bxma.optimization.risk_parity import (
    RiskParityOptimizer,
    HierarchicalRiskParity,
    NestedClusteredOptimization,
)
from bxma.optimization.robust import (
    BlackLittermanOptimizer,
    RobustMeanVariance,
    EntropyPoolingOptimizer,
)
from bxma.optimization.cvar import (
    MeanCVaROptimizer,
    MinCVaROptimizer,
)
from bxma.optimization.ml_optimizer import (
    NeuralPortfolioOptimizer,
    DifferentiableOptimizer,
    RLPortfolioOptimizer,
)

__all__ = [
    "MeanVarianceOptimizer",
    "MinVarianceOptimizer",
    "MaxSharpeOptimizer",
    "MaxDiversificationOptimizer",
    "RiskParityOptimizer",
    "HierarchicalRiskParity",
    "NestedClusteredOptimization",
    "BlackLittermanOptimizer",
    "RobustMeanVariance",
    "EntropyPoolingOptimizer",
    "MeanCVaROptimizer",
    "MinCVaROptimizer",
    "NeuralPortfolioOptimizer",
    "DifferentiableOptimizer",
    "RLPortfolioOptimizer",
]
