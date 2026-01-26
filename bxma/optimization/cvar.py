"""
CVaR-Based Portfolio Optimization for BXMA Risk/Quant Platform.

Implements CVaR (Expected Shortfall) optimization:
- Mean-CVaR Optimization
- Min-CVaR Optimization

References:
- "Optimization of Conditional Value-at-Risk" (Rockafellar & Uryasev, 2000)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import cvxpy as cp

from bxma.optimization.classical import (
    PortfolioOptimizer,
    OptimizationConstraints,
    OptimizationResult,
)


class MeanCVaROptimizer(PortfolioOptimizer):
    """
    Mean-CVaR Portfolio Optimization.
    
    Maximizes expected return subject to CVaR constraint.
    """
    
    def __init__(
        self,
        confidence_level: float = 0.95,
        cvar_limit: float = 0.05,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.confidence_level = confidence_level
        self.cvar_limit = cvar_limit
    
    def optimize(
        self,
        expected_returns: NDArray[np.float64],
        covariance: NDArray[np.float64],
        constraints: OptimizationConstraints | None = None,
        scenarios: NDArray[np.float64] | None = None,
    ) -> OptimizationResult:
        """Optimize Mean-CVaR portfolio."""
        import time
        
        start_time = time.time()
        n_assets = len(expected_returns)
        constraints = constraints or OptimizationConstraints()
        
        # Generate scenarios if not provided
        if scenarios is None:
            n_scenarios = 1000
            np.random.seed(42)
            L = np.linalg.cholesky(covariance)
            Z = np.random.randn(n_scenarios, n_assets)
            scenarios = expected_returns + Z @ L.T
        
        n_scenarios = len(scenarios)
        alpha = self.confidence_level
        
        # Variables
        w = cp.Variable(n_assets)
        var = cp.Variable()
        u = cp.Variable(n_scenarios)  # Auxiliary for CVaR
        
        # CVaR constraints (Rockafellar-Uryasev formulation)
        portfolio_returns = scenarios @ w
        
        cons = [
            u >= 0,
            u >= -portfolio_returns - var,
            cp.sum(w) == 1,
            w >= constraints.min_weight,
            w <= constraints.max_weight,
        ]
        
        # CVaR = var + (1/(1-alpha)) * E[max(-r_p - var, 0)]
        cvar = var + cp.sum(u) / (n_scenarios * (1 - alpha))
        cons.append(cvar <= self.cvar_limit)
        
        # Maximize expected return
        objective = cp.Maximize(expected_returns @ w)
        
        problem = cp.Problem(objective, cons)
        problem.solve(solver=cp.ECOS)
        
        solve_time = (time.time() - start_time) * 1000
        
        if problem.status in ["optimal", "optimal_inaccurate"]:
            weights = w.value
            return OptimizationResult(
                weights=weights,
                expected_return=float(expected_returns @ weights),
                expected_risk=float(np.sqrt(weights @ covariance @ weights)),
                sharpe_ratio=float(expected_returns @ weights / np.sqrt(weights @ covariance @ weights)),
                status=problem.status,
                optimal=True,
                solve_time_ms=solve_time,
            )
        else:
            weights = np.ones(n_assets) / n_assets
            return OptimizationResult(
                weights=weights,
                expected_return=float(expected_returns @ weights),
                expected_risk=float(np.sqrt(weights @ covariance @ weights)),
                sharpe_ratio=0.0,
                status=problem.status,
                optimal=False,
                solve_time_ms=solve_time,
            )


class MinCVaROptimizer(PortfolioOptimizer):
    """
    Minimum CVaR Portfolio Optimization.
    
    Minimizes CVaR (Expected Shortfall) directly.
    """
    
    def __init__(
        self,
        confidence_level: float = 0.95,
        target_return: float | None = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.confidence_level = confidence_level
        self.target_return = target_return
    
    def optimize(
        self,
        expected_returns: NDArray[np.float64],
        covariance: NDArray[np.float64],
        constraints: OptimizationConstraints | None = None,
        scenarios: NDArray[np.float64] | None = None,
    ) -> OptimizationResult:
        """Optimize Min-CVaR portfolio."""
        import time
        
        start_time = time.time()
        n_assets = len(expected_returns)
        constraints = constraints or OptimizationConstraints()
        
        # Generate scenarios if not provided
        if scenarios is None:
            n_scenarios = 1000
            np.random.seed(42)
            L = np.linalg.cholesky(covariance)
            Z = np.random.randn(n_scenarios, n_assets)
            scenarios = expected_returns + Z @ L.T
        
        n_scenarios = len(scenarios)
        alpha = self.confidence_level
        
        # Variables
        w = cp.Variable(n_assets)
        var = cp.Variable()
        u = cp.Variable(n_scenarios)
        
        portfolio_returns = scenarios @ w
        
        cons = [
            u >= 0,
            u >= -portfolio_returns - var,
            cp.sum(w) == 1,
            w >= constraints.min_weight,
            w <= constraints.max_weight,
        ]
        
        # Target return constraint
        if self.target_return is not None:
            cons.append(expected_returns @ w >= self.target_return)
        
        # Minimize CVaR
        cvar = var + cp.sum(u) / (n_scenarios * (1 - alpha))
        objective = cp.Minimize(cvar)
        
        problem = cp.Problem(objective, cons)
        problem.solve(solver=cp.ECOS)
        
        solve_time = (time.time() - start_time) * 1000
        
        if problem.status in ["optimal", "optimal_inaccurate"]:
            weights = w.value
            return OptimizationResult(
                weights=weights,
                expected_return=float(expected_returns @ weights),
                expected_risk=float(np.sqrt(weights @ covariance @ weights)),
                sharpe_ratio=float(expected_returns @ weights / np.sqrt(weights @ covariance @ weights)),
                status=problem.status,
                optimal=True,
                solve_time_ms=solve_time,
            )
        else:
            weights = np.ones(n_assets) / n_assets
            return OptimizationResult(
                weights=weights,
                expected_return=float(expected_returns @ weights),
                expected_risk=float(np.sqrt(weights @ covariance @ weights)),
                sharpe_ratio=0.0,
                status=problem.status,
                optimal=False,
                solve_time_ms=solve_time,
            )
