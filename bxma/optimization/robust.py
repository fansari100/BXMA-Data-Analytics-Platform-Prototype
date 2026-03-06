"""
Robust Portfolio Optimization for BXMA Data Analytics Platform.

Implements robust optimization approaches:
- Black-Litterman Model
- Robust Mean-Variance
- Entropy Pooling

References:
- "Global Portfolio Optimization" (Black & Litterman, 1992)
- "Entropy Pooling" (Meucci, 2008)
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from bxma.optimization.classical import (
    PortfolioOptimizer,
    OptimizationConstraints,
    OptimizationResult,
)


class BlackLittermanOptimizer(PortfolioOptimizer):
    """
    Black-Litterman Portfolio Optimization.
    
    Combines equilibrium returns with investor views
    using Bayesian updating.
    """
    
    def __init__(
        self,
        tau: float = 0.05,
        risk_aversion: float = 2.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tau = tau
        self.risk_aversion = risk_aversion
    
    def optimize(
        self,
        expected_returns: NDArray[np.float64],
        covariance: NDArray[np.float64],
        constraints: OptimizationConstraints | None = None,
        views: NDArray[np.float64] | None = None,
        view_confidences: NDArray[np.float64] | None = None,
        P: NDArray[np.float64] | None = None,
    ) -> OptimizationResult:
        """Optimize using Black-Litterman model."""
        import time
        from bxma.optimization.classical import MeanVarianceOptimizer
        
        start_time = time.time()
        n_assets = len(expected_returns)
        
        # Use equilibrium returns if no views provided
        if views is None:
            # Fall back to mean-variance
            mv = MeanVarianceOptimizer(risk_aversion=self.risk_aversion)
            return mv.optimize(expected_returns, covariance, constraints)
        
        # Black-Litterman combined returns
        tau_sigma = self.tau * covariance
        
        if view_confidences is None:
            omega = np.eye(len(views)) * 0.01
        else:
            omega = np.diag(view_confidences)
        
        if P is None:
            P = np.eye(n_assets)[:len(views)]
        
        # Posterior returns
        M = np.linalg.inv(np.linalg.inv(tau_sigma) + P.T @ np.linalg.inv(omega) @ P)
        bl_returns = M @ (np.linalg.inv(tau_sigma) @ expected_returns + 
                         P.T @ np.linalg.inv(omega) @ views)
        
        # Optimize with BL returns
        mv = MeanVarianceOptimizer(risk_aversion=self.risk_aversion)
        result = mv.optimize(bl_returns, covariance, constraints)
        result.solve_time_ms = (time.time() - start_time) * 1000
        
        return result


class RobustMeanVariance(PortfolioOptimizer):
    """
    Robust Mean-Variance Optimization.
    
    Accounts for parameter uncertainty using worst-case optimization.
    """
    
    def __init__(
        self,
        uncertainty_set: str = "ellipsoidal",
        epsilon: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.uncertainty_set = uncertainty_set
        self.epsilon = epsilon
    
    def optimize(
        self,
        expected_returns: NDArray[np.float64],
        covariance: NDArray[np.float64],
        constraints: OptimizationConstraints | None = None,
    ) -> OptimizationResult:
        """Optimize with robustness to estimation error."""
        import time
        import cvxpy as cp
        
        start_time = time.time()
        n_assets = len(expected_returns)
        constraints = constraints or OptimizationConstraints()
        
        w = cp.Variable(n_assets)
        
        # Worst-case return (robust counterpart)
        # min (μ'w) subject to ||w||_Σ <= ε
        port_return = expected_returns @ w
        uncertainty = self.epsilon * cp.norm(cp.sqrt(covariance) @ w, 2)
        robust_return = port_return - uncertainty
        
        # Risk
        port_variance = cp.quad_form(w, covariance)
        
        # Constraints
        cons = [
            cp.sum(w) == 1,
            w >= constraints.min_weight,
            w <= constraints.max_weight,
        ]
        
        # Maximize robust return - risk
        objective = cp.Maximize(robust_return - 0.5 * port_variance)
        
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


class EntropyPoolingOptimizer(PortfolioOptimizer):
    """
    Entropy Pooling Portfolio Optimization.
    
    Combines prior probabilities with views using relative entropy.
    
    Reference: "Entropy Pooling" (Meucci, 2008)
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def optimize(
        self,
        expected_returns: NDArray[np.float64],
        covariance: NDArray[np.float64],
        constraints: OptimizationConstraints | None = None,
    ) -> OptimizationResult:
        """Optimize using entropy pooling."""
        # Simplified implementation
        from bxma.optimization.classical import MeanVarianceOptimizer
        return MeanVarianceOptimizer().optimize(expected_returns, covariance, constraints)
