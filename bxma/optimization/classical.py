"""
Classical Portfolio Optimization for BXMA Risk/Quant Platform.

Implements foundational optimization approaches with CVXPY:
- Mean-Variance Optimization (Markowitz)
- Minimum Variance Portfolio
- Maximum Sharpe Ratio Portfolio
- Maximum Diversification Portfolio
- Target Return/Risk Portfolios

All optimizers use disciplined convex programming via CVXPY
for guaranteed global optimality and numerical stability.

Reference:
- "Portfolio Selection" (Markowitz, 1952)
- "The Maximum Diversification Index" (Choueifaty & Coignard, 2008)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal
import numpy as np
from numpy.typing import NDArray
import cvxpy as cp


@dataclass
class OptimizationConstraints:
    """Portfolio constraints specification."""
    
    # Weight bounds
    min_weight: float = 0.0
    max_weight: float = 1.0
    
    # Long/short
    allow_short: bool = False
    max_short_weight: float = 0.0
    
    # Leverage
    max_gross_exposure: float = 1.0
    
    # Sector/group constraints
    sector_limits: dict[str, tuple[float, float]] | None = None
    
    # Turnover
    max_turnover: float | None = None
    current_weights: NDArray[np.float64] | None = None
    
    # Factor constraints
    factor_limits: dict[str, tuple[float, float]] | None = None
    factor_loadings: NDArray[np.float64] | None = None
    
    # Cardinality
    min_assets: int | None = None
    max_assets: int | None = None


@dataclass 
class OptimizationResult:
    """Portfolio optimization result."""
    
    weights: NDArray[np.float64]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    
    # Solver information
    status: str
    optimal: bool
    solve_time_ms: float
    
    # Constraint satisfaction
    turnover: float | None = None
    gross_exposure: float | None = None
    
    # Risk decomposition
    risk_contributions: NDArray[np.float64] | None = None
    marginal_risks: NDArray[np.float64] | None = None


class PortfolioOptimizer(ABC):
    """Abstract base class for portfolio optimizers."""
    
    def __init__(
        self,
        solver: str = "CLARABEL",
        verbose: bool = False,
    ):
        """
        Initialize optimizer.
        
        Args:
            solver: CVXPY solver (CLARABEL, OSQP, ECOS, SCS)
            verbose: Print solver output
        """
        self.solver = solver
        self.verbose = verbose
    
    @abstractmethod
    def optimize(
        self,
        expected_returns: NDArray[np.float64],
        covariance: NDArray[np.float64],
        constraints: OptimizationConstraints | None = None,
    ) -> OptimizationResult:
        """
        Optimize portfolio.
        
        Args:
            expected_returns: Expected returns vector (N,)
            covariance: Covariance matrix (N x N)
            constraints: Portfolio constraints
            
        Returns:
            OptimizationResult with optimal weights
        """
        pass
    
    def _add_base_constraints(
        self,
        w: cp.Variable,
        constraints: OptimizationConstraints,
        n_assets: int,
    ) -> list:
        """Add common constraints to optimization problem."""
        cons = []
        
        # Weight bounds
        if constraints.allow_short:
            cons.append(w >= -constraints.max_short_weight)
            cons.append(w <= constraints.max_weight)
        else:
            cons.append(w >= constraints.min_weight)
            cons.append(w <= constraints.max_weight)
        
        # Sum to one (or gross exposure)
        if constraints.allow_short:
            cons.append(cp.sum(w) == 1)  # Net exposure = 1
            cons.append(cp.norm(w, 1) <= constraints.max_gross_exposure)
        else:
            cons.append(cp.sum(w) == 1)
        
        # Turnover constraint
        if constraints.max_turnover is not None and constraints.current_weights is not None:
            cons.append(
                cp.norm(w - constraints.current_weights, 1) <= 2 * constraints.max_turnover
            )
        
        # Factor constraints
        if constraints.factor_loadings is not None and constraints.factor_limits is not None:
            B = constraints.factor_loadings
            for factor_name, (lb, ub) in constraints.factor_limits.items():
                factor_idx = list(constraints.factor_limits.keys()).index(factor_name)
                if factor_idx < B.shape[1]:
                    factor_exposure = B[:, factor_idx] @ w
                    cons.append(factor_exposure >= lb)
                    cons.append(factor_exposure <= ub)
        
        return cons
    
    def _compute_risk_contributions(
        self,
        weights: NDArray[np.float64],
        covariance: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute marginal and component risk contributions."""
        port_var = weights @ covariance @ weights
        port_std = np.sqrt(port_var)
        
        # Marginal risk
        marginal = (covariance @ weights) / port_std
        
        # Risk contribution
        risk_contrib = weights * marginal
        
        return risk_contrib


class MeanVarianceOptimizer(PortfolioOptimizer):
    """
    Mean-Variance Optimization (Markowitz).
    
    Maximizes expected utility: E[r] - (λ/2) * Var[r]
    
    Or equivalently, minimizes variance for target return,
    or maximizes return for target risk.
    """
    
    def __init__(
        self,
        risk_aversion: float = 1.0,
        target_return: float | None = None,
        target_risk: float | None = None,
        **kwargs
    ):
        """
        Initialize Mean-Variance optimizer.
        
        Args:
            risk_aversion: Risk aversion parameter λ
            target_return: Target portfolio return (optional)
            target_risk: Target portfolio risk (optional)
        """
        super().__init__(**kwargs)
        self.risk_aversion = risk_aversion
        self.target_return = target_return
        self.target_risk = target_risk
    
    def optimize(
        self,
        expected_returns: NDArray[np.float64],
        covariance: NDArray[np.float64],
        constraints: OptimizationConstraints | None = None,
    ) -> OptimizationResult:
        """Optimize using mean-variance framework."""
        import time
        start_time = time.time()
        
        n_assets = len(expected_returns)
        constraints = constraints or OptimizationConstraints()
        
        # Decision variable
        w = cp.Variable(n_assets)
        
        # Portfolio return and risk
        port_return = expected_returns @ w
        port_variance = cp.quad_form(w, covariance)
        
        # Build constraints
        cons = self._add_base_constraints(w, constraints, n_assets)
        
        # Objective based on mode
        if self.target_return is not None:
            # Minimize variance subject to target return
            cons.append(port_return >= self.target_return)
            objective = cp.Minimize(port_variance)
            
        elif self.target_risk is not None:
            # Maximize return subject to target risk
            cons.append(port_variance <= self.target_risk ** 2)
            objective = cp.Maximize(port_return)
            
        else:
            # Maximize utility: E[r] - (λ/2) * Var[r]
            objective = cp.Maximize(
                port_return - (self.risk_aversion / 2) * port_variance
            )
        
        # Solve
        problem = cp.Problem(objective, cons)
        
        try:
            problem.solve(solver=getattr(cp, self.solver), verbose=self.verbose)
        except:
            # Fallback solver
            problem.solve(solver=cp.ECOS, verbose=self.verbose)
        
        solve_time = (time.time() - start_time) * 1000
        
        if problem.status in ["optimal", "optimal_inaccurate"]:
            weights = w.value
            exp_ret = float(expected_returns @ weights)
            exp_risk = float(np.sqrt(weights @ covariance @ weights))
            sharpe = exp_ret / exp_risk if exp_risk > 0 else 0
            
            risk_contrib = self._compute_risk_contributions(weights, covariance)
            
            return OptimizationResult(
                weights=weights,
                expected_return=exp_ret,
                expected_risk=exp_risk,
                sharpe_ratio=sharpe,
                status=problem.status,
                optimal=problem.status == "optimal",
                solve_time_ms=solve_time,
                gross_exposure=float(np.sum(np.abs(weights))),
                risk_contributions=risk_contrib,
            )
        else:
            # Return equal weight on failure
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


class MinVarianceOptimizer(PortfolioOptimizer):
    """
    Global Minimum Variance Portfolio.
    
    Minimizes portfolio variance without return constraint.
    Often performs well out-of-sample due to reduced estimation error.
    """
    
    def optimize(
        self,
        expected_returns: NDArray[np.float64],
        covariance: NDArray[np.float64],
        constraints: OptimizationConstraints | None = None,
    ) -> OptimizationResult:
        """Find minimum variance portfolio."""
        import time
        start_time = time.time()
        
        n_assets = len(expected_returns)
        constraints = constraints or OptimizationConstraints()
        
        w = cp.Variable(n_assets)
        port_variance = cp.quad_form(w, covariance)
        
        cons = self._add_base_constraints(w, constraints, n_assets)
        
        problem = cp.Problem(cp.Minimize(port_variance), cons)
        
        try:
            problem.solve(solver=getattr(cp, self.solver), verbose=self.verbose)
        except:
            problem.solve(solver=cp.ECOS, verbose=self.verbose)
        
        solve_time = (time.time() - start_time) * 1000
        
        if problem.status in ["optimal", "optimal_inaccurate"]:
            weights = w.value
            exp_ret = float(expected_returns @ weights)
            exp_risk = float(np.sqrt(weights @ covariance @ weights))
            sharpe = exp_ret / exp_risk if exp_risk > 0 else 0
            
            return OptimizationResult(
                weights=weights,
                expected_return=exp_ret,
                expected_risk=exp_risk,
                sharpe_ratio=sharpe,
                status=problem.status,
                optimal=True,
                solve_time_ms=solve_time,
                risk_contributions=self._compute_risk_contributions(weights, covariance),
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


class MaxSharpeOptimizer(PortfolioOptimizer):
    """
    Maximum Sharpe Ratio Portfolio (Tangency Portfolio).
    
    Maximizes risk-adjusted return: (E[r] - rf) / σ
    
    Uses reformulation as convex optimization problem.
    """
    
    def __init__(self, risk_free_rate: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.risk_free_rate = risk_free_rate
    
    def optimize(
        self,
        expected_returns: NDArray[np.float64],
        covariance: NDArray[np.float64],
        constraints: OptimizationConstraints | None = None,
    ) -> OptimizationResult:
        """Find maximum Sharpe ratio portfolio."""
        import time
        start_time = time.time()
        
        n_assets = len(expected_returns)
        constraints = constraints or OptimizationConstraints()
        
        # Excess returns
        excess_returns = expected_returns - self.risk_free_rate
        
        # Check if any positive excess returns
        if np.max(excess_returns) <= 0:
            # Return minimum variance portfolio
            return MinVarianceOptimizer(solver=self.solver).optimize(
                expected_returns, covariance, constraints
            )
        
        # Reformulate for convexity (Cornuejols & Tutuncu approach)
        # Maximize (μ'w - rf) / sqrt(w'Σw)
        # Equivalent to: minimize w'Σw s.t. (μ - rf)'w = 1, w >= 0
        
        y = cp.Variable(n_assets)
        kappa = cp.Variable()  # Scaling factor
        
        # Reformulated constraints
        cons = [
            excess_returns @ y == 1,  # Normalization
            kappa >= 0,
        ]
        
        # Bounds scaled by kappa
        if constraints.allow_short:
            cons.append(y >= -constraints.max_short_weight * kappa)
            cons.append(y <= constraints.max_weight * kappa)
        else:
            cons.append(y >= constraints.min_weight * kappa)
            cons.append(y <= constraints.max_weight * kappa)
        
        cons.append(cp.sum(y) == kappa)  # Sum to kappa
        
        # Minimize variance
        objective = cp.Minimize(cp.quad_form(y, covariance))
        
        problem = cp.Problem(objective, cons)
        
        try:
            problem.solve(solver=getattr(cp, self.solver), verbose=self.verbose)
        except:
            problem.solve(solver=cp.ECOS, verbose=self.verbose)
        
        solve_time = (time.time() - start_time) * 1000
        
        if problem.status in ["optimal", "optimal_inaccurate"] and kappa.value > 1e-8:
            # Recover original weights
            weights = y.value / kappa.value
            weights = weights / np.sum(weights)  # Ensure sum to 1
            
            exp_ret = float(expected_returns @ weights)
            exp_risk = float(np.sqrt(weights @ covariance @ weights))
            sharpe = (exp_ret - self.risk_free_rate) / exp_risk if exp_risk > 0 else 0
            
            return OptimizationResult(
                weights=weights,
                expected_return=exp_ret,
                expected_risk=exp_risk,
                sharpe_ratio=sharpe,
                status=problem.status,
                optimal=True,
                solve_time_ms=solve_time,
                risk_contributions=self._compute_risk_contributions(weights, covariance),
            )
        else:
            # Fallback to minimum variance
            return MinVarianceOptimizer(solver=self.solver).optimize(
                expected_returns, covariance, constraints
            )


class MaxDiversificationOptimizer(PortfolioOptimizer):
    """
    Maximum Diversification Portfolio.
    
    Maximizes diversification ratio: Σ w_i σ_i / σ_p
    
    The ratio of weighted average volatility to portfolio volatility.
    Higher ratio = more diversification benefit.
    
    Reference:
    - "Toward Maximum Diversification" (Choueifaty & Coignard, 2008)
    """
    
    def optimize(
        self,
        expected_returns: NDArray[np.float64],
        covariance: NDArray[np.float64],
        constraints: OptimizationConstraints | None = None,
    ) -> OptimizationResult:
        """Find maximum diversification portfolio."""
        import time
        start_time = time.time()
        
        n_assets = len(expected_returns)
        constraints = constraints or OptimizationConstraints()
        
        # Asset volatilities
        vols = np.sqrt(np.diag(covariance))
        
        # Reformulation: minimize portfolio variance s.t. Σ w_i σ_i = 1
        y = cp.Variable(n_assets)
        kappa = cp.Variable()
        
        cons = [
            vols @ y == 1,  # Normalization
            kappa >= 0,
        ]
        
        if constraints.allow_short:
            cons.append(y >= -constraints.max_short_weight * kappa)
            cons.append(y <= constraints.max_weight * kappa)
        else:
            cons.append(y >= constraints.min_weight * kappa)
            cons.append(y <= constraints.max_weight * kappa)
        
        cons.append(cp.sum(y) == kappa)
        
        objective = cp.Minimize(cp.quad_form(y, covariance))
        
        problem = cp.Problem(objective, cons)
        
        try:
            problem.solve(solver=getattr(cp, self.solver), verbose=self.verbose)
        except:
            problem.solve(solver=cp.ECOS, verbose=self.verbose)
        
        solve_time = (time.time() - start_time) * 1000
        
        if problem.status in ["optimal", "optimal_inaccurate"] and kappa.value > 1e-8:
            weights = y.value / kappa.value
            weights = weights / np.sum(weights)
            
            exp_ret = float(expected_returns @ weights)
            exp_risk = float(np.sqrt(weights @ covariance @ weights))
            sharpe = exp_ret / exp_risk if exp_risk > 0 else 0
            
            # Diversification ratio
            div_ratio = float(vols @ weights) / exp_risk if exp_risk > 0 else 1.0
            
            return OptimizationResult(
                weights=weights,
                expected_return=exp_ret,
                expected_risk=exp_risk,
                sharpe_ratio=sharpe,
                status=problem.status,
                optimal=True,
                solve_time_ms=solve_time,
                risk_contributions=self._compute_risk_contributions(weights, covariance),
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
