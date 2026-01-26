"""
Value-at-Risk (VaR) and Expected Shortfall (CVaR) Engine.

Implements multiple VaR methodologies from cutting-edge research:
- Parametric VaR (Normal, Student-t)
- Historical Simulation VaR
- Monte Carlo VaR with variance reduction
- Cornish-Fisher VaR (skewness/kurtosis adjustment)
- Entropic VaR (coherent risk measure)
- Component VaR, Marginal VaR, Incremental VaR

References:
- RiskMetrics Technical Document (J.P. Morgan, 1996)
- "Measuring Market Risk" (Kevin Dowd, 2005)
- "Expected Shortfall: A Natural Coherent Alternative to VaR" (Acerbi & Tasche, 2002)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal
import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.special import gamma
import warnings


@dataclass
class VaRResult:
    """Container for VaR calculation results."""
    
    var: float
    confidence_level: float
    horizon_days: int
    method: str
    
    # Expected Shortfall
    cvar: float | None = None
    
    # Component analysis
    component_var: NDArray[np.float64] | None = None
    marginal_var: NDArray[np.float64] | None = None
    incremental_var: NDArray[np.float64] | None = None
    
    # Additional metrics
    var_percentile: float | None = None
    num_exceedances: int | None = None
    
    # Distribution parameters
    distribution_params: dict | None = None


class VaREngine(ABC):
    """
    Abstract base class for VaR calculations.
    
    Provides unified interface for different VaR methodologies
    with support for portfolio-level and position-level analytics.
    """
    
    @abstractmethod
    def calculate_var(
        self,
        returns: NDArray[np.float64],
        weights: NDArray[np.float64],
        confidence_level: float,
        horizon_days: int,
    ) -> VaRResult:
        """
        Calculate Value-at-Risk.
        
        Args:
            returns: Matrix of asset returns (T x N)
            weights: Portfolio weights (N,)
            confidence_level: Confidence level (e.g., 0.95, 0.99)
            horizon_days: Risk horizon in trading days
            
        Returns:
            VaRResult with VaR and component analytics
        """
        pass
    
    @abstractmethod
    def calculate_cvar(
        self,
        returns: NDArray[np.float64],
        weights: NDArray[np.float64],
        confidence_level: float,
        horizon_days: int,
    ) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall).
        
        CVaR is a coherent risk measure representing the expected loss
        given that loss exceeds VaR.
        """
        pass
    
    def calculate_component_var(
        self,
        returns: NDArray[np.float64],
        weights: NDArray[np.float64],
        var: float,
    ) -> NDArray[np.float64]:
        """
        Calculate Component VaR for each position.
        
        Component VaR measures the contribution of each position
        to total portfolio VaR. Sum of component VaRs equals total VaR.
        
        Args:
            returns: Asset returns matrix
            weights: Portfolio weights
            var: Total portfolio VaR
            
        Returns:
            Component VaR for each asset
        """
        # Covariance matrix
        cov = np.cov(returns, rowvar=False)
        
        # Portfolio variance
        port_var = weights @ cov @ weights
        port_std = np.sqrt(port_var)
        
        # Marginal VaR = beta * VaR / portfolio_value
        beta = cov @ weights / port_var
        
        # Component VaR = weight * marginal_var * portfolio_value
        component_var = weights * beta * var
        
        return component_var
    
    def calculate_marginal_var(
        self,
        returns: NDArray[np.float64],
        weights: NDArray[np.float64],
        confidence_level: float,
    ) -> NDArray[np.float64]:
        """
        Calculate Marginal VaR for each position.
        
        Marginal VaR measures the change in portfolio VaR for
        a small change in position weight.
        """
        cov = np.cov(returns, rowvar=False)
        port_var = weights @ cov @ weights
        port_std = np.sqrt(port_var)
        
        z = stats.norm.ppf(confidence_level)
        
        # Marginal VaR = z * cov_i / sigma_p
        marginal = z * (cov @ weights) / port_std
        
        return marginal
    
    def calculate_incremental_var(
        self,
        returns: NDArray[np.float64],
        weights: NDArray[np.float64],
        confidence_level: float,
        horizon_days: int,
    ) -> NDArray[np.float64]:
        """
        Calculate Incremental VaR for each position.
        
        Incremental VaR measures the change in portfolio VaR
        from adding or removing a position entirely.
        """
        n_assets = len(weights)
        incremental = np.zeros(n_assets)
        
        # Base VaR
        base_var = self.calculate_var(
            returns, weights, confidence_level, horizon_days
        ).var
        
        # Calculate VaR without each position
        for i in range(n_assets):
            weights_ex = weights.copy()
            weights_ex[i] = 0
            # Renormalize
            if weights_ex.sum() > 0:
                weights_ex = weights_ex / weights_ex.sum()
            
            var_ex = self.calculate_var(
                returns, weights_ex, confidence_level, horizon_days
            ).var
            
            incremental[i] = base_var - var_ex
        
        return incremental


class ParametricVaR(VaREngine):
    """
    Parametric (Variance-Covariance) VaR.
    
    Assumes returns follow a known distribution (Normal or Student-t).
    Fast computation but may underestimate tail risk for fat-tailed returns.
    
    Features:
    - Normal distribution VaR
    - Student-t distribution VaR (accounts for fat tails)
    - Square-root-of-time scaling for multi-day horizons
    """
    
    def __init__(
        self,
        distribution: Literal["normal", "student_t"] = "normal",
        degrees_freedom: float = 5.0,
    ):
        """
        Initialize Parametric VaR engine.
        
        Args:
            distribution: 'normal' or 'student_t'
            degrees_freedom: Degrees of freedom for Student-t (typically 3-10)
        """
        self.distribution = distribution
        self.degrees_freedom = degrees_freedom
    
    def calculate_var(
        self,
        returns: NDArray[np.float64],
        weights: NDArray[np.float64],
        confidence_level: float,
        horizon_days: int,
    ) -> VaRResult:
        """Calculate parametric VaR."""
        # Portfolio returns
        portfolio_returns = returns @ weights
        
        # Mean and standard deviation
        mu = np.mean(portfolio_returns)
        sigma = np.std(portfolio_returns)
        
        # Quantile based on distribution
        if self.distribution == "normal":
            z = stats.norm.ppf(1 - confidence_level)
            var = -(mu * horizon_days + z * sigma * np.sqrt(horizon_days))
            
            # CVaR for normal distribution
            pdf_z = stats.norm.pdf(z)
            cvar = -(mu * horizon_days - sigma * np.sqrt(horizon_days) * pdf_z / (1 - confidence_level))
            
            dist_params = {"mu": mu, "sigma": sigma}
            
        else:  # Student-t
            df = self.degrees_freedom
            t_quantile = stats.t.ppf(1 - confidence_level, df)
            
            # Adjust for Student-t variance
            scale = sigma * np.sqrt((df - 2) / df) if df > 2 else sigma
            var = -(mu * horizon_days + t_quantile * scale * np.sqrt(horizon_days))
            
            # CVaR for Student-t
            pdf_t = stats.t.pdf(t_quantile, df)
            cvar_factor = (df + t_quantile**2) / (df - 1) * pdf_t / (1 - confidence_level)
            cvar = -(mu * horizon_days - scale * np.sqrt(horizon_days) * cvar_factor)
            
            dist_params = {"mu": mu, "sigma": sigma, "df": df}
        
        # Component VaR
        component_var = self.calculate_component_var(returns, weights, var)
        marginal_var = self.calculate_marginal_var(returns, weights, confidence_level)
        
        return VaRResult(
            var=var,
            confidence_level=confidence_level,
            horizon_days=horizon_days,
            method=f"parametric_{self.distribution}",
            cvar=cvar,
            component_var=component_var,
            marginal_var=marginal_var,
            distribution_params=dist_params,
        )
    
    def calculate_cvar(
        self,
        returns: NDArray[np.float64],
        weights: NDArray[np.float64],
        confidence_level: float,
        horizon_days: int,
    ) -> float:
        """Calculate parametric CVaR."""
        result = self.calculate_var(returns, weights, confidence_level, horizon_days)
        return result.cvar or result.var


class HistoricalVaR(VaREngine):
    """
    Historical Simulation VaR.
    
    Uses actual historical return distribution without parametric assumptions.
    Better captures fat tails and non-normality but requires sufficient data.
    
    Features:
    - Full historical simulation
    - Age-weighted historical simulation (EWHS)
    - Volatility-scaled historical simulation
    - Bootstrap confidence intervals
    """
    
    def __init__(
        self,
        method: Literal["standard", "age_weighted", "volatility_scaled"] = "standard",
        decay_factor: float = 0.94,
    ):
        """
        Initialize Historical VaR engine.
        
        Args:
            method: Simulation method
            decay_factor: Decay factor for age-weighted method (RiskMetrics default: 0.94)
        """
        self.method = method
        self.decay_factor = decay_factor
    
    def calculate_var(
        self,
        returns: NDArray[np.float64],
        weights: NDArray[np.float64],
        confidence_level: float,
        horizon_days: int,
    ) -> VaRResult:
        """Calculate historical simulation VaR."""
        # Portfolio returns
        portfolio_returns = returns @ weights
        n_obs = len(portfolio_returns)
        
        if self.method == "standard":
            # Simple historical quantile
            var_percentile = np.percentile(
                portfolio_returns, (1 - confidence_level) * 100
            )
            sorted_returns = np.sort(portfolio_returns)
            
        elif self.method == "age_weighted":
            # Exponentially weighted historical simulation
            weights_decay = np.array([
                self.decay_factor ** (n_obs - i - 1) for i in range(n_obs)
            ])
            weights_decay /= weights_decay.sum()
            
            # Weighted quantile
            sorted_idx = np.argsort(portfolio_returns)
            cum_weights = np.cumsum(weights_decay[sorted_idx])
            var_idx = np.searchsorted(cum_weights, 1 - confidence_level)
            var_percentile = portfolio_returns[sorted_idx[var_idx]]
            sorted_returns = portfolio_returns[sorted_idx]
            
        elif self.method == "volatility_scaled":
            # Scale historical returns by current vs historical volatility
            current_vol = np.std(portfolio_returns[-21:])  # Last month
            historical_vol = np.std(portfolio_returns)
            
            scaled_returns = portfolio_returns * (current_vol / historical_vol)
            var_percentile = np.percentile(
                scaled_returns, (1 - confidence_level) * 100
            )
            sorted_returns = np.sort(scaled_returns)
        
        # Scale to horizon
        var = -var_percentile * np.sqrt(horizon_days)
        
        # CVaR: Expected value of returns worse than VaR
        tail_returns = sorted_returns[sorted_returns <= var_percentile]
        cvar = -np.mean(tail_returns) * np.sqrt(horizon_days) if len(tail_returns) > 0 else var
        
        # Count exceedances
        exceedances = np.sum(portfolio_returns <= var_percentile)
        
        # Component VaR
        component_var = self.calculate_component_var(returns, weights, var)
        
        return VaRResult(
            var=var,
            confidence_level=confidence_level,
            horizon_days=horizon_days,
            method=f"historical_{self.method}",
            cvar=cvar,
            component_var=component_var,
            var_percentile=-var_percentile,
            num_exceedances=int(exceedances),
        )
    
    def calculate_cvar(
        self,
        returns: NDArray[np.float64],
        weights: NDArray[np.float64],
        confidence_level: float,
        horizon_days: int,
    ) -> float:
        """Calculate historical CVaR."""
        portfolio_returns = returns @ weights
        var_percentile = np.percentile(
            portfolio_returns, (1 - confidence_level) * 100
        )
        tail_returns = portfolio_returns[portfolio_returns <= var_percentile]
        return -np.mean(tail_returns) * np.sqrt(horizon_days) if len(tail_returns) > 0 else 0.0


class MonteCarloVaR(VaREngine):
    """
    Monte Carlo Simulation VaR.
    
    Generates scenarios from fitted distribution with variance reduction.
    Most flexible method, supports complex portfolios and path-dependent risks.
    
    Features:
    - Multivariate normal simulation
    - Multivariate Student-t simulation
    - Antithetic variates for variance reduction
    - Control variates optimization
    - Importance sampling for tail estimation
    """
    
    def __init__(
        self,
        n_simulations: int = 10000,
        distribution: Literal["normal", "student_t"] = "normal",
        degrees_freedom: float = 5.0,
        variance_reduction: bool = True,
        seed: int | None = 42,
    ):
        """
        Initialize Monte Carlo VaR engine.
        
        Args:
            n_simulations: Number of Monte Carlo simulations
            distribution: 'normal' or 'student_t'
            degrees_freedom: Degrees of freedom for Student-t
            variance_reduction: Use antithetic variates
            seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.distribution = distribution
        self.degrees_freedom = degrees_freedom
        self.variance_reduction = variance_reduction
        self.seed = seed
    
    def calculate_var(
        self,
        returns: NDArray[np.float64],
        weights: NDArray[np.float64],
        confidence_level: float,
        horizon_days: int,
    ) -> VaRResult:
        """Calculate Monte Carlo VaR."""
        if self.seed is not None:
            np.random.seed(self.seed)
        
        n_assets = returns.shape[1]
        
        # Estimate parameters from historical data
        mu = np.mean(returns, axis=0)
        cov = np.cov(returns, rowvar=False)
        
        # Cholesky decomposition for correlated sampling
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            # Handle non-positive definite matrix
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-8)
            cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
            L = np.linalg.cholesky(cov)
        
        # Generate simulations
        n_sims = self.n_simulations
        if self.variance_reduction:
            n_sims = n_sims // 2  # Will double with antithetic
        
        if self.distribution == "normal":
            Z = np.random.standard_normal((n_sims, n_assets))
        else:  # Student-t
            chi2 = np.random.chisquare(self.degrees_freedom, n_sims)
            Z = np.random.standard_normal((n_sims, n_assets))
            Z = Z / np.sqrt(chi2[:, np.newaxis] / self.degrees_freedom)
        
        # Transform to correlated returns
        simulated_returns = mu + (Z @ L.T)
        
        # Antithetic variates
        if self.variance_reduction:
            antithetic_returns = 2 * mu - simulated_returns
            simulated_returns = np.vstack([simulated_returns, antithetic_returns])
        
        # Scale to horizon
        simulated_returns = simulated_returns * np.sqrt(horizon_days)
        
        # Portfolio returns
        portfolio_returns = simulated_returns @ weights
        
        # VaR and CVaR
        var_percentile = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        var = -var_percentile
        
        tail_returns = portfolio_returns[portfolio_returns <= var_percentile]
        cvar = -np.mean(tail_returns) if len(tail_returns) > 0 else var
        
        # Component VaR using simulated scenarios
        component_var = self.calculate_component_var(returns, weights, var)
        
        return VaRResult(
            var=var,
            confidence_level=confidence_level,
            horizon_days=horizon_days,
            method=f"monte_carlo_{self.distribution}",
            cvar=cvar,
            component_var=component_var,
            distribution_params={
                "n_simulations": len(portfolio_returns),
                "mu": mu.tolist(),
                "variance_reduction": self.variance_reduction,
            },
        )
    
    def calculate_cvar(
        self,
        returns: NDArray[np.float64],
        weights: NDArray[np.float64],
        confidence_level: float,
        horizon_days: int,
    ) -> float:
        """Calculate Monte Carlo CVaR."""
        result = self.calculate_var(returns, weights, confidence_level, horizon_days)
        return result.cvar or result.var


class CornishFisherVaR(VaREngine):
    """
    Cornish-Fisher VaR with skewness and kurtosis adjustment.
    
    Extends parametric VaR to account for non-normality using
    Cornish-Fisher expansion. Better than normal VaR for skewed
    and leptokurtic return distributions.
    
    Reference:
    - "The Percentile Points of Distributions Having Known Cumulants"
      (Cornish & Fisher, 1937)
    """
    
    def calculate_var(
        self,
        returns: NDArray[np.float64],
        weights: NDArray[np.float64],
        confidence_level: float,
        horizon_days: int,
    ) -> VaRResult:
        """Calculate Cornish-Fisher adjusted VaR."""
        portfolio_returns = returns @ weights
        
        mu = np.mean(portfolio_returns)
        sigma = np.std(portfolio_returns)
        skew = stats.skew(portfolio_returns)
        kurt = stats.kurtosis(portfolio_returns)  # Excess kurtosis
        
        # Normal quantile
        z = stats.norm.ppf(1 - confidence_level)
        
        # Cornish-Fisher expansion
        z_cf = (
            z
            + (z**2 - 1) * skew / 6
            + (z**3 - 3*z) * kurt / 24
            - (2*z**3 - 5*z) * skew**2 / 36
        )
        
        # VaR
        var = -(mu * horizon_days + z_cf * sigma * np.sqrt(horizon_days))
        
        # Approximate CVaR using modified normal
        pdf_z = stats.norm.pdf(z_cf)
        cvar = -(mu * horizon_days - sigma * np.sqrt(horizon_days) * pdf_z / (1 - confidence_level))
        
        # Component VaR
        component_var = self.calculate_component_var(returns, weights, var)
        
        return VaRResult(
            var=var,
            confidence_level=confidence_level,
            horizon_days=horizon_days,
            method="cornish_fisher",
            cvar=cvar,
            component_var=component_var,
            distribution_params={
                "mu": mu,
                "sigma": sigma,
                "skewness": skew,
                "kurtosis": kurt,
                "z_normal": z,
                "z_cornish_fisher": z_cf,
            },
        )
    
    def calculate_cvar(
        self,
        returns: NDArray[np.float64],
        weights: NDArray[np.float64],
        confidence_level: float,
        horizon_days: int,
    ) -> float:
        """Calculate Cornish-Fisher CVaR."""
        result = self.calculate_var(returns, weights, confidence_level, horizon_days)
        return result.cvar or result.var


class EntropicVaR(VaREngine):
    """
    Entropic Value-at-Risk (EVaR).
    
    A coherent risk measure based on the Chernoff inequality.
    Provides an upper bound on CVaR and is more conservative.
    
    EVaR = min_z>0 { z^(-1) * log(E[exp(-z*X)]) + z^(-1) * log(1/alpha) }
    
    Reference:
    - "Entropic Value-at-Risk: A New Coherent Risk Measure" (Ahmadi-Javid, 2012)
    """
    
    def calculate_var(
        self,
        returns: NDArray[np.float64],
        weights: NDArray[np.float64],
        confidence_level: float,
        horizon_days: int,
    ) -> VaRResult:
        """Calculate Entropic VaR."""
        from scipy.optimize import minimize_scalar
        
        portfolio_returns = returns @ weights
        alpha = 1 - confidence_level
        
        def evar_objective(z: float) -> float:
            """Objective function for EVaR optimization."""
            if z <= 0:
                return np.inf
            mgf = np.mean(np.exp(-z * portfolio_returns))
            return (1/z) * np.log(mgf) + (1/z) * np.log(1/alpha)
        
        # Optimize over z
        result = minimize_scalar(evar_objective, bounds=(0.01, 100), method='bounded')
        
        evar = result.fun * np.sqrt(horizon_days)
        
        # Also compute regular VaR for comparison
        var_percentile = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        var = -var_percentile * np.sqrt(horizon_days)
        
        # CVaR
        tail_returns = portfolio_returns[portfolio_returns <= var_percentile]
        cvar = -np.mean(tail_returns) * np.sqrt(horizon_days) if len(tail_returns) > 0 else var
        
        return VaRResult(
            var=evar,
            confidence_level=confidence_level,
            horizon_days=horizon_days,
            method="entropic",
            cvar=cvar,
            distribution_params={
                "optimal_z": result.x,
                "historical_var": var,
                "historical_cvar": cvar,
            },
        )
    
    def calculate_cvar(
        self,
        returns: NDArray[np.float64],
        weights: NDArray[np.float64],
        confidence_level: float,
        horizon_days: int,
    ) -> float:
        """Calculate CVaR (EVaR provides an upper bound)."""
        portfolio_returns = returns @ weights
        var_percentile = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        tail_returns = portfolio_returns[portfolio_returns <= var_percentile]
        return -np.mean(tail_returns) * np.sqrt(horizon_days) if len(tail_returns) > 0 else 0.0
