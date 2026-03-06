"""
Factor-Based Attribution for BXMA Data Analytics Platform.

Implements factor attribution methodologies:
- Factor Return Attribution
- Risk Attribution

References:
- "Multi-Factor Attribution" (Barra, 1998)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray


@dataclass
class FactorAttributionResult:
    """Result of factor-based attribution."""
    
    total_return: float
    factor_return: float
    specific_return: float
    
    # Factor contributions
    factor_contributions: dict[str, float] = field(default_factory=dict)
    
    # Factor exposures used
    factor_exposures: dict[str, float] = field(default_factory=dict)
    
    # Factor returns
    factor_returns: dict[str, float] = field(default_factory=dict)


class FactorAttribution:
    """
    Factor-Based Performance Attribution.
    
    Decomposes returns into factor contributions and
    security-specific (alpha) returns.
    """
    
    def __init__(self):
        pass
    
    def calculate(
        self,
        weights: NDArray[np.float64],
        returns: NDArray[np.float64],
        factor_loadings: NDArray[np.float64],
        factor_returns: NDArray[np.float64],
        factor_names: list[str] | None = None,
    ) -> FactorAttributionResult:
        """
        Calculate factor attribution.
        
        Args:
            weights: Portfolio weights
            returns: Asset returns
            factor_loadings: Factor loadings (N_assets x K_factors)
            factor_returns: Factor returns
            factor_names: Factor names
            
        Returns:
            FactorAttributionResult
        """
        n_factors = len(factor_returns)
        
        if factor_names is None:
            factor_names = [f"Factor_{i}" for i in range(n_factors)]
        
        # Portfolio factor exposures
        port_exposures = factor_loadings.T @ weights
        
        # Factor contribution to return
        factor_contrib = port_exposures * factor_returns
        total_factor_return = float(np.sum(factor_contrib))
        
        # Total portfolio return
        total_return = float(weights @ returns)
        
        # Specific return (alpha)
        specific_return = total_return - total_factor_return
        
        # Build result
        contributions = {
            factor_names[i]: float(factor_contrib[i])
            for i in range(n_factors)
        }
        
        exposures = {
            factor_names[i]: float(port_exposures[i])
            for i in range(n_factors)
        }
        
        f_returns = {
            factor_names[i]: float(factor_returns[i])
            for i in range(n_factors)
        }
        
        return FactorAttributionResult(
            total_return=total_return,
            factor_return=total_factor_return,
            specific_return=specific_return,
            factor_contributions=contributions,
            factor_exposures=exposures,
            factor_returns=f_returns,
        )


@dataclass
class RiskAttributionResult:
    """Result of risk attribution."""
    
    total_risk: float
    factor_risk: float
    specific_risk: float
    
    # Factor risk contributions
    factor_risk_contributions: dict[str, float] = field(default_factory=dict)
    
    # Percentage breakdown
    factor_risk_pct: float = 0.0
    specific_risk_pct: float = 0.0


class RiskAttribution:
    """
    Risk Attribution Analysis.
    
    Decomposes portfolio risk into factor and specific components.
    """
    
    def __init__(self):
        pass
    
    def calculate(
        self,
        weights: NDArray[np.float64],
        factor_loadings: NDArray[np.float64],
        factor_covariance: NDArray[np.float64],
        specific_variances: NDArray[np.float64],
        factor_names: list[str] | None = None,
    ) -> RiskAttributionResult:
        """
        Calculate risk attribution.
        
        Args:
            weights: Portfolio weights
            factor_loadings: Factor loadings (N x K)
            factor_covariance: Factor covariance (K x K)
            specific_variances: Specific variances (N,)
            factor_names: Factor names
            
        Returns:
            RiskAttributionResult
        """
        n_factors = factor_loadings.shape[1]
        
        if factor_names is None:
            factor_names = [f"Factor_{i}" for i in range(n_factors)]
        
        # Portfolio factor exposures
        port_exposures = factor_loadings.T @ weights
        
        # Factor variance
        factor_var = float(port_exposures @ factor_covariance @ port_exposures)
        factor_risk = np.sqrt(factor_var)
        
        # Specific variance
        specific_var = float(weights @ np.diag(specific_variances) @ weights)
        specific_risk = np.sqrt(specific_var)
        
        # Total risk
        total_var = factor_var + specific_var
        total_risk = np.sqrt(total_var)
        
        # Marginal contribution by factor
        marginal = factor_covariance @ port_exposures
        factor_contributions = {
            factor_names[i]: float(port_exposures[i] * marginal[i] / total_var)
            for i in range(n_factors)
        }
        
        return RiskAttributionResult(
            total_risk=total_risk,
            factor_risk=factor_risk,
            specific_risk=specific_risk,
            factor_risk_contributions=factor_contributions,
            factor_risk_pct=factor_var / total_var * 100,
            specific_risk_pct=specific_var / total_var * 100,
        )
