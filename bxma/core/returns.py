"""
Returns calculation and transformation utilities for BXMA Data Analytics Platform.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class ReturnsEngine:
    """
    Comprehensive returns calculation engine.
    
    Provides:
    - Simple and log return calculations
    - Multi-period compounding
    - Risk-adjusted return metrics
    - Annualization
    """
    
    def simple_returns(
        self,
        prices: NDArray[np.float64],
        periods: int = 1,
    ) -> NDArray[np.float64]:
        """Compute simple returns from price series."""
        return prices[periods:] / prices[:-periods] - 1
    
    def log_returns(
        self,
        prices: NDArray[np.float64],
        periods: int = 1,
    ) -> NDArray[np.float64]:
        """Compute log returns from price series."""
        return np.log(prices[periods:] / prices[:-periods])
    
    def compound_returns(
        self,
        returns: NDArray[np.float64],
    ) -> float:
        """Compound returns geometrically."""
        return float(np.prod(1 + returns) - 1)
    
    def annualize_returns(
        self,
        returns: NDArray[np.float64],
        periods_per_year: int = 252,
    ) -> float:
        """Annualize returns."""
        total = np.prod(1 + returns) - 1
        years = len(returns) / periods_per_year
        return float((1 + total) ** (1 / years) - 1) if years > 0 else 0.0
    
    def annualize_volatility(
        self,
        returns: NDArray[np.float64],
        periods_per_year: int = 252,
    ) -> float:
        """Annualize volatility."""
        return float(np.std(returns) * np.sqrt(periods_per_year))
    
    def sharpe_ratio(
        self,
        returns: NDArray[np.float64],
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> float:
        """Compute annualized Sharpe ratio."""
        excess = returns - risk_free_rate / periods_per_year
        mean_excess = np.mean(excess) * periods_per_year
        vol = self.annualize_volatility(returns, periods_per_year)
        return float(mean_excess / vol) if vol > 0 else 0.0
    
    def sortino_ratio(
        self,
        returns: NDArray[np.float64],
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> float:
        """Compute annualized Sortino ratio."""
        excess = returns - risk_free_rate / periods_per_year
        mean_excess = np.mean(excess) * periods_per_year
        downside = np.minimum(excess, 0)
        downside_std = np.std(downside) * np.sqrt(periods_per_year)
        return float(mean_excess / downside_std) if downside_std > 0 else 0.0
    
    def max_drawdown(
        self,
        returns: NDArray[np.float64],
    ) -> tuple[float, int, int]:
        """
        Compute maximum drawdown.
        
        Returns: (max_dd, peak_idx, trough_idx)
        """
        cum = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum)
        drawdowns = cum / running_max - 1
        
        trough = int(np.argmin(drawdowns))
        peak = int(np.argmax(cum[:trough + 1])) if trough > 0 else 0
        
        return float(drawdowns[trough]), peak, trough
    
    def calmar_ratio(
        self,
        returns: NDArray[np.float64],
        periods_per_year: int = 252,
    ) -> float:
        """Compute Calmar ratio (return / max drawdown)."""
        ann_ret = self.annualize_returns(returns, periods_per_year)
        max_dd, _, _ = self.max_drawdown(returns)
        return float(ann_ret / abs(max_dd)) if max_dd != 0 else 0.0
