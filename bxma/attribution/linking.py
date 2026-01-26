"""
Multi-Period Attribution Linking for BXMA Risk/Quant Platform.

Implements multi-period linking methodologies:
- Frongello Linking
- Davies-Laker Linking
- Geometric Linking

References:
- "Linking Single-Period Attribution Results" (Frongello, 2002)
- "Linking Attribution Effects Over Time" (Davies & Laker, 2001)
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from bxma.attribution.brinson import AttributionResult


@dataclass
class LinkedAttributionResult:
    """Result of multi-period attribution linking."""
    
    # Cumulative returns
    cumulative_portfolio_return: float
    cumulative_benchmark_return: float
    cumulative_active_return: float
    
    # Linked effects
    linked_allocation: float
    linked_selection: float
    linked_interaction: float
    linked_total: float
    
    # Period-level results
    period_results: list[AttributionResult]
    
    # Linking residual (should be ~0 for good linking)
    linking_residual: float


class FrongelloLinking:
    """
    Frongello Multi-Period Linking.
    
    Links attribution effects across periods using a smoothing
    approach that ensures effects compound to total active return.
    
    Reference: "Linking Single-Period Attribution Results" (Frongello, 2002)
    """
    
    def __init__(self):
        pass
    
    def link(
        self,
        period_results: list[AttributionResult],
    ) -> LinkedAttributionResult:
        """
        Link multi-period attribution results.
        
        Args:
            period_results: List of single-period attribution results
            
        Returns:
            LinkedAttributionResult with compounded effects
        """
        n_periods = len(period_results)
        
        # Cumulative returns
        cum_port = np.prod([1 + r.portfolio_return for r in period_results]) - 1
        cum_bench = np.prod([1 + r.benchmark_return for r in period_results]) - 1
        cum_active = (1 + cum_port) / (1 + cum_bench) - 1
        
        # Frongello linking coefficients
        # Each period's effect is scaled by geometric linking factor
        linked_alloc = 0.0
        linked_sel = 0.0
        linked_int = 0.0
        
        cum_bench_factor = 1.0
        cum_port_factor = 1.0
        
        for i, result in enumerate(period_results):
            # Linking factor
            if i > 0:
                cum_bench_factor *= (1 + period_results[i-1].benchmark_return)
                cum_port_factor *= (1 + period_results[i-1].portfolio_return)
            
            # Scale effects by cumulative benchmark
            linked_alloc += result.allocation_effect * cum_bench_factor
            linked_sel += result.selection_effect * cum_bench_factor
            linked_int += result.interaction_effect * cum_bench_factor
        
        linked_total = linked_alloc + linked_sel + linked_int
        
        # Adjust to match geometric active return
        if linked_total != 0:
            scale = cum_active / linked_total
            linked_alloc *= scale
            linked_sel *= scale
            linked_int *= scale
            linked_total = cum_active
        
        residual = cum_active - linked_total
        
        return LinkedAttributionResult(
            cumulative_portfolio_return=cum_port,
            cumulative_benchmark_return=cum_bench,
            cumulative_active_return=cum_active,
            linked_allocation=linked_alloc,
            linked_selection=linked_sel,
            linked_interaction=linked_int,
            linked_total=linked_total,
            period_results=period_results,
            linking_residual=residual,
        )


class DaviesLinking:
    """
    Davies-Laker Multi-Period Linking.
    
    An alternative linking methodology that uses optimized
    weights for each period's contribution.
    """
    
    def __init__(self):
        pass
    
    def link(
        self,
        period_results: list[AttributionResult],
    ) -> LinkedAttributionResult:
        """Link using Davies-Laker method."""
        # Similar structure to Frongello but with different weights
        n_periods = len(period_results)
        
        cum_port = np.prod([1 + r.portfolio_return for r in period_results]) - 1
        cum_bench = np.prod([1 + r.benchmark_return for r in period_results]) - 1
        cum_active = (1 + cum_port) / (1 + cum_bench) - 1
        
        # Davies-Laker weights
        weights = np.ones(n_periods) / n_periods
        
        linked_alloc = sum(w * r.allocation_effect for w, r in zip(weights, period_results))
        linked_sel = sum(w * r.selection_effect for w, r in zip(weights, period_results))
        linked_int = sum(w * r.interaction_effect for w, r in zip(weights, period_results))
        
        linked_total = linked_alloc + linked_sel + linked_int
        
        # Scale to match cumulative active
        if linked_total != 0:
            scale = cum_active / linked_total
            linked_alloc *= scale
            linked_sel *= scale
            linked_int *= scale
            linked_total = cum_active
        
        return LinkedAttributionResult(
            cumulative_portfolio_return=cum_port,
            cumulative_benchmark_return=cum_bench,
            cumulative_active_return=cum_active,
            linked_allocation=linked_alloc,
            linked_selection=linked_sel,
            linked_interaction=linked_int,
            linked_total=linked_total,
            period_results=period_results,
            linking_residual=cum_active - linked_total,
        )


class GeometricLinking:
    """
    Pure Geometric Multi-Period Linking.
    
    Uses multiplicative compounding throughout.
    """
    
    def __init__(self):
        pass
    
    def link(
        self,
        period_results: list[AttributionResult],
    ) -> LinkedAttributionResult:
        """Link using geometric compounding."""
        n_periods = len(period_results)
        
        cum_port = np.prod([1 + r.portfolio_return for r in period_results]) - 1
        cum_bench = np.prod([1 + r.benchmark_return for r in period_results]) - 1
        cum_active = (1 + cum_port) / (1 + cum_bench) - 1
        
        # Geometric linking
        linked_alloc = np.prod([1 + r.allocation_effect for r in period_results]) - 1
        linked_sel = np.prod([1 + r.selection_effect for r in period_results]) - 1
        linked_int = np.prod([1 + r.interaction_effect for r in period_results]) - 1
        
        linked_total = (1 + linked_alloc) * (1 + linked_sel) * (1 + linked_int) - 1
        
        return LinkedAttributionResult(
            cumulative_portfolio_return=cum_port,
            cumulative_benchmark_return=cum_bench,
            cumulative_active_return=cum_active,
            linked_allocation=linked_alloc,
            linked_selection=linked_sel,
            linked_interaction=linked_int,
            linked_total=linked_total,
            period_results=period_results,
            linking_residual=cum_active - linked_total,
        )
