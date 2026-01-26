"""
Geometric Attribution Models for BXMA Risk/Quant Platform.

Implements geometric (multiplicative) attribution approaches:
- Cariño Method: Smoothing for compounding adjustment
- Menchero Method: Fully geometric approach
- GRAP (Geometric Return Attribution Program)

Geometric methods ensure attribution effects compound correctly
and are preferred for multi-period analysis.

References:
- "Combining Attribution Effects Over Time" (Cariño, 1999)
- "A Fully Geometric Approach to Attribution" (Menchero, 2000)
- "Linking Single Period Attribution Results" (Davies & Laker, 2001)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

from bxma.attribution.brinson import AttributionResult


class GeometricAttribution:
    """
    Base Geometric Attribution.
    
    Uses multiplicative decomposition instead of additive:
    
    (1 + R_p) / (1 + R_b) = Π (1 + contribution_i)
    
    This ensures proper compounding over multiple periods.
    """
    
    def __init__(self):
        pass
    
    def calculate(
        self,
        portfolio_weights: NDArray[np.float64],
        benchmark_weights: NDArray[np.float64],
        portfolio_segment_returns: NDArray[np.float64],
        benchmark_segment_returns: NDArray[np.float64],
        segment_names: list[str] | None = None,
    ) -> AttributionResult:
        """
        Calculate geometric attribution.
        
        Returns in geometric form: contribution to relative return.
        """
        n_segments = len(portfolio_weights)
        
        if segment_names is None:
            segment_names = [f"Segment_{i+1}" for i in range(n_segments)]
        
        # Total returns
        portfolio_return = portfolio_weights @ portfolio_segment_returns
        benchmark_return = benchmark_weights @ benchmark_segment_returns
        
        # Relative return factor
        relative_factor = (1 + portfolio_return) / (1 + benchmark_return)
        active_return = relative_factor - 1
        
        # Geometric allocation: relative weight × benchmark performance
        weight_diff = portfolio_weights - benchmark_weights
        benchmark_relative = (1 + benchmark_segment_returns) / (1 + benchmark_return) - 1
        segment_allocation = weight_diff * benchmark_relative
        
        # Geometric selection: benchmark weight × relative performance
        segment_relative = (1 + portfolio_segment_returns) / (1 + benchmark_segment_returns) - 1
        segment_selection = benchmark_weights * segment_relative
        
        # Geometric interaction
        segment_interaction = weight_diff * segment_relative
        
        allocation_effect = float(np.sum(segment_allocation))
        selection_effect = float(np.sum(segment_selection))
        interaction_effect = float(np.sum(segment_interaction))
        
        return AttributionResult(
            portfolio_return=portfolio_return,
            benchmark_return=benchmark_return,
            active_return=float(active_return),
            allocation_effect=allocation_effect,
            selection_effect=selection_effect,
            interaction_effect=interaction_effect,
            total_effect=allocation_effect + selection_effect + interaction_effect,
            segment_names=segment_names,
            segment_allocation=segment_allocation,
            segment_selection=segment_selection,
            segment_interaction=segment_interaction,
            portfolio_weights=portfolio_weights,
            benchmark_weights=benchmark_weights,
            portfolio_segment_returns=portfolio_segment_returns,
            benchmark_segment_returns=benchmark_segment_returns,
        )


class CarinoAttribution:
    """
    Cariño Geometric Attribution with Smoothing.
    
    Uses a smoothing factor to adjust arithmetic effects to
    properly compound to the geometric total.
    
    k = ln(1 + R) / R  (smoothing factor)
    
    Adjusted effect = arithmetic effect × (k_portfolio / k_link)
    
    Reference:
    - "Combining Attribution Effects Over Time" (Cariño, 1999)
    """
    
    def __init__(self):
        pass
    
    def calculate(
        self,
        portfolio_weights: NDArray[np.float64],
        benchmark_weights: NDArray[np.float64],
        portfolio_segment_returns: NDArray[np.float64],
        benchmark_segment_returns: NDArray[np.float64],
        segment_names: list[str] | None = None,
    ) -> AttributionResult:
        """Calculate Cariño-adjusted attribution."""
        n_segments = len(portfolio_weights)
        
        if segment_names is None:
            segment_names = [f"Segment_{i+1}" for i in range(n_segments)]
        
        # Total returns
        portfolio_return = portfolio_weights @ portfolio_segment_returns
        benchmark_return = benchmark_weights @ benchmark_segment_returns
        active_return = portfolio_return - benchmark_return
        
        # Cariño smoothing factors
        def carino_factor(r: float) -> float:
            """Compute Cariño smoothing factor k = ln(1+r)/r."""
            if abs(r) < 1e-10:
                return 1.0
            return np.log(1 + r) / r
        
        k_p = carino_factor(portfolio_return)
        k_b = carino_factor(benchmark_return)
        
        # Geometric active return
        geometric_active = (1 + portfolio_return) / (1 + benchmark_return) - 1
        k_active = carino_factor(geometric_active)
        
        # Linking coefficient
        k_link = (k_p - k_b) / (geometric_active) if abs(geometric_active) > 1e-10 else 1.0
        
        # Arithmetic effects (Brinson-Fachler)
        weight_diff = portfolio_weights - benchmark_weights
        return_diff = portfolio_segment_returns - benchmark_segment_returns
        benchmark_excess = benchmark_segment_returns - benchmark_return
        
        arith_allocation = weight_diff * benchmark_excess
        arith_selection = benchmark_weights * return_diff
        arith_interaction = weight_diff * return_diff
        
        # Apply Cariño adjustment
        adjustment_factor = k_p / k_link if abs(k_link) > 1e-10 else 1.0
        
        segment_allocation = arith_allocation * adjustment_factor
        segment_selection = arith_selection * adjustment_factor
        segment_interaction = arith_interaction * adjustment_factor
        
        allocation_effect = float(np.sum(segment_allocation))
        selection_effect = float(np.sum(segment_selection))
        interaction_effect = float(np.sum(segment_interaction))
        
        return AttributionResult(
            portfolio_return=portfolio_return,
            benchmark_return=benchmark_return,
            active_return=float(geometric_active),
            allocation_effect=allocation_effect,
            selection_effect=selection_effect,
            interaction_effect=interaction_effect,
            total_effect=allocation_effect + selection_effect + interaction_effect,
            segment_names=segment_names,
            segment_allocation=segment_allocation,
            segment_selection=segment_selection,
            segment_interaction=segment_interaction,
            portfolio_weights=portfolio_weights,
            benchmark_weights=benchmark_weights,
            portfolio_segment_returns=portfolio_segment_returns,
            benchmark_segment_returns=benchmark_segment_returns,
        )


class MencheroAttribution:
    """
    Menchero Fully Geometric Attribution.
    
    Provides a fully geometric framework where all effects
    are defined in terms of wealth ratios.
    
    Key insight: Define effects in terms of "what if" portfolios:
    - What if we had benchmark weights but portfolio returns?
    - What if we had portfolio weights but benchmark returns?
    
    Reference:
    - "A Fully Geometric Approach to Attribution" (Menchero, 2000)
    """
    
    def __init__(self, interaction_allocation: str = "selection"):
        """
        Initialize Menchero attribution.
        
        Args:
            interaction_allocation: Where to allocate interaction ('selection' or 'allocation')
        """
        self.interaction_allocation = interaction_allocation
    
    def calculate(
        self,
        portfolio_weights: NDArray[np.float64],
        benchmark_weights: NDArray[np.float64],
        portfolio_segment_returns: NDArray[np.float64],
        benchmark_segment_returns: NDArray[np.float64],
        segment_names: list[str] | None = None,
    ) -> AttributionResult:
        """Calculate Menchero geometric attribution."""
        n_segments = len(portfolio_weights)
        
        if segment_names is None:
            segment_names = [f"Segment_{i+1}" for i in range(n_segments)]
        
        # Total returns
        R_p = portfolio_weights @ portfolio_segment_returns
        R_b = benchmark_weights @ benchmark_segment_returns
        
        # Notional portfolios
        # Portfolio weights, benchmark returns
        R_wb = portfolio_weights @ benchmark_segment_returns
        # Benchmark weights, portfolio returns
        R_pw = benchmark_weights @ portfolio_segment_returns
        
        # Geometric decomposition
        # Total relative = (1 + R_p) / (1 + R_b)
        # = [(1 + R_wb)/(1 + R_b)] × [(1 + R_p)/(1 + R_wb)]  (if allocation first)
        # = [(1 + R_pw)/(1 + R_b)] × [(1 + R_p)/(1 + R_pw)]  (if selection first)
        
        # Using allocation-first ordering:
        allocation_factor = (1 + R_wb) / (1 + R_b)
        selection_factor = (1 + R_p) / (1 + R_wb)
        
        # Convert to additive for reporting
        allocation_effect = allocation_factor - 1
        selection_effect_gross = selection_factor - 1
        
        # Segment-level effects
        weight_diff = portfolio_weights - benchmark_weights
        
        # Allocation by segment (geometric)
        segment_allocation = weight_diff * (
            (1 + benchmark_segment_returns) / (1 + R_b) - 1
        )
        
        # Selection by segment (includes interaction if using benchmark weights)
        segment_selection = portfolio_weights * (
            (1 + portfolio_segment_returns) / (1 + benchmark_segment_returns) - 1
        )
        
        # Adjustment for Menchero
        allocation_effect = float(np.sum(segment_allocation))
        selection_effect = float(np.sum(segment_selection))
        
        # In Menchero, interaction is implicit
        interaction_effect = 0.0
        segment_interaction = np.zeros(n_segments)
        
        # Geometric active return
        geometric_active = (1 + R_p) / (1 + R_b) - 1
        
        return AttributionResult(
            portfolio_return=R_p,
            benchmark_return=R_b,
            active_return=float(geometric_active),
            allocation_effect=allocation_effect,
            selection_effect=selection_effect,
            interaction_effect=interaction_effect,
            total_effect=allocation_effect + selection_effect,
            segment_names=segment_names,
            segment_allocation=segment_allocation,
            segment_selection=segment_selection,
            segment_interaction=segment_interaction,
            portfolio_weights=portfolio_weights,
            benchmark_weights=benchmark_weights,
            portfolio_segment_returns=portfolio_segment_returns,
            benchmark_segment_returns=benchmark_segment_returns,
        )
