"""
Brinson Attribution Models for BXMA Data Analytics Platform.

Implements the foundational arithmetic attribution approaches:
- Brinson-Fachler (1985): Interaction separate from selection
- Brinson-Hood-Beebower (1986): No explicit interaction term

These decompose active return into:
- Allocation Effect: Over/under-weighting asset classes
- Selection Effect: Security selection within asset classes
- Interaction Effect: Combined allocation and selection

References:
- "Measuring Non-U.S. Equity Portfolio Performance"
  (Brinson & Fachler, 1985)
- "Determinants of Portfolio Performance"
  (Brinson, Hood & Beebower, 1986)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import numpy as np
from numpy.typing import NDArray


@dataclass
class AttributionResult:
    """Container for attribution analysis results."""
    
    # Returns
    portfolio_return: float
    benchmark_return: float
    active_return: float
    
    # Effects
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    total_effect: float
    
    # By segment
    segment_names: list[str]
    segment_allocation: NDArray[np.float64]
    segment_selection: NDArray[np.float64]
    segment_interaction: NDArray[np.float64]
    
    # Weights
    portfolio_weights: NDArray[np.float64]
    benchmark_weights: NDArray[np.float64]
    
    # Returns by segment
    portfolio_segment_returns: NDArray[np.float64]
    benchmark_segment_returns: NDArray[np.float64]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for reporting."""
        return {
            "summary": {
                "portfolio_return": self.portfolio_return,
                "benchmark_return": self.benchmark_return,
                "active_return": self.active_return,
                "allocation_effect": self.allocation_effect,
                "selection_effect": self.selection_effect,
                "interaction_effect": self.interaction_effect,
                "total_effect": self.total_effect,
            },
            "by_segment": {
                name: {
                    "allocation": float(self.segment_allocation[i]),
                    "selection": float(self.segment_selection[i]),
                    "interaction": float(self.segment_interaction[i]),
                    "portfolio_weight": float(self.portfolio_weights[i]),
                    "benchmark_weight": float(self.benchmark_weights[i]),
                    "portfolio_return": float(self.portfolio_segment_returns[i]),
                    "benchmark_return": float(self.benchmark_segment_returns[i]),
                }
                for i, name in enumerate(self.segment_names)
            },
        }


class BrinsonFachlerAttribution:
    """
    Brinson-Fachler Attribution Model.
    
    Decomposes active return into three distinct effects:
    
    Allocation: Σ (w_p,i - w_b,i) × (r_b,i - R_b)
    Selection:  Σ w_b,i × (r_p,i - r_b,i)
    Interaction: Σ (w_p,i - w_b,i) × (r_p,i - r_b,i)
    
    where:
    - w_p,i = portfolio weight in segment i
    - w_b,i = benchmark weight in segment i
    - r_p,i = portfolio return in segment i
    - r_b,i = benchmark return in segment i
    - R_b = total benchmark return
    
    Key property: Allocation + Selection + Interaction = Active Return
    """
    
    def __init__(self):
        """Initialize Brinson-Fachler attribution."""
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
        Calculate Brinson-Fachler attribution.
        
        Args:
            portfolio_weights: Portfolio weights by segment (N,)
            benchmark_weights: Benchmark weights by segment (N,)
            portfolio_segment_returns: Portfolio returns by segment (N,)
            benchmark_segment_returns: Benchmark returns by segment (N,)
            segment_names: Names for each segment
            
        Returns:
            AttributionResult with decomposition
        """
        n_segments = len(portfolio_weights)
        
        if segment_names is None:
            segment_names = [f"Segment_{i+1}" for i in range(n_segments)]
        
        # Total returns
        portfolio_return = portfolio_weights @ portfolio_segment_returns
        benchmark_return = benchmark_weights @ benchmark_segment_returns
        active_return = portfolio_return - benchmark_return
        
        # Weight differences
        weight_diff = portfolio_weights - benchmark_weights
        
        # Return differences
        return_diff = portfolio_segment_returns - benchmark_segment_returns
        
        # Allocation effect: over/under-weighting × benchmark excess return
        benchmark_excess = benchmark_segment_returns - benchmark_return
        segment_allocation = weight_diff * benchmark_excess
        allocation_effect = float(np.sum(segment_allocation))
        
        # Selection effect: benchmark weight × security selection
        segment_selection = benchmark_weights * return_diff
        selection_effect = float(np.sum(segment_selection))
        
        # Interaction effect: weight diff × security selection
        segment_interaction = weight_diff * return_diff
        interaction_effect = float(np.sum(segment_interaction))
        
        # Total effect
        total_effect = allocation_effect + selection_effect + interaction_effect
        
        return AttributionResult(
            portfolio_return=portfolio_return,
            benchmark_return=benchmark_return,
            active_return=active_return,
            allocation_effect=allocation_effect,
            selection_effect=selection_effect,
            interaction_effect=interaction_effect,
            total_effect=total_effect,
            segment_names=segment_names,
            segment_allocation=segment_allocation,
            segment_selection=segment_selection,
            segment_interaction=segment_interaction,
            portfolio_weights=portfolio_weights,
            benchmark_weights=benchmark_weights,
            portfolio_segment_returns=portfolio_segment_returns,
            benchmark_segment_returns=benchmark_segment_returns,
        )


class BrinsonHoodBeebowerAttribution:
    """
    Brinson-Hood-Beebower Attribution Model.
    
    Simpler two-effect model that includes interaction in selection:
    
    Allocation: Σ (w_p,i - w_b,i) × r_b,i
    Selection:  Σ w_p,i × (r_p,i - r_b,i)
    
    Note: This formulation attributes interaction to both effects.
    
    Reference:
    - "Determinants of Portfolio Performance" (Brinson et al., 1986)
    """
    
    def __init__(self):
        """Initialize BHB attribution."""
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
        Calculate BHB attribution.
        
        Uses portfolio weights for selection (unlike Brinson-Fachler).
        """
        n_segments = len(portfolio_weights)
        
        if segment_names is None:
            segment_names = [f"Segment_{i+1}" for i in range(n_segments)]
        
        # Total returns
        portfolio_return = portfolio_weights @ portfolio_segment_returns
        benchmark_return = benchmark_weights @ benchmark_segment_returns
        active_return = portfolio_return - benchmark_return
        
        # Weight and return differences
        weight_diff = portfolio_weights - benchmark_weights
        return_diff = portfolio_segment_returns - benchmark_segment_returns
        
        # Allocation effect: weight diff × benchmark returns
        segment_allocation = weight_diff * benchmark_segment_returns
        allocation_effect = float(np.sum(segment_allocation))
        
        # Selection effect: portfolio weight × return diff
        segment_selection = portfolio_weights * return_diff
        selection_effect = float(np.sum(segment_selection))
        
        # No explicit interaction
        segment_interaction = np.zeros(n_segments)
        interaction_effect = 0.0
        
        total_effect = allocation_effect + selection_effect
        
        return AttributionResult(
            portfolio_return=portfolio_return,
            benchmark_return=benchmark_return,
            active_return=active_return,
            allocation_effect=allocation_effect,
            selection_effect=selection_effect,
            interaction_effect=interaction_effect,
            total_effect=total_effect,
            segment_names=segment_names,
            segment_allocation=segment_allocation,
            segment_selection=segment_selection,
            segment_interaction=segment_interaction,
            portfolio_weights=portfolio_weights,
            benchmark_weights=benchmark_weights,
            portfolio_segment_returns=portfolio_segment_returns,
            benchmark_segment_returns=benchmark_segment_returns,
        )


class MultiLevelAttribution:
    """
    Multi-level Brinson Attribution.
    
    Extends Brinson to multiple hierarchical levels:
    - Asset Class level
    - Country/Region level
    - Sector level
    - Security level
    
    Each level captures the marginal effect of that decision.
    """
    
    def __init__(self, levels: list[str]):
        """
        Initialize multi-level attribution.
        
        Args:
            levels: Hierarchy of attribution levels (top to bottom)
        """
        self.levels = levels
        self.bf = BrinsonFachlerAttribution()
    
    def calculate(
        self,
        portfolio_data: dict,
        benchmark_data: dict,
    ) -> dict[str, AttributionResult]:
        """
        Calculate multi-level attribution.
        
        Args:
            portfolio_data: Nested dict with weights/returns by level
            benchmark_data: Nested dict with weights/returns by level
            
        Returns:
            Attribution results for each level
        """
        results = {}
        
        for level in self.levels:
            # Extract level-specific data
            p_weights = np.array(portfolio_data[level]["weights"])
            b_weights = np.array(benchmark_data[level]["weights"])
            p_returns = np.array(portfolio_data[level]["returns"])
            b_returns = np.array(benchmark_data[level]["returns"])
            names = portfolio_data[level].get("names", None)
            
            result = self.bf.calculate(
                p_weights, b_weights, p_returns, b_returns, names
            )
            results[level] = result
        
        return results
