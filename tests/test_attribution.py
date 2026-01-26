"""
Test Suite for Performance Attribution Module.
"""

import numpy as np
import pytest

from bxma.attribution.brinson import (
    BrinsonFachlerAttribution,
    BrinsonHoodBeebowerAttribution,
)
from bxma.attribution.geometric import CarinoAttribution, MencheroAttribution
from bxma.attribution.linking import FrongelloLinking, GeometricLinking


class TestBrinsonFachlerAttribution:
    """Tests for Brinson-Fachler attribution."""
    
    def test_basic_attribution(self, sample_portfolio_data):
        """Test basic Brinson-Fachler attribution."""
        bf = BrinsonFachlerAttribution()
        
        result = bf.calculate(
            sample_portfolio_data["portfolio_weights"],
            sample_portfolio_data["benchmark_weights"],
            sample_portfolio_data["portfolio_returns"],
            sample_portfolio_data["benchmark_returns"],
            sample_portfolio_data["segment_names"],
        )
        
        # Active return should equal sum of effects
        calculated_active = result.allocation_effect + result.selection_effect + result.interaction_effect
        assert abs(result.active_return - calculated_active) < 1e-10
    
    def test_active_return_calculation(self, sample_portfolio_data):
        """Test active return calculation."""
        bf = BrinsonFachlerAttribution()
        
        result = bf.calculate(
            sample_portfolio_data["portfolio_weights"],
            sample_portfolio_data["benchmark_weights"],
            sample_portfolio_data["portfolio_returns"],
            sample_portfolio_data["benchmark_returns"],
        )
        
        # Manual calculation of active return
        p_return = np.dot(sample_portfolio_data["portfolio_weights"], 
                         sample_portfolio_data["portfolio_returns"])
        b_return = np.dot(sample_portfolio_data["benchmark_weights"], 
                         sample_portfolio_data["benchmark_returns"])
        expected_active = p_return - b_return
        
        assert abs(result.active_return - expected_active) < 1e-10
    
    def test_segment_attribution_sums_to_total(self, sample_portfolio_data):
        """Test that segment attributions sum to totals."""
        bf = BrinsonFachlerAttribution()
        
        result = bf.calculate(
            sample_portfolio_data["portfolio_weights"],
            sample_portfolio_data["benchmark_weights"],
            sample_portfolio_data["portfolio_returns"],
            sample_portfolio_data["benchmark_returns"],
            sample_portfolio_data["segment_names"],
        )
        
        # Sum of segment effects should equal total effects
        assert abs(result.segment_allocation.sum() - result.allocation_effect) < 1e-10
        assert abs(result.segment_selection.sum() - result.selection_effect) < 1e-10
        assert abs(result.segment_interaction.sum() - result.interaction_effect) < 1e-10


class TestBrinsonHoodBeebowerAttribution:
    """Tests for Brinson-Hood-Beebower attribution."""
    
    def test_basic_attribution(self, sample_portfolio_data):
        """Test basic BHB attribution."""
        bhb = BrinsonHoodBeebowerAttribution()
        
        result = bhb.calculate(
            sample_portfolio_data["portfolio_weights"],
            sample_portfolio_data["benchmark_weights"],
            sample_portfolio_data["portfolio_returns"],
            sample_portfolio_data["benchmark_returns"],
        )
        
        # Active return should equal sum of effects (no interaction in BHB)
        calculated_active = result.allocation_effect + result.selection_effect
        assert abs(result.active_return - calculated_active) < 1e-10
    
    def test_no_interaction_effect(self, sample_portfolio_data):
        """Test that BHB has no interaction effect."""
        bhb = BrinsonHoodBeebowerAttribution()
        
        result = bhb.calculate(
            sample_portfolio_data["portfolio_weights"],
            sample_portfolio_data["benchmark_weights"],
            sample_portfolio_data["portfolio_returns"],
            sample_portfolio_data["benchmark_returns"],
        )
        
        assert result.interaction_effect == 0


class TestGeometricAttribution:
    """Tests for geometric attribution methods."""
    
    def test_carino_attribution(self):
        """Test Carino geometric attribution."""
        carino = CarinoAttribution()
        
        # Multi-period data
        portfolio_returns = [0.02, -0.01, 0.03, 0.01]
        benchmark_returns = [0.015, -0.008, 0.025, 0.012]
        
        result = carino.calculate(portfolio_returns, benchmark_returns)
        
        # Geometric return calculation
        p_geometric = np.prod([1 + r for r in portfolio_returns]) - 1
        b_geometric = np.prod([1 + r for r in benchmark_returns]) - 1
        
        assert abs(result.portfolio_return - p_geometric) < 1e-10
        assert abs(result.benchmark_return - b_geometric) < 1e-10
    
    def test_menchero_attribution(self):
        """Test Menchero geometric attribution."""
        menchero = MencheroAttribution()
        
        portfolio_returns = [0.02, -0.01, 0.03, 0.01]
        benchmark_returns = [0.015, -0.008, 0.025, 0.012]
        
        result = menchero.calculate(portfolio_returns, benchmark_returns)
        
        # Should properly compound
        assert result.portfolio_return > 0


class TestAttributionLinking:
    """Tests for multi-period attribution linking."""
    
    def test_frongello_linking(self):
        """Test Frongello linking method."""
        linker = FrongelloLinking()
        
        # Period attributions
        allocations = [0.005, 0.003, -0.002, 0.004]
        selections = [0.008, -0.002, 0.005, 0.003]
        interactions = [0.001, 0.0, 0.001, -0.001]
        
        result = linker.link(allocations, selections, interactions)
        
        # Linked effects should exist
        assert "allocation" in result
        assert "selection" in result
        assert "interaction" in result
    
    def test_geometric_linking(self):
        """Test geometric linking method."""
        linker = GeometricLinking()
        
        allocations = [0.005, 0.003, -0.002, 0.004]
        selections = [0.008, -0.002, 0.005, 0.003]
        
        result = linker.link(allocations, selections)
        
        # Should properly compound
        assert "allocation" in result
        assert "selection" in result


class TestAttributionEdgeCases:
    """Tests for edge cases in attribution."""
    
    def test_zero_weights(self):
        """Test attribution with some zero weights."""
        bf = BrinsonFachlerAttribution()
        
        portfolio_weights = np.array([0.5, 0.3, 0.2, 0.0])
        benchmark_weights = np.array([0.25, 0.25, 0.25, 0.25])
        portfolio_returns = np.array([0.08, 0.05, 0.12, 0.0])
        benchmark_returns = np.array([0.06, 0.04, 0.10, 0.02])
        
        result = bf.calculate(
            portfolio_weights, benchmark_weights,
            portfolio_returns, benchmark_returns
        )
        
        # Should handle zero weights gracefully
        assert np.isfinite(result.active_return)
    
    def test_identical_portfolios(self):
        """Test attribution when portfolio equals benchmark."""
        bf = BrinsonFachlerAttribution()
        
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        returns = np.array([0.06, 0.04, 0.10, 0.02])
        
        result = bf.calculate(weights, weights, returns, returns)
        
        # No active return when identical
        assert abs(result.active_return) < 1e-10
        assert abs(result.allocation_effect) < 1e-10
        assert abs(result.selection_effect) < 1e-10
    
    def test_negative_returns(self):
        """Test attribution with negative returns."""
        bf = BrinsonFachlerAttribution()
        
        portfolio_weights = np.array([0.4, 0.3, 0.2, 0.1])
        benchmark_weights = np.array([0.25, 0.25, 0.25, 0.25])
        portfolio_returns = np.array([-0.05, -0.08, 0.02, -0.03])
        benchmark_returns = np.array([-0.04, -0.06, 0.01, -0.02])
        
        result = bf.calculate(
            portfolio_weights, benchmark_weights,
            portfolio_returns, benchmark_returns
        )
        
        # Should handle negative returns
        assert np.isfinite(result.active_return)
        assert np.isfinite(result.allocation_effect)
