"""
Test Suite for Portfolio Optimization Module.
"""

import numpy as np
import pytest

from bxma.optimization.classical import (
    MeanVarianceOptimizer,
    MinVarianceOptimizer,
    MaxSharpeOptimizer,
)
from bxma.optimization.risk_parity import (
    RiskParityOptimizer,
    HierarchicalRiskParity,
)


class TestMeanVarianceOptimization:
    """Tests for Mean-Variance optimization."""
    
    def test_basic_optimization(self, sample_expected_returns, sample_covariance):
        """Test basic mean-variance optimization."""
        optimizer = MeanVarianceOptimizer(risk_aversion=1.0)
        result = optimizer.optimize(sample_expected_returns, sample_covariance)
        
        # Weights should sum to 1
        assert abs(result.weights.sum() - 1.0) < 1e-6
        
        # All weights should be non-negative (long-only)
        assert all(result.weights >= -1e-6)
        
        # Status should indicate success
        assert result.status in ["optimal", "optimal_inaccurate"]
    
    def test_higher_risk_aversion_lower_risk(self, sample_expected_returns, sample_covariance):
        """Test that higher risk aversion leads to lower portfolio risk."""
        optimizer_low = MeanVarianceOptimizer(risk_aversion=0.5)
        optimizer_high = MeanVarianceOptimizer(risk_aversion=5.0)
        
        result_low = optimizer_low.optimize(sample_expected_returns, sample_covariance)
        result_high = optimizer_high.optimize(sample_expected_returns, sample_covariance)
        
        # Higher risk aversion should lead to lower risk
        assert result_high.expected_risk <= result_low.expected_risk


class TestMinVarianceOptimization:
    """Tests for Minimum Variance optimization."""
    
    def test_min_variance_optimization(self, sample_expected_returns, sample_covariance):
        """Test minimum variance optimization."""
        optimizer = MinVarianceOptimizer()
        result = optimizer.optimize(sample_expected_returns, sample_covariance)
        
        # Weights should sum to 1
        assert abs(result.weights.sum() - 1.0) < 1e-6
        
        # Should have low variance
        portfolio_var = result.weights @ sample_covariance @ result.weights
        assert portfolio_var < np.trace(sample_covariance) / len(sample_expected_returns)
    
    def test_min_variance_lower_than_equal_weight(self, sample_expected_returns, sample_covariance):
        """Test that min variance has lower risk than equal weight."""
        optimizer = MinVarianceOptimizer()
        result = optimizer.optimize(sample_expected_returns, sample_covariance)
        
        n = len(sample_expected_returns)
        equal_weights = np.ones(n) / n
        equal_weight_risk = np.sqrt(equal_weights @ sample_covariance @ equal_weights)
        
        assert result.expected_risk <= equal_weight_risk


class TestMaxSharpeOptimization:
    """Tests for Maximum Sharpe Ratio optimization."""
    
    def test_max_sharpe_optimization(self, sample_expected_returns, sample_covariance):
        """Test maximum Sharpe ratio optimization."""
        optimizer = MaxSharpeOptimizer(risk_free_rate=0.02)
        result = optimizer.optimize(sample_expected_returns, sample_covariance)
        
        # Weights should sum to 1
        assert abs(result.weights.sum() - 1.0) < 1e-6
        
        # Sharpe ratio should be positive with positive risk premium
        if result.expected_return > 0.02:
            assert result.sharpe_ratio > 0


class TestRiskParityOptimization:
    """Tests for Risk Parity optimization."""
    
    def test_risk_parity_optimization(self, sample_expected_returns, sample_covariance):
        """Test risk parity optimization."""
        optimizer = RiskParityOptimizer()
        result = optimizer.optimize(sample_expected_returns, sample_covariance)
        
        # Weights should sum to 1
        assert abs(result.weights.sum() - 1.0) < 1e-6
        
        # All weights should be positive
        assert all(result.weights > 0)
    
    def test_risk_contributions_equal(self, sample_expected_returns, sample_covariance):
        """Test that risk contributions are approximately equal."""
        optimizer = RiskParityOptimizer()
        result = optimizer.optimize(sample_expected_returns, sample_covariance)
        
        if result.risk_contributions is not None:
            # Risk contributions should be approximately equal
            rc = result.risk_contributions
            mean_rc = np.mean(rc)
            max_deviation = np.max(np.abs(rc - mean_rc))
            
            # Allow 10% deviation
            assert max_deviation / mean_rc < 0.10


class TestHierarchicalRiskParity:
    """Tests for Hierarchical Risk Parity optimization."""
    
    def test_hrp_optimization(self, sample_expected_returns, sample_covariance):
        """Test HRP optimization."""
        optimizer = HierarchicalRiskParity()
        result = optimizer.optimize(sample_expected_returns, sample_covariance)
        
        # Weights should sum to 1
        assert abs(result.weights.sum() - 1.0) < 1e-6
        
        # All weights should be non-negative
        assert all(result.weights >= 0)
    
    def test_hrp_diversification(self, sample_expected_returns, sample_covariance):
        """Test that HRP provides diversification."""
        optimizer = HierarchicalRiskParity()
        result = optimizer.optimize(sample_expected_returns, sample_covariance)
        
        # No single asset should dominate
        assert np.max(result.weights) < 0.5
        
        # Multiple assets should have meaningful weights
        significant_weights = np.sum(result.weights > 0.05)
        assert significant_weights >= 3
    
    def test_hrp_stability(self, sample_expected_returns, sample_covariance):
        """Test HRP weight stability under small perturbations."""
        optimizer = HierarchicalRiskParity()
        result1 = optimizer.optimize(sample_expected_returns, sample_covariance)
        
        # Small perturbation to covariance
        perturbed_cov = sample_covariance * 1.01
        result2 = optimizer.optimize(sample_expected_returns, perturbed_cov)
        
        # Weights should be similar
        weight_diff = np.max(np.abs(result1.weights - result2.weights))
        assert weight_diff < 0.1


class TestOptimizationConstraints:
    """Tests for optimization with constraints."""
    
    def test_weight_bounds(self, sample_expected_returns, sample_covariance):
        """Test optimization with weight bounds."""
        optimizer = MeanVarianceOptimizer(
            risk_aversion=1.0,
            min_weight=0.02,
            max_weight=0.30
        )
        result = optimizer.optimize(sample_expected_returns, sample_covariance)
        
        # Check bounds are respected
        assert all(result.weights >= 0.02 - 1e-6)
        assert all(result.weights <= 0.30 + 1e-6)
    
    def test_sector_constraints(self, sample_expected_returns, sample_covariance):
        """Test optimization with sector constraints (simulated)."""
        # This would require sector mapping
        optimizer = MeanVarianceOptimizer(risk_aversion=1.0)
        result = optimizer.optimize(sample_expected_returns, sample_covariance)
        
        # Basic test - verify optimization completes
        assert result.weights.sum() > 0


class TestOptimizationEdgeCases:
    """Tests for edge cases in optimization."""
    
    def test_single_asset(self):
        """Test optimization with single asset."""
        expected_returns = np.array([0.08])
        covariance = np.array([[0.04]])
        
        optimizer = MinVarianceOptimizer()
        result = optimizer.optimize(expected_returns, covariance)
        
        assert abs(result.weights[0] - 1.0) < 1e-6
    
    def test_high_correlation_assets(self):
        """Test optimization with highly correlated assets."""
        n = 5
        expected_returns = np.array([0.05, 0.06, 0.07, 0.05, 0.06])
        
        # Create highly correlated covariance matrix
        correlation = np.ones((n, n)) * 0.95
        np.fill_diagonal(correlation, 1.0)
        vols = np.array([0.15, 0.16, 0.17, 0.15, 0.16])
        covariance = np.outer(vols, vols) * correlation
        
        optimizer = MinVarianceOptimizer()
        result = optimizer.optimize(expected_returns, covariance)
        
        # Should still produce valid weights
        assert abs(result.weights.sum() - 1.0) < 1e-6
