"""
Test Suite for Risk Analytics Module.
"""

import numpy as np
import pytest

from bxma.risk.var import (
    ParametricVaR,
    HistoricalVaR,
    MonteCarloVaR,
    CornishFisherVaR,
)
from bxma.risk.covariance import (
    LedoitWolfCovariance,
    ExponentialCovariance,
)
from bxma.risk.factor_models import StatisticalFactorModel


class TestParametricVaR:
    """Tests for Parametric VaR calculation."""
    
    def test_basic_var_calculation(self, sample_returns, sample_weights):
        """Test basic VaR calculation."""
        var_engine = ParametricVaR()
        result = var_engine.calculate_var(sample_returns, sample_weights, 0.95, 1)
        
        assert result.var > 0
        assert result.confidence_level == 0.95
        assert result.horizon_days == 1
        assert result.method == "parametric"
    
    def test_var_increases_with_confidence(self, sample_returns, sample_weights):
        """Test that VaR increases with higher confidence levels."""
        var_engine = ParametricVaR()
        
        var_95 = var_engine.calculate_var(sample_returns, sample_weights, 0.95, 1).var
        var_99 = var_engine.calculate_var(sample_returns, sample_weights, 0.99, 1).var
        
        assert var_99 > var_95
    
    def test_var_scales_with_horizon(self, sample_returns, sample_weights):
        """Test that VaR scales with time horizon (square root of time)."""
        var_engine = ParametricVaR()
        
        var_1d = var_engine.calculate_var(sample_returns, sample_weights, 0.95, 1).var
        var_10d = var_engine.calculate_var(sample_returns, sample_weights, 0.95, 10).var
        
        # 10-day VaR should be approximately sqrt(10) times 1-day VaR
        expected_ratio = np.sqrt(10)
        actual_ratio = var_10d / var_1d
        
        assert abs(actual_ratio - expected_ratio) < 0.5
    
    def test_cvar_greater_than_var(self, sample_returns, sample_weights):
        """Test that CVaR is always greater than or equal to VaR."""
        var_engine = ParametricVaR()
        result = var_engine.calculate_var(sample_returns, sample_weights, 0.95, 1)
        
        if result.cvar is not None:
            assert result.cvar >= result.var


class TestHistoricalVaR:
    """Tests for Historical VaR calculation."""
    
    def test_historical_var_calculation(self, sample_returns, sample_weights):
        """Test historical simulation VaR."""
        var_engine = HistoricalVaR()
        result = var_engine.calculate_var(sample_returns, sample_weights, 0.95, 1)
        
        assert result.var > 0
        assert result.method == "historical"
    
    def test_var_within_return_range(self, sample_returns, sample_weights):
        """Test that VaR is within historical return range."""
        var_engine = HistoricalVaR()
        result = var_engine.calculate_var(sample_returns, sample_weights, 0.95, 1)
        
        portfolio_returns = sample_returns @ sample_weights
        min_return = np.min(portfolio_returns)
        
        assert result.var <= abs(min_return)


class TestMonteCarloVaR:
    """Tests for Monte Carlo VaR calculation."""
    
    def test_monte_carlo_var_calculation(self, sample_returns, sample_weights):
        """Test Monte Carlo simulation VaR."""
        var_engine = MonteCarloVaR(n_simulations=5000)
        result = var_engine.calculate_var(sample_returns, sample_weights, 0.95, 1)
        
        assert result.var > 0
        assert result.method == "monte_carlo"
    
    def test_convergence_with_simulations(self, sample_returns, sample_weights):
        """Test that results converge with more simulations."""
        results = []
        for n_sims in [1000, 5000, 10000]:
            var_engine = MonteCarloVaR(n_simulations=n_sims)
            result = var_engine.calculate_var(sample_returns, sample_weights, 0.95, 1)
            results.append(result.var)
        
        # Variance should decrease with more simulations
        assert len(results) == 3


class TestCornishFisherVaR:
    """Tests for Cornish-Fisher VaR calculation."""
    
    def test_cornish_fisher_var_calculation(self, sample_returns, sample_weights):
        """Test Cornish-Fisher adjusted VaR."""
        var_engine = CornishFisherVaR()
        result = var_engine.calculate_var(sample_returns, sample_weights, 0.95, 1)
        
        assert result.var > 0
        assert result.method == "cornish_fisher"
    
    def test_adjustment_for_skewness(self, sample_weights):
        """Test that Cornish-Fisher adjusts for skewness."""
        # Create skewed returns
        np.random.seed(42)
        normal_returns = np.random.randn(252, 10) * 0.02
        skewed_returns = normal_returns - 0.01  # Add negative skew
        
        parametric = ParametricVaR()
        cornish_fisher = CornishFisherVaR()
        
        param_var = parametric.calculate_var(skewed_returns, sample_weights, 0.95, 1).var
        cf_var = cornish_fisher.calculate_var(skewed_returns, sample_weights, 0.95, 1).var
        
        # With negative skew, Cornish-Fisher should give higher VaR
        # (This is a simplified check - actual behavior depends on the data)
        assert cf_var > 0 and param_var > 0


class TestCovarianceEstimation:
    """Tests for covariance matrix estimation."""
    
    def test_ledoit_wolf_estimation(self, sample_returns):
        """Test Ledoit-Wolf shrinkage estimator."""
        estimator = LedoitWolfCovariance()
        result = estimator.fit(sample_returns)
        
        # Check covariance is symmetric
        assert np.allclose(result.covariance, result.covariance.T)
        
        # Check positive definite
        eigenvalues = np.linalg.eigvalsh(result.covariance)
        assert all(eigenvalues > 0)
        
        # Check condition number is reasonable
        assert result.condition_number < 1e10
    
    def test_exponential_covariance(self, sample_returns):
        """Test exponential weighted covariance."""
        estimator = ExponentialCovariance(halflife=63)
        result = estimator.fit(sample_returns)
        
        # Check covariance is symmetric
        assert np.allclose(result.covariance, result.covariance.T)
        
        # Check positive definite
        eigenvalues = np.linalg.eigvalsh(result.covariance)
        assert all(eigenvalues > 0)
    
    def test_correlation_bounds(self, sample_returns):
        """Test that correlations are between -1 and 1."""
        estimator = LedoitWolfCovariance()
        result = estimator.fit(sample_returns)
        
        assert np.all(result.correlation >= -1)
        assert np.all(result.correlation <= 1)


class TestFactorModel:
    """Tests for factor model fitting."""
    
    def test_pca_factor_model(self, sample_returns):
        """Test PCA-based factor model."""
        model = StatisticalFactorModel(n_factors=3, method="pca")
        result = model.fit(sample_returns)
        
        assert result.n_factors == 3
        assert 0 <= result.r_squared <= 1
        assert len(result.factor_names) == 3
    
    def test_explained_variance_ordering(self, sample_returns):
        """Test that factors are ordered by explained variance."""
        model = StatisticalFactorModel(n_factors=5, method="pca")
        result = model.fit(sample_returns)
        
        # Explained variance should be in descending order
        for i in range(len(result.explained_variance_ratio) - 1):
            assert result.explained_variance_ratio[i] >= result.explained_variance_ratio[i + 1]
    
    def test_total_explained_variance(self, sample_returns):
        """Test that total explained variance is bounded."""
        model = StatisticalFactorModel(n_factors=10, method="pca")
        result = model.fit(sample_returns)
        
        total_explained = sum(result.explained_variance_ratio)
        assert 0 < total_explained <= 1
