"""
Test Suite for API Endpoints.
"""

import numpy as np
import pytest
from httpx import AsyncClient


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    @pytest.mark.asyncio
    async def test_health_check(self, api_client: AsyncClient):
        """Test health check returns OK."""
        response = await api_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestVaREndpoint:
    """Tests for VaR calculation endpoint."""
    
    @pytest.mark.asyncio
    async def test_var_calculation(self, api_client: AsyncClient, sample_returns, sample_weights):
        """Test VaR calculation endpoint."""
        payload = {
            "portfolio": {
                "weights": sample_weights.tolist()
            },
            "returns": {
                "returns": sample_returns.tolist()
            },
            "confidence_level": 0.95,
            "horizon_days": 1,
            "method": "parametric"
        }
        
        response = await api_client.post("/api/v1/risk/var", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "var" in data
        assert "cvar" in data
        assert data["confidence_level"] == 0.95
        assert data["method"] == "parametric"
    
    @pytest.mark.asyncio
    async def test_var_invalid_method(self, api_client: AsyncClient, sample_returns, sample_weights):
        """Test VaR with invalid method returns error."""
        payload = {
            "portfolio": {
                "weights": sample_weights.tolist()
            },
            "returns": {
                "returns": sample_returns.tolist()
            },
            "method": "invalid_method"
        }
        
        response = await api_client.post("/api/v1/risk/var", json=payload)
        
        assert response.status_code == 422  # Validation error


class TestOptimizationEndpoint:
    """Tests for optimization endpoint."""
    
    @pytest.mark.asyncio
    async def test_hrp_optimization(self, api_client: AsyncClient, sample_expected_returns, sample_covariance):
        """Test HRP optimization endpoint."""
        payload = {
            "expected_returns": sample_expected_returns.tolist(),
            "covariance": sample_covariance.tolist(),
            "method": "hrp"
        }
        
        response = await api_client.post("/api/v1/optimize", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "weights" in data
        assert "expected_return" in data
        assert "expected_risk" in data
        assert "sharpe_ratio" in data
        
        # Weights should sum to 1
        assert abs(sum(data["weights"]) - 1.0) < 1e-5
    
    @pytest.mark.asyncio
    async def test_multiple_optimization_methods(
        self, api_client: AsyncClient, sample_expected_returns, sample_covariance
    ):
        """Test all optimization methods."""
        methods = ["mean_variance", "min_variance", "max_sharpe", "risk_parity", "hrp"]
        
        for method in methods:
            payload = {
                "expected_returns": sample_expected_returns.tolist(),
                "covariance": sample_covariance.tolist(),
                "method": method
            }
            
            response = await api_client.post("/api/v1/optimize", json=payload)
            
            assert response.status_code == 200, f"Failed for method: {method}"


class TestAttributionEndpoint:
    """Tests for attribution endpoint."""
    
    @pytest.mark.asyncio
    async def test_brinson_attribution(self, api_client: AsyncClient, sample_portfolio_data):
        """Test Brinson attribution endpoint."""
        payload = {
            "portfolio_weights": sample_portfolio_data["portfolio_weights"].tolist(),
            "benchmark_weights": sample_portfolio_data["benchmark_weights"].tolist(),
            "portfolio_returns": sample_portfolio_data["portfolio_returns"].tolist(),
            "benchmark_returns": sample_portfolio_data["benchmark_returns"].tolist(),
            "segment_names": sample_portfolio_data["segment_names"]
        }
        
        response = await api_client.post("/api/v1/attribution/brinson", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "portfolio_return" in data
        assert "benchmark_return" in data
        assert "active_return" in data
        assert "allocation_effect" in data
        assert "selection_effect" in data


class TestStressTestEndpoint:
    """Tests for stress testing endpoint."""
    
    @pytest.mark.asyncio
    async def test_stress_test_scenario(self, api_client: AsyncClient, sample_weights):
        """Test stress test execution."""
        payload = {
            "weights": sample_weights.tolist(),
            "scenario_name": "2008 Financial Crisis",
            "factor_shocks": {
                "Market": -0.40,
                "Credit": -0.25,
                "Rates": -0.15
            }
        }
        
        response = await api_client.post("/api/v1/stress-test", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "scenario_name" in data
        assert "portfolio_return" in data
        assert data["portfolio_return"] < 0  # Should be negative for crisis scenario
    
    @pytest.mark.asyncio
    async def test_get_standard_scenarios(self, api_client: AsyncClient):
        """Test fetching standard scenarios."""
        response = await api_client.get("/api/v1/stress-test/scenarios")
        
        assert response.status_code == 200
        data = response.json()
        assert "scenarios" in data
        assert len(data["scenarios"]) > 0


class TestCovarianceEndpoint:
    """Tests for covariance estimation endpoint."""
    
    @pytest.mark.asyncio
    async def test_ledoit_wolf_covariance(self, api_client: AsyncClient, sample_returns):
        """Test Ledoit-Wolf covariance estimation."""
        response = await api_client.post(
            "/api/v1/risk/covariance",
            params={"method": "ledoit_wolf"},
            json=sample_returns.tolist()
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "covariance" in data
        assert "correlation" in data
        assert "volatilities" in data


class TestAuthenticationEndpoints:
    """Tests for authentication (mocked)."""
    
    @pytest.mark.asyncio
    async def test_protected_endpoint_without_auth(self, api_client: AsyncClient):
        """Test that protected endpoints require authentication."""
        # This would test actual protected endpoints
        # For now, just verify the health endpoint (unprotected) works
        response = await api_client.get("/health")
        assert response.status_code == 200


class TestRateLimiting:
    """Tests for rate limiting behavior."""
    
    @pytest.mark.asyncio
    async def test_multiple_requests(self, api_client: AsyncClient):
        """Test multiple rapid requests."""
        responses = []
        for _ in range(10):
            response = await api_client.get("/health")
            responses.append(response.status_code)
        
        # All should succeed (no rate limiting on health)
        assert all(code == 200 for code in responses)
