"""
Pytest Configuration and Fixtures for BXMA Platform.
"""

import asyncio
from datetime import datetime
from typing import Generator
import pytest
import numpy as np
from httpx import AsyncClient, ASGITransport

# Test configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Sample data fixtures
@pytest.fixture
def sample_returns() -> np.ndarray:
    """Generate sample return data for testing."""
    np.random.seed(42)
    return np.random.randn(252, 10) * 0.02


@pytest.fixture
def sample_weights() -> np.ndarray:
    """Generate sample portfolio weights."""
    weights = np.array([0.15, 0.12, 0.10, 0.10, 0.15, 0.12, 0.08, 0.08, 0.05, 0.05])
    return weights / weights.sum()


@pytest.fixture
def sample_covariance(sample_returns: np.ndarray) -> np.ndarray:
    """Generate sample covariance matrix."""
    return np.cov(sample_returns, rowvar=False)


@pytest.fixture
def sample_expected_returns() -> np.ndarray:
    """Generate sample expected returns."""
    return np.array([0.08, 0.10, 0.12, 0.07, 0.05, 0.06, 0.09, 0.11, 0.04, 0.03])


@pytest.fixture
def sample_portfolio_data():
    """Sample portfolio data for attribution tests."""
    return {
        "portfolio_weights": np.array([0.40, 0.30, 0.20, 0.10]),
        "benchmark_weights": np.array([0.25, 0.25, 0.25, 0.25]),
        "portfolio_returns": np.array([0.08, 0.05, 0.12, 0.02]),
        "benchmark_returns": np.array([0.06, 0.04, 0.10, 0.02]),
        "segment_names": ["US Equity", "Int'l Equity", "Fixed Income", "Alternatives"],
    }


@pytest.fixture
def sample_factor_loadings() -> np.ndarray:
    """Generate sample factor loadings matrix."""
    np.random.seed(42)
    n_assets = 10
    n_factors = 5
    loadings = np.random.randn(n_assets, n_factors) * 0.5
    # Ensure market factor has positive loadings
    loadings[:, 0] = np.abs(loadings[:, 0]) + 0.5
    return loadings


# API test fixtures
@pytest.fixture
async def api_client() -> Generator:
    """Create async test client for API testing."""
    from backend.main import app
    
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        yield client


@pytest.fixture
def auth_headers() -> dict:
    """Generate test authentication headers."""
    from backend.auth.jwt import create_access_token
    
    token = create_access_token(
        user_id="test-user-id",
        email="test@bxma.com",
        roles=["analyst"],
        permissions=["risk:read", "risk:calculate"]
    )
    return {"Authorization": f"Bearer {token}"}


# Mock data fixtures
@pytest.fixture
def mock_market_data():
    """Generate mock market data for testing."""
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(252)]
    return {
        "dates": dates,
        "prices": {
            "SPY": np.cumprod(1 + np.random.randn(252) * 0.01) * 400,
            "AGG": np.cumprod(1 + np.random.randn(252) * 0.003) * 100,
            "GLD": np.cumprod(1 + np.random.randn(252) * 0.008) * 180,
        }
    }


from datetime import timedelta
