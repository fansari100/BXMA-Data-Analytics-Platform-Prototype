"""
RiskMetrics Integration for BXMA Risk/Quant Platform.

Provides connectivity to RiskMetrics data and risk calculations:
- Covariance matrix retrieval
- Factor model data
- VaR/CVaR calculations
- Stress test scenarios
- Historical simulation data

Reference:
- "RiskMetrics Technical Document" (J.P. Morgan, 1996)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal
import numpy as np
from numpy.typing import NDArray
import httpx
import asyncio


@dataclass
class RiskMetricsData:
    """Container for RiskMetrics data."""
    
    as_of_date: date
    
    # Covariance data
    covariance_matrix: NDArray[np.float64] | None = None
    correlation_matrix: NDArray[np.float64] | None = None
    volatilities: NDArray[np.float64] | None = None
    
    # Factor data
    factor_loadings: NDArray[np.float64] | None = None
    factor_covariance: NDArray[np.float64] | None = None
    factor_names: list[str] | None = None
    
    # Risk metrics
    var_95: float | None = None
    var_99: float | None = None
    cvar_95: float | None = None
    cvar_99: float | None = None
    
    # Metadata
    decay_factor: float = 0.94
    estimation_window: int = 252


class RiskMetricsConnector:
    """
    Connector for RiskMetrics data services.
    
    Provides integration with RiskMetrics API for:
    - Daily covariance matrices
    - Factor model exposures and returns
    - Historical volatility estimates
    - Stress test scenarios
    """
    
    def __init__(
        self,
        api_url: str = "",
        api_key: str = "",
        decay_factor: float = 0.94,
    ):
        """
        Initialize RiskMetrics connector.
        
        Args:
            api_url: RiskMetrics API endpoint
            api_key: API authentication key
            decay_factor: Exponential decay factor (default 0.94)
        """
        self.api_url = api_url
        self.api_key = api_key
        self.decay_factor = decay_factor
        self._client = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30.0,
            )
        return self._client
    
    async def fetch_covariance(
        self,
        assets: list[str],
        as_of_date: date | None = None,
    ) -> RiskMetricsData:
        """
        Fetch covariance matrix from RiskMetrics.
        
        Args:
            assets: List of asset identifiers
            as_of_date: Date for covariance estimate
            
        Returns:
            RiskMetricsData with covariance matrix
        """
        if not self.api_url:
            # Return simulated data for development
            return self._simulate_covariance(assets, as_of_date)
        
        client = await self._get_client()
        
        response = await client.post(
            f"{self.api_url}/covariance",
            json={
                "assets": assets,
                "as_of_date": (as_of_date or date.today()).isoformat(),
                "decay_factor": self.decay_factor,
            },
        )
        response.raise_for_status()
        
        data = response.json()
        
        return RiskMetricsData(
            as_of_date=as_of_date or date.today(),
            covariance_matrix=np.array(data["covariance"]),
            correlation_matrix=np.array(data["correlation"]),
            volatilities=np.array(data["volatilities"]),
            decay_factor=self.decay_factor,
        )
    
    async def fetch_factor_model(
        self,
        assets: list[str],
        model: str = "global_equity",
        as_of_date: date | None = None,
    ) -> RiskMetricsData:
        """
        Fetch factor model data from RiskMetrics.
        
        Args:
            assets: List of asset identifiers
            model: Factor model name
            as_of_date: Date for factor exposures
            
        Returns:
            RiskMetricsData with factor model
        """
        if not self.api_url:
            return self._simulate_factor_model(assets, as_of_date)
        
        client = await self._get_client()
        
        response = await client.post(
            f"{self.api_url}/factors/{model}",
            json={
                "assets": assets,
                "as_of_date": (as_of_date or date.today()).isoformat(),
            },
        )
        response.raise_for_status()
        
        data = response.json()
        
        return RiskMetricsData(
            as_of_date=as_of_date or date.today(),
            factor_loadings=np.array(data["loadings"]),
            factor_covariance=np.array(data["factor_covariance"]),
            factor_names=data["factor_names"],
        )
    
    async def fetch_var(
        self,
        portfolio_weights: NDArray[np.float64],
        assets: list[str],
        confidence_levels: list[float] = [0.95, 0.99],
        horizon_days: int = 1,
    ) -> dict[str, float]:
        """
        Fetch VaR calculations from RiskMetrics.
        
        Args:
            portfolio_weights: Portfolio weights
            assets: Asset identifiers
            confidence_levels: VaR confidence levels
            horizon_days: Risk horizon
            
        Returns:
            Dictionary of VaR and CVaR values
        """
        if not self.api_url:
            return self._simulate_var(portfolio_weights, assets, confidence_levels)
        
        client = await self._get_client()
        
        response = await client.post(
            f"{self.api_url}/var",
            json={
                "weights": portfolio_weights.tolist(),
                "assets": assets,
                "confidence_levels": confidence_levels,
                "horizon_days": horizon_days,
            },
        )
        response.raise_for_status()
        
        return response.json()
    
    async def fetch_stress_scenarios(
        self,
        scenario_names: list[str] | None = None,
    ) -> dict[str, dict]:
        """
        Fetch stress test scenarios from RiskMetrics.
        
        Args:
            scenario_names: Specific scenarios to fetch (None = all)
            
        Returns:
            Dictionary of scenario definitions
        """
        if not self.api_url:
            return self._default_stress_scenarios()
        
        client = await self._get_client()
        
        params = {}
        if scenario_names:
            params["scenarios"] = ",".join(scenario_names)
        
        response = await client.get(
            f"{self.api_url}/stress-scenarios",
            params=params,
        )
        response.raise_for_status()
        
        return response.json()
    
    def _simulate_covariance(
        self,
        assets: list[str],
        as_of_date: date | None,
    ) -> RiskMetricsData:
        """Simulate covariance data for development."""
        n = len(assets)
        
        # Generate random correlation matrix
        A = np.random.randn(n, n)
        corr = A @ A.T
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
        
        # Generate volatilities (annualized 15-40%)
        vols = np.random.uniform(0.15, 0.40, n)
        
        # Covariance from correlation and volatilities
        cov = np.outer(vols, vols) * corr
        
        return RiskMetricsData(
            as_of_date=as_of_date or date.today(),
            covariance_matrix=cov,
            correlation_matrix=corr,
            volatilities=vols,
            decay_factor=self.decay_factor,
        )
    
    def _simulate_factor_model(
        self,
        assets: list[str],
        as_of_date: date | None,
    ) -> RiskMetricsData:
        """Simulate factor model data for development."""
        n_assets = len(assets)
        n_factors = 10
        
        # Factor names
        factor_names = [
            "Market", "Size", "Value", "Momentum", "Volatility",
            "Quality", "Growth", "Yield", "Liquidity", "Currency"
        ]
        
        # Random factor loadings
        loadings = np.random.randn(n_assets, n_factors) * 0.5
        loadings[:, 0] = np.abs(loadings[:, 0]) + 0.5  # Market beta > 0
        
        # Factor covariance (diagonal dominant)
        factor_cov = np.eye(n_factors) * 0.04  # 20% vol
        factor_cov[0, 0] = 0.04  # Market factor
        
        return RiskMetricsData(
            as_of_date=as_of_date or date.today(),
            factor_loadings=loadings,
            factor_covariance=factor_cov,
            factor_names=factor_names,
        )
    
    def _simulate_var(
        self,
        weights: NDArray[np.float64],
        assets: list[str],
        confidence_levels: list[float],
    ) -> dict[str, float]:
        """Simulate VaR calculations for development."""
        # Assume 20% portfolio volatility
        portfolio_vol = 0.20
        daily_vol = portfolio_vol / np.sqrt(252)
        
        from scipy import stats
        
        result = {}
        for cl in confidence_levels:
            z = stats.norm.ppf(cl)
            var = z * daily_vol
            cvar = daily_vol * stats.norm.pdf(z) / (1 - cl)
            
            result[f"var_{int(cl*100)}"] = float(var)
            result[f"cvar_{int(cl*100)}"] = float(cvar)
        
        return result
    
    def _default_stress_scenarios(self) -> dict[str, dict]:
        """Return default stress test scenarios."""
        return {
            "2008_financial_crisis": {
                "name": "2008 Global Financial Crisis",
                "type": "historical",
                "start_date": "2008-09-01",
                "end_date": "2009-03-31",
                "shocks": {
                    "equity": -0.50,
                    "credit_spread": 0.03,
                    "volatility": 0.40,
                    "liquidity": -0.30,
                },
            },
            "2020_covid_crash": {
                "name": "COVID-19 Market Crash",
                "type": "historical",
                "start_date": "2020-02-19",
                "end_date": "2020-03-23",
                "shocks": {
                    "equity": -0.34,
                    "credit_spread": 0.025,
                    "volatility": 0.50,
                    "oil": -0.65,
                },
            },
            "2022_rate_shock": {
                "name": "2022 Interest Rate Shock",
                "type": "historical",
                "start_date": "2022-01-01",
                "end_date": "2022-10-31",
                "shocks": {
                    "equity": -0.25,
                    "bonds": -0.15,
                    "rates": 0.04,
                    "growth_value_spread": 0.30,
                },
            },
            "hypothetical_em_crisis": {
                "name": "Emerging Markets Crisis",
                "type": "hypothetical",
                "shocks": {
                    "em_equity": -0.40,
                    "em_fx": -0.20,
                    "em_credit_spread": 0.04,
                    "dm_equity": -0.15,
                },
            },
            "hypothetical_credit_crisis": {
                "name": "Credit Crisis",
                "type": "hypothetical",
                "shocks": {
                    "ig_spread": 0.02,
                    "hy_spread": 0.06,
                    "equity": -0.20,
                    "volatility": 0.25,
                },
            },
        }
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


def compute_riskmetrics_covariance(
    returns: NDArray[np.float64],
    decay_factor: float = 0.94,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute RiskMetrics exponentially weighted covariance.
    
    Standard RiskMetrics methodology with λ = 0.94.
    
    Args:
        returns: Asset returns (T x N)
        decay_factor: Exponential decay factor λ
        
    Returns:
        (covariance_matrix, volatilities)
    """
    T, N = returns.shape
    
    # Compute weights
    weights = np.array([
        (1 - decay_factor) * decay_factor ** (T - t - 1)
        for t in range(T)
    ])
    weights /= weights.sum()
    
    # Weighted mean
    mean = np.average(returns, axis=0, weights=weights)
    
    # Centered returns
    centered = returns - mean
    
    # Weighted covariance
    cov = np.zeros((N, N))
    for t in range(T):
        cov += weights[t] * np.outer(centered[t], centered[t])
    
    # Volatilities
    vols = np.sqrt(np.diag(cov))
    
    return cov, vols
