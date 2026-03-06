"""
Scenario Analysis Engine for BXMA Data Analytics Platform.

Implements multiple stress testing methodologies:
- Historical scenario replay
- Hypothetical scenario construction
- Factor-based stress testing
- Reverse stress testing

References:
- Basel III stress testing requirements
- CCAR/DFAST methodologies
- "Stress Testing and Risk Integration in Banks" (Rösch & Scheule, 2008)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal
import numpy as np
from numpy.typing import NDArray


@dataclass
class ScenarioDefinition:
    """Definition of a stress scenario."""
    
    name: str
    description: str
    scenario_type: Literal["historical", "hypothetical", "factor_shock"]
    
    # For historical scenarios
    start_date: date | None = None
    end_date: date | None = None
    
    # Factor shocks (factor_name -> shock magnitude)
    factor_shocks: dict[str, float] = field(default_factory=dict)
    
    # Asset-level shocks (asset_id -> shock)
    asset_shocks: dict[str, float] = field(default_factory=dict)
    
    # Correlation adjustments
    correlation_multiplier: float = 1.0
    
    # Liquidity adjustments
    liquidity_haircut: float = 0.0


@dataclass
class ScenarioResult:
    """Result of scenario analysis."""
    
    scenario_name: str
    scenario_type: str
    
    # Portfolio impact
    portfolio_return: float
    portfolio_var: float
    portfolio_cvar: float
    
    # Position-level impacts
    position_impacts: dict[str, float] = field(default_factory=dict)
    
    # Factor attribution
    factor_contributions: dict[str, float] = field(default_factory=dict)
    
    # Comparison to normal conditions
    percentile_rank: float | None = None
    z_score: float | None = None
    
    # Liquidity metrics
    liquidation_cost: float | None = None
    days_to_liquidate: float | None = None


class ScenarioEngine:
    """
    Unified scenario analysis engine.
    
    Runs portfolios through various stress scenarios and
    computes impact metrics.
    """
    
    def __init__(self):
        """Initialize scenario engine."""
        self._historical_data = {}
        self._factor_loadings = None
    
    def load_historical_data(
        self,
        returns: NDArray[np.float64],
        dates: list[date],
        asset_ids: list[str],
    ) -> None:
        """
        Load historical return data for scenario replay.
        
        Args:
            returns: Historical returns (T x N)
            dates: Date for each observation
            asset_ids: Asset identifiers
        """
        self._historical_data = {
            "returns": returns,
            "dates": dates,
            "asset_ids": asset_ids,
        }
    
    def set_factor_model(
        self,
        loadings: NDArray[np.float64],
        factor_names: list[str],
    ) -> None:
        """
        Set factor model for factor-based scenarios.
        
        Args:
            loadings: Factor loadings (N_assets x K_factors)
            factor_names: Factor names
        """
        self._factor_loadings = loadings
        self._factor_names = factor_names
    
    def run_scenario(
        self,
        weights: NDArray[np.float64],
        scenario: ScenarioDefinition,
        covariance: NDArray[np.float64] | None = None,
    ) -> ScenarioResult:
        """
        Run a single stress scenario.
        
        Args:
            weights: Portfolio weights
            scenario: Scenario definition
            covariance: Covariance matrix (for hypothetical scenarios)
            
        Returns:
            ScenarioResult with impact analysis
        """
        if scenario.scenario_type == "historical":
            return self._run_historical_scenario(weights, scenario)
        elif scenario.scenario_type == "hypothetical":
            return self._run_hypothetical_scenario(weights, scenario, covariance)
        elif scenario.scenario_type == "factor_shock":
            return self._run_factor_shock_scenario(weights, scenario)
        else:
            raise ValueError(f"Unknown scenario type: {scenario.scenario_type}")
    
    def run_scenarios(
        self,
        weights: NDArray[np.float64],
        scenarios: list[ScenarioDefinition],
        covariance: NDArray[np.float64] | None = None,
    ) -> list[ScenarioResult]:
        """Run multiple scenarios."""
        return [
            self.run_scenario(weights, scenario, covariance)
            for scenario in scenarios
        ]
    
    def _run_historical_scenario(
        self,
        weights: NDArray[np.float64],
        scenario: ScenarioDefinition,
    ) -> ScenarioResult:
        """Run historical scenario replay."""
        if not self._historical_data:
            raise ValueError("Historical data not loaded")
        
        returns = self._historical_data["returns"]
        dates = self._historical_data["dates"]
        asset_ids = self._historical_data["asset_ids"]
        
        # Find date range
        start_idx = next(
            (i for i, d in enumerate(dates) if d >= scenario.start_date),
            0
        )
        end_idx = next(
            (i for i, d in enumerate(dates) if d > scenario.end_date),
            len(dates)
        )
        
        # Extract scenario returns
        scenario_returns = returns[start_idx:end_idx]
        
        # Compound returns over scenario period
        portfolio_returns = scenario_returns @ weights
        cumulative_return = float(np.prod(1 + portfolio_returns) - 1)
        
        # Position-level impacts
        asset_cumulative = np.prod(1 + scenario_returns, axis=0) - 1
        position_impacts = {
            asset_ids[i]: float(weights[i] * asset_cumulative[i])
            for i in range(len(weights))
        }
        
        # Risk metrics for the period
        portfolio_vol = float(np.std(portfolio_returns) * np.sqrt(252))
        var_95 = float(np.percentile(portfolio_returns, 5))
        cvar_95 = float(np.mean(portfolio_returns[portfolio_returns <= var_95]))
        
        return ScenarioResult(
            scenario_name=scenario.name,
            scenario_type="historical",
            portfolio_return=cumulative_return,
            portfolio_var=var_95,
            portfolio_cvar=cvar_95,
            position_impacts=position_impacts,
        )
    
    def _run_hypothetical_scenario(
        self,
        weights: NDArray[np.float64],
        scenario: ScenarioDefinition,
        covariance: NDArray[np.float64] | None,
    ) -> ScenarioResult:
        """Run hypothetical scenario with specified shocks."""
        n_assets = len(weights)
        
        # Apply asset shocks
        asset_returns = np.zeros(n_assets)
        asset_ids = self._historical_data.get("asset_ids", [f"Asset_{i}" for i in range(n_assets)])
        
        for i, asset_id in enumerate(asset_ids):
            if asset_id in scenario.asset_shocks:
                asset_returns[i] = scenario.asset_shocks[asset_id]
        
        # Apply correlation stress
        if covariance is not None and scenario.correlation_multiplier != 1.0:
            # Extract correlations
            vols = np.sqrt(np.diag(covariance))
            corr = covariance / np.outer(vols, vols)
            
            # Stress correlations toward 1 (systemic stress)
            stressed_corr = scenario.correlation_multiplier * corr + \
                           (1 - scenario.correlation_multiplier) * np.ones_like(corr)
            
            # This would affect risk metrics
        
        # Portfolio return under scenario
        portfolio_return = float(weights @ asset_returns)
        
        # Apply liquidity haircut
        if scenario.liquidity_haircut > 0:
            liquidation_cost = scenario.liquidity_haircut * np.abs(portfolio_return)
            portfolio_return -= liquidation_cost
        else:
            liquidation_cost = 0.0
        
        position_impacts = {
            asset_ids[i]: float(weights[i] * asset_returns[i])
            for i in range(n_assets)
        }
        
        return ScenarioResult(
            scenario_name=scenario.name,
            scenario_type="hypothetical",
            portfolio_return=portfolio_return,
            portfolio_var=portfolio_return,  # Point estimate
            portfolio_cvar=portfolio_return * 1.2,  # Rough estimate
            position_impacts=position_impacts,
            liquidation_cost=liquidation_cost,
        )
    
    def _run_factor_shock_scenario(
        self,
        weights: NDArray[np.float64],
        scenario: ScenarioDefinition,
    ) -> ScenarioResult:
        """Run factor-based stress scenario."""
        if self._factor_loadings is None:
            raise ValueError("Factor model not set")
        
        n_factors = self._factor_loadings.shape[1]
        
        # Build factor shock vector
        factor_shocks = np.zeros(n_factors)
        for i, factor_name in enumerate(self._factor_names):
            if factor_name in scenario.factor_shocks:
                factor_shocks[i] = scenario.factor_shocks[factor_name]
        
        # Compute asset returns from factor shocks
        # r_i = Σ β_ij * f_j
        asset_returns = self._factor_loadings @ factor_shocks
        
        # Portfolio return
        portfolio_return = float(weights @ asset_returns)
        
        # Factor contribution to portfolio return
        portfolio_factor_exposures = self._factor_loadings.T @ weights
        factor_contributions = {
            self._factor_names[i]: float(portfolio_factor_exposures[i] * factor_shocks[i])
            for i in range(n_factors)
        }
        
        # Position impacts
        asset_ids = self._historical_data.get(
            "asset_ids",
            [f"Asset_{i}" for i in range(len(weights))]
        )
        position_impacts = {
            asset_ids[i]: float(weights[i] * asset_returns[i])
            for i in range(len(weights))
        }
        
        return ScenarioResult(
            scenario_name=scenario.name,
            scenario_type="factor_shock",
            portfolio_return=portfolio_return,
            portfolio_var=portfolio_return,
            portfolio_cvar=portfolio_return * 1.2,
            position_impacts=position_impacts,
            factor_contributions=factor_contributions,
        )
    
    def reverse_stress_test(
        self,
        weights: NDArray[np.float64],
        target_loss: float,
        covariance: NDArray[np.float64],
        n_scenarios: int = 1000,
    ) -> list[ScenarioDefinition]:
        """
        Find scenarios that produce a specified loss.
        
        Reverse stress testing identifies scenarios that would
        cause the portfolio to breach a loss threshold.
        
        Args:
            weights: Portfolio weights
            target_loss: Target loss threshold (negative)
            covariance: Covariance matrix
            n_scenarios: Number of scenarios to generate
            
        Returns:
            List of scenarios producing ~target_loss
        """
        results = []
        
        # Generate random factor shocks
        if self._factor_loadings is not None:
            n_factors = self._factor_loadings.shape[1]
            
            for i in range(n_scenarios):
                # Random factor shocks
                factor_shocks = np.random.randn(n_factors) * 0.1
                
                # Scale to achieve target loss
                asset_returns = self._factor_loadings @ factor_shocks
                portfolio_return = weights @ asset_returns
                
                if portfolio_return != 0:
                    scale = target_loss / portfolio_return
                    factor_shocks *= scale
                    
                    # Check if reasonable
                    if np.max(np.abs(factor_shocks)) < 0.5:  # Max 50% shock
                        scenario = ScenarioDefinition(
                            name=f"Reverse_Stress_{i}",
                            description=f"Scenario producing {target_loss:.1%} loss",
                            scenario_type="factor_shock",
                            factor_shocks={
                                self._factor_names[j]: float(factor_shocks[j])
                                for j in range(n_factors)
                            },
                        )
                        results.append(scenario)
        
        return results[:10]  # Return top 10


# Predefined scenarios
STANDARD_SCENARIOS = [
    ScenarioDefinition(
        name="2008_Global_Financial_Crisis",
        description="Lehman Brothers collapse and global credit crisis",
        scenario_type="historical",
        start_date=date(2008, 9, 1),
        end_date=date(2009, 3, 9),
    ),
    ScenarioDefinition(
        name="2020_COVID_Market_Crash",
        description="COVID-19 pandemic market selloff",
        scenario_type="historical",
        start_date=date(2020, 2, 19),
        end_date=date(2020, 3, 23),
    ),
    ScenarioDefinition(
        name="2022_Rate_Shock",
        description="Federal Reserve aggressive rate hiking cycle",
        scenario_type="historical",
        start_date=date(2022, 1, 3),
        end_date=date(2022, 10, 12),
    ),
    ScenarioDefinition(
        name="Equity_Crash_20pct",
        description="Hypothetical 20% equity market decline",
        scenario_type="factor_shock",
        factor_shocks={"Market": -0.20},
    ),
    ScenarioDefinition(
        name="Credit_Spread_Widening",
        description="Credit spreads widen 300bps",
        scenario_type="factor_shock",
        factor_shocks={"Credit_Spread": 0.03, "Market": -0.10},
    ),
    ScenarioDefinition(
        name="Interest_Rate_Shock",
        description="Rates rise 200bps",
        scenario_type="factor_shock",
        factor_shocks={"Interest_Rate": 0.02, "Duration": -0.08},
    ),
    ScenarioDefinition(
        name="Correlation_Crisis",
        description="Correlations spike to 1 (systemic stress)",
        scenario_type="hypothetical",
        correlation_multiplier=0.9,
        asset_shocks={},  # Correlations only
    ),
    ScenarioDefinition(
        name="Liquidity_Crisis",
        description="Market liquidity dries up",
        scenario_type="hypothetical",
        liquidity_haircut=0.05,
        asset_shocks={},
    ),
]
