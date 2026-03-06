"""
Portfolio and Position models for BXMA Data Analytics Platform.
Provides core data structures for multi-asset portfolio management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Iterator, Sequence
import numpy as np
from numpy.typing import NDArray

from bxma.core.types import (
    AssetClass,
    Strategy,
    FactorType,
    FactorExposure,
    WeightVector,
    CovarianceMatrix,
    RiskMetrics,
    PerformanceMetrics,
)


@dataclass
class SecurityIdentifier:
    """Comprehensive security identification."""
    
    ticker: str
    cusip: str | None = None
    isin: str | None = None
    sedol: str | None = None
    bloomberg_id: str | None = None
    ric: str | None = None  # Reuters Instrument Code
    internal_id: str | None = None
    
    def __hash__(self) -> int:
        return hash(self.ticker)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SecurityIdentifier):
            return False
        return self.ticker == other.ticker


@dataclass
class Position:
    """
    Single position within a portfolio.
    
    Represents a holding with full metadata for risk and attribution analysis.
    """
    
    # Identification
    security_id: SecurityIdentifier
    name: str
    
    # Classification
    asset_class: AssetClass
    sector: str | None = None
    industry: str | None = None
    country: str | None = None
    currency: str = "USD"
    
    # Position details
    quantity: float = 0.0
    market_value: float = 0.0
    weight: float = 0.0
    cost_basis: float = 0.0
    
    # Pricing
    current_price: float = 0.0
    previous_price: float = 0.0
    
    # Returns
    daily_return: float = 0.0
    mtd_return: float = 0.0
    qtd_return: float = 0.0
    ytd_return: float = 0.0
    
    # Risk metrics
    volatility: float = 0.0
    beta: float = 1.0
    
    # Factor exposures
    factor_exposures: dict[FactorType, float] = field(default_factory=dict)
    
    # Liquidity
    avg_daily_volume: float = 0.0
    days_to_liquidate: float = 0.0
    bid_ask_spread_bps: float = 0.0
    
    # Timestamps
    as_of_date: date = field(default_factory=date.today)
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        return self.market_value - self.cost_basis
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized P&L as percentage."""
        if self.cost_basis == 0:
            return 0.0
        return (self.market_value - self.cost_basis) / self.cost_basis
    
    @property
    def contribution_to_risk(self) -> float:
        """Rough contribution to portfolio risk (requires portfolio context for accurate calc)."""
        return self.weight * self.volatility * self.beta


@dataclass
class BenchmarkInfo:
    """Benchmark specification for relative performance."""
    
    name: str
    ticker: str
    weights: dict[str, float] = field(default_factory=dict)
    
    # Returns
    daily_return: float = 0.0
    mtd_return: float = 0.0
    qtd_return: float = 0.0
    ytd_return: float = 0.0
    
    # Risk
    volatility: float = 0.0
    
    # Factor exposures
    factor_exposures: dict[FactorType, float] = field(default_factory=dict)


@dataclass
class Portfolio:
    """
    Multi-asset portfolio with full analytics capabilities.
    
    Designed for BXMA's diversified strategies across Absolute Return,
    Multi-Strategy, Total Portfolio Management, and Public Real Assets.
    """
    
    # Identification
    portfolio_id: str
    name: str
    strategy: Strategy
    
    # Holdings
    positions: dict[str, Position] = field(default_factory=dict)
    cash: float = 0.0
    
    # Portfolio-level metrics
    total_nav: float = 0.0
    inception_date: date = field(default_factory=date.today)
    
    # Benchmark
    benchmark: BenchmarkInfo | None = None
    
    # Constraints
    max_leverage: float = 1.0
    
    # Timestamps
    as_of_date: date = field(default_factory=date.today)
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Cached analytics
    _covariance_matrix: CovarianceMatrix | None = field(default=None, repr=False)
    _risk_metrics: RiskMetrics | None = field(default=None, repr=False)
    _performance_metrics: PerformanceMetrics | None = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        """Validate and compute derived quantities."""
        self._recalculate_weights()
    
    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================
    
    def add_position(self, position: Position) -> None:
        """Add or update a position in the portfolio."""
        key = position.security_id.ticker
        self.positions[key] = position
        self._recalculate_weights()
        self._invalidate_cache()
    
    def remove_position(self, ticker: str) -> Position | None:
        """Remove a position from the portfolio."""
        position = self.positions.pop(ticker, None)
        if position:
            self._recalculate_weights()
            self._invalidate_cache()
        return position
    
    def get_position(self, ticker: str) -> Position | None:
        """Get a position by ticker."""
        return self.positions.get(ticker)
    
    def __iter__(self) -> Iterator[Position]:
        """Iterate over positions."""
        return iter(self.positions.values())
    
    def __len__(self) -> int:
        """Number of positions."""
        return len(self.positions)
    
    def __contains__(self, ticker: str) -> bool:
        """Check if ticker is in portfolio."""
        return ticker in self.positions
    
    # =========================================================================
    # WEIGHT CALCULATIONS
    # =========================================================================
    
    def _recalculate_weights(self) -> None:
        """Recalculate position weights based on market values."""
        total_mv = sum(p.market_value for p in self.positions.values()) + self.cash
        self.total_nav = total_mv
        
        if total_mv > 0:
            for position in self.positions.values():
                position.weight = position.market_value / total_mv
    
    def get_weights(self) -> WeightVector:
        """Get weight vector for all positions."""
        if not self.positions:
            return np.array([])
        return np.array([p.weight for p in self.positions.values()])
    
    def get_weights_dict(self) -> dict[str, float]:
        """Get weights as dictionary."""
        return {ticker: p.weight for ticker, p in self.positions.items()}
    
    def set_weights(self, weights: dict[str, float]) -> None:
        """
        Set target weights and rebalance positions.
        
        Args:
            weights: Dictionary mapping ticker to target weight
        """
        for ticker, weight in weights.items():
            if ticker in self.positions:
                self.positions[ticker].weight = weight
                self.positions[ticker].market_value = weight * self.total_nav
        
        self._invalidate_cache()
    
    # =========================================================================
    # AGGREGATIONS
    # =========================================================================
    
    def get_asset_class_weights(self) -> dict[AssetClass, float]:
        """Get weights aggregated by asset class."""
        weights: dict[AssetClass, float] = {}
        for position in self.positions.values():
            ac = position.asset_class
            weights[ac] = weights.get(ac, 0.0) + position.weight
        return weights
    
    def get_sector_weights(self) -> dict[str, float]:
        """Get weights aggregated by sector."""
        weights: dict[str, float] = {}
        for position in self.positions.values():
            sector = position.sector or "Unknown"
            weights[sector] = weights.get(sector, 0.0) + position.weight
        return weights
    
    def get_country_weights(self) -> dict[str, float]:
        """Get weights aggregated by country."""
        weights: dict[str, float] = {}
        for position in self.positions.values():
            country = position.country or "Unknown"
            weights[country] = weights.get(country, 0.0) + position.weight
        return weights
    
    def get_currency_weights(self) -> dict[str, float]:
        """Get weights aggregated by currency."""
        weights: dict[str, float] = {}
        for position in self.positions.values():
            weights[position.currency] = weights.get(position.currency, 0.0) + position.weight
        return weights
    
    def get_factor_exposures(self) -> dict[FactorType, float]:
        """
        Get portfolio-level factor exposures.
        
        Computed as weight-weighted average of position exposures.
        """
        exposures: dict[FactorType, float] = {}
        
        for position in self.positions.values():
            for factor, exposure in position.factor_exposures.items():
                weighted_exp = position.weight * exposure
                exposures[factor] = exposures.get(factor, 0.0) + weighted_exp
        
        return exposures
    
    # =========================================================================
    # RETURNS
    # =========================================================================
    
    @property
    def daily_return(self) -> float:
        """Portfolio daily return (weight-weighted)."""
        return sum(p.weight * p.daily_return for p in self.positions.values())
    
    @property
    def mtd_return(self) -> float:
        """Portfolio MTD return."""
        return sum(p.weight * p.mtd_return for p in self.positions.values())
    
    @property
    def ytd_return(self) -> float:
        """Portfolio YTD return."""
        return sum(p.weight * p.ytd_return for p in self.positions.values())
    
    @property
    def active_return(self) -> float | None:
        """Active return vs benchmark (daily)."""
        if self.benchmark is None:
            return None
        return self.daily_return - self.benchmark.daily_return
    
    # =========================================================================
    # RISK METRICS (BASIC)
    # =========================================================================
    
    @property
    def gross_exposure(self) -> float:
        """Gross exposure (sum of absolute weights)."""
        return sum(abs(p.weight) for p in self.positions.values())
    
    @property
    def net_exposure(self) -> float:
        """Net exposure (sum of weights)."""
        return sum(p.weight for p in self.positions.values())
    
    @property
    def leverage(self) -> float:
        """Portfolio leverage."""
        return self.gross_exposure
    
    @property
    def long_exposure(self) -> float:
        """Total long exposure."""
        return sum(p.weight for p in self.positions.values() if p.weight > 0)
    
    @property
    def short_exposure(self) -> float:
        """Total short exposure."""
        return abs(sum(p.weight for p in self.positions.values() if p.weight < 0))
    
    @property
    def concentration(self) -> float:
        """Herfindahl concentration index."""
        return sum(p.weight ** 2 for p in self.positions.values())
    
    @property
    def effective_n(self) -> float:
        """Effective number of positions (inverse Herfindahl)."""
        hhi = self.concentration
        return 1.0 / hhi if hhi > 0 else 0.0
    
    # =========================================================================
    # CACHE MANAGEMENT
    # =========================================================================
    
    def _invalidate_cache(self) -> None:
        """Invalidate cached analytics."""
        self._covariance_matrix = None
        self._risk_metrics = None
        self._performance_metrics = None
    
    # =========================================================================
    # SERIALIZATION
    # =========================================================================
    
    def to_dict(self) -> dict:
        """Serialize portfolio to dictionary."""
        return {
            "portfolio_id": self.portfolio_id,
            "name": self.name,
            "strategy": self.strategy.name,
            "total_nav": self.total_nav,
            "cash": self.cash,
            "as_of_date": self.as_of_date.isoformat(),
            "positions": {
                ticker: {
                    "ticker": p.security_id.ticker,
                    "name": p.name,
                    "asset_class": p.asset_class.name,
                    "weight": p.weight,
                    "market_value": p.market_value,
                    "daily_return": p.daily_return,
                }
                for ticker, p in self.positions.items()
            },
            "metrics": {
                "gross_exposure": self.gross_exposure,
                "net_exposure": self.net_exposure,
                "leverage": self.leverage,
                "concentration": self.concentration,
                "effective_n": self.effective_n,
            },
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> Portfolio:
        """Deserialize portfolio from dictionary."""
        portfolio = cls(
            portfolio_id=data["portfolio_id"],
            name=data["name"],
            strategy=Strategy[data["strategy"]],
            cash=data.get("cash", 0.0),
            inception_date=date.fromisoformat(data.get("inception_date", date.today().isoformat())),
        )
        
        for ticker, pos_data in data.get("positions", {}).items():
            position = Position(
                security_id=SecurityIdentifier(ticker=pos_data["ticker"]),
                name=pos_data["name"],
                asset_class=AssetClass[pos_data["asset_class"]],
                weight=pos_data.get("weight", 0.0),
                market_value=pos_data.get("market_value", 0.0),
                daily_return=pos_data.get("daily_return", 0.0),
            )
            portfolio.add_position(position)
        
        return portfolio


@dataclass
class PortfolioSnapshot:
    """
    Point-in-time snapshot of portfolio for historical analysis.
    
    Used for backtesting, attribution, and regulatory reporting.
    """
    
    portfolio_id: str
    snapshot_date: date
    snapshot_time: datetime
    
    # Holdings snapshot
    positions: dict[str, dict] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    
    # Aggregate metrics at snapshot time
    total_nav: float = 0.0
    daily_return: float = 0.0
    cumulative_return: float = 0.0
    
    # Risk metrics at snapshot time
    volatility: float = 0.0
    var_95: float = 0.0
    max_drawdown: float = 0.0
    
    # Factor exposures at snapshot time
    factor_exposures: dict[str, float] = field(default_factory=dict)


class PortfolioHistory:
    """
    Time series of portfolio snapshots for historical analysis.
    """
    
    def __init__(self, portfolio_id: str) -> None:
        self.portfolio_id = portfolio_id
        self.snapshots: list[PortfolioSnapshot] = []
    
    def add_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        """Add a snapshot to history."""
        self.snapshots.append(snapshot)
        self.snapshots.sort(key=lambda s: s.snapshot_date)
    
    def get_snapshot(self, as_of_date: date) -> PortfolioSnapshot | None:
        """Get snapshot for a specific date."""
        for snapshot in reversed(self.snapshots):
            if snapshot.snapshot_date <= as_of_date:
                return snapshot
        return None
    
    def get_returns_series(self) -> NDArray[np.float64]:
        """Get daily returns as numpy array."""
        return np.array([s.daily_return for s in self.snapshots])
    
    def get_nav_series(self) -> NDArray[np.float64]:
        """Get NAV series as numpy array."""
        return np.array([s.total_nav for s in self.snapshots])
    
    def get_dates(self) -> list[date]:
        """Get list of snapshot dates."""
        return [s.snapshot_date for s in self.snapshots]
