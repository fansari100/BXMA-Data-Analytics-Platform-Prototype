"""
Type definitions and enums for BXMA Risk/Quant Platform.
Provides strict typing for multi-asset portfolio analytics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum, auto
from typing import TypeAlias, TypeVar, Generic, Literal
import numpy as np
from numpy.typing import NDArray


# =============================================================================
# TYPE ALIASES
# =============================================================================
FloatArray: TypeAlias = NDArray[np.float64]
WeightVector: TypeAlias = NDArray[np.float64]
CovarianceMatrix: TypeAlias = NDArray[np.float64]
CorrelationMatrix: TypeAlias = NDArray[np.float64]
FactorLoadings: TypeAlias = NDArray[np.float64]
ReturnsSeries: TypeAlias = NDArray[np.float64]

T = TypeVar("T")


# =============================================================================
# ASSET CLASSIFICATION
# =============================================================================
class AssetClass(Enum):
    """Multi-asset class definitions for BXMA strategies."""
    
    # Equity
    EQUITY_DEVELOPED = auto()
    EQUITY_EMERGING = auto()
    EQUITY_FRONTIER = auto()
    EQUITY_SMALL_CAP = auto()
    EQUITY_LARGE_CAP = auto()
    EQUITY_VALUE = auto()
    EQUITY_GROWTH = auto()
    EQUITY_MOMENTUM = auto()
    
    # Fixed Income
    FIXED_INCOME_SOVEREIGN = auto()
    FIXED_INCOME_CORPORATE_IG = auto()
    FIXED_INCOME_CORPORATE_HY = auto()
    FIXED_INCOME_MUNICIPAL = auto()
    FIXED_INCOME_TIPS = auto()
    FIXED_INCOME_EM_DEBT = auto()
    FIXED_INCOME_CONVERTIBLE = auto()
    FIXED_INCOME_ABS = auto()
    FIXED_INCOME_MBS = auto()
    FIXED_INCOME_CLO = auto()
    
    # Alternatives
    ALTERNATIVE_HEDGE_FUND = auto()
    ALTERNATIVE_PRIVATE_EQUITY = auto()
    ALTERNATIVE_PRIVATE_CREDIT = auto()
    ALTERNATIVE_REAL_ESTATE = auto()
    ALTERNATIVE_INFRASTRUCTURE = auto()
    ALTERNATIVE_COMMODITIES = auto()
    ALTERNATIVE_NATURAL_RESOURCES = auto()
    
    # Derivatives
    DERIVATIVE_EQUITY_OPTION = auto()
    DERIVATIVE_INDEX_OPTION = auto()
    DERIVATIVE_INTEREST_RATE_SWAP = auto()
    DERIVATIVE_CREDIT_DEFAULT_SWAP = auto()
    DERIVATIVE_TOTAL_RETURN_SWAP = auto()
    DERIVATIVE_FUTURES = auto()
    DERIVATIVE_FORWARDS = auto()
    DERIVATIVE_VARIANCE_SWAP = auto()
    
    # Currency
    CURRENCY_G10 = auto()
    CURRENCY_EM = auto()
    CURRENCY_CRYPTO = auto()
    
    # Cash
    CASH = auto()


class Strategy(Enum):
    """BXMA business strategy classification."""
    
    ABSOLUTE_RETURN = auto()
    MULTI_STRATEGY = auto()
    TOTAL_PORTFOLIO_MANAGEMENT = auto()
    PUBLIC_REAL_ASSETS = auto()
    TACTICAL_OPPORTUNITIES = auto()
    SYSTEMATIC_MACRO = auto()
    RELATIVE_VALUE = auto()
    EVENT_DRIVEN = auto()


# =============================================================================
# RISK MEASURES
# =============================================================================
class RiskMeasure(Enum):
    """Risk measure types for multi-asset risk modeling."""
    
    # Variance-based
    VOLATILITY = auto()
    VARIANCE = auto()
    TRACKING_ERROR = auto()
    BETA = auto()
    
    # Downside risk
    VAR_PARAMETRIC = auto()
    VAR_HISTORICAL = auto()
    VAR_MONTE_CARLO = auto()
    VAR_CORNISH_FISHER = auto()
    CVAR = auto()  # Conditional VaR / Expected Shortfall
    EXPECTED_SHORTFALL = auto()
    ENTROPIC_VAR = auto()
    
    # Drawdown
    MAX_DRAWDOWN = auto()
    AVERAGE_DRAWDOWN = auto()
    CALMAR_RATIO = auto()
    ULCER_INDEX = auto()
    
    # Higher moments
    SKEWNESS = auto()
    KURTOSIS = auto()
    COSKEWNESS = auto()
    COKURTOSIS = auto()
    
    # Factor-based
    FACTOR_VAR = auto()
    SPECIFIC_RISK = auto()
    SYSTEMATIC_RISK = auto()
    MARGINAL_VAR = auto()
    COMPONENT_VAR = auto()
    INCREMENTAL_VAR = auto()
    
    # Tail risk
    TAIL_RISK = auto()
    EXTREME_VALUE_VAR = auto()
    EXPECTED_TAIL_LOSS = auto()
    
    # Liquidity risk
    LIQUIDITY_ADJUSTED_VAR = auto()
    BID_ASK_SPREAD_RISK = auto()
    MARKET_IMPACT = auto()
    
    # Concentration
    HERFINDAHL_INDEX = auto()
    EFFECTIVE_NUMBER_OF_BETS = auto()
    PORTFOLIO_CONCENTRATION = auto()


class RiskHorizon(Enum):
    """Risk calculation horizons."""
    
    DAILY = 1
    WEEKLY = 5
    BIWEEKLY = 10
    MONTHLY = 21
    QUARTERLY = 63
    SEMI_ANNUAL = 126
    ANNUAL = 252


# =============================================================================
# OPTIMIZATION OBJECTIVES
# =============================================================================
class OptimizationObjective(Enum):
    """Portfolio optimization objectives."""
    
    # Classical
    MAX_SHARPE = auto()
    MIN_VARIANCE = auto()
    MAX_RETURN = auto()
    MIN_TRACKING_ERROR = auto()
    
    # Risk parity
    RISK_PARITY = auto()
    INVERSE_VOLATILITY = auto()
    EQUAL_RISK_CONTRIBUTION = auto()
    HIERARCHICAL_RISK_PARITY = auto()
    NESTED_CLUSTERED_OPTIMIZATION = auto()
    
    # Robust
    ROBUST_MAX_SHARPE = auto()
    ROBUST_MIN_VARIANCE = auto()
    MAX_DIVERSIFICATION = auto()
    MIN_CORRELATION = auto()
    
    # CVaR-based
    MIN_CVAR = auto()
    MAX_CVAR_RETURN = auto()
    MEAN_CVAR = auto()
    
    # Factor-based
    TARGET_FACTOR_EXPOSURE = auto()
    FACTOR_RISK_PARITY = auto()
    MIN_FACTOR_RISK = auto()
    
    # Multi-objective
    MULTI_OBJECTIVE_PARETO = auto()
    LEXICOGRAPHIC = auto()
    
    # ML-based
    REINFORCEMENT_LEARNING = auto()
    NEURAL_NETWORK = auto()
    DIFFERENTIABLE_OPTIMIZATION = auto()


# =============================================================================
# ATTRIBUTION METHODS
# =============================================================================
class AttributionMethod(Enum):
    """Performance attribution methodologies."""
    
    # Arithmetic
    BRINSON_FACHLER = auto()
    BRINSON_HOOD_BEEBOWER = auto()
    
    # Geometric
    GEOMETRIC_SMOOTHING = auto()
    CARIÑO = auto()
    MENCHERO = auto()
    GRAP = auto()  # Geometric Return Attribution Program
    
    # Multi-period
    LINKING_FRONGELLO = auto()
    LINKING_DAVIES = auto()
    LINKING_OPTIMIZED = auto()
    
    # Factor-based
    FACTOR_ATTRIBUTION = auto()
    FUNDAMENTAL_ATTRIBUTION = auto()
    RISK_ATTRIBUTION = auto()
    
    # Fixed income
    CAMPISI = auto()
    DURATION_ATTRIBUTION = auto()
    KEY_RATE_DURATION = auto()
    
    # Derivatives
    GREEK_ATTRIBUTION = auto()
    OPTIONS_ATTRIBUTION = auto()
    
    # Multi-asset
    ASSET_ALLOCATION = auto()
    SECURITY_SELECTION = auto()
    INTERACTION = auto()
    CURRENCY = auto()


# =============================================================================
# MARKET REGIMES
# =============================================================================
class RegimeState(Enum):
    """Market regime classification for regime-switching models."""
    
    # Volatility regimes
    LOW_VOLATILITY = auto()
    NORMAL_VOLATILITY = auto()
    HIGH_VOLATILITY = auto()
    CRISIS = auto()
    
    # Trend regimes
    BULL_MARKET = auto()
    BEAR_MARKET = auto()
    SIDEWAYS = auto()
    
    # Economic regimes
    EXPANSION = auto()
    LATE_CYCLE = auto()
    RECESSION = auto()
    RECOVERY = auto()
    
    # Correlation regimes
    RISK_ON = auto()
    RISK_OFF = auto()
    DECORRELATED = auto()
    
    # Liquidity regimes
    HIGH_LIQUIDITY = auto()
    LOW_LIQUIDITY = auto()
    LIQUIDITY_CRISIS = auto()


# =============================================================================
# FACTOR DEFINITIONS
# =============================================================================
class FactorType(Enum):
    """Factor model classifications."""
    
    # Macroeconomic
    MACRO_GDP_GROWTH = auto()
    MACRO_INFLATION = auto()
    MACRO_INTEREST_RATE = auto()
    MACRO_CREDIT_SPREAD = auto()
    MACRO_YIELD_CURVE = auto()
    MACRO_FX = auto()
    MACRO_COMMODITY = auto()
    MACRO_VOLATILITY = auto()
    
    # Style factors (Equity)
    STYLE_VALUE = auto()
    STYLE_GROWTH = auto()
    STYLE_MOMENTUM = auto()
    STYLE_SIZE = auto()
    STYLE_QUALITY = auto()
    STYLE_LOW_VOLATILITY = auto()
    STYLE_DIVIDEND_YIELD = auto()
    STYLE_BETA = auto()
    STYLE_LIQUIDITY = auto()
    
    # Fixed income factors
    FI_DURATION = auto()
    FI_CONVEXITY = auto()
    FI_CREDIT_SPREAD = auto()
    FI_TERM_STRUCTURE = auto()
    FI_PREPAYMENT = auto()
    
    # Alternative factors
    ALT_CARRY = auto()
    ALT_TREND = auto()
    ALT_MEAN_REVERSION = auto()
    ALT_DISPERSION = auto()


@dataclass(frozen=True)
class FactorExposure:
    """Factor exposure for a single asset or portfolio."""
    
    factor_type: FactorType
    exposure: float
    t_statistic: float | None = None
    standard_error: float | None = None
    confidence_interval: tuple[float, float] | None = None


# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass
class RiskMetrics:
    """Comprehensive risk metrics container."""
    
    # Basic
    volatility: float
    variance: float
    
    # VaR/CVaR at multiple confidence levels
    var_95: float
    var_99: float
    var_99_5: float
    cvar_95: float
    cvar_99: float
    
    # Drawdown
    max_drawdown: float
    average_drawdown: float
    current_drawdown: float
    
    # Higher moments
    skewness: float
    kurtosis: float
    
    # Factor-based
    systematic_risk: float
    specific_risk: float
    tracking_error: float | None = None
    
    # Concentration
    herfindahl_index: float | None = None
    effective_bets: float | None = None
    
    # Tail metrics
    tail_ratio: float | None = None
    gain_to_pain_ratio: float | None = None
    
    # Timestamp
    as_of_date: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics container."""
    
    # Returns
    total_return: float
    annualized_return: float
    cumulative_return: float
    
    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Win/Loss
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    
    # Period statistics
    best_period: float
    worst_period: float
    avg_period: float
    
    # Optional fields with defaults
    information_ratio: float | None = None
    treynor_ratio: float | None = None
    
    # Attribution
    alpha: float | None = None
    beta: float | None = None
    r_squared: float | None = None
    
    # Timestamp
    as_of_date: datetime = field(default_factory=datetime.now)


@dataclass
class OptimizationResult:
    """Portfolio optimization result container."""
    
    weights: WeightVector
    objective_value: float
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    
    # Constraints satisfaction
    constraints_satisfied: bool
    
    # Solver information
    solver_status: str
    iterations: int
    solve_time_ms: float
    
    # Optional fields with defaults
    constraint_violations: dict[str, float] = field(default_factory=dict)
    factor_exposures: dict[FactorType, float] = field(default_factory=dict)
    
    # Risk decomposition
    risk_contributions: WeightVector | None = None
    marginal_risks: WeightVector | None = None


@dataclass
class AttributionResult:
    """Performance attribution result container."""
    
    total_return: float
    benchmark_return: float
    active_return: float
    
    # Decomposition
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    currency_effect: float | None = None
    
    # Factor attribution
    factor_contributions: dict[FactorType, float] = field(default_factory=dict)
    specific_return: float | None = None
    
    # By asset class
    asset_class_attribution: dict[AssetClass, dict[str, float]] = field(default_factory=dict)
    
    # Confidence intervals
    attribution_std_errors: dict[str, float] = field(default_factory=dict)


@dataclass
class StressTestResult:
    """Stress test scenario result."""
    
    scenario_name: str
    scenario_type: Literal["historical", "hypothetical", "factor_shock"]
    
    portfolio_return: float
    portfolio_var: float
    portfolio_cvar: float
    
    # Factor shocks applied
    factor_shocks: dict[FactorType, float] = field(default_factory=dict)
    
    # Asset-level impacts
    asset_impacts: dict[str, float] = field(default_factory=dict)
    
    # Comparison to historical
    percentile_rank: float | None = None
    z_score: float | None = None
