"""
Configuration management for BXMA Data Analytics Platform.
Supports multiple environments, data sources, and computation settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any
import json
import os


class Environment(Enum):
    """Deployment environment."""
    
    DEVELOPMENT = auto()
    STAGING = auto()
    PRODUCTION = auto()
    RESEARCH = auto()


class ComputeBackend(Enum):
    """Computation backend selection."""
    
    NUMPY = auto()
    JAX = auto()
    TORCH = auto()
    CUPY = auto()  # GPU via CuPy
    DASK = auto()  # Distributed
    RAY = auto()  # Distributed


@dataclass
class DataSourceConfig:
    """Configuration for external data sources."""
    
    # RiskMetrics
    riskmetrics_api_url: str = ""
    riskmetrics_api_key: str = ""
    
    # Bloomberg
    bloomberg_host: str = "localhost"
    bloomberg_port: int = 8194
    
    # Market data
    market_data_provider: str = "refinitiv"
    market_data_api_key: str = ""
    
    # Database connections
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "bxma"
    postgres_user: str = ""
    postgres_password: str = ""
    
    # Redis cache
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # Kafka streaming
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_schema_registry: str = "http://localhost:8081"


@dataclass
class RiskConfig:
    """Configuration for risk calculations."""
    
    # VaR/CVaR
    confidence_levels: list[float] = field(default_factory=lambda: [0.95, 0.99, 0.995])
    var_method: str = "historical"  # parametric, historical, monte_carlo
    var_lookback_days: int = 252
    
    # Monte Carlo
    monte_carlo_simulations: int = 10000
    monte_carlo_seed: int | None = 42
    
    # Factor model
    factor_model_type: str = "statistical"  # statistical, fundamental
    num_factors: int = 10
    factor_estimation_window: int = 756  # 3 years
    
    # Covariance estimation
    covariance_method: str = "ledoit_wolf"  # sample, ledoit_wolf, shrinkage, dcc
    covariance_lookback: int = 252
    exponential_decay_halflife: int | None = 63
    
    # Stress testing
    stress_test_scenarios: list[str] = field(
        default_factory=lambda: [
            "2008_financial_crisis",
            "2020_covid_crash",
            "2022_rate_shock",
            "hypothetical_em_crisis",
            "hypothetical_credit_crisis",
        ]
    )
    
    # Tail risk
    extreme_value_threshold: float = 0.05  # 5th percentile for EVT
    
    # Liquidity
    liquidity_horizon_days: int = 10
    market_impact_model: str = "almgren_chriss"


@dataclass
class OptimizationConfig:
    """Configuration for portfolio optimization."""
    
    # Solver settings
    solver: str = "CLARABEL"  # OSQP, ECOS, SCS, CLARABEL
    solver_verbose: bool = False
    solver_max_iterations: int = 10000
    solver_tolerance: float = 1e-8
    
    # Constraints
    max_position_size: float = 0.20  # 20% max single position
    min_position_size: float = 0.0
    max_sector_exposure: float = 0.40
    max_factor_exposure: float = 2.0
    turnover_constraint: float = 0.50  # 50% max turnover
    
    # Transaction costs
    transaction_cost_bps: float = 10.0  # 10 basis points
    slippage_bps: float = 5.0
    
    # Regularization
    l1_regularization: float = 0.0
    l2_regularization: float = 0.001
    
    # Robust optimization
    uncertainty_set: str = "ellipsoidal"  # box, ellipsoidal, factor
    robustness_parameter: float = 0.1
    
    # Multi-period
    rebalancing_frequency: str = "monthly"
    lookahead_periods: int = 12


@dataclass
class MLConfig:
    """Configuration for machine learning models."""
    
    # Device
    device: str = "cuda"  # cpu, cuda, mps
    mixed_precision: bool = True
    
    # LSTM volatility forecasting
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 3
    lstm_dropout: float = 0.2
    lstm_sequence_length: int = 60
    
    # Transformer
    transformer_d_model: int = 256
    transformer_nhead: int = 8
    transformer_num_layers: int = 6
    transformer_dropout: float = 0.1
    
    # Training
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    epochs: int = 100
    early_stopping_patience: int = 10
    
    # Regime detection
    num_regimes: int = 4
    regime_model: str = "hmm"  # hmm, gmm, neural
    
    # Bayesian
    num_posterior_samples: int = 1000
    mcmc_warmup: int = 500


@dataclass
class ReportingConfig:
    """Configuration for risk reporting."""
    
    # Report generation
    report_output_dir: str = "./reports"
    report_format: str = "html"  # html, pdf, excel
    
    # Dashboard
    dashboard_port: int = 8050
    dashboard_debug: bool = False
    
    # Alerts
    var_breach_threshold: float = 0.95
    drawdown_alert_threshold: float = -0.10
    
    # Email notifications
    smtp_host: str = ""
    smtp_port: int = 587
    alert_recipients: list[str] = field(default_factory=list)


@dataclass
class BXMAConfig:
    """
    Master configuration for BXMA Data Analytics Platform.
    
    Consolidates all subsystem configurations and provides
    environment-aware loading and validation.
    """
    
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    
    # Computation
    compute_backend: ComputeBackend = ComputeBackend.NUMPY
    num_workers: int = 4
    use_gpu: bool = False
    
    # Sub-configurations
    data_sources: DataSourceConfig = field(default_factory=DataSourceConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    
    # Logging
    log_level: str = "INFO"
    log_file: str | None = None
    
    # Cache
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    
    @classmethod
    def from_file(cls, config_path: str | Path) -> BXMAConfig:
        """Load configuration from JSON file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path) as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> BXMAConfig:
        """Create configuration from dictionary."""
        # Parse environment
        env_str = config_dict.get("environment", "development").upper()
        environment = Environment[env_str]
        
        # Parse compute backend
        backend_str = config_dict.get("compute_backend", "numpy").upper()
        compute_backend = ComputeBackend[backend_str]
        
        # Parse sub-configurations
        data_sources = DataSourceConfig(**config_dict.get("data_sources", {}))
        risk = RiskConfig(**config_dict.get("risk", {}))
        optimization = OptimizationConfig(**config_dict.get("optimization", {}))
        ml = MLConfig(**config_dict.get("ml", {}))
        reporting = ReportingConfig(**config_dict.get("reporting", {}))
        
        return cls(
            environment=environment,
            compute_backend=compute_backend,
            num_workers=config_dict.get("num_workers", 4),
            use_gpu=config_dict.get("use_gpu", False),
            data_sources=data_sources,
            risk=risk,
            optimization=optimization,
            ml=ml,
            reporting=reporting,
            log_level=config_dict.get("log_level", "INFO"),
            log_file=config_dict.get("log_file"),
            cache_enabled=config_dict.get("cache_enabled", True),
            cache_ttl_seconds=config_dict.get("cache_ttl_seconds", 3600),
        )
    
    @classmethod
    def from_env(cls) -> BXMAConfig:
        """
        Create configuration from environment variables.
        
        Useful for containerized deployments and CI/CD.
        """
        env_str = os.getenv("BXMA_ENVIRONMENT", "development").upper()
        environment = Environment[env_str]
        
        backend_str = os.getenv("BXMA_COMPUTE_BACKEND", "numpy").upper()
        compute_backend = ComputeBackend[backend_str]
        
        # Data sources from env
        data_sources = DataSourceConfig(
            riskmetrics_api_url=os.getenv("BXMA_RISKMETRICS_URL", ""),
            riskmetrics_api_key=os.getenv("BXMA_RISKMETRICS_KEY", ""),
            postgres_host=os.getenv("BXMA_POSTGRES_HOST", "localhost"),
            postgres_port=int(os.getenv("BXMA_POSTGRES_PORT", "5432")),
            postgres_db=os.getenv("BXMA_POSTGRES_DB", "bxma"),
            postgres_user=os.getenv("BXMA_POSTGRES_USER", ""),
            postgres_password=os.getenv("BXMA_POSTGRES_PASSWORD", ""),
            redis_host=os.getenv("BXMA_REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("BXMA_REDIS_PORT", "6379")),
            kafka_bootstrap_servers=os.getenv("BXMA_KAFKA_SERVERS", "localhost:9092"),
        )
        
        return cls(
            environment=environment,
            compute_backend=compute_backend,
            num_workers=int(os.getenv("BXMA_NUM_WORKERS", "4")),
            use_gpu=os.getenv("BXMA_USE_GPU", "false").lower() == "true",
            data_sources=data_sources,
            log_level=os.getenv("BXMA_LOG_LEVEL", "INFO"),
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to dictionary."""
        return {
            "environment": self.environment.name.lower(),
            "compute_backend": self.compute_backend.name.lower(),
            "num_workers": self.num_workers,
            "use_gpu": self.use_gpu,
            "data_sources": {
                k: v for k, v in self.data_sources.__dict__.items()
                if not k.endswith("_password") and not k.endswith("_key")
            },
            "risk": self.risk.__dict__,
            "optimization": self.optimization.__dict__,
            "ml": self.ml.__dict__,
            "reporting": self.reporting.__dict__,
            "log_level": self.log_level,
            "cache_enabled": self.cache_enabled,
            "cache_ttl_seconds": self.cache_ttl_seconds,
        }
    
    def validate(self) -> list[str]:
        """
        Validate configuration and return list of issues.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []
        
        # Validate confidence levels
        for cl in self.risk.confidence_levels:
            if not 0 < cl < 1:
                issues.append(f"Invalid confidence level: {cl}. Must be between 0 and 1.")
        
        # Validate position constraints
        if self.optimization.max_position_size <= self.optimization.min_position_size:
            issues.append("max_position_size must be greater than min_position_size")
        
        # Validate GPU settings
        if self.use_gpu and self.compute_backend == ComputeBackend.NUMPY:
            issues.append("GPU enabled but compute_backend is NUMPY. Use JAX, TORCH, or CUPY.")
        
        # Validate ML settings
        if self.ml.device == "cuda" and not self.use_gpu:
            issues.append("ML device is 'cuda' but use_gpu is False")
        
        return issues
