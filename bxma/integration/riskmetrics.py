"""
RiskMetrics Integration Adapter
===============================

Integrates with MSCI RiskMetrics for:
- Factor covariance data
- Risk model parameters
- Benchmark data
- ESG scores

Handles the legacy SOAP/XML interface and translates
to the Titan-X internal format.

Author: BXMA Quant Team
Date: January 2026
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Literal
from enum import Enum, auto


class RiskModel(Enum):
    """RiskMetrics risk models."""
    BARRA_USE4 = "USE4"
    BARRA_GEM3 = "GEM3"
    BARRA_CNE5 = "CNE5"
    AXIOMA_WW21 = "WW21"
    AXIOMA_US4 = "US4"


class FactorCategory(Enum):
    """Factor categories."""
    STYLE = auto()
    INDUSTRY = auto()
    COUNTRY = auto()
    CURRENCY = auto()
    CUSTOM = auto()


@dataclass
class RiskMetricsConfig:
    """Configuration for RiskMetrics connection."""
    
    # Connection
    endpoint: str = "https://api.riskmetrics.com/v2"
    api_key: str = ""
    client_id: str = ""
    
    # Model selection
    risk_model: RiskModel = RiskModel.BARRA_USE4
    
    # Data options
    include_covariance: bool = True
    include_exposures: bool = True
    include_residuals: bool = True
    
    # History
    history_start: date | None = None
    history_end: date | None = None


@dataclass
class FactorDefinition:
    """Definition of a risk factor."""
    
    factor_id: str
    name: str
    category: FactorCategory
    
    description: str = ""
    sector: str = ""
    region: str = ""
    
    # Statistics
    mean_return: float = 0.0
    volatility: float = 0.0
    half_life_days: int = 60


@dataclass
class AssetExposure:
    """Factor exposures for an asset."""
    
    asset_id: str
    as_of_date: date
    
    # Exposures
    factor_exposures: dict[str, float] = field(default_factory=dict)
    
    # Specific risk
    specific_risk: float = 0.0
    specific_return: float = 0.0
    
    # Metadata
    model_version: str = ""


@dataclass
class CovarianceData:
    """Factor covariance matrix data."""
    
    as_of_date: date
    model: RiskModel
    
    # Factors
    factor_ids: list[str] = field(default_factory=list)
    
    # Covariance matrix
    covariance: NDArray[np.float64] | None = None
    
    # Correlation matrix
    correlation: NDArray[np.float64] | None = None
    
    # Factor volatilities
    factor_volatilities: dict[str, float] = field(default_factory=dict)


@dataclass
class RiskMetricsData:
    """Complete RiskMetrics data package."""
    
    config: RiskMetricsConfig
    as_of_date: date
    
    # Factor definitions
    factors: list[FactorDefinition] = field(default_factory=list)
    
    # Covariance
    covariance_data: CovarianceData | None = None
    
    # Asset exposures
    asset_exposures: dict[str, AssetExposure] = field(default_factory=dict)
    
    # Metadata
    fetch_timestamp: datetime = field(default_factory=datetime.now)
    data_quality_score: float = 1.0


class RiskMetricsAdapter:
    """
    Adapter for RiskMetrics data integration.
    
    Handles:
    - Authentication
    - Data fetching
    - Format conversion
    - Caching
    """
    
    def __init__(self, config: RiskMetricsConfig):
        self.config = config
        self._cache: dict[str, Any] = {}
        self._connected = False
    
    def connect(self) -> bool:
        """Establish connection to RiskMetrics API."""
        # In production, this would authenticate with the API
        self._connected = True
        return True
    
    def disconnect(self):
        """Close connection."""
        self._connected = False
    
    def fetch_factors(self) -> list[FactorDefinition]:
        """Fetch factor definitions from RiskMetrics."""
        # In production, this would call the API
        # Returning sample factors for the Barra USE4 model
        
        style_factors = [
            FactorDefinition("MOM", "Momentum", FactorCategory.STYLE, volatility=0.03),
            FactorDefinition("VOL", "Volatility", FactorCategory.STYLE, volatility=0.04),
            FactorDefinition("SIZE", "Size", FactorCategory.STYLE, volatility=0.02),
            FactorDefinition("VALUE", "Value", FactorCategory.STYLE, volatility=0.025),
            FactorDefinition("GROWTH", "Growth", FactorCategory.STYLE, volatility=0.028),
            FactorDefinition("QUALITY", "Quality", FactorCategory.STYLE, volatility=0.02),
            FactorDefinition("LEVERAGE", "Leverage", FactorCategory.STYLE, volatility=0.03),
            FactorDefinition("LIQUIDITY", "Liquidity", FactorCategory.STYLE, volatility=0.025),
        ]
        
        industry_factors = [
            FactorDefinition("TECH", "Technology", FactorCategory.INDUSTRY, volatility=0.035),
            FactorDefinition("FINA", "Financials", FactorCategory.INDUSTRY, volatility=0.03),
            FactorDefinition("HLTH", "Healthcare", FactorCategory.INDUSTRY, volatility=0.028),
            FactorDefinition("CONS", "Consumer", FactorCategory.INDUSTRY, volatility=0.025),
            FactorDefinition("ENRG", "Energy", FactorCategory.INDUSTRY, volatility=0.04),
            FactorDefinition("INDU", "Industrials", FactorCategory.INDUSTRY, volatility=0.03),
            FactorDefinition("UTIL", "Utilities", FactorCategory.INDUSTRY, volatility=0.02),
            FactorDefinition("RLST", "Real Estate", FactorCategory.INDUSTRY, volatility=0.025),
        ]
        
        return style_factors + industry_factors
    
    def fetch_covariance(self, as_of_date: date | None = None) -> CovarianceData:
        """Fetch factor covariance matrix."""
        if as_of_date is None:
            as_of_date = date.today()
        
        factors = self.fetch_factors()
        factor_ids = [f.factor_id for f in factors]
        n_factors = len(factor_ids)
        
        # Generate sample covariance (in production, from API)
        # Using exponentially weighted correlation structure
        correlation = np.eye(n_factors)
        
        # Add some structure
        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                if factors[i].category == factors[j].category:
                    correlation[i, j] = 0.3 + np.random.uniform(0, 0.2)
                else:
                    correlation[i, j] = np.random.uniform(-0.1, 0.2)
                correlation[j, i] = correlation[i, j]
        
        # Convert to covariance
        vols = np.array([f.volatility for f in factors])
        covariance = np.outer(vols, vols) * correlation
        
        factor_vols = {f.factor_id: f.volatility for f in factors}
        
        return CovarianceData(
            as_of_date=as_of_date,
            model=self.config.risk_model,
            factor_ids=factor_ids,
            covariance=covariance,
            correlation=correlation,
            factor_volatilities=factor_vols,
        )
    
    def fetch_exposures(
        self,
        asset_ids: list[str],
        as_of_date: date | None = None,
    ) -> dict[str, AssetExposure]:
        """Fetch factor exposures for assets."""
        if as_of_date is None:
            as_of_date = date.today()
        
        factors = self.fetch_factors()
        factor_ids = [f.factor_id for f in factors]
        
        exposures = {}
        
        for asset_id in asset_ids:
            # Generate sample exposures (in production, from API)
            factor_exp = {}
            for fid in factor_ids:
                factor_exp[fid] = np.random.randn() * 0.5
            
            exposures[asset_id] = AssetExposure(
                asset_id=asset_id,
                as_of_date=as_of_date,
                factor_exposures=factor_exp,
                specific_risk=np.random.uniform(0.02, 0.06),
            )
        
        return exposures
    
    def fetch_all(
        self,
        asset_ids: list[str] | None = None,
        as_of_date: date | None = None,
    ) -> RiskMetricsData:
        """Fetch complete RiskMetrics data package."""
        if as_of_date is None:
            as_of_date = date.today()
        
        factors = self.fetch_factors()
        covariance = self.fetch_covariance(as_of_date)
        
        exposures = {}
        if asset_ids:
            exposures = self.fetch_exposures(asset_ids, as_of_date)
        
        return RiskMetricsData(
            config=self.config,
            as_of_date=as_of_date,
            factors=factors,
            covariance_data=covariance,
            asset_exposures=exposures,
        )
    
    def to_titan_format(self, data: RiskMetricsData) -> dict:
        """Convert RiskMetrics data to Titan-X internal format."""
        return {
            "model": data.config.risk_model.value,
            "as_of_date": data.as_of_date.isoformat(),
            "factors": [
                {
                    "id": f.factor_id,
                    "name": f.name,
                    "category": f.category.name,
                    "volatility": f.volatility,
                }
                for f in data.factors
            ],
            "covariance": {
                "factor_ids": data.covariance_data.factor_ids if data.covariance_data else [],
                "matrix": data.covariance_data.covariance.tolist() if data.covariance_data and data.covariance_data.covariance is not None else [],
            },
            "exposures": {
                asset_id: {
                    "factors": exp.factor_exposures,
                    "specific_risk": exp.specific_risk,
                }
                for asset_id, exp in data.asset_exposures.items()
            },
        }
