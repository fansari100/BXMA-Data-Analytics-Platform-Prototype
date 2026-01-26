"""
Macro Market Data Module
========================

Handles macro and asset-class specific market data:
- Economic indicators
- Central bank data
- Yield curves
- FX rates
- Commodity prices
- Equity indices
- Credit spreads

CRITICAL REQUIREMENT: Experience working with macro and 
asset class specific market data.

Author: BXMA Quant Team
Date: January 2026
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Literal, Any
from enum import Enum, auto


class MacroIndicatorType(Enum):
    """Types of macroeconomic indicators."""
    # Economic activity
    GDP = auto()
    INDUSTRIAL_PRODUCTION = auto()
    RETAIL_SALES = auto()
    PMI_MANUFACTURING = auto()
    PMI_SERVICES = auto()
    
    # Labor market
    UNEMPLOYMENT_RATE = auto()
    NONFARM_PAYROLLS = auto()
    INITIAL_CLAIMS = auto()
    LABOR_PARTICIPATION = auto()
    
    # Inflation
    CPI = auto()
    CORE_CPI = auto()
    PPI = auto()
    PCE = auto()
    
    # Central bank
    FED_FUNDS_RATE = auto()
    ECB_DEPOSIT_RATE = auto()
    BOJ_RATE = auto()
    BOE_RATE = auto()
    
    # Sentiment
    CONSUMER_CONFIDENCE = auto()
    BUSINESS_CONFIDENCE = auto()
    ISM_MANUFACTURING = auto()
    
    # Housing
    HOUSING_STARTS = auto()
    HOME_PRICES = auto()
    MORTGAGE_RATES = auto()
    
    # Trade
    TRADE_BALANCE = auto()
    CURRENT_ACCOUNT = auto()


class AssetClassType(Enum):
    """Asset class categories."""
    EQUITY = auto()
    FIXED_INCOME = auto()
    FX = auto()
    COMMODITIES = auto()
    CREDIT = auto()
    REAL_ESTATE = auto()
    ALTERNATIVES = auto()


@dataclass
class MacroDataPoint:
    """A single macro data point."""
    
    indicator: MacroIndicatorType
    region: str  # US, EU, JP, UK, CN, etc.
    
    # Value
    value: float
    previous_value: float | None = None
    consensus_estimate: float | None = None
    
    # Date
    release_date: date = field(default_factory=date.today)
    reference_period: str = ""  # e.g., "2025-Q4", "2025-12"
    
    # Metadata
    source: str = ""
    revision: int = 0
    is_preliminary: bool = False
    
    @property
    def surprise(self) -> float | None:
        """Calculate surprise vs consensus."""
        if self.consensus_estimate is not None:
            return self.value - self.consensus_estimate
        return None
    
    @property
    def change(self) -> float | None:
        """Calculate change from previous."""
        if self.previous_value is not None:
            return self.value - self.previous_value
        return None


@dataclass
class YieldCurve:
    """Yield curve data."""
    
    currency: str  # USD, EUR, JPY, etc.
    curve_type: Literal["government", "swap", "corporate", "ois"] = "government"
    
    # Tenors and rates
    tenors: list[str] = field(default_factory=list)  # 1M, 3M, 6M, 1Y, 2Y, 5Y, 10Y, 30Y
    rates: list[float] = field(default_factory=list)
    
    # Date
    as_of_date: date = field(default_factory=date.today)
    
    # Derived metrics
    slope_2s10s: float | None = None
    slope_5s30s: float | None = None
    curvature: float | None = None
    
    def get_rate(self, tenor: str) -> float | None:
        """Get rate for a specific tenor."""
        if tenor in self.tenors:
            idx = self.tenors.index(tenor)
            return self.rates[idx]
        return None
    
    def interpolate(self, target_years: float) -> float:
        """Interpolate rate for a given maturity in years."""
        tenor_years = []
        for t in self.tenors:
            if t.endswith("M"):
                tenor_years.append(int(t[:-1]) / 12)
            elif t.endswith("Y"):
                tenor_years.append(int(t[:-1]))
        
        return float(np.interp(target_years, tenor_years, self.rates))


@dataclass
class FXRate:
    """Foreign exchange rate data."""
    
    base_currency: str
    quote_currency: str
    
    # Rates
    spot: float
    forward_1m: float | None = None
    forward_3m: float | None = None
    forward_1y: float | None = None
    
    # Volatility
    implied_vol_1m: float | None = None
    implied_vol_3m: float | None = None
    
    # Date
    as_of_date: date = field(default_factory=date.today)
    
    @property
    def pair(self) -> str:
        """Get currency pair string."""
        return f"{self.base_currency}/{self.quote_currency}"


@dataclass
class CreditSpread:
    """Credit spread data."""
    
    index_name: str  # CDX.NA.IG, CDX.NA.HY, iTraxx Europe, etc.
    
    # Spread
    spread_bps: float
    previous_spread_bps: float | None = None
    
    # Metrics
    duration: float | None = None
    dv01: float | None = None
    
    # Components
    sector_spreads: dict[str, float] = field(default_factory=dict)
    
    # Date
    as_of_date: date = field(default_factory=date.today)


@dataclass
class CommodityPrice:
    """Commodity price data."""
    
    commodity: str  # WTI, Brent, Gold, Silver, Copper, etc.
    unit: str  # USD/bbl, USD/oz, USD/lb, etc.
    
    # Prices
    spot: float
    front_month: float | None = None
    back_month: float | None = None
    
    # Curve metrics
    contango_backwardation: float | None = None  # Front - Back
    
    # Inventory
    inventory_change: float | None = None
    
    # Date
    as_of_date: date = field(default_factory=date.today)


@dataclass
class EquityIndex:
    """Equity index data."""
    
    index_name: str  # SPX, NDX, RTY, VIX, etc.
    
    # Price data
    price: float
    previous_close: float
    
    # Performance
    daily_return: float | None = None
    mtd_return: float | None = None
    ytd_return: float | None = None
    
    # Valuation
    pe_ratio: float | None = None
    earnings_yield: float | None = None
    dividend_yield: float | None = None
    
    # Volatility
    realized_vol_20d: float | None = None
    implied_vol: float | None = None  # e.g., VIX for SPX
    
    # Technicals
    ma_50: float | None = None
    ma_200: float | None = None
    rsi_14: float | None = None
    
    # Date
    as_of_date: date = field(default_factory=date.today)


class MacroDataProvider:
    """
    Provider for macro and asset class specific market data.
    
    Integrates with:
    - Bloomberg
    - Reuters
    - FRED
    - Central bank APIs
    - Proprietary sources
    """
    
    def __init__(self):
        self._cache: dict[str, Any] = {}
        self._indicators: dict[str, list[MacroDataPoint]] = {}
        self._yield_curves: dict[str, YieldCurve] = {}
        self._fx_rates: dict[str, FXRate] = {}
        self._credit_spreads: dict[str, CreditSpread] = {}
        self._commodities: dict[str, CommodityPrice] = {}
        self._indices: dict[str, EquityIndex] = {}
    
    def load_macro_indicator(
        self,
        indicator: MacroIndicatorType,
        region: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[MacroDataPoint]:
        """Load historical macro indicator data."""
        # In production, fetch from data source
        # For now, generate sample data
        key = f"{indicator.name}_{region}"
        
        if key not in self._indicators:
            self._indicators[key] = self._generate_sample_macro_data(
                indicator, region
            )
        
        return self._indicators[key]
    
    def get_yield_curve(
        self,
        currency: str,
        curve_type: str = "government",
        as_of_date: date | None = None,
    ) -> YieldCurve:
        """Get yield curve data."""
        key = f"{currency}_{curve_type}"
        
        if key not in self._yield_curves:
            self._yield_curves[key] = self._generate_sample_yield_curve(
                currency, curve_type
            )
        
        return self._yield_curves[key]
    
    def get_fx_rate(
        self,
        base_currency: str,
        quote_currency: str,
    ) -> FXRate:
        """Get FX rate data."""
        pair = f"{base_currency}/{quote_currency}"
        
        if pair not in self._fx_rates:
            self._fx_rates[pair] = self._generate_sample_fx_rate(
                base_currency, quote_currency
            )
        
        return self._fx_rates[pair]
    
    def get_credit_spread(self, index_name: str) -> CreditSpread:
        """Get credit spread data."""
        if index_name not in self._credit_spreads:
            self._credit_spreads[index_name] = self._generate_sample_credit_spread(
                index_name
            )
        
        return self._credit_spreads[index_name]
    
    def get_commodity_price(self, commodity: str) -> CommodityPrice:
        """Get commodity price data."""
        if commodity not in self._commodities:
            self._commodities[commodity] = self._generate_sample_commodity(commodity)
        
        return self._commodities[commodity]
    
    def get_equity_index(self, index_name: str) -> EquityIndex:
        """Get equity index data."""
        if index_name not in self._indices:
            self._indices[index_name] = self._generate_sample_index(index_name)
        
        return self._indices[index_name]
    
    def get_market_snapshot(self) -> dict[str, Any]:
        """Get comprehensive market snapshot."""
        return {
            "timestamp": datetime.now().isoformat(),
            "equity_indices": {
                "SPX": self.get_equity_index("SPX"),
                "NDX": self.get_equity_index("NDX"),
                "RTY": self.get_equity_index("RTY"),
                "VIX": self.get_equity_index("VIX"),
            },
            "yield_curves": {
                "USD": self.get_yield_curve("USD"),
                "EUR": self.get_yield_curve("EUR"),
            },
            "fx_rates": {
                "EUR/USD": self.get_fx_rate("EUR", "USD"),
                "USD/JPY": self.get_fx_rate("USD", "JPY"),
                "GBP/USD": self.get_fx_rate("GBP", "USD"),
            },
            "commodities": {
                "WTI": self.get_commodity_price("WTI"),
                "Gold": self.get_commodity_price("Gold"),
            },
            "credit": {
                "CDX.NA.IG": self.get_credit_spread("CDX.NA.IG"),
                "CDX.NA.HY": self.get_credit_spread("CDX.NA.HY"),
            },
        }
    
    def get_economic_calendar(
        self,
        start_date: date,
        end_date: date,
        region: str | None = None,
    ) -> list[dict]:
        """Get economic calendar for a date range."""
        # Sample economic calendar
        calendar = [
            {
                "date": "2026-01-27",
                "time": "08:30 ET",
                "indicator": "Durable Goods Orders",
                "region": "US",
                "period": "Dec",
                "consensus": "1.0%",
                "previous": "0.5%",
                "importance": "high",
            },
            {
                "date": "2026-01-28",
                "time": "10:00 ET",
                "indicator": "Consumer Confidence",
                "region": "US",
                "period": "Jan",
                "consensus": "105.0",
                "previous": "104.7",
                "importance": "high",
            },
            {
                "date": "2026-01-29",
                "time": "14:00 ET",
                "indicator": "FOMC Decision",
                "region": "US",
                "period": "Jan",
                "consensus": "4.25%",
                "previous": "4.25%",
                "importance": "critical",
            },
        ]
        
        if region:
            calendar = [e for e in calendar if e["region"] == region]
        
        return calendar
    
    def _generate_sample_macro_data(
        self,
        indicator: MacroIndicatorType,
        region: str,
    ) -> list[MacroDataPoint]:
        """Generate sample macro data for testing."""
        # Sample values by indicator
        base_values = {
            MacroIndicatorType.GDP: 2.5,
            MacroIndicatorType.CPI: 3.2,
            MacroIndicatorType.UNEMPLOYMENT_RATE: 4.1,
            MacroIndicatorType.FED_FUNDS_RATE: 4.25,
            MacroIndicatorType.PMI_MANUFACTURING: 52.5,
        }
        
        base = base_values.get(indicator, 100.0)
        
        return [
            MacroDataPoint(
                indicator=indicator,
                region=region,
                value=base + np.random.randn() * 0.5,
                previous_value=base,
                consensus_estimate=base + np.random.randn() * 0.2,
                source="Sample",
            )
        ]
    
    def _generate_sample_yield_curve(
        self,
        currency: str,
        curve_type: str,
    ) -> YieldCurve:
        """Generate sample yield curve."""
        tenors = ["1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"]
        
        # Base rates by currency
        base_rates = {
            "USD": [4.3, 4.4, 4.5, 4.4, 4.2, 4.1, 4.3, 4.5],
            "EUR": [2.8, 2.9, 3.0, 3.0, 2.8, 2.7, 2.9, 3.1],
            "JPY": [0.1, 0.1, 0.2, 0.3, 0.5, 0.8, 1.1, 1.5],
        }
        
        rates = base_rates.get(currency, base_rates["USD"])
        
        return YieldCurve(
            currency=currency,
            curve_type=curve_type,
            tenors=tenors,
            rates=rates,
            slope_2s10s=rates[6] - rates[4],  # 10Y - 2Y
            slope_5s30s=rates[7] - rates[5],  # 30Y - 5Y
        )
    
    def _generate_sample_fx_rate(
        self,
        base_currency: str,
        quote_currency: str,
    ) -> FXRate:
        """Generate sample FX rate."""
        spot_rates = {
            ("EUR", "USD"): 1.0850,
            ("USD", "JPY"): 148.50,
            ("GBP", "USD"): 1.2650,
            ("USD", "CHF"): 0.8720,
        }
        
        spot = spot_rates.get((base_currency, quote_currency), 1.0)
        
        return FXRate(
            base_currency=base_currency,
            quote_currency=quote_currency,
            spot=spot,
            forward_1m=spot * 1.001,
            forward_3m=spot * 1.003,
            forward_1y=spot * 1.012,
            implied_vol_1m=8.5,
            implied_vol_3m=9.2,
        )
    
    def _generate_sample_credit_spread(self, index_name: str) -> CreditSpread:
        """Generate sample credit spread."""
        spreads = {
            "CDX.NA.IG": 55.0,
            "CDX.NA.HY": 350.0,
            "iTraxx Europe": 60.0,
        }
        
        spread = spreads.get(index_name, 100.0)
        
        return CreditSpread(
            index_name=index_name,
            spread_bps=spread,
            previous_spread_bps=spread + np.random.randn() * 5,
            duration=5.0,
            dv01=4500.0,
        )
    
    def _generate_sample_commodity(self, commodity: str) -> CommodityPrice:
        """Generate sample commodity price."""
        prices = {
            "WTI": (75.50, "USD/bbl"),
            "Brent": (79.80, "USD/bbl"),
            "Gold": (2650.0, "USD/oz"),
            "Silver": (31.50, "USD/oz"),
            "Copper": (4.15, "USD/lb"),
        }
        
        price, unit = prices.get(commodity, (100.0, "USD"))
        
        return CommodityPrice(
            commodity=commodity,
            unit=unit,
            spot=price,
            front_month=price * 1.002,
            back_month=price * 1.015,
            contango_backwardation=-price * 0.013,
        )
    
    def _generate_sample_index(self, index_name: str) -> EquityIndex:
        """Generate sample equity index."""
        indices = {
            "SPX": (5950.0, 22.5, 0.18, 1.35),  # price, pe, vol, div
            "NDX": (21000.0, 35.0, 0.22, 0.65),
            "RTY": (2250.0, 28.0, 0.25, 1.15),
            "VIX": (15.5, None, None, None),
        }
        
        price, pe, vol, div = indices.get(index_name, (100.0, 20.0, 0.15, 1.5))
        
        return EquityIndex(
            index_name=index_name,
            price=price,
            previous_close=price * 0.998,
            daily_return=0.002,
            mtd_return=0.025,
            ytd_return=0.018,
            pe_ratio=pe,
            dividend_yield=div,
            realized_vol_20d=vol,
            implied_vol=15.5 if index_name == "SPX" else None,
            ma_50=price * 0.98,
            ma_200=price * 0.95,
            rsi_14=55.0,
        )
