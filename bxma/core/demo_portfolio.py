"""
BXMA Demo Portfolio Configuration
A realistic multi-asset portfolio for demonstration purposes (January 2026)
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List
import numpy as np

@dataclass
class PortfolioHolding:
    """Individual holding in the portfolio"""
    ticker: str
    name: str
    asset_class: str
    sector: str
    weight: float  # Target weight as decimal
    shares: int = 0
    cost_basis: float = 0.0
    current_price: float = 0.0
    
    @property
    def market_value(self) -> float:
        return self.shares * self.current_price
    
    @property
    def gain_loss(self) -> float:
        return self.market_value - (self.shares * self.cost_basis)
    
    @property
    def gain_loss_pct(self) -> float:
        if self.cost_basis > 0:
            return (self.current_price - self.cost_basis) / self.cost_basis
        return 0.0


@dataclass
class DemoPortfolio:
    """
    BXMA Multi-Asset Demo Portfolio
    Total AUM: ~$100 Million
    Strategy: Diversified Multi-Asset with Risk Parity tilt
    """
    name: str = "BXMA Multi-Asset Strategy"
    inception_date: date = field(default_factory=lambda: date(2020, 1, 2))
    base_currency: str = "USD"
    total_aum: float = 100_000_000.0  # $100M
    
    # Holdings with target weights
    holdings: List[PortfolioHolding] = field(default_factory=list)
    
    # Benchmark
    benchmark_ticker: str = "SPY"  # S&P 500 as primary benchmark
    
    def __post_init__(self):
        if not self.holdings:
            self.holdings = self._initialize_holdings()
    
    def _initialize_holdings(self) -> List[PortfolioHolding]:
        """Initialize the demo portfolio with realistic holdings"""
        return [
            # US Equities (35% total)
            PortfolioHolding(
                ticker="SPY", name="SPDR S&P 500 ETF", 
                asset_class="Equity", sector="US Large Cap",
                weight=0.15, shares=28000, cost_basis=420.50
            ),
            PortfolioHolding(
                ticker="QQQ", name="Invesco QQQ Trust",
                asset_class="Equity", sector="US Large Cap Tech",
                weight=0.08, shares=15000, cost_basis=380.25
            ),
            PortfolioHolding(
                ticker="IWM", name="iShares Russell 2000 ETF",
                asset_class="Equity", sector="US Small Cap",
                weight=0.05, shares=22000, cost_basis=195.00
            ),
            PortfolioHolding(
                ticker="VTV", name="Vanguard Value ETF",
                asset_class="Equity", sector="US Value",
                weight=0.04, shares=25000, cost_basis=145.00
            ),
            PortfolioHolding(
                ticker="MTUM", name="iShares MSCI USA Momentum Factor ETF",
                asset_class="Equity", sector="US Momentum",
                weight=0.03, shares=15000, cost_basis=175.00
            ),
            
            # International Equities (20% total)
            PortfolioHolding(
                ticker="EFA", name="iShares MSCI EAFE ETF",
                asset_class="Equity", sector="International Developed",
                weight=0.10, shares=120000, cost_basis=72.50
            ),
            PortfolioHolding(
                ticker="EEM", name="iShares MSCI Emerging Markets ETF",
                asset_class="Equity", sector="Emerging Markets",
                weight=0.06, shares=135000, cost_basis=40.25
            ),
            PortfolioHolding(
                ticker="VWO", name="Vanguard FTSE Emerging Markets ETF",
                asset_class="Equity", sector="Emerging Markets",
                weight=0.04, shares=85000, cost_basis=42.00
            ),
            
            # Fixed Income (25% total)
            PortfolioHolding(
                ticker="TLT", name="iShares 20+ Year Treasury Bond ETF",
                asset_class="Fixed Income", sector="US Govt Long",
                weight=0.08, shares=75000, cost_basis=98.50
            ),
            PortfolioHolding(
                ticker="IEF", name="iShares 7-10 Year Treasury Bond ETF",
                asset_class="Fixed Income", sector="US Govt Intermediate",
                weight=0.06, shares=55000, cost_basis=95.00
            ),
            PortfolioHolding(
                ticker="LQD", name="iShares iBoxx $ Investment Grade Corporate Bond ETF",
                asset_class="Fixed Income", sector="Corporate IG",
                weight=0.06, shares=50000, cost_basis=108.00
            ),
            PortfolioHolding(
                ticker="HYG", name="iShares iBoxx $ High Yield Corporate Bond ETF",
                asset_class="Fixed Income", sector="Corporate HY",
                weight=0.03, shares=35000, cost_basis=76.50
            ),
            PortfolioHolding(
                ticker="TIP", name="iShares TIPS Bond ETF",
                asset_class="Fixed Income", sector="TIPS",
                weight=0.02, shares=18000, cost_basis=105.00
            ),
            
            # Alternatives (12% total)
            PortfolioHolding(
                ticker="GLD", name="SPDR Gold Shares",
                asset_class="Alternatives", sector="Commodities - Gold",
                weight=0.05, shares=25000, cost_basis=180.00
            ),
            PortfolioHolding(
                ticker="VNQ", name="Vanguard Real Estate ETF",
                asset_class="Alternatives", sector="Real Estate",
                weight=0.04, shares=40000, cost_basis=85.00
            ),
            PortfolioHolding(
                ticker="DBC", name="Invesco DB Commodity Index Tracking Fund",
                asset_class="Alternatives", sector="Commodities - Broad",
                weight=0.03, shares=120000, cost_basis=22.50
            ),
            
            # Cash (8% total - represented by short-term treasuries)
            PortfolioHolding(
                ticker="SHV", name="iShares Short Treasury Bond ETF",
                asset_class="Cash", sector="Money Market",
                weight=0.05, shares=70000, cost_basis=110.00
            ),
            PortfolioHolding(
                ticker="BIL", name="SPDR Bloomberg 1-3 Month T-Bill ETF",
                asset_class="Cash", sector="T-Bills",
                weight=0.03, shares=32000, cost_basis=91.50
            ),
        ]
    
    def get_holdings_by_asset_class(self) -> Dict[str, List[PortfolioHolding]]:
        """Group holdings by asset class"""
        result = {}
        for h in self.holdings:
            if h.asset_class not in result:
                result[h.asset_class] = []
            result[h.asset_class].append(h)
        return result
    
    def get_tickers(self) -> List[str]:
        """Get list of all tickers"""
        return [h.ticker for h in self.holdings]
    
    def get_weights(self) -> np.ndarray:
        """Get weights as numpy array"""
        return np.array([h.weight for h in self.holdings])
    
    def update_prices(self, prices: Dict[str, float]):
        """Update current prices for all holdings"""
        for holding in self.holdings:
            if holding.ticker in prices:
                holding.current_price = prices[holding.ticker]
    
    def get_asset_allocation(self) -> Dict[str, float]:
        """Calculate actual asset allocation based on current prices"""
        total_value = sum(h.market_value for h in self.holdings)
        if total_value == 0:
            return {h.asset_class: h.weight for h in self.holdings}
        
        allocation = {}
        for h in self.holdings:
            if h.asset_class not in allocation:
                allocation[h.asset_class] = 0
            allocation[h.asset_class] += h.market_value / total_value
        return allocation


# Create singleton instance
DEMO_PORTFOLIO = DemoPortfolio()


# Risk-free rate (as of Jan 2026 - approximate 10Y Treasury yield)
RISK_FREE_RATE = 0.0428  # 4.28%

# Market indices for reference
MARKET_INDICES = {
    "SPY": "S&P 500",
    "QQQ": "NASDAQ 100",
    "IWM": "Russell 2000",
    "EFA": "MSCI EAFE",
    "EEM": "MSCI EM",
    "AGG": "Bloomberg US Agg Bond",
    "^VIX": "CBOE Volatility Index",
    "^TNX": "10-Year Treasury Yield",
}
