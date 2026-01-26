"""
Data Feed Integrations
======================

Adapters for external data providers:
- Bloomberg
- Reuters
- Alternative data providers

Handles:
- Real-time streaming
- Historical data retrieval
- Data normalization

Author: BXMA Quant Team
Date: January 2026
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Callable, Iterator, AsyncIterator
from enum import Enum, auto
import asyncio


class FeedType(Enum):
    """Types of data feeds."""
    MARKET_DATA = auto()
    REFERENCE_DATA = auto()
    NEWS = auto()
    ANALYTICS = auto()
    ALTERNATIVE = auto()


class UpdateFrequency(Enum):
    """Data update frequencies."""
    TICK = auto()
    SECOND = auto()
    MINUTE = auto()
    HOUR = auto()
    DAILY = auto()
    INTRADAY = auto()


@dataclass
class FeedConfig:
    """Configuration for a data feed."""
    
    feed_id: str
    name: str
    provider: str
    feed_type: FeedType
    
    # Connection
    endpoint: str = ""
    port: int = 0
    
    # Authentication
    api_key: str = ""
    credentials: dict[str, str] = field(default_factory=dict)
    
    # Subscription
    symbols: list[str] = field(default_factory=list)
    fields: list[str] = field(default_factory=list)
    
    # Options
    frequency: UpdateFrequency = UpdateFrequency.TICK
    buffer_size: int = 10000
    
    # Error handling
    reconnect_attempts: int = 3
    reconnect_delay_ms: int = 1000


@dataclass
class MarketTick:
    """A single market data tick."""
    
    symbol: str
    timestamp: datetime
    
    # Prices
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    
    # Sizes
    bid_size: int = 0
    ask_size: int = 0
    last_size: int = 0
    
    # Derived
    mid: float = 0.0
    spread: float = 0.0
    
    # Flags
    is_trade: bool = False
    is_quote: bool = False
    exchange: str = ""


@dataclass
class OHLCV:
    """OHLCV bar data."""
    
    symbol: str
    timestamp: datetime
    
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    
    # Additional
    vwap: float = 0.0
    n_trades: int = 0


@dataclass
class NewsItem:
    """A news item."""
    
    id: str
    timestamp: datetime
    
    headline: str = ""
    body: str = ""
    source: str = ""
    
    # Classification
    categories: list[str] = field(default_factory=list)
    tickers: list[str] = field(default_factory=list)
    
    # Sentiment (pre-computed or raw)
    sentiment_score: float | None = None
    relevance_score: float | None = None


class DataFeedAdapter:
    """Base class for data feed adapters."""
    
    def __init__(self, config: FeedConfig):
        self.config = config
        self._connected = False
        self._callbacks: list[Callable] = []
    
    def connect(self) -> bool:
        """Connect to the feed."""
        raise NotImplementedError
    
    def disconnect(self):
        """Disconnect from the feed."""
        raise NotImplementedError
    
    def subscribe(self, symbols: list[str], fields: list[str] | None = None):
        """Subscribe to symbols."""
        raise NotImplementedError
    
    def unsubscribe(self, symbols: list[str]):
        """Unsubscribe from symbols."""
        raise NotImplementedError
    
    def on_data(self, callback: Callable[[MarketTick], None]):
        """Register data callback."""
        self._callbacks.append(callback)
    
    def get_historical(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        frequency: UpdateFrequency = UpdateFrequency.DAILY,
    ) -> list[OHLCV]:
        """Get historical data."""
        raise NotImplementedError


class BloombergAdapter(DataFeedAdapter):
    """
    Bloomberg data feed adapter.
    
    In production, uses Bloomberg API (blpapi).
    """
    
    def __init__(self, config: FeedConfig):
        super().__init__(config)
        self._session = None
    
    def connect(self) -> bool:
        """Connect to Bloomberg."""
        # In production:
        # import blpapi
        # self._session = blpapi.Session()
        # self._session.start()
        self._connected = True
        return True
    
    def disconnect(self):
        """Disconnect from Bloomberg."""
        self._connected = False
    
    def subscribe(self, symbols: list[str], fields: list[str] | None = None):
        """Subscribe to Bloomberg symbols."""
        if fields is None:
            fields = ["LAST_PRICE", "BID", "ASK", "VOLUME"]
        
        # In production:
        # subscriptions = blpapi.SubscriptionList()
        # for symbol in symbols:
        #     subscriptions.add(symbol, fields)
        # self._session.subscribe(subscriptions)
        
        self.config.symbols = symbols
        self.config.fields = fields
    
    def unsubscribe(self, symbols: list[str]):
        """Unsubscribe from Bloomberg symbols."""
        pass
    
    def get_historical(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        frequency: UpdateFrequency = UpdateFrequency.DAILY,
    ) -> list[OHLCV]:
        """Get historical data from Bloomberg."""
        # In production, uses refDataService
        # Generate sample data
        
        n_days = (end - start).days
        bars = []
        
        price = 100.0
        for i in range(n_days):
            dt = start + timedelta(days=i)
            
            # Random walk
            change = np.random.randn() * 0.02
            price *= (1 + change)
            
            bars.append(OHLCV(
                symbol=symbol,
                timestamp=dt,
                open=price * (1 + np.random.randn() * 0.005),
                high=price * (1 + abs(np.random.randn()) * 0.01),
                low=price * (1 - abs(np.random.randn()) * 0.01),
                close=price,
                volume=np.random.randint(1000000, 10000000),
            ))
        
        return bars
    
    def get_reference_data(
        self,
        symbols: list[str],
        fields: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Get reference data from Bloomberg."""
        # In production, uses refDataService
        result = {}
        
        for symbol in symbols:
            result[symbol] = {}
            for field in fields:
                if field == "NAME":
                    result[symbol][field] = f"{symbol} Corp"
                elif field == "MARKET_CAP":
                    result[symbol][field] = np.random.uniform(1e9, 1e12)
                elif field == "PE_RATIO":
                    result[symbol][field] = np.random.uniform(10, 30)
                elif field == "DIVIDEND_YIELD":
                    result[symbol][field] = np.random.uniform(0, 0.05)
        
        return result


class ReutersAdapter(DataFeedAdapter):
    """
    Reuters/Refinitiv data feed adapter.
    
    In production, uses Elektron Real-Time or Refinitiv Data Platform.
    """
    
    def __init__(self, config: FeedConfig):
        super().__init__(config)
    
    def connect(self) -> bool:
        """Connect to Reuters."""
        self._connected = True
        return True
    
    def disconnect(self):
        """Disconnect from Reuters."""
        self._connected = False
    
    def subscribe(self, symbols: list[str], fields: list[str] | None = None):
        """Subscribe to Reuters symbols."""
        if fields is None:
            fields = ["BID", "ASK", "TRDPRC_1", "TRDVOL_1"]
        
        self.config.symbols = symbols
        self.config.fields = fields
    
    def unsubscribe(self, symbols: list[str]):
        """Unsubscribe from Reuters symbols."""
        pass
    
    def get_historical(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        frequency: UpdateFrequency = UpdateFrequency.DAILY,
    ) -> list[OHLCV]:
        """Get historical data from Reuters."""
        # Similar to Bloomberg
        n_days = (end - start).days
        bars = []
        
        price = 100.0
        for i in range(n_days):
            dt = start + timedelta(days=i)
            change = np.random.randn() * 0.02
            price *= (1 + change)
            
            bars.append(OHLCV(
                symbol=symbol,
                timestamp=dt,
                open=price * (1 + np.random.randn() * 0.005),
                high=price * (1 + abs(np.random.randn()) * 0.01),
                low=price * (1 - abs(np.random.randn()) * 0.01),
                close=price,
                volume=np.random.randint(1000000, 10000000),
            ))
        
        return bars
    
    def get_news(
        self,
        query: str | None = None,
        symbols: list[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        max_items: int = 100,
    ) -> list[NewsItem]:
        """Get news from Reuters."""
        # In production, uses News API
        news = []
        
        for i in range(min(max_items, 10)):
            news.append(NewsItem(
                id=f"NEWS-{i}",
                timestamp=datetime.now() - timedelta(hours=i),
                headline=f"Sample headline {i}",
                body="Sample news body...",
                source="Reuters",
                categories=["Markets"],
                tickers=symbols or [],
                sentiment_score=np.random.uniform(-1, 1),
            ))
        
        return news


class DataFeedManager:
    """
    Manages multiple data feeds.
    
    Provides:
    - Unified interface to multiple providers
    - Failover handling
    - Data normalization
    - Caching
    """
    
    def __init__(self):
        self._feeds: dict[str, DataFeedAdapter] = {}
        self._primary_feed: str | None = None
        self._cache: dict[str, Any] = {}
    
    def register_feed(
        self,
        feed_id: str,
        adapter: DataFeedAdapter,
        is_primary: bool = False,
    ):
        """Register a data feed."""
        self._feeds[feed_id] = adapter
        if is_primary or self._primary_feed is None:
            self._primary_feed = feed_id
    
    def connect_all(self):
        """Connect to all feeds."""
        for feed_id, adapter in self._feeds.items():
            try:
                adapter.connect()
            except Exception as e:
                print(f"Failed to connect to {feed_id}: {e}")
    
    def disconnect_all(self):
        """Disconnect from all feeds."""
        for adapter in self._feeds.values():
            adapter.disconnect()
    
    def get_market_data(
        self,
        symbol: str,
        feed_id: str | None = None,
    ) -> MarketTick | None:
        """Get current market data for a symbol."""
        target_feed = feed_id or self._primary_feed
        if target_feed is None:
            return None
        
        # In production, would return live tick
        return MarketTick(
            symbol=symbol,
            timestamp=datetime.now(),
            bid=100.0,
            ask=100.1,
            last=100.05,
            mid=100.05,
            spread=0.1,
        )
    
    def get_historical(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        frequency: UpdateFrequency = UpdateFrequency.DAILY,
        feed_id: str | None = None,
    ) -> list[OHLCV]:
        """Get historical data with failover."""
        target_feed = feed_id or self._primary_feed
        
        if target_feed and target_feed in self._feeds:
            try:
                return self._feeds[target_feed].get_historical(
                    symbol, start, end, frequency
                )
            except Exception:
                pass
        
        # Failover to other feeds
        for fid, adapter in self._feeds.items():
            if fid != target_feed:
                try:
                    return adapter.get_historical(symbol, start, end, frequency)
                except Exception:
                    continue
        
        return []
    
    def normalize_symbol(self, symbol: str, source: str, target: str) -> str:
        """Normalize symbol between providers."""
        # In production, would use mapping tables
        mappings = {
            ("bloomberg", "reuters"): {
                "AAPL US Equity": "AAPL.O",
                "MSFT US Equity": "MSFT.O",
            },
            ("reuters", "bloomberg"): {
                "AAPL.O": "AAPL US Equity",
                "MSFT.O": "MSFT US Equity",
            },
        }
        
        mapping = mappings.get((source.lower(), target.lower()), {})
        return mapping.get(symbol, symbol)
