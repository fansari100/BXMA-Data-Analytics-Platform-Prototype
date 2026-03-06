"""
High-Performance Data Infrastructure for BXMA Data Analytics Platform.

Leverages cutting-edge technologies:
- Polars: Rust-based DataFrame (10-100x faster than pandas)
- DuckDB: In-process OLAP database
- Apache Arrow: Zero-copy columnar data format
- yfinance: Live market data feeds
- PostgreSQL/TimescaleDB for time-series persistence
- Macro and asset-class specific market data

Designed for scalable, computationally efficient analytics.
"""

# Lazy imports to avoid dependency issues
def __getattr__(name):
    if name == "DataEngine":
        from bxma.data.engine import DataEngine
        return DataEngine
    elif name == "market_data_service":
        from bxma.data.live_market_data import market_data_service
        return market_data_service
    elif name == "LiveMarketDataService":
        from bxma.data.live_market_data import LiveMarketDataService
        return LiveMarketDataService
    elif name == "MacroDataProvider":
        from bxma.data.macro import MacroDataProvider
        return MacroDataProvider
    elif name == "LargeDatasetManager":
        from bxma.data.large_datasets import LargeDatasetManager
        return LargeDatasetManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DataEngine",
    "market_data_service",
    "LiveMarketDataService",
    "MacroDataProvider",
    "LargeDatasetManager",
]
