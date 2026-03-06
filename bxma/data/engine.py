"""
High-Performance Data Engine for BXMA Data Analytics Platform.

Built on Polars + DuckDB + Arrow for maximum performance:
- 10-100x faster than pandas for large datasets
- SQL queries on DataFrames via DuckDB
- Zero-copy data sharing via Apache Arrow
- Lazy evaluation for query optimization
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, Sequence
import tempfile

import numpy as np
from numpy.typing import NDArray
import polars as pl
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from bxma.core.config import BXMAConfig


class DataEngine:
    """
    High-performance data engine combining Polars, DuckDB, and Arrow.
    
    Features:
    - Lazy evaluation with query optimization
    - SQL interface via DuckDB
    - Efficient time-series operations
    - Multi-source data aggregation
    - Parallel processing
    
    Performance Characteristics:
    - 10-100x faster than pandas for aggregations
    - Sub-millisecond query latency for cached data
    - Linear scaling with data size
    - Memory-efficient via lazy evaluation
    """
    
    def __init__(self, config: BXMAConfig | None = None) -> None:
        """
        Initialize the data engine.
        
        Args:
            config: BXMA configuration object
        """
        self.config = config or BXMAConfig()
        
        # Initialize DuckDB connection
        self._db = duckdb.connect(":memory:")
        self._configure_duckdb()
        
        # Data cache
        self._cache: dict[str, pl.LazyFrame] = {}
        self._arrow_cache: dict[str, pa.Table] = {}
        
    def _configure_duckdb(self) -> None:
        """Configure DuckDB for optimal performance."""
        # Enable parallel execution
        self._db.execute(f"SET threads TO {self.config.num_workers}")
        
        # Enable progress bar for long queries
        self._db.execute("SET enable_progress_bar = true")
        
        # Optimize memory usage
        self._db.execute("SET memory_limit = '8GB'")
        
        # Enable query optimization
        self._db.execute("SET enable_optimizer = true")
    
    # =========================================================================
    # DATA LOADING
    # =========================================================================
    
    def load_parquet(
        self,
        path: str | Path,
        columns: list[str] | None = None,
        filters: list[tuple] | None = None,
        lazy: bool = True,
    ) -> pl.LazyFrame | pl.DataFrame:
        """
        Load data from Parquet file with predicate pushdown.
        
        Args:
            path: Path to parquet file or directory
            columns: Columns to load (None = all)
            filters: Row filters for predicate pushdown
            lazy: Return LazyFrame for deferred execution
            
        Returns:
            Polars LazyFrame or DataFrame
        """
        path = Path(path)
        
        if lazy:
            lf = pl.scan_parquet(
                path,
                n_rows=None,
                cache=True,
                parallel="auto",
                rechunk=True,
            )
            
            if columns:
                lf = lf.select(columns)
            
            return lf
        else:
            return pl.read_parquet(path, columns=columns)
    
    def load_csv(
        self,
        path: str | Path,
        schema: dict[str, pl.DataType] | None = None,
        lazy: bool = True,
    ) -> pl.LazyFrame | pl.DataFrame:
        """
        Load data from CSV with schema inference optimization.
        
        Args:
            path: Path to CSV file
            schema: Optional explicit schema
            lazy: Return LazyFrame for deferred execution
            
        Returns:
            Polars LazyFrame or DataFrame
        """
        if lazy:
            return pl.scan_csv(
                path,
                dtypes=schema,
                infer_schema_length=10000,
                n_rows=None,
                cache=True,
            )
        else:
            return pl.read_csv(path, dtypes=schema)
    
    def from_pandas(self, df) -> pl.DataFrame:
        """
        Convert pandas DataFrame to Polars (zero-copy when possible).
        
        Args:
            df: pandas DataFrame
            
        Returns:
            Polars DataFrame
        """
        return pl.from_pandas(df)
    
    def to_pandas(self, df: pl.DataFrame):
        """
        Convert Polars DataFrame to pandas (zero-copy when possible).
        
        Args:
            df: Polars DataFrame
            
        Returns:
            pandas DataFrame
        """
        return df.to_pandas()
    
    def from_arrow(self, table: pa.Table) -> pl.DataFrame:
        """
        Convert Arrow table to Polars (zero-copy).
        
        Args:
            table: PyArrow Table
            
        Returns:
            Polars DataFrame
        """
        return pl.from_arrow(table)
    
    def to_arrow(self, df: pl.DataFrame) -> pa.Table:
        """
        Convert Polars DataFrame to Arrow (zero-copy).
        
        Args:
            df: Polars DataFrame
            
        Returns:
            PyArrow Table
        """
        return df.to_arrow()
    
    # =========================================================================
    # SQL INTERFACE (via DuckDB)
    # =========================================================================
    
    def sql(self, query: str, **params: Any) -> pl.DataFrame:
        """
        Execute SQL query on registered tables.
        
        Uses DuckDB for SQL execution with Polars interop.
        
        Args:
            query: SQL query string
            **params: Query parameters
            
        Returns:
            Polars DataFrame with results
        """
        result = self._db.execute(query, params).fetchdf()
        return pl.from_pandas(result)
    
    def register_table(
        self,
        name: str,
        data: pl.DataFrame | pl.LazyFrame | pa.Table,
    ) -> None:
        """
        Register a table for SQL queries.
        
        Args:
            name: Table name for SQL queries
            data: Data to register
        """
        if isinstance(data, pl.LazyFrame):
            data = data.collect()
        
        if isinstance(data, pl.DataFrame):
            arrow_table = data.to_arrow()
        else:
            arrow_table = data
        
        self._db.register(name, arrow_table)
        self._arrow_cache[name] = arrow_table
    
    def unregister_table(self, name: str) -> None:
        """Unregister a table from SQL interface."""
        self._db.unregister(name)
        self._arrow_cache.pop(name, None)
    
    # =========================================================================
    # TIME SERIES OPERATIONS
    # =========================================================================
    
    def resample(
        self,
        df: pl.LazyFrame | pl.DataFrame,
        date_col: str,
        freq: str,
        agg_exprs: dict[str, str],
    ) -> pl.LazyFrame | pl.DataFrame:
        """
        Resample time series to different frequency.
        
        Args:
            df: Input DataFrame
            date_col: Date column name
            freq: Target frequency ('1d', '1w', '1mo', etc.)
            agg_exprs: Aggregation expressions {col: agg_func}
            
        Returns:
            Resampled DataFrame
        """
        is_lazy = isinstance(df, pl.LazyFrame)
        if not is_lazy:
            df = df.lazy()
        
        # Build aggregation expressions
        aggs = []
        for col, func in agg_exprs.items():
            if func == "sum":
                aggs.append(pl.col(col).sum().alias(col))
            elif func == "mean":
                aggs.append(pl.col(col).mean().alias(col))
            elif func == "last":
                aggs.append(pl.col(col).last().alias(col))
            elif func == "first":
                aggs.append(pl.col(col).first().alias(col))
            elif func == "std":
                aggs.append(pl.col(col).std().alias(col))
            elif func == "min":
                aggs.append(pl.col(col).min().alias(col))
            elif func == "max":
                aggs.append(pl.col(col).max().alias(col))
        
        result = df.group_by_dynamic(date_col, every=freq).agg(aggs)
        
        return result if is_lazy else result.collect()
    
    def rolling_stats(
        self,
        df: pl.LazyFrame | pl.DataFrame,
        value_col: str,
        window: int,
        stats: list[str] = ["mean", "std", "min", "max"],
    ) -> pl.LazyFrame | pl.DataFrame:
        """
        Compute rolling statistics efficiently.
        
        Args:
            df: Input DataFrame
            value_col: Column to compute stats on
            window: Rolling window size
            stats: Statistics to compute
            
        Returns:
            DataFrame with rolling statistics
        """
        is_lazy = isinstance(df, pl.LazyFrame)
        if not is_lazy:
            df = df.lazy()
        
        exprs = []
        col = pl.col(value_col)
        
        if "mean" in stats:
            exprs.append(col.rolling_mean(window).alias(f"{value_col}_rolling_mean"))
        if "std" in stats:
            exprs.append(col.rolling_std(window).alias(f"{value_col}_rolling_std"))
        if "min" in stats:
            exprs.append(col.rolling_min(window).alias(f"{value_col}_rolling_min"))
        if "max" in stats:
            exprs.append(col.rolling_max(window).alias(f"{value_col}_rolling_max"))
        if "sum" in stats:
            exprs.append(col.rolling_sum(window).alias(f"{value_col}_rolling_sum"))
        if "skew" in stats:
            exprs.append(col.rolling_skew(window).alias(f"{value_col}_rolling_skew"))
        
        result = df.with_columns(exprs)
        
        return result if is_lazy else result.collect()
    
    def ewm_stats(
        self,
        df: pl.LazyFrame | pl.DataFrame,
        value_col: str,
        halflife: int,
    ) -> pl.LazyFrame | pl.DataFrame:
        """
        Compute exponentially weighted statistics.
        
        Args:
            df: Input DataFrame
            value_col: Column to compute stats on
            halflife: Halflife for exponential decay
            
        Returns:
            DataFrame with EWM statistics
        """
        is_lazy = isinstance(df, pl.LazyFrame)
        if not is_lazy:
            df = df.lazy()
        
        alpha = 1 - np.exp(-np.log(2) / halflife)
        
        result = df.with_columns([
            pl.col(value_col)
            .ewm_mean(alpha=alpha)
            .alias(f"{value_col}_ewm_mean"),
            pl.col(value_col)
            .ewm_std(alpha=alpha)
            .alias(f"{value_col}_ewm_std"),
        ])
        
        return result if is_lazy else result.collect()
    
    # =========================================================================
    # RETURNS CALCULATIONS
    # =========================================================================
    
    def compute_returns(
        self,
        df: pl.LazyFrame | pl.DataFrame,
        price_col: str,
        method: Literal["simple", "log"] = "simple",
        periods: int = 1,
    ) -> pl.LazyFrame | pl.DataFrame:
        """
        Compute returns from price series.
        
        Args:
            df: Input DataFrame with prices
            price_col: Price column name
            method: 'simple' or 'log' returns
            periods: Lookback periods
            
        Returns:
            DataFrame with returns column added
        """
        is_lazy = isinstance(df, pl.LazyFrame)
        if not is_lazy:
            df = df.lazy()
        
        if method == "simple":
            returns_expr = (
                pl.col(price_col) / pl.col(price_col).shift(periods) - 1
            ).alias(f"{price_col}_return")
        else:  # log returns
            returns_expr = (
                pl.col(price_col).log() - pl.col(price_col).shift(periods).log()
            ).alias(f"{price_col}_return")
        
        result = df.with_columns(returns_expr)
        
        return result if is_lazy else result.collect()
    
    def compute_covariance_matrix(
        self,
        returns_df: pl.DataFrame,
        asset_cols: list[str],
        method: Literal["sample", "ledoit_wolf", "exponential"] = "sample",
        halflife: int | None = None,
    ) -> NDArray[np.float64]:
        """
        Compute covariance matrix with various estimators.
        
        Args:
            returns_df: DataFrame with returns
            asset_cols: Asset column names
            method: Estimation method
            halflife: Halflife for exponential weighting
            
        Returns:
            Covariance matrix as numpy array
        """
        # Extract returns matrix
        returns = returns_df.select(asset_cols).to_numpy()
        
        if method == "sample":
            return np.cov(returns, rowvar=False)
        
        elif method == "exponential":
            if halflife is None:
                halflife = 63  # Default ~3 months
            
            n_obs = len(returns)
            weights = np.array([
                np.exp(-np.log(2) * (n_obs - i - 1) / halflife)
                for i in range(n_obs)
            ])
            weights /= weights.sum()
            
            # Weighted mean
            weighted_mean = np.average(returns, axis=0, weights=weights)
            
            # Weighted covariance
            centered = returns - weighted_mean
            cov = np.zeros((len(asset_cols), len(asset_cols)))
            for i in range(n_obs):
                cov += weights[i] * np.outer(centered[i], centered[i])
            
            return cov
        
        elif method == "ledoit_wolf":
            # Ledoit-Wolf shrinkage estimator
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(returns)
            return lw.covariance_
        
        else:
            raise ValueError(f"Unknown covariance method: {method}")
    
    # =========================================================================
    # CACHING
    # =========================================================================
    
    def cache(self, name: str, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Cache a LazyFrame for reuse.
        
        Args:
            name: Cache key
            data: LazyFrame to cache
            
        Returns:
            Cached LazyFrame
        """
        self._cache[name] = data
        return data
    
    def get_cached(self, name: str) -> pl.LazyFrame | None:
        """Get cached LazyFrame."""
        return self._cache.get(name)
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self._arrow_cache.clear()
    
    # =========================================================================
    # DATA EXPORT
    # =========================================================================
    
    def to_parquet(
        self,
        df: pl.DataFrame | pl.LazyFrame,
        path: str | Path,
        compression: str = "zstd",
    ) -> None:
        """
        Write DataFrame to Parquet with optimal compression.
        
        Args:
            df: DataFrame to write
            path: Output path
            compression: Compression codec
        """
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        
        df.write_parquet(path, compression=compression)
    
    def to_csv(
        self,
        df: pl.DataFrame | pl.LazyFrame,
        path: str | Path,
    ) -> None:
        """Write DataFrame to CSV."""
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        
        df.write_csv(path)


class ReturnsEngine:
    """
    Specialized engine for return calculations and transformations.
    
    Provides comprehensive return analytics:
    - Simple and log returns
    - Multi-period compounding
    - Risk-adjusted returns
    - Geometric linking
    """
    
    def __init__(self, data_engine: DataEngine) -> None:
        self.engine = data_engine
    
    def simple_returns(
        self,
        prices: NDArray[np.float64],
        periods: int = 1,
    ) -> NDArray[np.float64]:
        """Compute simple returns from price series."""
        return prices[periods:] / prices[:-periods] - 1
    
    def log_returns(
        self,
        prices: NDArray[np.float64],
        periods: int = 1,
    ) -> NDArray[np.float64]:
        """Compute log returns from price series."""
        return np.log(prices[periods:] / prices[:-periods])
    
    def compound_returns(
        self,
        returns: NDArray[np.float64],
        method: Literal["geometric", "arithmetic"] = "geometric",
    ) -> float:
        """Compound returns over period."""
        if method == "geometric":
            return np.prod(1 + returns) - 1
        else:
            return np.sum(returns)
    
    def annualize_returns(
        self,
        returns: NDArray[np.float64],
        periods_per_year: int = 252,
    ) -> float:
        """Annualize returns."""
        total_return = np.prod(1 + returns) - 1
        n_periods = len(returns)
        years = n_periods / periods_per_year
        return (1 + total_return) ** (1 / years) - 1
    
    def annualize_volatility(
        self,
        returns: NDArray[np.float64],
        periods_per_year: int = 252,
    ) -> float:
        """Annualize volatility."""
        return np.std(returns) * np.sqrt(periods_per_year)
    
    def sharpe_ratio(
        self,
        returns: NDArray[np.float64],
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> float:
        """Compute annualized Sharpe ratio."""
        excess_returns = returns - risk_free_rate / periods_per_year
        mean_excess = np.mean(excess_returns) * periods_per_year
        vol = self.annualize_volatility(returns, periods_per_year)
        return mean_excess / vol if vol > 0 else 0.0
    
    def sortino_ratio(
        self,
        returns: NDArray[np.float64],
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> float:
        """Compute annualized Sortino ratio."""
        excess_returns = returns - risk_free_rate / periods_per_year
        mean_excess = np.mean(excess_returns) * periods_per_year
        
        # Downside deviation
        downside = np.minimum(excess_returns, 0)
        downside_std = np.std(downside) * np.sqrt(periods_per_year)
        
        return mean_excess / downside_std if downside_std > 0 else 0.0
    
    def max_drawdown(
        self,
        returns: NDArray[np.float64],
    ) -> tuple[float, int, int]:
        """
        Compute maximum drawdown.
        
        Returns:
            (max_dd, peak_idx, trough_idx)
        """
        cum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = cum_returns / running_max - 1
        
        trough_idx = np.argmin(drawdowns)
        peak_idx = np.argmax(cum_returns[:trough_idx + 1]) if trough_idx > 0 else 0
        
        return float(drawdowns[trough_idx]), int(peak_idx), int(trough_idx)
