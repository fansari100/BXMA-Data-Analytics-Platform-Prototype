"""
Large Dataset Management
========================

Efficient handling of large-scale financial datasets:
- Streaming data processing
- Chunked operations
- Memory-efficient transformations
- Parallel processing
- Data validation and cleaning

CRITICAL REQUIREMENT: Experience sourcing, cleaning, managing,
and analyzing large data sets.

Author: BXMA Quant Team
Date: January 2026
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Generator, Callable, Any, Literal
from pathlib import Path
import json


@dataclass
class DatasetConfig:
    """Configuration for large dataset operations."""
    
    # Memory management
    chunk_size: int = 100000
    max_memory_mb: int = 4096
    
    # Parallel processing
    n_workers: int = 4
    use_multiprocessing: bool = True
    
    # Data types
    optimize_dtypes: bool = True
    categorical_threshold: float = 0.5  # Convert to categorical if unique ratio < threshold
    
    # Caching
    enable_cache: bool = True
    cache_path: str = ".cache/datasets"
    
    # Validation
    validate_on_load: bool = True
    drop_duplicates: bool = True
    handle_missing: Literal["drop", "fill", "raise"] = "fill"


@dataclass
class DataQualityReport:
    """Report on data quality metrics."""
    
    total_rows: int = 0
    total_columns: int = 0
    
    # Missing data
    missing_counts: dict[str, int] = field(default_factory=dict)
    missing_percentages: dict[str, float] = field(default_factory=dict)
    
    # Duplicates
    duplicate_rows: int = 0
    
    # Data types
    column_types: dict[str, str] = field(default_factory=dict)
    
    # Statistics
    numeric_stats: dict[str, dict] = field(default_factory=dict)
    
    # Anomalies
    outliers_detected: dict[str, int] = field(default_factory=dict)
    
    # Timestamp
    generated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "missing_counts": self.missing_counts,
            "missing_percentages": self.missing_percentages,
            "duplicate_rows": self.duplicate_rows,
            "column_types": self.column_types,
            "numeric_stats": self.numeric_stats,
            "outliers_detected": self.outliers_detected,
            "generated_at": self.generated_at.isoformat(),
        }


class LargeDatasetManager:
    """
    Manager for large-scale dataset operations.
    
    Provides:
    - Memory-efficient data loading
    - Streaming transformations
    - Parallel processing
    - Data validation and cleaning
    """
    
    def __init__(self, config: DatasetConfig | None = None):
        self.config = config or DatasetConfig()
        self._cache: dict[str, Any] = {}
    
    def load_csv_chunked(
        self,
        filepath: str | Path,
        columns: list[str] | None = None,
        dtype: dict[str, type] | None = None,
        parse_dates: list[str] | None = None,
    ) -> Generator[NDArray, None, None]:
        """
        Load CSV file in chunks.
        
        Args:
            filepath: Path to CSV file
            columns: Columns to load (None for all)
            dtype: Column data types
            parse_dates: Columns to parse as dates
            
        Yields:
            Numpy arrays for each chunk
        """
        try:
            import polars as pl
            
            # Use Polars lazy frame for memory efficiency
            scanner = pl.scan_csv(
                str(filepath),
                has_header=True,
                n_rows=None,
            )
            
            if columns:
                scanner = scanner.select(columns)
            
            # Process in chunks
            df = scanner.collect()
            n_chunks = max(1, len(df) // self.config.chunk_size)
            
            for i in range(n_chunks):
                start = i * self.config.chunk_size
                end = min((i + 1) * self.config.chunk_size, len(df))
                chunk = df.slice(start, end - start)
                yield chunk.to_numpy()
                
        except ImportError:
            # Fallback to numpy
            data = np.genfromtxt(
                filepath,
                delimiter=",",
                names=True,
                max_rows=self.config.chunk_size,
            )
            yield data
    
    def load_parquet_optimized(
        self,
        filepath: str | Path,
        columns: list[str] | None = None,
        filters: list[tuple] | None = None,
    ) -> NDArray:
        """
        Load Parquet file with optimized settings.
        
        Uses predicate pushdown and column pruning.
        """
        try:
            import polars as pl
            
            df = pl.read_parquet(
                str(filepath),
                columns=columns,
            )
            
            return df.to_numpy()
            
        except ImportError:
            import pyarrow.parquet as pq
            
            table = pq.read_table(
                str(filepath),
                columns=columns,
            )
            
            return table.to_pandas().values
    
    def stream_transform(
        self,
        data_generator: Generator[NDArray, None, None],
        transform_fn: Callable[[NDArray], NDArray],
    ) -> Generator[NDArray, None, None]:
        """
        Apply transformation to streaming data.
        
        Args:
            data_generator: Generator yielding data chunks
            transform_fn: Function to apply to each chunk
            
        Yields:
            Transformed chunks
        """
        for chunk in data_generator:
            yield transform_fn(chunk)
    
    def parallel_process(
        self,
        data: NDArray,
        process_fn: Callable[[NDArray], NDArray],
        n_workers: int | None = None,
    ) -> NDArray:
        """
        Process data in parallel across multiple workers.
        
        Args:
            data: Input data
            process_fn: Function to apply
            n_workers: Number of workers (None for config default)
            
        Returns:
            Processed data
        """
        n_workers = n_workers or self.config.n_workers
        
        if not self.config.use_multiprocessing or n_workers <= 1:
            return process_fn(data)
        
        # Split data into chunks
        chunks = np.array_split(data, n_workers)
        
        # Process chunks (in production, use multiprocessing)
        results = [process_fn(chunk) for chunk in chunks]
        
        return np.concatenate(results)
    
    def validate_data(
        self,
        data: NDArray,
        column_names: list[str] | None = None,
    ) -> DataQualityReport:
        """
        Validate data quality.
        
        Args:
            data: Input data array
            column_names: Optional column names
            
        Returns:
            Data quality report
        """
        report = DataQualityReport(
            total_rows=data.shape[0],
            total_columns=data.shape[1] if data.ndim > 1 else 1,
        )
        
        if column_names is None:
            column_names = [f"col_{i}" for i in range(report.total_columns)]
        
        # Analyze each column
        for i, col_name in enumerate(column_names):
            if data.ndim == 1:
                col_data = data
            else:
                col_data = data[:, i]
            
            # Missing values
            if np.issubdtype(col_data.dtype, np.floating):
                missing = np.sum(np.isnan(col_data))
            else:
                missing = np.sum(col_data == None)  # noqa
            
            report.missing_counts[col_name] = int(missing)
            report.missing_percentages[col_name] = missing / len(col_data) * 100
            
            # Data type
            report.column_types[col_name] = str(col_data.dtype)
            
            # Numeric statistics
            if np.issubdtype(col_data.dtype, np.number):
                valid_data = col_data[~np.isnan(col_data)] if np.issubdtype(col_data.dtype, np.floating) else col_data
                
                if len(valid_data) > 0:
                    report.numeric_stats[col_name] = {
                        "mean": float(np.mean(valid_data)),
                        "std": float(np.std(valid_data)),
                        "min": float(np.min(valid_data)),
                        "max": float(np.max(valid_data)),
                        "median": float(np.median(valid_data)),
                    }
                    
                    # Detect outliers (>3 std from mean)
                    mean = np.mean(valid_data)
                    std = np.std(valid_data)
                    outliers = np.sum(np.abs(valid_data - mean) > 3 * std)
                    report.outliers_detected[col_name] = int(outliers)
        
        return report
    
    def clean_data(
        self,
        data: NDArray,
        column_names: list[str] | None = None,
        fill_values: dict[str, Any] | None = None,
    ) -> NDArray:
        """
        Clean data by handling missing values and outliers.
        
        Args:
            data: Input data
            column_names: Column names
            fill_values: Values to fill missing data
            
        Returns:
            Cleaned data
        """
        result = data.copy()
        
        # Handle missing values
        if self.config.handle_missing == "drop":
            # Drop rows with any missing values
            if np.issubdtype(result.dtype, np.floating):
                mask = ~np.any(np.isnan(result), axis=1) if result.ndim > 1 else ~np.isnan(result)
                result = result[mask]
        elif self.config.handle_missing == "fill":
            # Fill with provided values or column mean
            if np.issubdtype(result.dtype, np.floating):
                if result.ndim > 1:
                    for i in range(result.shape[1]):
                        col_data = result[:, i]
                        nan_mask = np.isnan(col_data)
                        if np.any(nan_mask):
                            if fill_values and column_names and column_names[i] in fill_values:
                                fill_val = fill_values[column_names[i]]
                            else:
                                fill_val = np.nanmean(col_data)
                            col_data[nan_mask] = fill_val
                else:
                    nan_mask = np.isnan(result)
                    if np.any(nan_mask):
                        result[nan_mask] = np.nanmean(result)
        
        # Drop duplicates
        if self.config.drop_duplicates and result.ndim > 1:
            result = np.unique(result, axis=0)
        
        return result
    
    def calculate_returns(
        self,
        prices: NDArray,
        method: Literal["simple", "log"] = "simple",
    ) -> NDArray:
        """
        Calculate returns from price series.
        
        Args:
            prices: Price array (time x assets)
            method: Return calculation method
            
        Returns:
            Returns array
        """
        if method == "log":
            returns = np.diff(np.log(prices), axis=0)
        else:
            returns = np.diff(prices, axis=0) / prices[:-1]
        
        return returns
    
    def calculate_rolling_statistics(
        self,
        data: NDArray,
        window: int,
        statistic: Literal["mean", "std", "sum", "min", "max"] = "mean",
    ) -> NDArray:
        """
        Calculate rolling statistics efficiently.
        
        Args:
            data: Input data (1D or 2D)
            window: Rolling window size
            statistic: Statistic to calculate
            
        Returns:
            Rolling statistic array
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        n_rows, n_cols = data.shape
        result = np.full((n_rows, n_cols), np.nan)
        
        for i in range(window - 1, n_rows):
            window_data = data[i - window + 1:i + 1]
            
            if statistic == "mean":
                result[i] = np.mean(window_data, axis=0)
            elif statistic == "std":
                result[i] = np.std(window_data, axis=0)
            elif statistic == "sum":
                result[i] = np.sum(window_data, axis=0)
            elif statistic == "min":
                result[i] = np.min(window_data, axis=0)
            elif statistic == "max":
                result[i] = np.max(window_data, axis=0)
        
        return result.squeeze()
    
    def resample_timeseries(
        self,
        data: NDArray,
        dates: list[date],
        target_frequency: Literal["weekly", "monthly", "quarterly"] = "monthly",
        aggregation: Literal["last", "mean", "sum"] = "last",
    ) -> tuple[NDArray, list[date]]:
        """
        Resample time series to lower frequency.
        
        Args:
            data: Input data
            dates: Date index
            target_frequency: Target frequency
            aggregation: Aggregation method
            
        Returns:
            Resampled data and dates
        """
        # Group by period
        periods: dict[str, list[int]] = {}
        
        for i, d in enumerate(dates):
            if target_frequency == "weekly":
                key = f"{d.year}-W{d.isocalendar()[1]:02d}"
            elif target_frequency == "monthly":
                key = f"{d.year}-{d.month:02d}"
            elif target_frequency == "quarterly":
                quarter = (d.month - 1) // 3 + 1
                key = f"{d.year}-Q{quarter}"
            else:
                key = str(d)
            
            if key not in periods:
                periods[key] = []
            periods[key].append(i)
        
        # Aggregate
        resampled_data = []
        resampled_dates = []
        
        for key, indices in sorted(periods.items()):
            period_data = data[indices]
            
            if aggregation == "last":
                agg_data = period_data[-1]
            elif aggregation == "mean":
                agg_data = np.mean(period_data, axis=0)
            elif aggregation == "sum":
                agg_data = np.sum(period_data, axis=0)
            else:
                agg_data = period_data[-1]
            
            resampled_data.append(agg_data)
            resampled_dates.append(dates[indices[-1]])
        
        return np.array(resampled_data), resampled_dates
    
    def merge_datasets(
        self,
        left: NDArray,
        right: NDArray,
        left_key_col: int = 0,
        right_key_col: int = 0,
        how: Literal["inner", "left", "right", "outer"] = "inner",
    ) -> NDArray:
        """
        Merge two datasets on a key column.
        
        Args:
            left: Left dataset
            right: Right dataset
            left_key_col: Key column index in left
            right_key_col: Key column index in right
            how: Merge type
            
        Returns:
            Merged dataset
        """
        # Get keys
        left_keys = left[:, left_key_col] if left.ndim > 1 else left
        right_keys = right[:, right_key_col] if right.ndim > 1 else right
        
        if how == "inner":
            common_keys = np.intersect1d(left_keys, right_keys)
            left_mask = np.isin(left_keys, common_keys)
            right_mask = np.isin(right_keys, common_keys)
            
            left_filtered = left[left_mask]
            right_filtered = right[right_mask]
            
            # Sort both by key
            left_order = np.argsort(left_filtered[:, left_key_col] if left.ndim > 1 else left_filtered)
            right_order = np.argsort(right_filtered[:, right_key_col] if right.ndim > 1 else right_filtered)
            
            left_sorted = left_filtered[left_order]
            right_sorted = right_filtered[right_order]
            
            # Remove key column from right before concat
            if right.ndim > 1:
                right_cols = [i for i in range(right.shape[1]) if i != right_key_col]
                right_sorted = right_sorted[:, right_cols]
            
            return np.hstack([left_sorted, right_sorted])
        
        # For other merge types, would need more complex implementation
        raise NotImplementedError(f"Merge type '{how}' not implemented")


# Utility functions for data operations
def detect_file_format(filepath: str | Path) -> str:
    """Detect file format from extension."""
    path = Path(filepath)
    ext = path.suffix.lower()
    
    format_map = {
        ".csv": "csv",
        ".tsv": "tsv",
        ".parquet": "parquet",
        ".arrow": "arrow",
        ".json": "json",
        ".feather": "feather",
        ".h5": "hdf5",
        ".hdf5": "hdf5",
    }
    
    return format_map.get(ext, "unknown")


def estimate_memory_usage(
    n_rows: int,
    n_cols: int,
    dtype: np.dtype = np.float64,
) -> int:
    """Estimate memory usage in bytes."""
    return n_rows * n_cols * dtype.itemsize


def optimal_chunk_size(
    n_rows: int,
    n_cols: int,
    dtype: np.dtype = np.float64,
    target_memory_mb: int = 100,
) -> int:
    """Calculate optimal chunk size for target memory."""
    bytes_per_row = n_cols * dtype.itemsize
    target_bytes = target_memory_mb * 1024 * 1024
    
    return max(1, int(target_bytes / bytes_per_row))
