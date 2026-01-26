"""
Time Series Downsampling Algorithms
===================================

Implements visually-optimal downsampling for large time series.

Algorithms:
- LTTB: Largest-Triangle-Three-Buckets (preserves visual shape)
- MinMax: Preserves extrema (good for financial data)
- Adaptive: Automatically selects best algorithm

These algorithms run on both CPU (for preprocessing) and GPU
(for real-time zoom/pan via WebGPU compute shaders).

References:
- Sveinn Steinarsson (2013): Downsampling Time Series for Visual Representation

Author: BXMA Quant Team
Date: January 2026
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Protocol


class Downsampler(Protocol):
    """Protocol for downsampling algorithms."""
    
    def downsample(
        self,
        x: NDArray,
        y: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Downsample x, y arrays."""
        ...


class LTTBDownsampler:
    """
    Largest-Triangle-Three-Buckets downsampling.
    
    Preserves the visual appearance of the time series by selecting
    points that maximize the triangle area with neighboring buckets.
    
    This is the gold standard for time series visualization downsampling.
    """
    
    def __init__(self, target_points: int):
        """
        Args:
            target_points: Number of points to downsample to
        """
        self.target_points = target_points
    
    def downsample(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Downsample using LTTB algorithm.
        
        Args:
            x: X values (typically timestamps)
            y: Y values
            
        Returns:
            Tuple of (downsampled_x, downsampled_y)
        """
        n = len(x)
        
        if n <= self.target_points:
            return x.copy(), y.copy()
        
        # Always include first and last points
        sampled_x = [x[0]]
        sampled_y = [y[0]]
        
        # Bucket size
        bucket_size = (n - 2) / (self.target_points - 2)
        
        # Previous selected point
        a_x = x[0]
        a_y = y[0]
        
        for i in range(self.target_points - 2):
            # Calculate bucket boundaries
            bucket_start = int((i + 0) * bucket_size) + 1
            bucket_end = int((i + 1) * bucket_size) + 1
            
            # Calculate average of next bucket
            next_start = int((i + 1) * bucket_size) + 1
            next_end = int((i + 2) * bucket_size) + 1
            if next_end > n - 1:
                next_end = n - 1
            
            avg_x = np.mean(x[next_start:next_end + 1])
            avg_y = np.mean(y[next_start:next_end + 1])
            
            # Find point with maximum triangle area
            max_area = -1.0
            max_idx = bucket_start
            
            for j in range(bucket_start, min(bucket_end, n - 1)):
                # Triangle area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
                area = abs(
                    (a_x - avg_x) * (y[j] - a_y) -
                    (a_x - x[j]) * (avg_y - a_y)
                ) * 0.5
                
                if area > max_area:
                    max_area = area
                    max_idx = j
            
            sampled_x.append(x[max_idx])
            sampled_y.append(y[max_idx])
            
            # Update previous point
            a_x = x[max_idx]
            a_y = y[max_idx]
        
        # Add last point
        sampled_x.append(x[-1])
        sampled_y.append(y[-1])
        
        return np.array(sampled_x), np.array(sampled_y)


class MinMaxDownsampler:
    """
    Min-Max downsampling preserving extrema.
    
    For each bucket, keeps both the minimum and maximum values.
    This is particularly useful for financial data where extrema
    (highs and lows) are important.
    """
    
    def __init__(self, target_points: int):
        """
        Args:
            target_points: Approximate number of points to downsample to
                          (actual output will be ~2x this for min/max pairs)
        """
        self.target_points = target_points
    
    def downsample(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Downsample preserving min/max per bucket.
        
        Args:
            x: X values
            y: Y values
            
        Returns:
            Tuple of (downsampled_x, downsampled_y)
        """
        n = len(x)
        
        if n <= self.target_points:
            return x.copy(), y.copy()
        
        # Number of buckets
        n_buckets = self.target_points // 2
        bucket_size = n / n_buckets
        
        sampled_x = []
        sampled_y = []
        
        for i in range(n_buckets):
            start = int(i * bucket_size)
            end = int((i + 1) * bucket_size)
            end = min(end, n)
            
            if start >= end:
                continue
            
            bucket_x = x[start:end]
            bucket_y = y[start:end]
            
            min_idx = np.argmin(bucket_y)
            max_idx = np.argmax(bucket_y)
            
            # Add in temporal order
            if min_idx <= max_idx:
                sampled_x.extend([bucket_x[min_idx], bucket_x[max_idx]])
                sampled_y.extend([bucket_y[min_idx], bucket_y[max_idx]])
            else:
                sampled_x.extend([bucket_x[max_idx], bucket_x[min_idx]])
                sampled_y.extend([bucket_y[max_idx], bucket_y[min_idx]])
        
        return np.array(sampled_x), np.array(sampled_y)


class ModeDownsampler:
    """
    Mode-based downsampling for categorical/binned data.
    
    Selects the most frequent value in each bucket.
    """
    
    def __init__(self, target_points: int):
        self.target_points = target_points
    
    def downsample(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Downsample using mode per bucket."""
        n = len(x)
        
        if n <= self.target_points:
            return x.copy(), y.copy()
        
        bucket_size = n / self.target_points
        
        sampled_x = []
        sampled_y = []
        
        for i in range(self.target_points):
            start = int(i * bucket_size)
            end = int((i + 1) * bucket_size)
            end = min(end, n)
            
            if start >= end:
                continue
            
            bucket_x = x[start:end]
            bucket_y = y[start:end]
            
            # Use middle x
            mid_idx = len(bucket_x) // 2
            sampled_x.append(bucket_x[mid_idx])
            
            # Use mode of y (or median for continuous)
            sampled_y.append(np.median(bucket_y))
        
        return np.array(sampled_x), np.array(sampled_y)


class AdaptiveDownsampler:
    """
    Adaptive downsampling that selects the best algorithm.
    
    Automatically chooses between LTTB, MinMax, and simple sampling
    based on data characteristics.
    """
    
    def __init__(
        self,
        target_points: int,
        preserve_extrema: bool = True,
        financial_data: bool = True,
    ):
        """
        Args:
            target_points: Number of points to downsample to
            preserve_extrema: Whether to preserve min/max values
            financial_data: Whether this is financial data (affects algorithm choice)
        """
        self.target_points = target_points
        self.preserve_extrema = preserve_extrema
        self.financial_data = financial_data
        
        # Initialize downsamplers
        self._lttb = LTTBDownsampler(target_points)
        self._minmax = MinMaxDownsampler(target_points)
    
    def downsample(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Downsample using adaptively selected algorithm.
        
        Selection criteria:
        - Financial data with high volatility: MinMax
        - Smooth data: LTTB
        - Default for financial: MinMax
        """
        n = len(x)
        
        if n <= self.target_points:
            return x.copy(), y.copy()
        
        # Analyze data characteristics
        volatility = self._estimate_volatility(y)
        
        # Choose algorithm
        if self.financial_data and (self.preserve_extrema or volatility > 0.1):
            return self._minmax.downsample(x, y)
        else:
            return self._lttb.downsample(x, y)
    
    def _estimate_volatility(self, y: NDArray[np.float64]) -> float:
        """Estimate data volatility."""
        if len(y) < 2:
            return 0.0
        
        # Use coefficient of variation of returns
        returns = np.diff(y) / (np.abs(y[:-1]) + 1e-10)
        return float(np.std(returns))


class StreamingDownsampler:
    """
    Downsampling for streaming data.
    
    Maintains a fixed-size buffer and produces downsampled
    output as new data arrives.
    """
    
    def __init__(
        self,
        buffer_size: int = 100_000,
        output_size: int = 1000,
    ):
        """
        Args:
            buffer_size: Maximum points to keep in memory
            output_size: Number of points in downsampled output
        """
        self.buffer_size = buffer_size
        self.output_size = output_size
        
        self._x_buffer: list[float] = []
        self._y_buffer: list[float] = []
        self._downsampler = LTTBDownsampler(output_size)
    
    def add_point(self, x: float, y: float):
        """Add a new point to the stream."""
        self._x_buffer.append(x)
        self._y_buffer.append(y)
        
        # Trim if exceeding buffer
        if len(self._x_buffer) > self.buffer_size:
            # Keep last buffer_size points
            self._x_buffer = self._x_buffer[-self.buffer_size:]
            self._y_buffer = self._y_buffer[-self.buffer_size:]
    
    def add_points(self, x: NDArray[np.float64], y: NDArray[np.float64]):
        """Add multiple points."""
        self._x_buffer.extend(x.tolist())
        self._y_buffer.extend(y.tolist())
        
        # Trim
        if len(self._x_buffer) > self.buffer_size:
            self._x_buffer = self._x_buffer[-self.buffer_size:]
            self._y_buffer = self._y_buffer[-self.buffer_size:]
    
    def get_downsampled(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get current downsampled view."""
        if not self._x_buffer:
            return np.array([]), np.array([])
        
        x = np.array(self._x_buffer)
        y = np.array(self._y_buffer)
        
        return self._downsampler.downsample(x, y)
    
    @property
    def buffer_length(self) -> int:
        """Current buffer size."""
        return len(self._x_buffer)


@dataclass
class DownsamplingResult:
    """Result of downsampling operation."""
    
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    
    original_length: int
    downsampled_length: int
    compression_ratio: float
    algorithm: str
    
    # Quality metrics
    max_error: float = 0.0  # Maximum deviation from original
    mean_error: float = 0.0  # Mean deviation


def evaluate_downsampling_quality(
    original_x: NDArray[np.float64],
    original_y: NDArray[np.float64],
    downsampled_x: NDArray[np.float64],
    downsampled_y: NDArray[np.float64],
) -> dict:
    """
    Evaluate the quality of downsampling.
    
    Measures:
    - Shape preservation
    - Extrema preservation
    - Interpolation error
    """
    # Interpolate downsampled back to original x coordinates
    interpolated = np.interp(original_x, downsampled_x, downsampled_y)
    
    # Compute errors
    errors = np.abs(original_y - interpolated)
    
    # Extrema preservation
    orig_min = np.min(original_y)
    orig_max = np.max(original_y)
    ds_min = np.min(downsampled_y)
    ds_max = np.max(downsampled_y)
    
    return {
        "max_error": float(np.max(errors)),
        "mean_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "compression_ratio": len(original_x) / len(downsampled_x),
        "min_preserved": abs(orig_min - ds_min) < 0.01 * abs(orig_max - orig_min),
        "max_preserved": abs(orig_max - ds_max) < 0.01 * abs(orig_max - orig_min),
        "correlation": float(np.corrcoef(original_y, interpolated)[0, 1]),
    }
