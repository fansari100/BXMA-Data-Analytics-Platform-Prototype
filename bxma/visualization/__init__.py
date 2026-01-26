"""
BXMA Visualization Module
=========================

High-performance visualization for financial data using WebGPU.

Components:
- WebGPU compute shaders for client-side aggregation
- LTTB downsampling for time-series
- 3D volatility surface rendering
- Real-time streaming dashboards

Author: BXMA Quant Team
Date: January 2026
"""

from bxma.visualization.webgpu_renderer import (
    WebGPUConfig,
    ChartData,
    VolatilitySurface,
    WebGPURenderer,
    TimeSeriesChart,
    HeatmapChart,
    Surface3DChart,
)

from bxma.visualization.downsampling import (
    LTTBDownsampler,
    MinMaxDownsampler,
    AdaptiveDownsampler,
)

__all__ = [
    "WebGPUConfig",
    "ChartData",
    "VolatilitySurface",
    "WebGPURenderer",
    "TimeSeriesChart",
    "HeatmapChart",
    "Surface3DChart",
    "LTTBDownsampler",
    "MinMaxDownsampler",
    "AdaptiveDownsampler",
]
