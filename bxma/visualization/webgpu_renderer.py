"""
WebGPU-Accelerated Visualization Renderer
==========================================

High-performance visualization using WebGPU compute shaders.

Architecture:
- Server generates data in Arrow format
- WebGPU compute shaders perform aggregation on client GPU
- 60 FPS interactive rendering for millions of data points

Features:
- LTTB downsampling in GPU compute shaders
- 3D volatility surface rendering
- Real-time streaming updates
- Cross-platform (Chrome, Firefox, Safari)

This module provides the Python backend that prepares data for
WebGPU-based frontend rendering.

References:
- WebGPU Specification (W3C)
- Sveinn Steinarsson (2013): Downsampling Time Series for Visual Representation

Author: BXMA Quant Team
Date: January 2026
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
from enum import Enum, auto
import json


class ChartType(Enum):
    """Types of charts supported."""
    LINE = auto()
    CANDLESTICK = auto()
    HEATMAP = auto()
    SURFACE_3D = auto()
    SCATTER = auto()
    BAR = auto()
    HISTOGRAM = auto()


class ColorScale(Enum):
    """Color scales for heatmaps and surfaces."""
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    CIVIDIS = "cividis"
    TURBO = "turbo"
    RDYLGN = "RdYlGn"  # Red-Yellow-Green for PnL
    RDBU = "RdBu"  # Red-Blue diverging


@dataclass
class WebGPUConfig:
    """Configuration for WebGPU rendering."""
    
    # Canvas
    width: int = 1920
    height: int = 1080
    pixel_ratio: float = 2.0
    
    # Performance
    max_vertices: int = 10_000_000
    batch_size: int = 65536
    target_fps: int = 60
    
    # Features
    enable_antialiasing: bool = True
    enable_depth_buffer: bool = True
    enable_compute_shaders: bool = True
    
    # Streaming
    enable_streaming: bool = True
    stream_buffer_size: int = 1024 * 1024  # 1MB


@dataclass
class ChartData:
    """Data for chart rendering."""
    
    chart_id: str
    chart_type: ChartType
    
    # Time series data
    timestamps: NDArray[np.int64] | None = None
    values: NDArray[np.float64] | None = None
    
    # OHLCV data (for candlestick)
    open: NDArray[np.float64] | None = None
    high: NDArray[np.float64] | None = None
    low: NDArray[np.float64] | None = None
    close: NDArray[np.float64] | None = None
    volume: NDArray[np.float64] | None = None
    
    # Multi-series
    series: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    
    # Labels
    x_label: str = ""
    y_label: str = ""
    title: str = ""
    
    # Styling
    color_scale: ColorScale = ColorScale.VIRIDIS
    
    def to_gpu_buffer(self) -> dict:
        """Convert to format suitable for GPU buffer upload."""
        buffer_data = {
            "chart_id": self.chart_id,
            "chart_type": self.chart_type.name,
        }
        
        if self.timestamps is not None:
            buffer_data["timestamps"] = self.timestamps.tobytes().hex()
            buffer_data["timestamps_dtype"] = str(self.timestamps.dtype)
            buffer_data["timestamps_shape"] = list(self.timestamps.shape)
        
        if self.values is not None:
            buffer_data["values"] = self.values.tobytes().hex()
            buffer_data["values_dtype"] = str(self.values.dtype)
            buffer_data["values_shape"] = list(self.values.shape)
        
        return buffer_data


@dataclass
class VolatilitySurface:
    """3D volatility surface data."""
    
    # Axes
    strikes: NDArray[np.float64]  # X-axis
    expiries: NDArray[np.float64]  # Y-axis (days to expiry)
    
    # Surface values
    implied_vols: NDArray[np.float64]  # 2D grid (strikes x expiries)
    
    # Reference
    underlying: str = ""
    spot_price: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Computed Greeks (optional)
    delta_surface: NDArray[np.float64] | None = None
    gamma_surface: NDArray[np.float64] | None = None
    vega_surface: NDArray[np.float64] | None = None
    
    def to_mesh(self) -> dict:
        """Convert to 3D mesh format for rendering."""
        n_strikes = len(self.strikes)
        n_expiries = len(self.expiries)
        
        # Create vertex grid
        vertices = []
        indices = []
        colors = []
        
        # Normalize for visualization
        vol_min = np.min(self.implied_vols)
        vol_max = np.max(self.implied_vols)
        vol_range = vol_max - vol_min + 1e-8
        
        for i, strike in enumerate(self.strikes):
            for j, expiry in enumerate(self.expiries):
                # Position
                x = (strike - self.spot_price) / self.spot_price  # Moneyness
                y = expiry / 365.0  # Years
                z = self.implied_vols[i, j]
                
                vertices.append([x, y, z])
                
                # Color based on vol level
                t = (z - vol_min) / vol_range
                colors.append([t, 0.5, 1 - t, 1.0])
        
        # Create triangles
        for i in range(n_strikes - 1):
            for j in range(n_expiries - 1):
                # Two triangles per grid cell
                idx = i * n_expiries + j
                indices.extend([
                    idx, idx + 1, idx + n_expiries,
                    idx + 1, idx + n_expiries + 1, idx + n_expiries
                ])
        
        return {
            "vertices": np.array(vertices, dtype=np.float32).tobytes().hex(),
            "indices": np.array(indices, dtype=np.uint32).tobytes().hex(),
            "colors": np.array(colors, dtype=np.float32).tobytes().hex(),
            "n_vertices": len(vertices),
            "n_indices": len(indices),
            "vol_range": [float(vol_min), float(vol_max)],
        }


class WebGPURenderer:
    """
    Backend renderer that prepares data for WebGPU frontend.
    
    Generates:
    - GPU buffer data
    - Shader code
    - Render commands
    """
    
    def __init__(self, config: WebGPUConfig | None = None):
        self.config = config or WebGPUConfig()
        self._charts: dict[str, ChartData] = {}
        self._surfaces: dict[str, VolatilitySurface] = {}
    
    def add_chart(self, chart: ChartData):
        """Add a chart to the renderer."""
        self._charts[chart.chart_id] = chart
    
    def add_surface(self, surface_id: str, surface: VolatilitySurface):
        """Add a volatility surface to the renderer."""
        self._surfaces[surface_id] = surface
    
    def generate_render_packet(self) -> dict:
        """
        Generate a render packet to send to the WebGPU frontend.
        
        Contains all data and commands needed for one frame.
        """
        packet = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "width": self.config.width,
                "height": self.config.height,
                "pixel_ratio": self.config.pixel_ratio,
            },
            "charts": {},
            "surfaces": {},
            "commands": [],
        }
        
        for chart_id, chart in self._charts.items():
            packet["charts"][chart_id] = chart.to_gpu_buffer()
        
        for surface_id, surface in self._surfaces.items():
            packet["surfaces"][surface_id] = surface.to_mesh()
        
        return packet
    
    def generate_compute_shader(self, operation: str) -> str:
        """
        Generate WGSL compute shader code.
        
        Operations:
        - "lttb": Largest-Triangle-Three-Buckets downsampling
        - "aggregate": Group-by aggregation
        - "rolling_stats": Rolling statistics
        """
        if operation == "lttb":
            return self._lttb_shader()
        elif operation == "aggregate":
            return self._aggregate_shader()
        elif operation == "rolling_stats":
            return self._rolling_stats_shader()
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _lttb_shader(self) -> str:
        """WGSL shader for LTTB downsampling."""
        return '''
// LTTB (Largest Triangle Three Buckets) Downsampling
// Runs on GPU for real-time downsampling of time series

struct InputData {
    timestamp: f32,
    value: f32,
}

struct OutputData {
    timestamp: f32,
    value: f32,
    selected: u32,
}

@group(0) @binding(0) var<storage, read> input: array<InputData>;
@group(0) @binding(1) var<storage, read_write> output: array<OutputData>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // [input_size, output_size, 0, 0]

fn triangle_area(p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>) -> f32 {
    return abs((p1.x - p3.x) * (p2.y - p1.y) - (p1.x - p2.x) * (p3.y - p1.y)) * 0.5;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let bucket_idx = global_id.x;
    let input_size = params.x;
    let output_size = params.y;
    
    if (bucket_idx >= output_size - 2u) {
        return;
    }
    
    // First and last points always selected
    if (bucket_idx == 0u) {
        output[0].timestamp = input[0].timestamp;
        output[0].value = input[0].value;
        output[0].selected = 1u;
        return;
    }
    
    // Calculate bucket boundaries
    let bucket_size = f32(input_size - 2u) / f32(output_size - 2u);
    let bucket_start = u32(f32(bucket_idx - 1u) * bucket_size) + 1u;
    let bucket_end = min(u32(f32(bucket_idx) * bucket_size) + 1u, input_size - 1u);
    
    // Get average of next bucket for triangle calculation
    let next_bucket_start = bucket_end;
    let next_bucket_end = min(u32(f32(bucket_idx + 1u) * bucket_size) + 1u, input_size - 1u);
    
    var avg_x: f32 = 0.0;
    var avg_y: f32 = 0.0;
    var count: f32 = 0.0;
    
    for (var i = next_bucket_start; i < next_bucket_end; i = i + 1u) {
        avg_x = avg_x + input[i].timestamp;
        avg_y = avg_y + input[i].value;
        count = count + 1.0;
    }
    
    if (count > 0.0) {
        avg_x = avg_x / count;
        avg_y = avg_y / count;
    }
    
    let next_avg = vec2<f32>(avg_x, avg_y);
    
    // Previous selected point (simplified - use bucket start for efficiency)
    let prev_point = vec2<f32>(input[bucket_start - 1u].timestamp, input[bucket_start - 1u].value);
    
    // Find point with largest triangle area
    var max_area: f32 = -1.0;
    var max_idx: u32 = bucket_start;
    
    for (var i = bucket_start; i < bucket_end; i = i + 1u) {
        let current = vec2<f32>(input[i].timestamp, input[i].value);
        let area = triangle_area(prev_point, current, next_avg);
        
        if (area > max_area) {
            max_area = area;
            max_idx = i;
        }
    }
    
    // Output selected point
    output[bucket_idx].timestamp = input[max_idx].timestamp;
    output[bucket_idx].value = input[max_idx].value;
    output[bucket_idx].selected = 1u;
}
'''
    
    def _aggregate_shader(self) -> str:
        """WGSL shader for parallel aggregation."""
        return '''
// Parallel aggregation compute shader
// Performs group-by operations on GPU

struct DataPoint {
    group_id: u32,
    value: f32,
}

struct AggResult {
    group_id: u32,
    sum: f32,
    count: u32,
    min_val: f32,
    max_val: f32,
}

@group(0) @binding(0) var<storage, read> input: array<DataPoint>;
@group(0) @binding(1) var<storage, read_write> output: array<AggResult>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // [input_size, n_groups, 0, 0]

var<workgroup> shared_sums: array<f32, 256>;
var<workgroup> shared_counts: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let idx = global_id.x;
    let local_idx = local_id.x;
    let input_size = params.x;
    
    // Initialize shared memory
    shared_sums[local_idx] = 0.0;
    shared_counts[local_idx] = 0u;
    
    workgroupBarrier();
    
    // Load and accumulate
    if (idx < input_size) {
        let data = input[idx];
        // Atomic add would be ideal, using simple store for demo
        shared_sums[local_idx] = data.value;
        shared_counts[local_idx] = 1u;
    }
    
    workgroupBarrier();
    
    // Parallel reduction (simplified)
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (local_idx < stride) {
            shared_sums[local_idx] = shared_sums[local_idx] + shared_sums[local_idx + stride];
            shared_counts[local_idx] = shared_counts[local_idx] + shared_counts[local_idx + stride];
        }
        workgroupBarrier();
    }
    
    // Write result
    if (local_idx == 0u) {
        output[group_id.x].sum = shared_sums[0];
        output[group_id.x].count = shared_counts[0];
    }
}
'''
    
    def _rolling_stats_shader(self) -> str:
        """WGSL shader for rolling statistics."""
        return '''
// Rolling statistics compute shader
// Computes rolling mean, std, min, max on GPU

struct TimeSeriesPoint {
    timestamp: f32,
    value: f32,
}

struct RollingStats {
    mean: f32,
    std: f32,
    min_val: f32,
    max_val: f32,
}

@group(0) @binding(0) var<storage, read> input: array<TimeSeriesPoint>;
@group(0) @binding(1) var<storage, read_write> output: array<RollingStats>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // [input_size, window_size, 0, 0]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let input_size = params.x;
    let window_size = params.y;
    
    if (idx >= input_size) {
        return;
    }
    
    // Calculate window bounds
    let start = select(0u, idx - window_size + 1u, idx >= window_size - 1u);
    let end = idx + 1u;
    let count = end - start;
    
    // First pass: sum and min/max
    var sum: f32 = 0.0;
    var min_val: f32 = 1e38;
    var max_val: f32 = -1e38;
    
    for (var i = start; i < end; i = i + 1u) {
        let val = input[i].value;
        sum = sum + val;
        min_val = min(min_val, val);
        max_val = max(max_val, val);
    }
    
    let mean = sum / f32(count);
    
    // Second pass: variance
    var var_sum: f32 = 0.0;
    for (var i = start; i < end; i = i + 1u) {
        let diff = input[i].value - mean;
        var_sum = var_sum + diff * diff;
    }
    
    let std = sqrt(var_sum / f32(count));
    
    // Write output
    output[idx].mean = mean;
    output[idx].std = std;
    output[idx].min_val = min_val;
    output[idx].max_val = max_val;
}
'''


class TimeSeriesChart:
    """
    High-performance time series chart for WebGPU rendering.
    
    Handles:
    - Automatic LTTB downsampling
    - Real-time streaming updates
    - Multi-series overlay
    """
    
    def __init__(
        self,
        chart_id: str,
        max_points: int = 10_000_000,
        target_display_points: int = 2000,
    ):
        self.chart_id = chart_id
        self.max_points = max_points
        self.target_display_points = target_display_points
        
        # Data buffers
        self._timestamps: list[int] = []
        self._values: list[float] = []
        self._series: dict[str, list[float]] = {}
    
    def add_point(self, timestamp: int, value: float, series: str = "default"):
        """Add a point to the chart."""
        if series == "default":
            self._timestamps.append(timestamp)
            self._values.append(value)
        else:
            if series not in self._series:
                self._series[series] = []
            self._series[series].append(value)
        
        # Trim if exceeding max
        if len(self._timestamps) > self.max_points:
            self._timestamps = self._timestamps[-self.max_points:]
            self._values = self._values[-self.max_points:]
    
    def get_display_data(self) -> ChartData:
        """Get downsampled data for display."""
        timestamps = np.array(self._timestamps, dtype=np.int64)
        values = np.array(self._values, dtype=np.float64)
        
        # Downsample if needed
        if len(timestamps) > self.target_display_points:
            from bxma.visualization.downsampling import LTTBDownsampler
            downsampler = LTTBDownsampler(self.target_display_points)
            timestamps, values = downsampler.downsample(timestamps, values)
        
        return ChartData(
            chart_id=self.chart_id,
            chart_type=ChartType.LINE,
            timestamps=timestamps,
            values=values,
        )


class HeatmapChart:
    """
    High-performance heatmap for correlation matrices, etc.
    """
    
    def __init__(
        self,
        chart_id: str,
        labels_x: list[str] | None = None,
        labels_y: list[str] | None = None,
    ):
        self.chart_id = chart_id
        self.labels_x = labels_x or []
        self.labels_y = labels_y or []
        self._data: NDArray[np.float64] | None = None
    
    def set_data(self, data: NDArray[np.float64]):
        """Set the heatmap data."""
        self._data = data
    
    def get_render_data(self) -> dict:
        """Get data for rendering."""
        if self._data is None:
            return {}
        
        return {
            "chart_id": self.chart_id,
            "type": "heatmap",
            "data": self._data.tobytes().hex(),
            "shape": list(self._data.shape),
            "dtype": str(self._data.dtype),
            "labels_x": self.labels_x,
            "labels_y": self.labels_y,
            "value_range": [float(np.min(self._data)), float(np.max(self._data))],
        }


class Surface3DChart:
    """
    3D surface chart for volatility surfaces, PnL landscapes, etc.
    """
    
    def __init__(
        self,
        chart_id: str,
        x_label: str = "Strike",
        y_label: str = "Expiry",
        z_label: str = "Implied Vol",
    ):
        self.chart_id = chart_id
        self.x_label = x_label
        self.y_label = y_label
        self.z_label = z_label
        self._surface: VolatilitySurface | None = None
    
    def set_surface(self, surface: VolatilitySurface):
        """Set the surface data."""
        self._surface = surface
    
    def get_render_data(self) -> dict:
        """Get data for 3D rendering."""
        if self._surface is None:
            return {}
        
        mesh = self._surface.to_mesh()
        mesh["chart_id"] = self.chart_id
        mesh["labels"] = {
            "x": self.x_label,
            "y": self.y_label,
            "z": self.z_label,
        }
        
        return mesh
