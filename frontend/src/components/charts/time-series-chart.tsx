"use client";

import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { cn } from "@/lib/utils";

interface TimeSeriesData {
  timestamps: number[];
  values: number[];
  label?: string;
}

interface TimeSeriesChartProps {
  data: TimeSeriesData[];
  width?: number;
  height?: number;
  className?: string;
  showGrid?: boolean;
  showLegend?: boolean;
  showTooltip?: boolean;
  colors?: string[];
  title?: string;
  yAxisLabel?: string;
  enableZoom?: boolean;
  enableDownsampling?: boolean;
  targetPoints?: number;
}

// LTTB Downsampling Algorithm
function lttbDownsample(
  timestamps: number[],
  values: number[],
  targetPoints: number
): { timestamps: number[]; values: number[] } {
  const n = timestamps.length;

  if (n <= targetPoints) {
    return { timestamps: [...timestamps], values: [...values] };
  }

  const sampledTimestamps: number[] = [timestamps[0]];
  const sampledValues: number[] = [values[0]];

  const bucketSize = (n - 2) / (targetPoints - 2);

  let aX = timestamps[0];
  let aY = values[0];

  for (let i = 0; i < targetPoints - 2; i++) {
    const bucketStart = Math.floor((i + 0) * bucketSize) + 1;
    const bucketEnd = Math.floor((i + 1) * bucketSize) + 1;
    const nextStart = Math.floor((i + 1) * bucketSize) + 1;
    const nextEnd = Math.min(
      Math.floor((i + 2) * bucketSize) + 1,
      n - 1
    );

    // Average of next bucket
    let avgX = 0;
    let avgY = 0;
    for (let j = nextStart; j < nextEnd; j++) {
      avgX += timestamps[j];
      avgY += values[j];
    }
    avgX /= nextEnd - nextStart || 1;
    avgY /= nextEnd - nextStart || 1;

    // Find point with max triangle area
    let maxArea = -1;
    let maxIdx = bucketStart;

    for (let j = bucketStart; j < Math.min(bucketEnd, n - 1); j++) {
      const area =
        Math.abs(
          (aX - avgX) * (values[j] - aY) -
            (aX - timestamps[j]) * (avgY - aY)
        ) * 0.5;

      if (area > maxArea) {
        maxArea = area;
        maxIdx = j;
      }
    }

    sampledTimestamps.push(timestamps[maxIdx]);
    sampledValues.push(values[maxIdx]);

    aX = timestamps[maxIdx];
    aY = values[maxIdx];
  }

  sampledTimestamps.push(timestamps[n - 1]);
  sampledValues.push(values[n - 1]);

  return { timestamps: sampledTimestamps, values: sampledValues };
}

export function TimeSeriesChart({
  data,
  width = 800,
  height = 400,
  className,
  showGrid = true,
  showLegend = true,
  showTooltip = true,
  colors = [
    "#6366f1",
    "#22c55e",
    "#f59e0b",
    "#ef4444",
    "#8b5cf6",
    "#ec4899",
  ],
  title,
  yAxisLabel,
  enableZoom = true,
  enableDownsampling = true,
  targetPoints = 1000,
}: TimeSeriesChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    values: { label: string; value: number; color: string }[];
    timestamp: number;
  } | null>(null);
  const [zoomRange, setZoomRange] = useState<{
    start: number;
    end: number;
  } | null>(null);

  // Downsample data if needed
  const processedData = useMemo(() => {
    if (!enableDownsampling) return data;

    return data.map((series) => {
      const { timestamps, values } = lttbDownsample(
        series.timestamps,
        series.values,
        targetPoints
      );
      return { ...series, timestamps, values };
    });
  }, [data, enableDownsampling, targetPoints]);

  // Compute bounds
  const bounds = useMemo(() => {
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;

    for (const series of processedData) {
      for (let i = 0; i < series.timestamps.length; i++) {
        minX = Math.min(minX, series.timestamps[i]);
        maxX = Math.max(maxX, series.timestamps[i]);
        minY = Math.min(minY, series.values[i]);
        maxY = Math.max(maxY, series.values[i]);
      }
    }

    // Apply zoom
    if (zoomRange) {
      minX = zoomRange.start;
      maxX = zoomRange.end;
    }

    // Add padding
    const yPadding = (maxY - minY) * 0.1 || 1;
    minY -= yPadding;
    maxY += yPadding;

    return { minX, maxX, minY, maxY };
  }, [processedData, zoomRange]);

  // Render function
  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { minX, maxX, minY, maxY } = bounds;
    const padding = { top: 40, right: 20, bottom: 50, left: 70 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    // Clear
    ctx.fillStyle = "#0a0a0f";
    ctx.fillRect(0, 0, width, height);

    // Title
    if (title) {
      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 14px Inter, system-ui";
      ctx.fillText(title, padding.left, 25);
    }

    // Grid
    if (showGrid) {
      ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
      ctx.lineWidth = 1;

      // Horizontal lines
      const yTicks = 5;
      for (let i = 0; i <= yTicks; i++) {
        const y = padding.top + (chartHeight / yTicks) * i;
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(padding.left + chartWidth, y);
        ctx.stroke();
      }

      // Vertical lines
      const xTicks = 6;
      for (let i = 0; i <= xTicks; i++) {
        const x = padding.left + (chartWidth / xTicks) * i;
        ctx.beginPath();
        ctx.moveTo(x, padding.top);
        ctx.lineTo(x, padding.top + chartHeight);
        ctx.stroke();
      }
    }

    // Axes
    ctx.strokeStyle = "rgba(255, 255, 255, 0.3)";
    ctx.lineWidth = 1;

    // Y axis
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + chartHeight);
    ctx.stroke();

    // X axis
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top + chartHeight);
    ctx.lineTo(padding.left + chartWidth, padding.top + chartHeight);
    ctx.stroke();

    // Y axis labels
    ctx.fillStyle = "rgba(255, 255, 255, 0.7)";
    ctx.font = "11px Inter, system-ui";
    ctx.textAlign = "right";

    const yTicks = 5;
    for (let i = 0; i <= yTicks; i++) {
      const value = maxY - ((maxY - minY) / yTicks) * i;
      const y = padding.top + (chartHeight / yTicks) * i;
      ctx.fillText(value.toFixed(2), padding.left - 8, y + 4);
    }

    // X axis labels
    ctx.textAlign = "center";
    const xTicks = 6;
    for (let i = 0; i <= xTicks; i++) {
      const timestamp = minX + ((maxX - minX) / xTicks) * i;
      const x = padding.left + (chartWidth / xTicks) * i;
      const date = new Date(timestamp);
      ctx.fillText(
        date.toLocaleDateString("en-US", { month: "short", day: "numeric" }),
        x,
        padding.top + chartHeight + 20
      );
    }

    // Y axis label
    if (yAxisLabel) {
      ctx.save();
      ctx.translate(15, padding.top + chartHeight / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.fillStyle = "rgba(255, 255, 255, 0.7)";
      ctx.fillText(yAxisLabel, 0, 0);
      ctx.restore();
    }

    // Draw series
    for (let seriesIdx = 0; seriesIdx < processedData.length; seriesIdx++) {
      const series = processedData[seriesIdx];
      const color = colors[seriesIdx % colors.length];

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";

      ctx.beginPath();
      let started = false;

      for (let i = 0; i < series.timestamps.length; i++) {
        const ts = series.timestamps[i];
        const val = series.values[i];

        if (ts < minX || ts > maxX) continue;

        const x =
          padding.left + ((ts - minX) / (maxX - minX)) * chartWidth;
        const y =
          padding.top +
          chartHeight -
          ((val - minY) / (maxY - minY)) * chartHeight;

        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }

      ctx.stroke();
    }

    // Legend
    if (showLegend && processedData.some((s) => s.label)) {
      let legendX = padding.left + chartWidth - 100;
      let legendY = padding.top + 10;

      for (let i = 0; i < processedData.length; i++) {
        const series = processedData[i];
        if (!series.label) continue;

        const color = colors[i % colors.length];

        ctx.fillStyle = color;
        ctx.fillRect(legendX, legendY, 12, 12);

        ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
        ctx.font = "11px Inter, system-ui";
        ctx.textAlign = "left";
        ctx.fillText(series.label, legendX + 18, legendY + 10);

        legendY += 20;
      }
    }
  }, [
    processedData,
    bounds,
    width,
    height,
    showGrid,
    showLegend,
    colors,
    title,
    yAxisLabel,
  ]);

  // Handle mouse move for tooltip
  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!showTooltip) return;

      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      const { minX, maxX, minY, maxY } = bounds;
      const padding = { top: 40, right: 20, bottom: 50, left: 70 };
      const chartWidth = width - padding.left - padding.right;
      const chartHeight = height - padding.top - padding.bottom;

      // Check if in chart area
      if (
        x < padding.left ||
        x > padding.left + chartWidth ||
        y < padding.top ||
        y > padding.top + chartHeight
      ) {
        setTooltip(null);
        return;
      }

      // Find closest timestamp
      const timestamp =
        minX + ((x - padding.left) / chartWidth) * (maxX - minX);

      // Find values at this timestamp
      const values: { label: string; value: number; color: string }[] = [];

      for (let i = 0; i < processedData.length; i++) {
        const series = processedData[i];
        let closestIdx = 0;
        let closestDist = Infinity;

        for (let j = 0; j < series.timestamps.length; j++) {
          const dist = Math.abs(series.timestamps[j] - timestamp);
          if (dist < closestDist) {
            closestDist = dist;
            closestIdx = j;
          }
        }

        values.push({
          label: series.label || `Series ${i + 1}`,
          value: series.values[closestIdx],
          color: colors[i % colors.length],
        });
      }

      setTooltip({ x, y, values, timestamp });
    },
    [processedData, bounds, width, height, showTooltip, colors]
  );

  const handleMouseLeave = () => {
    setTooltip(null);
  };

  // Render on changes
  useEffect(() => {
    render();
  }, [render]);

  return (
    <div className={cn("relative", className)}>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="rounded-lg"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      />

      {/* Tooltip */}
      {tooltip && (
        <div
          className="absolute pointer-events-none bg-black/90 border border-white/20 rounded-lg px-3 py-2 text-sm z-10"
          style={{
            left: tooltip.x + 10,
            top: tooltip.y - 10,
            transform:
              tooltip.x > width - 150 ? "translateX(-100%)" : "none",
          }}
        >
          <div className="text-white/60 text-xs mb-1">
            {new Date(tooltip.timestamp).toLocaleString()}
          </div>
          {tooltip.values.map((v, i) => (
            <div key={i} className="flex items-center gap-2">
              <div
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: v.color }}
              />
              <span className="text-white/80">{v.label}:</span>
              <span className="text-white font-medium">
                {v.value.toFixed(4)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// Demo data generator
export function generateDemoTimeSeries(
  days: number = 365,
  series: number = 2
): TimeSeriesData[] {
  const now = Date.now();
  const msPerDay = 24 * 60 * 60 * 1000;

  const result: TimeSeriesData[] = [];

  for (let s = 0; s < series; s++) {
    const timestamps: number[] = [];
    const values: number[] = [];

    let value = 100 + Math.random() * 20;

    for (let d = 0; d < days; d++) {
      timestamps.push(now - (days - d) * msPerDay);

      // Random walk with drift
      const drift = 0.0003 * (s + 1);
      const volatility = 0.02 * (s + 1);
      value *= 1 + drift + (Math.random() - 0.5) * volatility;
      values.push(value);
    }

    result.push({
      timestamps,
      values,
      label: `Portfolio ${s + 1}`,
    });
  }

  return result;
}
