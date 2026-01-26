"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { cn } from "@/lib/utils";

interface HeatmapData {
  matrix: number[][];
  rowLabels: string[];
  colLabels: string[];
  title?: string;
}

interface HeatmapChartProps {
  data: HeatmapData;
  width?: number;
  height?: number;
  className?: string;
  colorScheme?: "rdbu" | "viridis" | "plasma" | "coolwarm";
  showValues?: boolean;
  symmetric?: boolean;
}

export function HeatmapChart({
  data,
  width = 600,
  height = 500,
  className,
  colorScheme = "rdbu",
  showValues = true,
  symmetric = true,
}: HeatmapChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredCell, setHoveredCell] = useState<{
    row: number;
    col: number;
    value: number;
  } | null>(null);

  // Color schemes
  const getColor = useCallback(
    (value: number, min: number, max: number) => {
      const t = (value - min) / (max - min || 1);

      const schemes = {
        rdbu: () => {
          if (t < 0.5) {
            // Blue to white
            const r = Math.floor(33 + t * 2 * (255 - 33));
            const g = Math.floor(102 + t * 2 * (255 - 102));
            const b = Math.floor(172 + t * 2 * (255 - 172));
            return `rgb(${r}, ${g}, ${b})`;
          } else {
            // White to red
            const r = 255;
            const g = Math.floor(255 - (t - 0.5) * 2 * (255 - 102));
            const b = Math.floor(255 - (t - 0.5) * 2 * (255 - 94));
            return `rgb(${r}, ${g}, ${b})`;
          }
        },
        viridis: () => {
          const r = Math.floor(68 + t * (253 - 68));
          const g = Math.floor(1 + t * (231 - 1));
          const b = Math.floor(84 + t * (37 - 84));
          return `rgb(${r}, ${g}, ${b})`;
        },
        plasma: () => {
          const r = Math.floor(13 + t * (240 - 13));
          const g = Math.floor(8 + t * (249 - 8));
          const b = Math.floor(135 + t * (33 - 135));
          return `rgb(${r}, ${g}, ${b})`;
        },
        coolwarm: () => {
          if (t < 0.5) {
            const r = Math.floor(59 + t * 2 * (221 - 59));
            const g = Math.floor(76 + t * 2 * (221 - 76));
            const b = Math.floor(192 + t * 2 * (221 - 192));
            return `rgb(${r}, ${g}, ${b})`;
          } else {
            const r = Math.floor(221 + (t - 0.5) * 2 * (180 - 221));
            const g = Math.floor(221 - (t - 0.5) * 2 * (221 - 4));
            const b = Math.floor(221 - (t - 0.5) * 2 * (221 - 38));
            return `rgb(${r}, ${g}, ${b})`;
          }
        },
      };

      return schemes[colorScheme]();
    },
    [colorScheme]
  );

  // Render
  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { matrix, rowLabels, colLabels, title } = data;
    const nRows = matrix.length;
    const nCols = matrix[0]?.length || 0;

    if (nRows === 0 || nCols === 0) return;

    // Dimensions
    const padding = { top: 60, right: 80, bottom: 30, left: 100 };
    const cellWidth = (width - padding.left - padding.right) / nCols;
    const cellHeight = (height - padding.top - padding.bottom) / nRows;

    // Find min/max
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (let i = 0; i < nRows; i++) {
      for (let j = 0; j < nCols; j++) {
        minVal = Math.min(minVal, matrix[i][j]);
        maxVal = Math.max(maxVal, matrix[i][j]);
      }
    }

    // For symmetric colorscale (correlation)
    if (symmetric) {
      const absMax = Math.max(Math.abs(minVal), Math.abs(maxVal));
      minVal = -absMax;
      maxVal = absMax;
    }

    // Clear
    ctx.fillStyle = "#0a0a0f";
    ctx.fillRect(0, 0, width, height);

    // Title
    if (title) {
      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 14px Inter, system-ui";
      ctx.textAlign = "center";
      ctx.fillText(title, width / 2, 25);
    }

    // Draw cells
    for (let i = 0; i < nRows; i++) {
      for (let j = 0; j < nCols; j++) {
        const value = matrix[i][j];
        const x = padding.left + j * cellWidth;
        const y = padding.top + i * cellHeight;

        // Cell background
        ctx.fillStyle = getColor(value, minVal, maxVal);
        ctx.fillRect(x, y, cellWidth - 1, cellHeight - 1);

        // Highlight hovered cell
        if (hoveredCell?.row === i && hoveredCell?.col === j) {
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 2;
          ctx.strokeRect(x, y, cellWidth - 1, cellHeight - 1);
        }

        // Cell value
        if (showValues && cellWidth > 30 && cellHeight > 20) {
          // Text color based on background brightness
          const brightness = Math.abs(value - (maxVal + minVal) / 2) / ((maxVal - minVal) / 2);
          ctx.fillStyle = brightness > 0.5 ? "#ffffff" : "#000000";
          ctx.font = "10px Inter, system-ui";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(
            value.toFixed(2),
            x + cellWidth / 2,
            y + cellHeight / 2
          );
        }
      }
    }

    // Row labels
    ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
    ctx.font = "11px Inter, system-ui";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";

    for (let i = 0; i < nRows; i++) {
      const y = padding.top + i * cellHeight + cellHeight / 2;
      const label = rowLabels[i] || `Row ${i}`;
      // Truncate long labels
      const truncated =
        label.length > 12 ? label.substring(0, 10) + "..." : label;
      ctx.fillText(truncated, padding.left - 8, y);
    }

    // Column labels (rotated)
    ctx.textAlign = "left";
    for (let j = 0; j < nCols; j++) {
      const x = padding.left + j * cellWidth + cellWidth / 2;
      const y = padding.top - 8;
      const label = colLabels[j] || `Col ${j}`;
      const truncated =
        label.length > 12 ? label.substring(0, 10) + "..." : label;

      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(-Math.PI / 4);
      ctx.fillText(truncated, 0, 0);
      ctx.restore();
    }

    // Color legend
    const legendWidth = 20;
    const legendHeight = height - padding.top - padding.bottom;
    const legendX = width - padding.right + 20;
    const legendY = padding.top;

    // Legend gradient
    for (let i = 0; i < legendHeight; i++) {
      const t = i / legendHeight;
      const value = maxVal - t * (maxVal - minVal);
      ctx.fillStyle = getColor(value, minVal, maxVal);
      ctx.fillRect(legendX, legendY + i, legendWidth, 1);
    }

    // Legend border
    ctx.strokeStyle = "rgba(255, 255, 255, 0.3)";
    ctx.lineWidth = 1;
    ctx.strokeRect(legendX, legendY, legendWidth, legendHeight);

    // Legend labels
    ctx.fillStyle = "rgba(255, 255, 255, 0.7)";
    ctx.font = "10px Inter, system-ui";
    ctx.textAlign = "left";
    ctx.fillText(maxVal.toFixed(2), legendX + legendWidth + 5, legendY + 5);
    ctx.fillText(
      ((maxVal + minVal) / 2).toFixed(2),
      legendX + legendWidth + 5,
      legendY + legendHeight / 2
    );
    ctx.fillText(
      minVal.toFixed(2),
      legendX + legendWidth + 5,
      legendY + legendHeight - 5
    );
  }, [data, width, height, getColor, showValues, symmetric, hoveredCell]);

  // Mouse handling
  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      const { matrix, rowLabels, colLabels } = data;
      const nRows = matrix.length;
      const nCols = matrix[0]?.length || 0;

      const padding = { top: 60, right: 80, bottom: 30, left: 100 };
      const cellWidth = (width - padding.left - padding.right) / nCols;
      const cellHeight = (height - padding.top - padding.bottom) / nRows;

      const col = Math.floor((x - padding.left) / cellWidth);
      const row = Math.floor((y - padding.top) / cellHeight);

      if (row >= 0 && row < nRows && col >= 0 && col < nCols) {
        setHoveredCell({ row, col, value: matrix[row][col] });
      } else {
        setHoveredCell(null);
      }
    },
    [data, width, height]
  );

  const handleMouseLeave = () => {
    setHoveredCell(null);
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
      {hoveredCell && (
        <div className="absolute top-4 right-24 bg-black/90 border border-white/20 rounded-lg px-3 py-2 text-sm">
          <div className="text-white/60 text-xs mb-1">
            {data.rowLabels[hoveredCell.row]} × {data.colLabels[hoveredCell.col]}
          </div>
          <div className="text-white font-medium">
            {hoveredCell.value.toFixed(4)}
          </div>
        </div>
      )}
    </div>
  );
}

// Demo data generator
export function generateDemoCorrelationMatrix(n: number = 10): HeatmapData {
  const assets = [
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "AGG",
    "TLT",
    "GLD",
    "USO",
    "VNQ",
  ].slice(0, n);

  const matrix: number[][] = [];

  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < n; j++) {
      if (i === j) {
        row.push(1.0);
      } else if (j < i) {
        row.push(matrix[j][i]);
      } else {
        // Generate realistic correlations
        let corr = 0.3 + Math.random() * 0.4;
        // Stocks tend to be more correlated
        if (i < 5 && j < 5) corr += 0.2;
        // Bonds negative correlation with stocks
        if ((i < 5 && j >= 5 && j <= 6) || (j < 5 && i >= 5 && i <= 6)) {
          corr = -0.3 + Math.random() * 0.3;
        }
        row.push(Math.max(-1, Math.min(1, corr)));
      }
    }
    matrix.push(row);
  }

  return {
    matrix,
    rowLabels: assets,
    colLabels: assets,
    title: "Asset Correlation Matrix",
  };
}
