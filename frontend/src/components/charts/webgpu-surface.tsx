"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { cn } from "@/lib/utils";

interface VolatilitySurfaceData {
  strikes: number[];
  expiries: number[];
  impliedVols: number[][];
  spotPrice: number;
  underlying: string;
}

interface WebGPUSurfaceProps {
  data: VolatilitySurfaceData | null;
  width?: number;
  height?: number;
  className?: string;
  colorScheme?: "viridis" | "plasma" | "turbo" | "rdylgn";
}

export function WebGPUSurface({
  data,
  width = 800,
  height = 600,
  className,
  colorScheme = "viridis",
}: WebGPUSurfaceProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [rotation, setRotation] = useState({ x: -30, y: 45 });
  const [isDragging, setIsDragging] = useState(false);
  const [lastMouse, setLastMouse] = useState({ x: 0, y: 0 });
  const [hoveredPoint, setHoveredPoint] = useState<{
    strike: number;
    expiry: number;
    vol: number;
  } | null>(null);

  // Color scheme functions
  const colorSchemes = {
    viridis: (t: number) => {
      const r = Math.floor(68 + t * (253 - 68));
      const g = Math.floor(1 + t * (231 - 1));
      const b = Math.floor(84 + t * (37 - 84));
      return `rgb(${r}, ${g}, ${b})`;
    },
    plasma: (t: number) => {
      const r = Math.floor(13 + t * (240 - 13));
      const g = Math.floor(8 + t * (249 - 8));
      const b = Math.floor(135 + t * (33 - 135));
      return `rgb(${r}, ${g}, ${b})`;
    },
    turbo: (t: number) => {
      const r = Math.floor(48 + t * (122));
      const g = Math.floor(18 + t * (237 - 18));
      const b = Math.floor(59 + t * (180 - 59));
      return `rgb(${r}, ${g}, ${b})`;
    },
    rdylgn: (t: number) => {
      if (t < 0.5) {
        const r = Math.floor(215 + t * 2 * (255 - 215));
        const g = Math.floor(48 + t * 2 * (255 - 48));
        const b = Math.floor(39 + t * 2 * (191 - 39));
        return `rgb(${r}, ${g}, ${b})`;
      } else {
        const r = Math.floor(255 + (t - 0.5) * 2 * (26 - 255));
        const g = Math.floor(255 + (t - 0.5) * 2 * (152 - 255));
        const b = Math.floor(191 + (t - 0.5) * 2 * (80 - 191));
        return `rgb(${r}, ${g}, ${b})`;
      }
    },
  };

  const getColor = colorSchemes[colorScheme];

  // 3D projection
  const project = useCallback(
    (x: number, y: number, z: number) => {
      const radX = (rotation.x * Math.PI) / 180;
      const radY = (rotation.y * Math.PI) / 180;

      // Rotate around Y axis
      const x1 = x * Math.cos(radY) - z * Math.sin(radY);
      const z1 = x * Math.sin(radY) + z * Math.cos(radY);

      // Rotate around X axis
      const y1 = y * Math.cos(radX) - z1 * Math.sin(radX);
      const z2 = y * Math.sin(radX) + z1 * Math.cos(radX);

      // Perspective projection
      const scale = 400 / (400 + z2);
      const projX = x1 * scale + width / 2;
      const projY = -y1 * scale + height / 2;

      return { x: projX, y: projY, depth: z2 };
    },
    [rotation, width, height]
  );

  // Render the surface
  const renderSurface = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = "#0a0a0f";
    ctx.fillRect(0, 0, width, height);

    const { strikes, expiries, impliedVols, spotPrice } = data;

    // Normalize data for rendering
    const minVol = Math.min(...impliedVols.flat());
    const maxVol = Math.max(...impliedVols.flat());
    const volRange = maxVol - minVol || 1;

    // Scale factors
    const xScale = 200 / (strikes.length - 1 || 1);
    const yScale = 200 / (expiries.length - 1 || 1);
    const zScale = 150 / volRange;

    // Build mesh points
    const points: {
      x: number;
      y: number;
      depth: number;
      vol: number;
      strike: number;
      expiry: number;
      row: number;
      col: number;
    }[] = [];

    for (let i = 0; i < strikes.length; i++) {
      for (let j = 0; j < expiries.length; j++) {
        const vol = impliedVols[i][j];
        const x = (i - strikes.length / 2) * xScale;
        const y = ((vol - minVol) / volRange) * zScale * 2 - zScale;
        const z = (j - expiries.length / 2) * yScale;

        const projected = project(x, y, z);
        points.push({
          ...projected,
          vol,
          strike: strikes[i],
          expiry: expiries[j],
          row: i,
          col: j,
        });
      }
    }

    // Sort by depth for painter's algorithm
    const quads: {
      points: typeof points;
      avgDepth: number;
      avgVol: number;
    }[] = [];

    for (let i = 0; i < strikes.length - 1; i++) {
      for (let j = 0; j < expiries.length - 1; j++) {
        const idx = i * expiries.length + j;
        const p1 = points[idx];
        const p2 = points[idx + 1];
        const p3 = points[idx + expiries.length + 1];
        const p4 = points[idx + expiries.length];

        quads.push({
          points: [p1, p2, p3, p4],
          avgDepth: (p1.depth + p2.depth + p3.depth + p4.depth) / 4,
          avgVol: (p1.vol + p2.vol + p3.vol + p4.vol) / 4,
        });
      }
    }

    // Sort back to front
    quads.sort((a, b) => b.avgDepth - a.avgDepth);

    // Draw quads
    for (const quad of quads) {
      const [p1, p2, p3, p4] = quad.points;
      const t = (quad.avgVol - minVol) / volRange;

      ctx.fillStyle = getColor(t);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
      ctx.lineWidth = 0.5;

      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.lineTo(p3.x, p3.y);
      ctx.lineTo(p4.x, p4.y);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = "rgba(255, 255, 255, 0.5)";
    ctx.lineWidth = 2;

    // X axis (Strike)
    const xStart = project(-120, -100, 0);
    const xEnd = project(120, -100, 0);
    ctx.beginPath();
    ctx.moveTo(xStart.x, xStart.y);
    ctx.lineTo(xEnd.x, xEnd.y);
    ctx.stroke();

    // Y axis (Vol)
    const yStart = project(-120, -100, 0);
    const yEnd = project(-120, 100, 0);
    ctx.beginPath();
    ctx.moveTo(yStart.x, yStart.y);
    ctx.lineTo(yEnd.x, yEnd.y);
    ctx.stroke();

    // Z axis (Expiry)
    const zStart = project(-120, -100, 0);
    const zEnd = project(-120, -100, 120);
    ctx.beginPath();
    ctx.moveTo(zStart.x, zStart.y);
    ctx.lineTo(zEnd.x, zEnd.y);
    ctx.stroke();

    // Labels
    ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
    ctx.font = "12px Inter, system-ui";

    const xLabel = project(0, -120, 0);
    ctx.fillText("Strike", xLabel.x - 20, xLabel.y);

    const yLabel = project(-140, 0, 0);
    ctx.fillText("Vol", yLabel.x, yLabel.y);

    const zLabel = project(-120, -120, 60);
    ctx.fillText("Expiry", zLabel.x, zLabel.y);

    // Title
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 14px Inter, system-ui";
    ctx.fillText(`${data.underlying} Volatility Surface`, 20, 30);

    ctx.font = "12px Inter, system-ui";
    ctx.fillStyle = "rgba(255, 255, 255, 0.7)";
    ctx.fillText(`Spot: $${spotPrice.toFixed(2)}`, 20, 50);
    ctx.fillText(
      `Vol Range: ${(minVol * 100).toFixed(1)}% - ${(maxVol * 100).toFixed(1)}%`,
      20,
      70
    );

    // Hovered point tooltip
    if (hoveredPoint) {
      ctx.fillStyle = "rgba(0, 0, 0, 0.8)";
      ctx.fillRect(width - 180, 20, 160, 70);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.3)";
      ctx.strokeRect(width - 180, 20, 160, 70);

      ctx.fillStyle = "#ffffff";
      ctx.font = "12px Inter, system-ui";
      ctx.fillText(`Strike: $${hoveredPoint.strike.toFixed(2)}`, width - 170, 40);
      ctx.fillText(`Expiry: ${hoveredPoint.expiry}d`, width - 170, 58);
      ctx.fillText(
        `IV: ${(hoveredPoint.vol * 100).toFixed(2)}%`,
        width - 170,
        76
      );
    }
  }, [data, width, height, project, getColor, hoveredPoint]);

  // Handle mouse events for rotation
  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setLastMouse({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      const dx = e.clientX - lastMouse.x;
      const dy = e.clientY - lastMouse.y;

      setRotation((prev) => ({
        x: Math.max(-90, Math.min(90, prev.x + dy * 0.5)),
        y: prev.y + dx * 0.5,
      }));

      setLastMouse({ x: e.clientX, y: e.clientY });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Render on data or rotation change
  useEffect(() => {
    renderSurface();
  }, [renderSurface]);

  return (
    <div className={cn("relative", className)}>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="rounded-lg cursor-grab active:cursor-grabbing"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      />
      <div className="absolute bottom-4 left-4 text-xs text-white/60">
        Drag to rotate • Scroll to zoom
      </div>
    </div>
  );
}

// Demo data generator
export function generateDemoVolSurface(): VolatilitySurfaceData {
  const spotPrice = 100;
  const strikes = Array.from({ length: 21 }, (_, i) => spotPrice * (0.8 + i * 0.02));
  const expiries = [7, 14, 30, 60, 90, 120, 180, 365];

  const impliedVols: number[][] = [];

  for (const strike of strikes) {
    const row: number[] = [];
    for (const expiry of expiries) {
      // Volatility smile + term structure
      const moneyness = Math.log(strike / spotPrice);
      const smile = 0.2 * Math.exp(-moneyness * moneyness * 10);
      const termStructure = 0.15 + 0.1 * Math.sqrt(expiry / 365);
      const vol = termStructure + smile + Math.random() * 0.01;
      row.push(vol);
    }
    impliedVols.push(row);
  }

  return {
    strikes,
    expiries,
    impliedVols,
    spotPrice,
    underlying: "SPY",
  };
}
