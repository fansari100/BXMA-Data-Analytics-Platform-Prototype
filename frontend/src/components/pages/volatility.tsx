"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  WebGPUSurface,
  generateDemoVolSurface,
} from "@/components/charts/webgpu-surface";

interface VolatilitySurfaceData {
  strikes: number[];
  expiries: number[];
  impliedVols: number[][];
  spotPrice: number;
  underlying: string;
}

const underlyings = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "TSLA"];

export function VolatilityPage() {
  const [selectedUnderlying, setSelectedUnderlying] = useState("SPY");
  const [surfaceData, setSurfaceData] = useState<VolatilitySurfaceData | null>(
    null
  );
  const [metrics, setMetrics] = useState({
    atmVol: 0,
    skew25d: 0,
    termSlope: 0,
    volOfVol: 0,
  });

  useEffect(() => {
    // Generate demo data for selected underlying
    const data = generateDemoVolSurface();
    data.underlying = selectedUnderlying;
    data.spotPrice = 100 + Math.random() * 400;
    setSurfaceData(data);

    // Calculate metrics
    const midStrikeIdx = Math.floor(data.strikes.length / 2);
    const atmVol = data.impliedVols[midStrikeIdx][3]; // 60-day ATM
    const otm25d = data.impliedVols[Math.floor(data.strikes.length * 0.25)][3];
    const skew25d = otm25d - atmVol;
    const termSlope =
      data.impliedVols[midStrikeIdx][7] - data.impliedVols[midStrikeIdx][0];

    setMetrics({
      atmVol: atmVol * 100,
      skew25d: skew25d * 100,
      termSlope: termSlope * 100,
      volOfVol: 2.5 + Math.random() * 2,
    });
  }, [selectedUnderlying]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">
            Volatility Surface
          </h1>
          <p className="text-muted-foreground">
            Interactive 3D volatility surface analysis
          </p>
        </div>

        {/* Underlying selector */}
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">Underlying:</span>
          <select
            value={selectedUnderlying}
            onChange={(e) => setSelectedUnderlying(e.target.value)}
            className="bg-background border border-input rounded-md px-3 py-2 text-sm"
          >
            {underlyings.map((u) => (
              <option key={u} value={u}>
                {u}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Metrics cards */}
      <div className="grid grid-cols-4 gap-4">
        <Card className="bg-card/50 border-border/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              ATM Vol (60d)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metrics.atmVol.toFixed(1)}%</div>
            <div className="text-xs text-muted-foreground mt-1">
              At-the-money implied volatility
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card/50 border-border/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              25Δ Skew
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div
              className={`text-2xl font-bold ${
                metrics.skew25d > 0 ? "text-red-500" : "text-green-500"
              }`}
            >
              {metrics.skew25d > 0 ? "+" : ""}
              {metrics.skew25d.toFixed(2)}%
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              OTM puts vs ATM vol
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card/50 border-border/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Term Structure
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div
              className={`text-2xl font-bold ${
                metrics.termSlope > 0 ? "text-amber-500" : "text-blue-500"
              }`}
            >
              {metrics.termSlope > 0 ? "+" : ""}
              {metrics.termSlope.toFixed(2)}%
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              1Y vs 1W ATM slope
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card/50 border-border/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Vol of Vol
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {metrics.volOfVol.toFixed(1)}%
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              Volatility surface instability
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main surface visualization */}
      <Card className="bg-card/30 border-border/50">
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>3D Volatility Surface</span>
            <div className="flex items-center gap-4 text-sm font-normal">
              <span className="text-muted-foreground">
                Spot: ${surfaceData?.spotPrice.toFixed(2)}
              </span>
              <span className="text-muted-foreground">
                Strikes: {surfaceData?.strikes.length}
              </span>
              <span className="text-muted-foreground">
                Expiries: {surfaceData?.expiries.length}
              </span>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent className="flex justify-center">
          <WebGPUSurface
            data={surfaceData}
            width={900}
            height={600}
            colorScheme="viridis"
          />
        </CardContent>
      </Card>

      {/* Additional analysis */}
      <div className="grid grid-cols-2 gap-6">
        <Card className="bg-card/30 border-border/50">
          <CardHeader>
            <CardTitle>Volatility Smile (60d)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[200px] flex items-center justify-center text-muted-foreground">
              Interactive smile chart would render here
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card/30 border-border/50">
          <CardHeader>
            <CardTitle>Term Structure (ATM)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[200px] flex items-center justify-center text-muted-foreground">
              Term structure chart would render here
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
