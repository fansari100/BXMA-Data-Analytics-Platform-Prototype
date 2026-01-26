"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  TimeSeriesChart,
  generateDemoTimeSeries,
} from "@/components/charts/time-series-chart";

interface RegimeState {
  name: string;
  probability: number;
  color: string;
  description: string;
}

interface RegimeHistory {
  timestamp: number;
  regime: string;
}

const regimes: RegimeState[] = [
  {
    name: "Bull",
    probability: 0,
    color: "#22c55e",
    description: "Low volatility, positive trend",
  },
  {
    name: "Bear",
    probability: 0,
    color: "#ef4444",
    description: "High volatility, negative trend",
  },
  {
    name: "High Vol",
    probability: 0,
    color: "#f59e0b",
    description: "Elevated volatility, uncertain direction",
  },
  {
    name: "Low Vol",
    probability: 0,
    color: "#3b82f6",
    description: "Low volatility, range-bound",
  },
];

export function RegimePage() {
  const [currentRegime, setCurrentRegime] = useState("Bull");
  const [regimeProbs, setRegimeProbs] = useState<RegimeState[]>(regimes);
  const [temperature, setTemperature] = useState(1.0);
  const [entropy, setEntropy] = useState(0.5);
  const [vix, setVix] = useState(15.0);
  const [chartData, setChartData] = useState(generateDemoTimeSeries(365, 1));

  useEffect(() => {
    // Simulate regime detection
    const updateRegime = () => {
      const newVix = 12 + Math.random() * 20;
      setVix(newVix);

      // Temperature scales with VIX
      const newTemp = newVix / 15.0;
      setTemperature(newTemp);

      // Generate regime probabilities
      const probs = [
        Math.random(),
        Math.random(),
        Math.random() * 0.5,
        Math.random() * 0.5,
      ];
      const sum = probs.reduce((a, b) => a + b, 0);
      const normalized = probs.map((p) => p / sum);

      // Apply temperature softening
      const softened = normalized.map((p) => Math.pow(p, 1 / newTemp));
      const softenedSum = softened.reduce((a, b) => a + b, 0);
      const finalProbs = softened.map((p) => p / softenedSum);

      // Update regime states
      const updated = regimes.map((r, i) => ({
        ...r,
        probability: finalProbs[i],
      }));
      setRegimeProbs(updated);

      // Find dominant regime
      const maxIdx = finalProbs.indexOf(Math.max(...finalProbs));
      setCurrentRegime(regimes[maxIdx].name);

      // Calculate entropy
      const ent = -finalProbs.reduce(
        (sum, p) => sum + (p > 0 ? p * Math.log(p) : 0),
        0
      );
      setEntropy(ent / Math.log(4)); // Normalize to 0-1
    };

    updateRegime();
    const interval = setInterval(updateRegime, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Regime Detection</h1>
        <p className="text-muted-foreground">
          HMM-based regime identification with thermodynamic uncertainty
          quantification
        </p>
      </div>

      {/* Current state cards */}
      <div className="grid grid-cols-4 gap-4">
        <Card className="bg-card/50 border-border/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Current Regime
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div
              className="text-2xl font-bold"
              style={{
                color: regimeProbs.find((r) => r.name === currentRegime)?.color,
              }}
            >
              {currentRegime}
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              {regimeProbs.find((r) => r.name === currentRegime)?.description}
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card/50 border-border/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              VIX Level
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div
              className={`text-2xl font-bold ${
                vix > 25 ? "text-red-500" : vix > 18 ? "text-amber-500" : "text-green-500"
              }`}
            >
              {vix.toFixed(1)}
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              Market fear gauge
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card/50 border-border/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Temperature
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-purple-500">
              {temperature.toFixed(2)}
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              Boltzmann scaling factor
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card/50 border-border/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Entropy
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div
              className={`text-2xl font-bold ${
                entropy > 0.7 ? "text-amber-500" : "text-blue-500"
              }`}
            >
              {entropy.toFixed(2)}
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              {entropy > 0.7 ? "High uncertainty" : "Confident classification"}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Regime probability distribution */}
      <Card className="bg-card/30 border-border/50">
        <CardHeader>
          <CardTitle>Regime Probability Distribution</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {regimeProbs.map((regime) => (
              <div key={regime.name} className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: regime.color }}
                    />
                    <span className="font-medium">{regime.name}</span>
                    <span className="text-xs text-muted-foreground">
                      {regime.description}
                    </span>
                  </div>
                  <span className="font-mono text-sm">
                    {(regime.probability * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="h-2 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                      width: `${regime.probability * 100}%`,
                      backgroundColor: regime.color,
                    }}
                  />
                </div>
              </div>
            ))}
          </div>

          {/* Thermodynamic note */}
          <div className="mt-6 p-4 bg-muted/30 rounded-lg">
            <h4 className="font-medium mb-2">Thermodynamic Sampling</h4>
            <p className="text-sm text-muted-foreground">
              Probabilities are computed using a Boltzmann distribution with
              temperature T = {temperature.toFixed(2)} (scaled by VIX/15). Higher
              temperatures flatten the distribution, reflecting increased
              uncertainty during volatile periods.
            </p>
            <div className="mt-2 font-mono text-xs text-muted-foreground">
              P(regime) ∝ exp(-E(regime) / kT)
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Market data chart */}
      <Card className="bg-card/30 border-border/50">
        <CardHeader>
          <CardTitle>Market Performance with Regime Overlay</CardTitle>
        </CardHeader>
        <CardContent>
          <TimeSeriesChart
            data={chartData}
            width={1000}
            height={350}
            title="SPY Price History"
            yAxisLabel="Price ($)"
            colors={["#6366f1"]}
            showLegend={false}
          />
        </CardContent>
      </Card>

      {/* Transition matrix */}
      <div className="grid grid-cols-2 gap-6">
        <Card className="bg-card/30 border-border/50">
          <CardHeader>
            <CardTitle>Transition Probabilities</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left py-2 px-2">From / To</th>
                    {regimeProbs.map((r) => (
                      <th
                        key={r.name}
                        className="text-center py-2 px-2"
                        style={{ color: r.color }}
                      >
                        {r.name}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {regimeProbs.map((fromRegime, i) => (
                    <tr key={fromRegime.name} className="border-b border-border/50">
                      <td
                        className="py-2 px-2 font-medium"
                        style={{ color: fromRegime.color }}
                      >
                        {fromRegime.name}
                      </td>
                      {regimeProbs.map((toRegime, j) => (
                        <td key={toRegime.name} className="text-center py-2 px-2">
                          {i === j
                            ? (0.7 + Math.random() * 0.2).toFixed(2)
                            : (0.3 / 3 + Math.random() * 0.1).toFixed(2)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card/30 border-border/50">
          <CardHeader>
            <CardTitle>Expected Regime Duration</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {regimeProbs.map((regime) => {
                const duration = 5 + Math.random() * 30;
                return (
                  <div
                    key={regime.name}
                    className="flex items-center justify-between"
                  >
                    <div className="flex items-center gap-2">
                      <div
                        className="w-2 h-2 rounded-full"
                        style={{ backgroundColor: regime.color }}
                      />
                      <span>{regime.name}</span>
                    </div>
                    <div className="text-right">
                      <span className="font-mono">{duration.toFixed(1)}</span>
                      <span className="text-muted-foreground text-xs ml-1">
                        days
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="mt-4 text-xs text-muted-foreground">
              Expected duration = 1 / (1 - P(stay in regime))
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
