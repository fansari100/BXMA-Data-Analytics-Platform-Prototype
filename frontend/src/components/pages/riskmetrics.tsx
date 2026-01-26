"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  HeatmapChart,
  generateDemoCorrelationMatrix,
} from "@/components/charts/heatmap";

interface FactorExposure {
  factor: string;
  exposure: number;
  contribution: number;
}

interface RiskDecomposition {
  factorRisk: number;
  specificRisk: number;
  totalRisk: number;
}

const factorModels = [
  { id: "BARRA_USE4", name: "Barra US Equity Model 4", provider: "MSCI/Barra" },
  { id: "BARRA_GEM3", name: "Barra Global Equity Model 3", provider: "MSCI/Barra" },
  { id: "BARRA_CNE5", name: "Barra China Equity Model 5", provider: "MSCI/Barra" },
  { id: "AXIOMA_WW21", name: "Axioma Worldwide Equity", provider: "Qontigo" },
  { id: "AXIOMA_US4", name: "Axioma US Equity Model 4", provider: "Qontigo" },
];

const styleFactors = [
  "Momentum",
  "Volatility",
  "Size",
  "Value",
  "Growth",
  "Quality",
  "Leverage",
  "Liquidity",
];

export function RiskMetricsPage() {
  const [selectedModel, setSelectedModel] = useState("BARRA_USE4");
  const [decayFactor, setDecayFactor] = useState(0.94);
  const [correlationData, setCorrelationData] = useState(
    generateDemoCorrelationMatrix(10)
  );
  const [factorExposures, setFactorExposures] = useState<FactorExposure[]>([]);
  const [riskDecomp, setRiskDecomp] = useState<RiskDecomposition>({
    factorRisk: 0,
    specificRisk: 0,
    totalRisk: 0,
  });
  const [varMetrics, setVarMetrics] = useState({
    var95: 0,
    var99: 0,
    cvar95: 0,
    cvar99: 0,
  });

  useEffect(() => {
    // Simulate RiskMetrics data
    const exposures: FactorExposure[] = styleFactors.map((factor) => ({
      factor,
      exposure: (Math.random() - 0.5) * 2,
      contribution: Math.random() * 0.3,
    }));
    setFactorExposures(exposures);

    const factorRisk = 0.12 + Math.random() * 0.05;
    const specificRisk = 0.03 + Math.random() * 0.02;
    setRiskDecomp({
      factorRisk,
      specificRisk,
      totalRisk: Math.sqrt(factorRisk ** 2 + specificRisk ** 2),
    });

    setVarMetrics({
      var95: 0.015 + Math.random() * 0.005,
      var99: 0.022 + Math.random() * 0.008,
      cvar95: 0.02 + Math.random() * 0.006,
      cvar99: 0.03 + Math.random() * 0.01,
    });
  }, [selectedModel]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">
            RiskMetrics Integration
          </h1>
          <p className="text-muted-foreground">
            MSCI/Barra factor models and RiskMetrics EWMA methodology
          </p>
        </div>

        {/* Model selector */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">Factor Model:</span>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="bg-background border border-input rounded-md px-3 py-2 text-sm"
            >
              {factorModels.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">λ (decay):</span>
            <input
              type="number"
              value={decayFactor}
              onChange={(e) => setDecayFactor(parseFloat(e.target.value))}
              step={0.01}
              min={0.9}
              max={0.99}
              className="bg-background border border-input rounded-md px-3 py-2 text-sm w-20"
            />
          </div>
        </div>
      </div>

      {/* Methodology banner */}
      <Card className="bg-gradient-to-r from-blue-500/10 to-cyan-500/10 border-blue-500/20">
        <CardContent className="py-4">
          <div className="flex items-start gap-4">
            <div className="p-2 bg-blue-500/20 rounded-lg">
              <svg
                className="w-6 h-6 text-blue-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
                />
              </svg>
            </div>
            <div>
              <h3 className="font-semibold text-blue-400">
                RiskMetrics™ Methodology
              </h3>
              <p className="text-sm text-muted-foreground mt-1">
                Industry-standard exponentially weighted moving average (EWMA) covariance
                estimation with decay factor λ = {decayFactor}. Based on the J.P. Morgan
                RiskMetrics Technical Document (1996).
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* VaR metrics */}
      <div className="grid grid-cols-4 gap-4">
        <Card className="bg-card/50 border-border/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              VaR (95%)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-amber-500">
              {(varMetrics.var95 * 100).toFixed(2)}%
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              1-day parametric
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card/50 border-border/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              VaR (99%)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-500">
              {(varMetrics.var99 * 100).toFixed(2)}%
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              1-day parametric
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card/50 border-border/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              CVaR (95%)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-orange-500">
              {(varMetrics.cvar95 * 100).toFixed(2)}%
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              Expected shortfall
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card/50 border-border/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              CVaR (99%)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-rose-500">
              {(varMetrics.cvar99 * 100).toFixed(2)}%
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              Expected shortfall
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Risk decomposition and factor exposures */}
      <div className="grid grid-cols-2 gap-6">
        {/* Risk decomposition */}
        <Card className="bg-card/30 border-border/50">
          <CardHeader>
            <CardTitle>Factor Risk Decomposition</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              {/* Total risk */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium">Total Active Risk</span>
                  <span className="text-xl font-bold">
                    {(riskDecomp.totalRisk * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="h-4 bg-muted rounded-full overflow-hidden flex">
                  <div
                    className="h-full bg-blue-500"
                    style={{
                      width: `${
                        (riskDecomp.factorRisk ** 2 / riskDecomp.totalRisk ** 2) *
                        100
                      }%`,
                    }}
                  />
                  <div
                    className="h-full bg-purple-500"
                    style={{
                      width: `${
                        (riskDecomp.specificRisk ** 2 / riskDecomp.totalRisk ** 2) *
                        100
                      }%`,
                    }}
                  />
                </div>
              </div>

              {/* Breakdown */}
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 bg-blue-500/10 rounded-lg">
                  <div className="text-sm text-muted-foreground">Factor Risk</div>
                  <div className="text-2xl font-bold text-blue-400">
                    {(riskDecomp.factorRisk * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">
                    {(
                      (riskDecomp.factorRisk ** 2 / riskDecomp.totalRisk ** 2) *
                      100
                    ).toFixed(0)}
                    % of variance
                  </div>
                </div>
                <div className="p-4 bg-purple-500/10 rounded-lg">
                  <div className="text-sm text-muted-foreground">Specific Risk</div>
                  <div className="text-2xl font-bold text-purple-400">
                    {(riskDecomp.specificRisk * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">
                    {(
                      (riskDecomp.specificRisk ** 2 / riskDecomp.totalRisk ** 2) *
                      100
                    ).toFixed(0)}
                    % of variance
                  </div>
                </div>
              </div>

              {/* Model info */}
              <div className="p-3 bg-muted/30 rounded-lg text-sm">
                <div className="font-medium mb-1">
                  {factorModels.find((m) => m.id === selectedModel)?.name}
                </div>
                <div className="text-muted-foreground">
                  Provider:{" "}
                  {factorModels.find((m) => m.id === selectedModel)?.provider}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Factor exposures */}
        <Card className="bg-card/30 border-border/50">
          <CardHeader>
            <CardTitle>Style Factor Exposures</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {factorExposures.map((exp) => (
                <div key={exp.factor} className="flex items-center gap-4">
                  <div className="w-24 text-sm text-muted-foreground">
                    {exp.factor}
                  </div>
                  <div className="flex-1">
                    <div className="h-2 bg-muted rounded-full overflow-hidden relative">
                      <div
                        className={`absolute top-0 h-full rounded-full ${
                          exp.exposure >= 0 ? "bg-green-500" : "bg-red-500"
                        }`}
                        style={{
                          left: exp.exposure >= 0 ? "50%" : `${50 + exp.exposure * 25}%`,
                          width: `${Math.abs(exp.exposure) * 25}%`,
                        }}
                      />
                      <div className="absolute top-0 left-1/2 w-px h-full bg-white/20" />
                    </div>
                  </div>
                  <div
                    className={`w-16 text-right font-mono text-sm ${
                      exp.exposure >= 0 ? "text-green-400" : "text-red-400"
                    }`}
                  >
                    {exp.exposure >= 0 ? "+" : ""}
                    {exp.exposure.toFixed(2)}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Correlation matrix */}
      <Card className="bg-card/30 border-border/50">
        <CardHeader>
          <CardTitle>EWMA Correlation Matrix (λ = {decayFactor})</CardTitle>
        </CardHeader>
        <CardContent className="flex justify-center">
          <HeatmapChart
            data={{
              ...correlationData,
              title: `RiskMetrics EWMA Correlation (λ=${decayFactor})`,
            }}
            width={700}
            height={550}
            colorScheme="rdbu"
            showValues={true}
            symmetric={true}
          />
        </CardContent>
      </Card>

      {/* Stress scenarios */}
      <Card className="bg-card/30 border-border/50">
        <CardHeader>
          <CardTitle>RiskMetrics Stress Scenarios</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4">
            {[
              {
                name: "2008 Financial Crisis",
                type: "Historical",
                equityShock: -50,
                creditShock: 300,
              },
              {
                name: "COVID-19 Crash",
                type: "Historical",
                equityShock: -34,
                creditShock: 250,
              },
              {
                name: "2022 Rate Shock",
                type: "Historical",
                equityShock: -25,
                creditShock: 150,
              },
              {
                name: "EM Crisis",
                type: "Hypothetical",
                equityShock: -40,
                creditShock: 400,
              },
              {
                name: "Credit Crisis",
                type: "Hypothetical",
                equityShock: -20,
                creditShock: 600,
              },
              {
                name: "Stagflation",
                type: "Hypothetical",
                equityShock: -30,
                creditShock: 200,
              },
            ].map((scenario) => (
              <div
                key={scenario.name}
                className="p-4 bg-muted/30 rounded-lg border border-border/50"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium">{scenario.name}</span>
                  <span
                    className={`text-xs px-2 py-0.5 rounded ${
                      scenario.type === "Historical"
                        ? "bg-blue-500/20 text-blue-400"
                        : "bg-amber-500/20 text-amber-400"
                    }`}
                  >
                    {scenario.type}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div>
                    <span className="text-muted-foreground">Equity:</span>
                    <span className="ml-2 text-red-400">
                      {scenario.equityShock}%
                    </span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Credit:</span>
                    <span className="ml-2 text-amber-400">
                      +{scenario.creditShock}bp
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
