"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import {
  Shield,
  AlertTriangle,
  TrendingDown,
  Activity,
  Layers,
  Calculator,
  RefreshCw,
} from "lucide-react";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  ComposedChart,
} from "recharts";
import { cn, formatPercent, formatNumber } from "@/lib/utils";

// Generate mock VaR history
const generateVaRHistory = () => {
  const data = [];
  const startDate = new Date("2024-01-01");
  
  for (let i = 0; i < 252; i++) {
    const date = new Date(startDate);
    date.setDate(date.getDate() + i);
    const var95 = 0.012 + Math.random() * 0.008;
    const var99 = var95 * 1.3 + Math.random() * 0.004;
    const actualReturn = (Math.random() - 0.48) * 0.025;
    data.push({
      date: date.toISOString().split("T")[0],
      var95: -var95,
      var99: -var99,
      actualReturn,
      breached: actualReturn < -var95,
    });
  }
  return data;
};

// Generate component VaR
const generateComponentVaR = () => [
  { asset: "US Large Cap", var: 0.0045, contribution: 32 },
  { asset: "Int'l Developed", var: 0.0038, contribution: 24 },
  { asset: "Emerging Markets", var: 0.0052, contribution: 18 },
  { asset: "IG Credit", var: 0.0012, contribution: 8 },
  { asset: "HY Credit", var: 0.0025, contribution: 10 },
  { asset: "Government Bonds", var: 0.0008, contribution: 5 },
  { asset: "Alternatives", var: 0.0015, contribution: 3 },
];

// Generate return distribution
const generateReturnDistribution = () => {
  const data = [];
  for (let i = -5; i <= 5; i += 0.25) {
    const frequency = Math.exp(-(i * i) / 2) * 100 + Math.random() * 10;
    data.push({
      return: i,
      frequency: frequency,
      var95: i === -1.75 ? frequency : null,
      var99: i === -2.5 ? frequency : null,
    });
  }
  return data;
};

export function RiskPage() {
  const [varHistory] = useState(generateVaRHistory);
  const [componentVaR] = useState(generateComponentVaR);
  const [distribution] = useState(generateReturnDistribution);
  const [selectedMethod, setSelectedMethod] = useState<"parametric" | "historical" | "monte_carlo">("parametric");
  const [isCalculating, setIsCalculating] = useState(false);

  const totalVaR95 = 0.0142;
  const totalVaR99 = 0.0198;
  const cvar95 = 0.0189;
  const breachCount = varHistory.filter(d => d.breached).length;
  const breachRate = breachCount / varHistory.length;

  const handleCalculate = () => {
    setIsCalculating(true);
    setTimeout(() => setIsCalculating(false), 1500);
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.1 },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 },
  };

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="space-y-6"
    >
      {/* Header */}
      <motion.div variants={itemVariants} className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-display font-bold text-dark-100">Risk Analytics</h2>
          <p className="text-dark-400 mt-1">Value-at-Risk and tail risk analysis</p>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={selectedMethod}
            onChange={(e) => setSelectedMethod(e.target.value as any)}
            className="px-4 py-2 bg-dark-800 border border-dark-700 rounded-lg text-dark-100 text-sm focus:outline-none focus:border-accent-cyan/50"
          >
            <option value="parametric">Parametric VaR</option>
            <option value="historical">Historical VaR</option>
            <option value="monte_carlo">Monte Carlo VaR</option>
          </select>
          <button
            onClick={handleCalculate}
            className="btn-primary flex items-center gap-2"
          >
            <RefreshCw className={cn("w-4 h-4", isCalculating && "animate-spin")} />
            Recalculate
          </button>
        </div>
      </motion.div>

      {/* Key metrics */}
      <motion.div variants={itemVariants} className="grid grid-cols-4 gap-4">
        <div className="metric-card">
          <div className="flex items-center gap-2 mb-2">
            <Shield className="w-4 h-4 text-accent-cyan" />
            <span className="text-sm text-dark-400">VaR (95%, 1D)</span>
          </div>
          <p className="text-2xl font-semibold text-dark-100">{formatPercent(totalVaR95)}</p>
          <p className="text-xs text-dark-500 mt-1">$1.28B at risk</p>
        </div>
        <div className="metric-card">
          <div className="flex items-center gap-2 mb-2">
            <Shield className="w-4 h-4 text-accent-violet" />
            <span className="text-sm text-dark-400">VaR (99%, 1D)</span>
          </div>
          <p className="text-2xl font-semibold text-dark-100">{formatPercent(totalVaR99)}</p>
          <p className="text-xs text-dark-500 mt-1">$1.78B at risk</p>
        </div>
        <div className="metric-card">
          <div className="flex items-center gap-2 mb-2">
            <TrendingDown className="w-4 h-4 text-accent-rose" />
            <span className="text-sm text-dark-400">CVaR (95%)</span>
          </div>
          <p className="text-2xl font-semibold text-dark-100">{formatPercent(cvar95)}</p>
          <p className="text-xs text-dark-500 mt-1">Expected Shortfall</p>
        </div>
        <div className="metric-card">
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle className="w-4 h-4 text-accent-amber" />
            <span className="text-sm text-dark-400">Breach Rate</span>
          </div>
          <p className="text-2xl font-semibold text-dark-100">{formatPercent(breachRate)}</p>
          <p className="text-xs text-dark-500 mt-1">{breachCount} breaches YTD</p>
        </div>
      </motion.div>

      {/* VaR history and distribution */}
      <div className="grid grid-cols-2 gap-6">
        <motion.div variants={itemVariants} className="glass-card">
          <h3 className="text-lg font-semibold text-dark-100 mb-4">VaR Backtesting</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={varHistory.slice(-90)}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis
                  dataKey="date"
                  stroke="#6a6d79"
                  fontSize={11}
                  tickFormatter={(value) => new Date(value).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
                />
                <YAxis
                  stroke="#6a6d79"
                  fontSize={11}
                  tickFormatter={(value) => `${(value * 100).toFixed(1)}%`}
                  domain={[-0.03, 0.02]}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "rgba(39, 40, 45, 0.95)",
                    border: "1px solid rgba(100, 100, 120, 0.3)",
                    borderRadius: "8px",
                  }}
                  formatter={(value: number) => [`${(value * 100).toFixed(2)}%`, ""]}
                />
                <ReferenceLine y={0} stroke="rgba(255,255,255,0.2)" />
                <Area
                  type="monotone"
                  dataKey="var95"
                  stroke="none"
                  fill="rgba(255, 107, 107, 0.2)"
                  name="95% VaR Bound"
                />
                <Line
                  type="monotone"
                  dataKey="var95"
                  stroke="#ff6b6b"
                  strokeWidth={1.5}
                  strokeDasharray="4 4"
                  dot={false}
                  name="VaR 95%"
                />
                <Bar
                  dataKey="actualReturn"
                  fill={(entry: any) => entry.breached ? "#ff6b6b" : "#00d4ff"}
                  name="Actual Return"
                  opacity={0.8}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        <motion.div variants={itemVariants} className="glass-card">
          <h3 className="text-lg font-semibold text-dark-100 mb-4">Return Distribution</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={distribution}>
                <defs>
                  <linearGradient id="distGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#00d4ff" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#00d4ff" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis
                  dataKey="return"
                  stroke="#6a6d79"
                  fontSize={11}
                  tickFormatter={(value) => `${value}σ`}
                />
                <YAxis stroke="#6a6d79" fontSize={11} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "rgba(39, 40, 45, 0.95)",
                    border: "1px solid rgba(100, 100, 120, 0.3)",
                    borderRadius: "8px",
                  }}
                />
                <ReferenceLine x={-1.65} stroke="#ff6b6b" strokeDasharray="4 4" label={{ value: "95% VaR", fill: "#ff6b6b", fontSize: 10 }} />
                <ReferenceLine x={-2.33} stroke="#a855f7" strokeDasharray="4 4" label={{ value: "99% VaR", fill: "#a855f7", fontSize: 10 }} />
                <Area
                  type="monotone"
                  dataKey="frequency"
                  stroke="#00d4ff"
                  strokeWidth={2}
                  fill="url(#distGradient)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
      </div>

      {/* Component VaR */}
      <motion.div variants={itemVariants} className="glass-card">
        <div className="flex items-center gap-2 mb-6">
          <Layers className="w-5 h-5 text-accent-cyan" />
          <h3 className="text-lg font-semibold text-dark-100">Component VaR Decomposition</h3>
        </div>
        <div className="grid grid-cols-2 gap-8">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={componentVaR} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis type="number" stroke="#6a6d79" fontSize={11} tickFormatter={(v) => `${(v * 100).toFixed(2)}%`} />
                <YAxis type="category" dataKey="asset" stroke="#6a6d79" fontSize={11} width={100} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "rgba(39, 40, 45, 0.95)",
                    border: "1px solid rgba(100, 100, 120, 0.3)",
                    borderRadius: "8px",
                  }}
                  formatter={(value: number) => [`${(value * 100).toFixed(3)}%`, "Component VaR"]}
                />
                <Bar dataKey="var" fill="#00d4ff" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={componentVaR} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis type="number" stroke="#6a6d79" fontSize={11} tickFormatter={(v) => `${v}%`} />
                <YAxis type="category" dataKey="asset" stroke="#6a6d79" fontSize={11} width={100} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "rgba(39, 40, 45, 0.95)",
                    border: "1px solid rgba(100, 100, 120, 0.3)",
                    borderRadius: "8px",
                  }}
                  formatter={(value: number) => [`${value}%`, "Contribution"]}
                />
                <Bar dataKey="contribution" fill="#00ff88" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}
