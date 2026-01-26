"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import {
  Target,
  Settings,
  Play,
  Download,
  Layers,
  TrendingUp,
  Shield,
  BarChart2,
} from "lucide-react";
import {
  ScatterChart,
  Scatter,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from "recharts";
import { cn, formatPercent, formatNumber } from "@/lib/utils";

// Generate efficient frontier data
const generateEfficientFrontier = () => {
  const data = [];
  for (let risk = 0.05; risk <= 0.25; risk += 0.01) {
    const baseReturn = 0.02 + risk * 0.6 - Math.pow(risk - 0.15, 2) * 0.5;
    data.push({
      risk: risk * 100,
      return: (baseReturn + Math.random() * 0.005) * 100,
      sharpe: baseReturn / risk,
    });
  }
  return data.sort((a, b) => a.risk - b.risk);
};

// Generate optimal weights
const generateOptimalWeights = (method: string) => {
  const assets = [
    "US Large Cap",
    "US Small Cap",
    "Int'l Developed",
    "Emerging Markets",
    "US Govt Bonds",
    "Corp IG",
    "Corp HY",
    "TIPS",
    "Real Estate",
    "Commodities",
  ];

  const baseWeights = {
    hrp: [0.18, 0.08, 0.15, 0.06, 0.20, 0.12, 0.05, 0.08, 0.05, 0.03],
    risk_parity: [0.10, 0.05, 0.08, 0.04, 0.30, 0.18, 0.08, 0.10, 0.04, 0.03],
    max_sharpe: [0.25, 0.12, 0.18, 0.08, 0.12, 0.10, 0.05, 0.05, 0.03, 0.02],
    min_variance: [0.08, 0.03, 0.06, 0.02, 0.35, 0.22, 0.06, 0.12, 0.04, 0.02],
  };

  const weights = baseWeights[method as keyof typeof baseWeights] || baseWeights.hrp;
  
  return assets.map((name, i) => ({
    name,
    weight: weights[i] * 100,
    riskContribution: weights[i] * (0.8 + Math.random() * 0.4) * 100,
  }));
};

// Generate portfolio stats
const generatePortfolioStats = (method: string) => {
  const stats = {
    hrp: { return: 0.082, risk: 0.112, sharpe: 0.73 },
    risk_parity: { return: 0.065, risk: 0.085, sharpe: 0.76 },
    max_sharpe: { return: 0.098, risk: 0.145, sharpe: 0.68 },
    min_variance: { return: 0.055, risk: 0.072, sharpe: 0.76 },
  };
  return stats[method as keyof typeof stats] || stats.hrp;
};

export function OptimizationPage() {
  const [efficientFrontier] = useState(generateEfficientFrontier);
  const [selectedMethod, setSelectedMethod] = useState("hrp");
  const [optimalWeights, setOptimalWeights] = useState(() => generateOptimalWeights("hrp"));
  const [portfolioStats, setPortfolioStats] = useState(() => generatePortfolioStats("hrp"));
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [constraints, setConstraints] = useState({
    maxWeight: 30,
    minWeight: 2,
    turnover: 15,
  });

  const handleOptimize = () => {
    setIsOptimizing(true);
    setTimeout(() => {
      setOptimalWeights(generateOptimalWeights(selectedMethod));
      setPortfolioStats(generatePortfolioStats(selectedMethod));
      setIsOptimizing(false);
    }, 2000);
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

  const methods = [
    { id: "hrp", name: "Hierarchical Risk Parity", description: "ML-based clustering approach" },
    { id: "risk_parity", name: "Risk Parity", description: "Equal risk contribution" },
    { id: "max_sharpe", name: "Max Sharpe", description: "Tangency portfolio" },
    { id: "min_variance", name: "Min Variance", description: "Global minimum variance" },
  ];

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
          <h2 className="text-2xl font-display font-bold text-dark-100">Portfolio Optimization</h2>
          <p className="text-dark-400 mt-1">Advanced optimization with multiple methodologies</p>
        </div>
        <div className="flex items-center gap-3">
          <button className="btn-secondary flex items-center gap-2">
            <Download className="w-4 h-4" />
            Export
          </button>
          <button
            onClick={handleOptimize}
            disabled={isOptimizing}
            className="btn-primary flex items-center gap-2"
          >
            <Play className={cn("w-4 h-4", isOptimizing && "animate-pulse")} />
            {isOptimizing ? "Optimizing..." : "Run Optimization"}
          </button>
        </div>
      </motion.div>

      {/* Method selection and constraints */}
      <div className="grid grid-cols-4 gap-6">
        {/* Methods */}
        <motion.div variants={itemVariants} className="col-span-3 glass-card">
          <h3 className="text-lg font-semibold text-dark-100 mb-4">Optimization Method</h3>
          <div className="grid grid-cols-4 gap-3">
            {methods.map((method) => (
              <button
                key={method.id}
                onClick={() => setSelectedMethod(method.id)}
                className={cn(
                  "p-4 rounded-lg border transition-all duration-200 text-left",
                  selectedMethod === method.id
                    ? "bg-accent-cyan/10 border-accent-cyan/50"
                    : "bg-dark-800/30 border-dark-700 hover:border-dark-600"
                )}
              >
                <p className={cn(
                  "font-medium text-sm",
                  selectedMethod === method.id ? "text-accent-cyan" : "text-dark-200"
                )}>
                  {method.name}
                </p>
                <p className="text-xs text-dark-500 mt-1">{method.description}</p>
              </button>
            ))}
          </div>
        </motion.div>

        {/* Constraints */}
        <motion.div variants={itemVariants} className="glass-card">
          <div className="flex items-center gap-2 mb-4">
            <Settings className="w-4 h-4 text-accent-cyan" />
            <h3 className="text-lg font-semibold text-dark-100">Constraints</h3>
          </div>
          <div className="space-y-4">
            <div>
              <label className="text-xs text-dark-400 block mb-1">Max Weight</label>
              <input
                type="number"
                value={constraints.maxWeight}
                onChange={(e) => setConstraints({ ...constraints, maxWeight: parseInt(e.target.value) })}
                className="input-field text-sm"
              />
            </div>
            <div>
              <label className="text-xs text-dark-400 block mb-1">Min Weight</label>
              <input
                type="number"
                value={constraints.minWeight}
                onChange={(e) => setConstraints({ ...constraints, minWeight: parseInt(e.target.value) })}
                className="input-field text-sm"
              />
            </div>
            <div>
              <label className="text-xs text-dark-400 block mb-1">Max Turnover</label>
              <input
                type="number"
                value={constraints.turnover}
                onChange={(e) => setConstraints({ ...constraints, turnover: parseInt(e.target.value) })}
                className="input-field text-sm"
              />
            </div>
          </div>
        </motion.div>
      </div>

      {/* Results */}
      <div className="grid grid-cols-3 gap-6">
        {/* Portfolio stats */}
        <motion.div variants={itemVariants} className="glass-card">
          <h3 className="text-lg font-semibold text-dark-100 mb-4">Optimal Portfolio</h3>
          <div className="space-y-6">
            <div className="flex items-center justify-between p-4 bg-dark-800/30 rounded-lg">
              <div className="flex items-center gap-3">
                <TrendingUp className="w-5 h-5 text-accent-emerald" />
                <span className="text-dark-400">Expected Return</span>
              </div>
              <span className="text-xl font-semibold text-dark-100">
                {formatPercent(portfolioStats.return)}
              </span>
            </div>
            <div className="flex items-center justify-between p-4 bg-dark-800/30 rounded-lg">
              <div className="flex items-center gap-3">
                <Shield className="w-5 h-5 text-accent-rose" />
                <span className="text-dark-400">Expected Risk</span>
              </div>
              <span className="text-xl font-semibold text-dark-100">
                {formatPercent(portfolioStats.risk)}
              </span>
            </div>
            <div className="flex items-center justify-between p-4 bg-dark-800/30 rounded-lg">
              <div className="flex items-center gap-3">
                <Target className="w-5 h-5 text-accent-cyan" />
                <span className="text-dark-400">Sharpe Ratio</span>
              </div>
              <span className="text-xl font-semibold text-dark-100">
                {portfolioStats.sharpe.toFixed(2)}
              </span>
            </div>
          </div>
        </motion.div>

        {/* Efficient frontier */}
        <motion.div variants={itemVariants} className="col-span-2 glass-card">
          <h3 className="text-lg font-semibold text-dark-100 mb-4">Efficient Frontier</h3>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis
                  type="number"
                  dataKey="risk"
                  name="Risk"
                  stroke="#6a6d79"
                  fontSize={11}
                  tickFormatter={(v) => `${v.toFixed(0)}%`}
                  domain={[5, 25]}
                  label={{ value: "Volatility (%)", position: "insideBottom", offset: -5, fill: "#6a6d79", fontSize: 11 }}
                />
                <YAxis
                  type="number"
                  dataKey="return"
                  name="Return"
                  stroke="#6a6d79"
                  fontSize={11}
                  tickFormatter={(v) => `${v.toFixed(0)}%`}
                  domain={[4, 12]}
                  label={{ value: "Return (%)", angle: -90, position: "insideLeft", fill: "#6a6d79", fontSize: 11 }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "rgba(39, 40, 45, 0.95)",
                    border: "1px solid rgba(100, 100, 120, 0.3)",
                    borderRadius: "8px",
                  }}
                  formatter={(value: number, name: string) => [`${value.toFixed(2)}%`, name]}
                />
                <Scatter name="Efficient Frontier" data={efficientFrontier} fill="#00d4ff" line={{ stroke: "#00d4ff", strokeWidth: 2 }}>
                  {efficientFrontier.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.sharpe > 0.7 ? "#00ff88" : "#00d4ff"} />
                  ))}
                </Scatter>
                {/* Current portfolio marker */}
                <Scatter
                  name="Optimal"
                  data={[{ risk: portfolioStats.risk * 100, return: portfolioStats.return * 100 }]}
                  fill="#ffaa00"
                  shape="star"
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
      </div>

      {/* Weights */}
      <motion.div variants={itemVariants} className="glass-card">
        <div className="flex items-center gap-2 mb-6">
          <Layers className="w-5 h-5 text-accent-cyan" />
          <h3 className="text-lg font-semibold text-dark-100">Optimal Allocation</h3>
        </div>
        <div className="grid grid-cols-2 gap-8">
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={optimalWeights} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis type="number" stroke="#6a6d79" fontSize={11} tickFormatter={(v) => `${v}%`} />
                <YAxis type="category" dataKey="name" stroke="#6a6d79" fontSize={11} width={100} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "rgba(39, 40, 45, 0.95)",
                    border: "1px solid rgba(100, 100, 120, 0.3)",
                    borderRadius: "8px",
                  }}
                  formatter={(value: number) => [`${value.toFixed(1)}%`, "Weight"]}
                />
                <Bar dataKey="weight" fill="#00d4ff" radius={[0, 4, 4, 0]} name="Target Weight" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={optimalWeights} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis type="number" stroke="#6a6d79" fontSize={11} tickFormatter={(v) => `${v}%`} />
                <YAxis type="category" dataKey="name" stroke="#6a6d79" fontSize={11} width={100} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "rgba(39, 40, 45, 0.95)",
                    border: "1px solid rgba(100, 100, 120, 0.3)",
                    borderRadius: "8px",
                  }}
                  formatter={(value: number) => [`${value.toFixed(1)}%`, "Risk Contribution"]}
                />
                <Bar dataKey="riskContribution" fill="#00ff88" radius={[0, 4, 4, 0]} name="Risk Contribution" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}
