"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  TrendingUp,
  TrendingDown,
  Shield,
  Target,
  Activity,
  AlertTriangle,
  DollarSign,
  Percent,
  BarChart2,
  PieChart,
  ArrowUpRight,
  ArrowDownRight,
} from "lucide-react";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart as RechartsPie,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { formatCompactNumber, formatPercent, formatNumber, cn, getChangeColor } from "@/lib/utils";

// Generate mock data
const generateNavHistory = () => {
  const data = [];
  let nav = 85000000000; // $85B
  const startDate = new Date("2024-01-01");
  
  for (let i = 0; i < 365; i++) {
    const date = new Date(startDate);
    date.setDate(date.getDate() + i);
    nav *= 1 + (Math.random() - 0.48) * 0.015;
    data.push({
      date: date.toISOString().split("T")[0],
      nav,
      benchmark: 85000000000 * (1 + 0.0003 * i + Math.random() * 0.005),
    });
  }
  return data;
};

const generateRiskMetrics = () => ({
  var95: 0.0142,
  cvar95: 0.0189,
  volatility: 0.128,
  sharpeRatio: 1.34,
  maxDrawdown: -0.0423,
  beta: 0.92,
  trackingError: 0.018,
  informationRatio: 0.89,
});

const generateFactorExposures = () => [
  { factor: "Market", exposure: 0.95, contribution: 0.68 },
  { factor: "Size", exposure: 0.12, contribution: 0.04 },
  { factor: "Value", exposure: -0.08, contribution: -0.02 },
  { factor: "Momentum", exposure: 0.23, contribution: 0.12 },
  { factor: "Quality", exposure: 0.18, contribution: 0.08 },
  { factor: "Low Vol", exposure: -0.15, contribution: -0.05 },
];

const generateAssetAllocation = () => [
  { name: "US Equities", value: 35, color: "#00d4ff" },
  { name: "Intl Equities", value: 20, color: "#00ff88" },
  { name: "Fixed Income", value: 25, color: "#ffaa00" },
  { name: "Alternatives", value: 12, color: "#a855f7" },
  { name: "Real Estate", value: 5, color: "#ff6b6b" },
  { name: "Cash", value: 3, color: "#64748b" },
];

const generateRecentAlerts = () => [
  { id: 1, type: "warning", message: "VaR breach in EM Equity sleeve", time: "2m ago" },
  { id: 2, type: "info", message: "Rebalancing triggered for Core Bond", time: "15m ago" },
  { id: 3, type: "success", message: "HRP optimization completed", time: "1h ago" },
  { id: 4, type: "warning", message: "Liquidity constraint binding", time: "2h ago" },
];

export function DashboardPage() {
  const [navHistory] = useState(generateNavHistory);
  const [riskMetrics] = useState(generateRiskMetrics);
  const [factorExposures] = useState(generateFactorExposures);
  const [assetAllocation] = useState(generateAssetAllocation);
  const [alerts] = useState(generateRecentAlerts);

  const currentNav = navHistory[navHistory.length - 1].nav;
  const previousNav = navHistory[navHistory.length - 2].nav;
  const dailyReturn = (currentNav - previousNav) / previousNav;
  const ytdReturn = (currentNav - navHistory[0].nav) / navHistory[0].nav;

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
      {/* Header metrics */}
      <motion.div variants={itemVariants} className="grid grid-cols-4 gap-4">
        <MetricCard
          title="Total NAV"
          value={formatCompactNumber(currentNav)}
          change={dailyReturn}
          icon={DollarSign}
        />
        <MetricCard
          title="Daily Return"
          value={formatPercent(dailyReturn)}
          change={dailyReturn}
          icon={dailyReturn >= 0 ? TrendingUp : TrendingDown}
        />
        <MetricCard
          title="YTD Return"
          value={formatPercent(ytdReturn)}
          change={ytdReturn}
          icon={ytdReturn >= 0 ? TrendingUp : TrendingDown}
        />
        <MetricCard
          title="Sharpe Ratio"
          value={riskMetrics.sharpeRatio.toFixed(2)}
          change={0.05}
          icon={Target}
        />
      </motion.div>

      {/* Main charts row */}
      <div className="grid grid-cols-3 gap-6">
        {/* NAV chart */}
        <motion.div variants={itemVariants} className="col-span-2 glass-card">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-lg font-semibold text-dark-100">Portfolio Performance</h3>
              <p className="text-sm text-dark-400">NAV vs Benchmark</p>
            </div>
            <div className="flex items-center gap-4 text-sm">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-accent-cyan rounded-full" />
                <span className="text-dark-400">Portfolio</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-dark-500 rounded-full" />
                <span className="text-dark-400">Benchmark</span>
              </div>
            </div>
          </div>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={navHistory.slice(-90)}>
                <defs>
                  <linearGradient id="navGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#00d4ff" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#00d4ff" stopOpacity={0} />
                  </linearGradient>
                </defs>
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
                  tickFormatter={(value) => `$${(value / 1e9).toFixed(0)}B`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "rgba(39, 40, 45, 0.95)",
                    border: "1px solid rgba(100, 100, 120, 0.3)",
                    borderRadius: "8px",
                  }}
                  formatter={(value: number) => [`$${(value / 1e9).toFixed(2)}B`, ""]}
                  labelFormatter={(label) => new Date(label).toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" })}
                />
                <Area
                  type="monotone"
                  dataKey="nav"
                  stroke="#00d4ff"
                  strokeWidth={2}
                  fill="url(#navGradient)"
                  name="Portfolio NAV"
                />
                <Line
                  type="monotone"
                  dataKey="benchmark"
                  stroke="#6a6d79"
                  strokeWidth={1.5}
                  strokeDasharray="4 4"
                  dot={false}
                  name="Benchmark"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Asset allocation */}
        <motion.div variants={itemVariants} className="glass-card">
          <h3 className="text-lg font-semibold text-dark-100 mb-6">Asset Allocation</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <RechartsPie>
                <Pie
                  data={assetAllocation}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={90}
                  paddingAngle={2}
                  dataKey="value"
                >
                  {assetAllocation.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: "rgba(39, 40, 45, 0.95)",
                    border: "1px solid rgba(100, 100, 120, 0.3)",
                    borderRadius: "8px",
                  }}
                  formatter={(value: number) => [`${value}%`, ""]}
                />
              </RechartsPie>
            </ResponsiveContainer>
          </div>
          <div className="grid grid-cols-2 gap-2 mt-4">
            {assetAllocation.map((item) => (
              <div key={item.name} className="flex items-center gap-2 text-xs">
                <div className="w-2 h-2 rounded-full" style={{ backgroundColor: item.color }} />
                <span className="text-dark-400 truncate">{item.name}</span>
                <span className="text-dark-200 font-medium ml-auto">{item.value}%</span>
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Risk metrics and factor exposures */}
      <div className="grid grid-cols-3 gap-6">
        {/* Risk metrics */}
        <motion.div variants={itemVariants} className="glass-card">
          <div className="flex items-center gap-2 mb-6">
            <Shield className="w-5 h-5 text-accent-cyan" />
            <h3 className="text-lg font-semibold text-dark-100">Risk Metrics</h3>
          </div>
          <div className="space-y-4">
            <RiskMetricRow label="VaR (95%)" value={formatPercent(riskMetrics.var95)} status="normal" />
            <RiskMetricRow label="CVaR (95%)" value={formatPercent(riskMetrics.cvar95)} status="normal" />
            <RiskMetricRow label="Volatility" value={formatPercent(riskMetrics.volatility)} status="normal" />
            <RiskMetricRow label="Max Drawdown" value={formatPercent(riskMetrics.maxDrawdown)} status="warning" />
            <RiskMetricRow label="Beta" value={riskMetrics.beta.toFixed(2)} status="normal" />
            <RiskMetricRow label="Tracking Error" value={formatPercent(riskMetrics.trackingError)} status="normal" />
            <RiskMetricRow label="Info Ratio" value={riskMetrics.informationRatio.toFixed(2)} status="good" />
          </div>
        </motion.div>

        {/* Factor exposures */}
        <motion.div variants={itemVariants} className="glass-card">
          <div className="flex items-center gap-2 mb-6">
            <BarChart2 className="w-5 h-5 text-accent-emerald" />
            <h3 className="text-lg font-semibold text-dark-100">Factor Exposures</h3>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={factorExposures} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis type="number" stroke="#6a6d79" fontSize={11} />
                <YAxis type="category" dataKey="factor" stroke="#6a6d79" fontSize={11} width={70} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "rgba(39, 40, 45, 0.95)",
                    border: "1px solid rgba(100, 100, 120, 0.3)",
                    borderRadius: "8px",
                  }}
                />
                <Bar
                  dataKey="exposure"
                  fill="#00d4ff"
                  radius={[0, 4, 4, 0]}
                  name="Exposure"
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Recent alerts */}
        <motion.div variants={itemVariants} className="glass-card">
          <div className="flex items-center gap-2 mb-6">
            <Activity className="w-5 h-5 text-accent-amber" />
            <h3 className="text-lg font-semibold text-dark-100">Recent Activity</h3>
          </div>
          <div className="space-y-3">
            {alerts.map((alert) => (
              <div
                key={alert.id}
                className={cn(
                  "flex items-start gap-3 p-3 rounded-lg",
                  alert.type === "warning" && "bg-accent-amber/10",
                  alert.type === "info" && "bg-accent-cyan/10",
                  alert.type === "success" && "bg-accent-emerald/10"
                )}
              >
                <AlertTriangle
                  className={cn(
                    "w-4 h-4 mt-0.5",
                    alert.type === "warning" && "text-accent-amber",
                    alert.type === "info" && "text-accent-cyan",
                    alert.type === "success" && "text-accent-emerald"
                  )}
                />
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-dark-200">{alert.message}</p>
                  <p className="text-xs text-dark-500 mt-1">{alert.time}</p>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </motion.div>
  );
}

// Sub-components
function MetricCard({
  title,
  value,
  change,
  icon: Icon,
}: {
  title: string;
  value: string;
  change: number;
  icon: React.ElementType;
}) {
  return (
    <div className="metric-card">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-dark-400 mb-1">{title}</p>
          <p className="text-2xl font-semibold text-dark-100">{value}</p>
        </div>
        <div className="p-2 bg-dark-800/50 rounded-lg">
          <Icon className="w-5 h-5 text-accent-cyan" />
        </div>
      </div>
      <div className={cn("flex items-center gap-1 mt-3 text-sm", getChangeColor(change))}>
        {change >= 0 ? (
          <ArrowUpRight className="w-4 h-4" />
        ) : (
          <ArrowDownRight className="w-4 h-4" />
        )}
        <span>{change >= 0 ? "+" : ""}{(change * 100).toFixed(2)}%</span>
        <span className="text-dark-500 ml-1">vs prev</span>
      </div>
    </div>
  );
}

function RiskMetricRow({
  label,
  value,
  status,
}: {
  label: string;
  value: string;
  status: "good" | "normal" | "warning" | "danger";
}) {
  const statusColors = {
    good: "bg-accent-emerald",
    normal: "bg-accent-cyan",
    warning: "bg-accent-amber",
    danger: "bg-accent-rose",
  };

  return (
    <div className="flex items-center justify-between">
      <span className="text-sm text-dark-400">{label}</span>
      <div className="flex items-center gap-2">
        <span className="text-sm font-mono font-medium text-dark-100">{value}</span>
        <div className={cn("w-2 h-2 rounded-full", statusColors[status])} />
      </div>
    </div>
  );
}
