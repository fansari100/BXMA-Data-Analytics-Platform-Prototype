"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import {
  BarChart3,
  Calendar,
  ArrowRight,
  TrendingUp,
  TrendingDown,
  Layers,
  Filter,
} from "lucide-react";
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  ComposedChart,
  Area,
} from "recharts";
import { cn, formatPercent, formatBps } from "@/lib/utils";

// Generate attribution data
const generateAttributionData = () => ({
  summary: {
    portfolioReturn: 0.0823,
    benchmarkReturn: 0.0712,
    activeReturn: 0.0111,
    allocationEffect: 0.0042,
    selectionEffect: 0.0058,
    interactionEffect: 0.0011,
  },
  segments: [
    { name: "US Large Cap", allocation: 0.0015, selection: 0.0022, interaction: 0.0003, total: 0.0040 },
    { name: "US Small Cap", allocation: 0.0008, selection: 0.0012, interaction: 0.0002, total: 0.0022 },
    { name: "Int'l Developed", allocation: 0.0012, selection: -0.0005, interaction: 0.0001, total: 0.0008 },
    { name: "Emerging Markets", allocation: 0.0005, selection: 0.0018, interaction: 0.0003, total: 0.0026 },
    { name: "Fixed Income", allocation: -0.0003, selection: 0.0008, interaction: 0.0001, total: 0.0006 },
    { name: "Alternatives", allocation: 0.0005, selection: 0.0003, interaction: 0.0001, total: 0.0009 },
  ],
});

// Generate time series attribution
const generateTimeSeriesAttribution = () => {
  const data = [];
  const startDate = new Date("2024-01-01");
  let cumAllocation = 0;
  let cumSelection = 0;
  
  for (let i = 0; i < 52; i++) {
    const date = new Date(startDate);
    date.setDate(date.getDate() + i * 7);
    
    const allocation = (Math.random() - 0.45) * 0.003;
    const selection = (Math.random() - 0.45) * 0.004;
    cumAllocation += allocation;
    cumSelection += selection;
    
    data.push({
      date: date.toISOString().split("T")[0],
      allocation,
      selection,
      cumAllocation,
      cumSelection,
      cumTotal: cumAllocation + cumSelection,
    });
  }
  return data;
};

export function AttributionPage() {
  const [attribution] = useState(generateAttributionData);
  const [timeSeriesData] = useState(generateTimeSeriesAttribution);
  const [selectedPeriod, setSelectedPeriod] = useState("ytd");
  const [view, setView] = useState<"summary" | "detailed">("summary");

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

  const { summary, segments } = attribution;

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
          <h2 className="text-2xl font-display font-bold text-dark-100">Performance Attribution</h2>
          <p className="text-dark-400 mt-1">Brinson-Fachler decomposition of active returns</p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center bg-dark-800/50 rounded-lg p-1">
            {["1M", "3M", "YTD", "1Y"].map((period) => (
              <button
                key={period}
                onClick={() => setSelectedPeriod(period.toLowerCase())}
                className={cn(
                  "px-3 py-1.5 text-sm rounded-md transition-all",
                  selectedPeriod === period.toLowerCase()
                    ? "bg-accent-cyan text-dark-950"
                    : "text-dark-400 hover:text-dark-100"
                )}
              >
                {period}
              </button>
            ))}
          </div>
          <div className="flex items-center bg-dark-800/50 rounded-lg p-1">
            <button
              onClick={() => setView("summary")}
              className={cn(
                "px-3 py-1.5 text-sm rounded-md transition-all",
                view === "summary"
                  ? "bg-accent-cyan text-dark-950"
                  : "text-dark-400 hover:text-dark-100"
              )}
            >
              Summary
            </button>
            <button
              onClick={() => setView("detailed")}
              className={cn(
                "px-3 py-1.5 text-sm rounded-md transition-all",
                view === "detailed"
                  ? "bg-accent-cyan text-dark-950"
                  : "text-dark-400 hover:text-dark-100"
              )}
            >
              Detailed
            </button>
          </div>
        </div>
      </motion.div>

      {/* Summary metrics */}
      <motion.div variants={itemVariants} className="grid grid-cols-6 gap-4">
        <div className="metric-card">
          <p className="text-sm text-dark-400 mb-1">Portfolio</p>
          <p className={cn("text-xl font-semibold", summary.portfolioReturn >= 0 ? "text-accent-emerald" : "text-accent-rose")}>
            {formatPercent(summary.portfolioReturn)}
          </p>
        </div>
        <div className="metric-card">
          <p className="text-sm text-dark-400 mb-1">Benchmark</p>
          <p className="text-xl font-semibold text-dark-100">{formatPercent(summary.benchmarkReturn)}</p>
        </div>
        <div className="metric-card border-l-2 border-accent-cyan">
          <p className="text-sm text-dark-400 mb-1">Active Return</p>
          <p className={cn("text-xl font-semibold", summary.activeReturn >= 0 ? "text-accent-emerald" : "text-accent-rose")}>
            {formatPercent(summary.activeReturn)}
          </p>
        </div>
        <div className="metric-card">
          <p className="text-sm text-dark-400 mb-1">Allocation</p>
          <p className={cn("text-xl font-semibold", summary.allocationEffect >= 0 ? "text-accent-emerald" : "text-accent-rose")}>
            {formatBps(summary.allocationEffect)}
          </p>
        </div>
        <div className="metric-card">
          <p className="text-sm text-dark-400 mb-1">Selection</p>
          <p className={cn("text-xl font-semibold", summary.selectionEffect >= 0 ? "text-accent-emerald" : "text-accent-rose")}>
            {formatBps(summary.selectionEffect)}
          </p>
        </div>
        <div className="metric-card">
          <p className="text-sm text-dark-400 mb-1">Interaction</p>
          <p className={cn("text-xl font-semibold", summary.interactionEffect >= 0 ? "text-accent-emerald" : "text-accent-rose")}>
            {formatBps(summary.interactionEffect)}
          </p>
        </div>
      </motion.div>

      {/* Attribution waterfall and time series */}
      <div className="grid grid-cols-2 gap-6">
        {/* Waterfall */}
        <motion.div variants={itemVariants} className="glass-card">
          <h3 className="text-lg font-semibold text-dark-100 mb-4">Active Return Decomposition</h3>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={[
                  { name: "Portfolio", value: summary.portfolioReturn * 100, fill: "#00d4ff" },
                  { name: "Benchmark", value: -summary.benchmarkReturn * 100, fill: "#6a6d79" },
                  { name: "Active", value: summary.activeReturn * 100, fill: "#00ff88" },
                ]}
                layout="vertical"
              >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis type="number" stroke="#6a6d79" fontSize={11} tickFormatter={(v) => `${v.toFixed(1)}%`} />
                <YAxis type="category" dataKey="name" stroke="#6a6d79" fontSize={11} width={80} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "rgba(39, 40, 45, 0.95)",
                    border: "1px solid rgba(100, 100, 120, 0.3)",
                    borderRadius: "8px",
                  }}
                  formatter={(value: number) => [`${Math.abs(value).toFixed(2)}%`, ""]}
                />
                <ReferenceLine x={0} stroke="rgba(255,255,255,0.2)" />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {[0, 1, 2].map((index) => (
                    <motion.rect
                      key={index}
                      initial={{ width: 0 }}
                      animate={{ width: "100%" }}
                      transition={{ duration: 0.5, delay: index * 0.1 }}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Time series */}
        <motion.div variants={itemVariants} className="glass-card">
          <h3 className="text-lg font-semibold text-dark-100 mb-4">Cumulative Attribution</h3>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={timeSeriesData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis
                  dataKey="date"
                  stroke="#6a6d79"
                  fontSize={11}
                  tickFormatter={(value) => new Date(value).toLocaleDateString("en-US", { month: "short" })}
                />
                <YAxis
                  stroke="#6a6d79"
                  fontSize={11}
                  tickFormatter={(value) => `${(value * 100).toFixed(1)}%`}
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
                  dataKey="cumAllocation"
                  stroke="#00d4ff"
                  fill="rgba(0, 212, 255, 0.2)"
                  name="Allocation"
                />
                <Area
                  type="monotone"
                  dataKey="cumSelection"
                  stroke="#00ff88"
                  fill="rgba(0, 255, 136, 0.2)"
                  name="Selection"
                />
                <Line
                  type="monotone"
                  dataKey="cumTotal"
                  stroke="#ffaa00"
                  strokeWidth={2}
                  dot={false}
                  name="Total Active"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
      </div>

      {/* Segment attribution */}
      <motion.div variants={itemVariants} className="glass-card">
        <div className="flex items-center gap-2 mb-6">
          <Layers className="w-5 h-5 text-accent-cyan" />
          <h3 className="text-lg font-semibold text-dark-100">Segment Attribution</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>Segment</th>
                <th className="text-right">Allocation Effect</th>
                <th className="text-right">Selection Effect</th>
                <th className="text-right">Interaction</th>
                <th className="text-right">Total Effect</th>
              </tr>
            </thead>
            <tbody>
              {segments.map((segment) => (
                <tr key={segment.name}>
                  <td className="font-medium">{segment.name}</td>
                  <td className={cn("text-right font-mono", segment.allocation >= 0 ? "text-accent-emerald" : "text-accent-rose")}>
                    {formatBps(segment.allocation)}
                  </td>
                  <td className={cn("text-right font-mono", segment.selection >= 0 ? "text-accent-emerald" : "text-accent-rose")}>
                    {formatBps(segment.selection)}
                  </td>
                  <td className={cn("text-right font-mono", segment.interaction >= 0 ? "text-accent-emerald" : "text-accent-rose")}>
                    {formatBps(segment.interaction)}
                  </td>
                  <td className={cn("text-right font-mono font-semibold", segment.total >= 0 ? "text-accent-emerald" : "text-accent-rose")}>
                    {formatBps(segment.total)}
                  </td>
                </tr>
              ))}
              <tr className="bg-dark-800/30">
                <td className="font-semibold">Total</td>
                <td className={cn("text-right font-mono font-semibold", summary.allocationEffect >= 0 ? "text-accent-emerald" : "text-accent-rose")}>
                  {formatBps(summary.allocationEffect)}
                </td>
                <td className={cn("text-right font-mono font-semibold", summary.selectionEffect >= 0 ? "text-accent-emerald" : "text-accent-rose")}>
                  {formatBps(summary.selectionEffect)}
                </td>
                <td className={cn("text-right font-mono font-semibold", summary.interactionEffect >= 0 ? "text-accent-emerald" : "text-accent-rose")}>
                  {formatBps(summary.interactionEffect)}
                </td>
                <td className={cn("text-right font-mono font-semibold", summary.activeReturn >= 0 ? "text-accent-emerald" : "text-accent-rose")}>
                  {formatBps(summary.activeReturn)}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </motion.div>
    </motion.div>
  );
}
