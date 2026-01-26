"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import {
  Zap,
  Play,
  AlertTriangle,
  TrendingDown,
  Settings,
  Plus,
  ChevronDown,
  ChevronUp,
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
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";
import { cn, formatPercent, formatCompactNumber } from "@/lib/utils";

// Standard scenarios
const standardScenarios = [
  {
    id: "gfc",
    name: "2008 Financial Crisis",
    description: "Global credit freeze, equity collapse",
    factorShocks: { Market: -0.40, Credit: -0.25, Rates: -0.15 },
    impact: -0.182,
    severity: "extreme",
  },
  {
    id: "covid",
    name: "COVID-19 Crash",
    description: "March 2020 pandemic shock",
    factorShocks: { Market: -0.35, Credit: -0.15, Vol: 0.50 },
    impact: -0.145,
    severity: "extreme",
  },
  {
    id: "rate_hike",
    name: "Rates +200bps",
    description: "Aggressive Fed tightening",
    factorShocks: { Rates: 0.02, Credit: -0.05, Market: -0.08 },
    impact: -0.068,
    severity: "moderate",
  },
  {
    id: "em_crisis",
    name: "EM Currency Crisis",
    description: "Emerging market contagion",
    factorShocks: { EM: -0.30, FX: -0.15, Credit: -0.10 },
    impact: -0.095,
    severity: "high",
  },
  {
    id: "deflation",
    name: "Deflation Shock",
    description: "Japan-style deflation scenario",
    factorShocks: { Rates: -0.02, Market: -0.15, Credit: -0.08 },
    impact: -0.072,
    severity: "moderate",
  },
  {
    id: "geopolitical",
    name: "Geopolitical Crisis",
    description: "Major regional conflict",
    factorShocks: { Market: -0.20, Oil: 0.40, Vol: 0.30 },
    impact: -0.112,
    severity: "high",
  },
];

// Position impacts
const generatePositionImpacts = (scenario: typeof standardScenarios[0]) => [
  { name: "US Equities", impact: scenario.impact * 0.35, contribution: 35 },
  { name: "Int'l Equities", impact: scenario.impact * 0.25, contribution: 25 },
  { name: "EM Equities", impact: scenario.impact * 0.18, contribution: 18 },
  { name: "Credit", impact: scenario.impact * 0.12, contribution: 12 },
  { name: "Govt Bonds", impact: scenario.impact * 0.05, contribution: 5 },
  { name: "Alternatives", impact: scenario.impact * 0.05, contribution: 5 },
];

// Factor sensitivities
const factorSensitivities = [
  { factor: "Market", sensitivity: 0.92 },
  { factor: "Size", sensitivity: 0.15 },
  { factor: "Value", sensitivity: -0.08 },
  { factor: "Credit", sensitivity: 0.45 },
  { factor: "Rates", sensitivity: -0.32 },
  { factor: "FX", sensitivity: 0.18 },
];

export function StressTestPage() {
  const [selectedScenario, setSelectedScenario] = useState(standardScenarios[0]);
  const [isRunning, setIsRunning] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [expandedScenario, setExpandedScenario] = useState<string | null>(null);
  const [customShocks, setCustomShocks] = useState({
    Market: 0,
    Credit: 0,
    Rates: 0,
    Vol: 0,
  });

  // Calculate custom scenario impact based on slider values
  const calculateCustomImpact = () => {
    // Weight each factor's contribution to portfolio impact
    const factorWeights = {
      Market: 0.45,  // Market has highest impact
      Credit: 0.25,
      Rates: 0.15,
      Vol: 0.15,
    };
    
    let totalImpact = 0;
    Object.entries(customShocks).forEach(([factor, shock]) => {
      const weight = factorWeights[factor as keyof typeof factorWeights] || 0.1;
      // Negative market/credit shocks hurt portfolio, positive vol hurts
      if (factor === "Vol") {
        totalImpact -= shock * weight; // Higher vol = negative impact
      } else {
        totalImpact += shock * weight; // Market/Credit/Rates shocks directly impact
      }
    });
    
    return totalImpact;
  };

  // Check if custom shocks are being used (any non-zero value)
  const hasCustomShocks = Object.values(customShocks).some(v => v !== 0);

  const handleRunStressTest = () => {
    setIsRunning(true);
    setTimeout(() => {
      // Use custom scenario if sliders have been adjusted, otherwise use selected scenario
      const effectiveImpact = hasCustomShocks 
        ? calculateCustomImpact() 
        : selectedScenario.impact;
      
      const effectiveScenario = hasCustomShocks
        ? {
            ...selectedScenario,
            name: "Custom Scenario",
            description: `Market: ${(customShocks.Market * 100).toFixed(0)}%, Credit: ${(customShocks.Credit * 100).toFixed(0)}%, Rates: ${(customShocks.Rates * 100).toFixed(0)}%, Vol: ${(customShocks.Vol * 100).toFixed(0)}%`,
            impact: effectiveImpact,
            factorShocks: customShocks,
          }
        : selectedScenario;
      
      setResults({
        portfolioImpact: effectiveImpact,
        dollarImpact: effectiveImpact * 90_000_000_000,
        positionImpacts: generatePositionImpacts(effectiveScenario),
        worstCase: effectiveImpact * 1.2,
        bestCase: effectiveImpact * 0.8,
        scenarioName: effectiveScenario.name,
        scenarioDescription: effectiveScenario.description,
        isCustom: hasCustomShocks,
      });
      setIsRunning(false);
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

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "extreme":
        return "text-accent-rose bg-accent-rose/10 border-accent-rose/30";
      case "high":
        return "text-accent-amber bg-accent-amber/10 border-accent-amber/30";
      case "moderate":
        return "text-accent-cyan bg-accent-cyan/10 border-accent-cyan/30";
      default:
        return "text-dark-400 bg-dark-800 border-dark-700";
    }
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
          <h2 className="text-2xl font-display font-bold text-dark-100">Stress Testing</h2>
          <p className="text-dark-400 mt-1">Scenario analysis and extreme event simulation</p>
        </div>
        <button
          onClick={handleRunStressTest}
          disabled={isRunning}
          className="btn-primary flex items-center gap-2"
        >
          <Play className={cn("w-4 h-4", isRunning && "animate-pulse")} />
          {isRunning ? "Running..." : "Run Stress Test"}
        </button>
      </motion.div>

      <div className="grid grid-cols-3 gap-6">
        {/* Scenario selection */}
        <motion.div variants={itemVariants} className="col-span-2 glass-card">
          <div className="flex items-center gap-2 mb-4">
            <Zap className="w-5 h-5 text-accent-amber" />
            <h3 className="text-lg font-semibold text-dark-100">Scenarios</h3>
          </div>
          <div className="space-y-3">
            {standardScenarios.map((scenario) => (
              <div
                key={scenario.id}
                className={cn(
                  "p-4 rounded-lg border cursor-pointer transition-all duration-200",
                  selectedScenario.id === scenario.id
                    ? "bg-dark-800/50 border-accent-cyan/50"
                    : "bg-dark-900/30 border-dark-700 hover:border-dark-600"
                )}
                onClick={() => setSelectedScenario(scenario)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className={cn(
                      "w-2 h-2 rounded-full",
                      selectedScenario.id === scenario.id ? "bg-accent-cyan" : "bg-dark-600"
                    )} />
                    <div>
                      <p className="font-medium text-dark-100">{scenario.name}</p>
                      <p className="text-sm text-dark-500">{scenario.description}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <span className={cn(
                      "px-2 py-1 text-xs rounded border",
                      getSeverityColor(scenario.severity)
                    )}>
                      {scenario.severity}
                    </span>
                    <span className={cn(
                      "text-lg font-mono font-semibold",
                      scenario.impact < 0 ? "text-accent-rose" : "text-accent-emerald"
                    )}>
                      {formatPercent(scenario.impact)}
                    </span>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setExpandedScenario(expandedScenario === scenario.id ? null : scenario.id);
                      }}
                    >
                      {expandedScenario === scenario.id ? (
                        <ChevronUp className="w-4 h-4 text-dark-400" />
                      ) : (
                        <ChevronDown className="w-4 h-4 text-dark-400" />
                      )}
                    </button>
                  </div>
                </div>
                {expandedScenario === scenario.id && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    className="mt-4 pt-4 border-t border-dark-700"
                  >
                    <div className="grid grid-cols-3 gap-4">
                      {Object.entries(scenario.factorShocks).map(([factor, shock]) => (
                        <div key={factor} className="text-sm">
                          <span className="text-dark-500">{factor}</span>
                          <span className={cn(
                            "ml-2 font-mono",
                            shock < 0 ? "text-accent-rose" : "text-accent-emerald"
                          )}>
                            {shock >= 0 ? "+" : ""}{(shock * 100).toFixed(0)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </motion.div>
                )}
              </div>
            ))}
          </div>
        </motion.div>

        {/* Factor sensitivities */}
        <motion.div variants={itemVariants} className="glass-card">
          <h3 className="text-lg font-semibold text-dark-100 mb-4">Factor Sensitivities</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={factorSensitivities}>
                <PolarGrid stroke="rgba(255,255,255,0.1)" />
                <PolarAngleAxis dataKey="factor" stroke="#6a6d79" fontSize={11} />
                <PolarRadiusAxis stroke="#6a6d79" fontSize={10} domain={[-0.5, 1]} />
                <Radar
                  name="Sensitivity"
                  dataKey="sensitivity"
                  stroke="#00d4ff"
                  fill="#00d4ff"
                  fillOpacity={0.3}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
      </div>

      {/* Results */}
      {results && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-6"
        >
          {/* Scenario indicator */}
          {results.isCustom && (
            <div className="bg-accent-cyan/10 border border-accent-cyan/30 rounded-lg p-4">
              <div className="flex items-center gap-2">
                <Settings className="w-5 h-5 text-accent-cyan" />
                <span className="font-semibold text-accent-cyan">Custom Scenario Applied</span>
              </div>
              <p className="text-dark-400 text-sm mt-1">{results.scenarioDescription}</p>
            </div>
          )}
          
          {/* Impact summary */}
          <div className="grid grid-cols-4 gap-4">
            <div className="metric-card">
              <div className="flex items-center gap-2 mb-2">
                <TrendingDown className="w-4 h-4 text-accent-rose" />
                <span className="text-sm text-dark-400">Portfolio Impact</span>
              </div>
              <p className={cn(
                "text-2xl font-semibold",
                results.portfolioImpact < 0 ? "text-accent-rose" : "text-accent-emerald"
              )}>
                {formatPercent(results.portfolioImpact)}
              </p>
            </div>
            <div className="metric-card">
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle className="w-4 h-4 text-accent-amber" />
                <span className="text-sm text-dark-400">Dollar Impact</span>
              </div>
              <p className="text-2xl font-semibold text-accent-rose">
                {formatCompactNumber(results.dollarImpact)}
              </p>
            </div>
            <div className="metric-card">
              <div className="flex items-center gap-2 mb-2">
                <TrendingDown className="w-4 h-4 text-dark-500" />
                <span className="text-sm text-dark-400">Worst Case</span>
              </div>
              <p className="text-2xl font-semibold text-dark-100">
                {formatPercent(results.worstCase)}
              </p>
            </div>
            <div className="metric-card">
              <div className="flex items-center gap-2 mb-2">
                <TrendingDown className="w-4 h-4 text-dark-500" />
                <span className="text-sm text-dark-400">Best Case</span>
              </div>
              <p className="text-2xl font-semibold text-dark-100">
                {formatPercent(results.bestCase)}
              </p>
            </div>
          </div>

          {/* Position impacts */}
          <div className="glass-card">
            <h3 className="text-lg font-semibold text-dark-100 mb-4">Position Impact Analysis</h3>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={results.positionImpacts} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis
                    type="number"
                    stroke="#6a6d79"
                    fontSize={11}
                    tickFormatter={(v) => `${(v * 100).toFixed(1)}%`}
                    domain={["dataMin", 0]}
                  />
                  <YAxis type="category" dataKey="name" stroke="#6a6d79" fontSize={11} width={100} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "rgba(39, 40, 45, 0.95)",
                      border: "1px solid rgba(100, 100, 120, 0.3)",
                      borderRadius: "8px",
                    }}
                    formatter={(value: number) => [`${(value * 100).toFixed(2)}%`, "Impact"]}
                  />
                  <ReferenceLine x={0} stroke="rgba(255,255,255,0.2)" />
                  <Bar
                    dataKey="impact"
                    fill="#ff6b6b"
                    radius={[4, 0, 0, 4]}
                    name="Impact"
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </motion.div>
      )}

      {/* Custom scenario builder */}
      <motion.div variants={itemVariants} className="glass-card">
        <div className="flex items-center gap-2 mb-4">
          <Settings className="w-5 h-5 text-accent-cyan" />
          <h3 className="text-lg font-semibold text-dark-100">Custom Scenario Builder</h3>
        </div>
        <div className="grid grid-cols-4 gap-4">
          {Object.entries(customShocks).map(([factor, value]) => (
            <div key={factor}>
              <label className="text-sm text-dark-400 block mb-2">{factor} Shock (%)</label>
              <input
                type="range"
                min="-50"
                max="50"
                value={value * 100}
                onChange={(e) =>
                  setCustomShocks({ ...customShocks, [factor]: parseInt(e.target.value) / 100 })
                }
                className="w-full"
              />
              <div className="flex justify-between text-xs text-dark-500 mt-1">
                <span>-50%</span>
                <span className={cn(
                  "font-mono",
                  value < 0 ? "text-accent-rose" : value > 0 ? "text-accent-emerald" : "text-dark-400"
                )}>
                  {value >= 0 ? "+" : ""}{(value * 100).toFixed(0)}%
                </span>
                <span>+50%</span>
              </div>
            </div>
          ))}
        </div>
        <div className="flex gap-3 mt-4">
          <button 
            onClick={() => setCustomShocks({ Market: 0, Credit: 0, Rates: 0, Vol: 0 })}
            className="btn-secondary flex items-center gap-2"
            disabled={!hasCustomShocks}
          >
            Reset to Zero
          </button>
          <button 
            onClick={handleRunStressTest}
            disabled={isRunning || !hasCustomShocks}
            className="btn-primary flex items-center gap-2"
          >
            <Play className="w-4 h-4" />
            Run Custom Scenario
          </button>
        </div>
        {hasCustomShocks && (
          <p className="text-sm text-accent-cyan mt-2">
            Custom shocks active: Estimated impact {formatPercent(calculateCustomImpact())}
          </p>
        )}
      </motion.div>
    </motion.div>
  );
}
