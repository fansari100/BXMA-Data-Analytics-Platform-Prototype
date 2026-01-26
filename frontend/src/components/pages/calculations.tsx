"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Calculator,
  ChevronDown,
  ChevronRight,
  RefreshCw,
  TrendingDown,
  BarChart3,
  Percent,
  Grid3X3,
  Target,
} from "lucide-react";
import { Math } from "@/components/ui/math";

interface CalculationStep {
  step: number;
  title: string;
  formula_latex: string;
  description: string;
  inputs: Record<string, string>;
  calculation: string;
  result: string;
}

interface CalculationResult {
  result: {
    name: string;
    value?: number;
    value_pct?: number;
    value_dollar?: number;
    formula_latex: string;
  };
  inputs: Record<string, number>;
  steps: CalculationStep[];
  timestamp: string;
}

const calculationTypes = [
  {
    id: "var",
    name: "Value at Risk (VaR)",
    icon: TrendingDown,
    description: "Maximum expected loss at a given confidence level",
    endpoint: "/api/v1/calculations/var",
    params: { confidence: 0.95, horizon: 1, method: "parametric" },
  },
  {
    id: "cvar",
    name: "Conditional VaR (CVaR)",
    icon: BarChart3,
    description: "Expected loss beyond VaR (Expected Shortfall)",
    endpoint: "/api/v1/calculations/cvar",
    params: { confidence: 0.95, horizon: 1 },
  },
  {
    id: "sharpe",
    name: "Sharpe Ratio",
    icon: Percent,
    description: "Risk-adjusted return measure",
    endpoint: "/api/v1/calculations/sharpe",
    params: {},
  },
  {
    id: "covariance",
    name: "Covariance Matrix",
    icon: Grid3X3,
    description: "Asset return covariance estimation",
    endpoint: "/api/v1/calculations/covariance",
    params: { method: "ewma", decay_factor: 0.94 },
  },
  {
    id: "optimization",
    name: "Portfolio Optimization",
    icon: Target,
    description: "Optimal weight allocation (HRP)",
    endpoint: "/api/v1/calculations/optimization",
    params: { method: "hrp" },
  },
];

function LatexFormula({ latex }: { latex: string }) {
  // Use KaTeX for proper mathematical formula rendering
  return (
    <div className="bg-dark-900/80 px-4 py-3 rounded-lg overflow-x-auto">
      <Math block>{latex}</Math>
    </div>
  );
}

function StepCard({ step, isExpanded, onToggle }: { 
  step: CalculationStep; 
  isExpanded: boolean;
  onToggle: () => void;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: step.step * 0.1 }}
      className="border border-dark-700/50 rounded-lg overflow-hidden"
    >
      <button
        onClick={onToggle}
        className="w-full flex items-center gap-4 p-4 bg-dark-800/30 hover:bg-dark-800/50 transition-colors text-left"
      >
        <div className="flex items-center justify-center w-8 h-8 rounded-full bg-accent-cyan/20 text-accent-cyan font-bold text-sm">
          {step.step}
        </div>
        <div className="flex-1">
          <h4 className="font-semibold text-dark-100">{step.title}</h4>
          <p className="text-sm text-dark-400 mt-0.5">{step.description}</p>
        </div>
        {isExpanded ? (
          <ChevronDown className="w-5 h-5 text-dark-400" />
        ) : (
          <ChevronRight className="w-5 h-5 text-dark-400" />
        )}
      </button>
      
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="p-4 space-y-4 bg-dark-900/30 border-t border-dark-700/50">
              {/* Formula */}
              <div>
                <h5 className="text-xs font-semibold uppercase tracking-wider text-dark-400 mb-2">
                  Formula
                </h5>
                <LatexFormula latex={step.formula_latex} />
              </div>
              
              {/* Inputs */}
              {Object.keys(step.inputs).length > 0 && (
                <div>
                  <h5 className="text-xs font-semibold uppercase tracking-wider text-dark-400 mb-2">
                    Inputs
                  </h5>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                    {Object.entries(step.inputs).map(([key, value]) => (
                      <div key={key} className="bg-dark-800/50 px-3 py-2 rounded">
                        <span className="text-dark-400 text-xs">{key}:</span>
                        <span className="ml-2 text-dark-100 font-mono text-sm">{value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              {/* Calculation */}
              <div>
                <h5 className="text-xs font-semibold uppercase tracking-wider text-dark-400 mb-2">
                  Calculation
                </h5>
                <p className="text-dark-300 bg-dark-800/50 px-3 py-2 rounded font-mono text-sm">
                  {step.calculation}
                </p>
              </div>
              
              {/* Result */}
              <div>
                <h5 className="text-xs font-semibold uppercase tracking-wider text-dark-400 mb-2">
                  Result
                </h5>
                <p className="text-accent-emerald bg-accent-emerald/10 px-3 py-2 rounded font-semibold">
                  {step.result}
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

export function CalculationsPage() {
  const [selectedCalc, setSelectedCalc] = useState(calculationTypes[0]);
  const [result, setResult] = useState<CalculationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set([1]));

  const fetchCalculation = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const params = new URLSearchParams(
        Object.entries(selectedCalc.params).map(([k, v]) => [k, String(v)])
      );
      const response = await fetch(
        `http://localhost:8000${selectedCalc.endpoint}?${params}`
      );
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setResult(data);
      // Expand first step by default
      setExpandedSteps(new Set([1]));
    } catch (err: any) {
      setError(err.message || "Failed to fetch calculation");
      console.error("Calculation error:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCalculation();
  }, [selectedCalc]);

  const toggleStep = (stepNum: number) => {
    setExpandedSteps((prev) => {
      const next = new Set(prev);
      if (next.has(stepNum)) {
        next.delete(stepNum);
      } else {
        next.add(stepNum);
      }
      return next;
    });
  };

  const expandAll = () => {
    if (result) {
      setExpandedSteps(new Set(result.steps.map((s) => s.step)));
    }
  };

  const collapseAll = () => {
    setExpandedSteps(new Set());
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight flex items-center gap-3">
            <Calculator className="w-7 h-7 text-accent-cyan" />
            Financial Calculations
          </h1>
          <p className="text-muted-foreground mt-1">
            Step-by-step mathematical derivations with actual portfolio data
          </p>
        </div>
        <Button
          onClick={fetchCalculation}
          disabled={loading}
          className="bg-accent-cyan hover:bg-accent-cyan/80 text-dark-950"
        >
          <RefreshCw className={`w-4 h-4 mr-2 ${loading ? "animate-spin" : ""}`} />
          Recalculate
        </Button>
      </div>

      {/* Calculation Type Selector */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        {calculationTypes.map((calc) => {
          const Icon = calc.icon;
          const isSelected = selectedCalc.id === calc.id;
          return (
            <button
              key={calc.id}
              onClick={() => setSelectedCalc(calc)}
              className={`p-4 rounded-xl border transition-all text-left ${
                isSelected
                  ? "bg-accent-cyan/10 border-accent-cyan/50 shadow-glow-sm"
                  : "bg-dark-800/30 border-dark-700/50 hover:border-dark-600"
              }`}
            >
              <Icon
                className={`w-6 h-6 mb-2 ${
                  isSelected ? "text-accent-cyan" : "text-dark-400"
                }`}
              />
              <h3
                className={`font-semibold text-sm ${
                  isSelected ? "text-accent-cyan" : "text-dark-200"
                }`}
              >
                {calc.name}
              </h3>
              <p className="text-xs text-dark-400 mt-1 line-clamp-2">
                {calc.description}
              </p>
            </button>
          );
        })}
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-red-400">
          {error}
        </div>
      )}

      {/* Result Display */}
      {result && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Result */}
          <Card className="lg:col-span-1 bg-gradient-to-br from-accent-cyan/10 to-accent-emerald/10 border-accent-cyan/30">
            <CardHeader>
              <CardTitle className="text-lg">Result</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="text-center">
                <p className="text-dark-400 text-sm mb-1">{result.result.name}</p>
                {result.result.value !== undefined && (
                  <p className="text-4xl font-bold text-accent-cyan">
                    {result.result.value.toFixed(4)}
                  </p>
                )}
                {result.result.value_pct !== undefined && (
                  <p className="text-4xl font-bold text-accent-cyan">
                    {result.result.value_pct.toFixed(4)}%
                  </p>
                )}
                {result.result.value_dollar !== undefined && (
                  <p className="text-lg text-dark-300 mt-1">
                    ${result.result.value_dollar.toLocaleString(undefined, {
                      minimumFractionDigits: 0,
                      maximumFractionDigits: 0,
                    })}
                  </p>
                )}
              </div>
              
              <div className="pt-4 border-t border-dark-700/50">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-dark-400 mb-2">
                  Master Formula
                </h4>
                <LatexFormula latex={result.result.formula_latex} />
              </div>

              {Object.keys(result.inputs).length > 0 && (
                <div className="pt-4 border-t border-dark-700/50">
                  <h4 className="text-xs font-semibold uppercase tracking-wider text-dark-400 mb-2">
                    Input Parameters
                  </h4>
                  <div className="space-y-2">
                    {Object.entries(result.inputs).map(([key, value]) => (
                      <div key={key} className="flex justify-between text-sm">
                        <span className="text-dark-400">{key.replace(/_/g, " ")}:</span>
                        <span className="text-dark-100 font-mono">
                          {typeof value === "number"
                            ? value.toLocaleString(undefined, {
                                minimumFractionDigits: 2,
                                maximumFractionDigits: 4,
                              })
                            : value}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Derivation Steps */}
          <Card className="lg:col-span-2 bg-dark-800/30 border-dark-700/50">
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="text-lg">
                Mathematical Derivation ({result.steps.length} steps)
              </CardTitle>
              <div className="flex gap-2">
                <Button variant="outline" size="sm" onClick={expandAll}>
                  Expand All
                </Button>
                <Button variant="outline" size="sm" onClick={collapseAll}>
                  Collapse All
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              {result.steps.map((step) => (
                <StepCard
                  key={step.step}
                  step={step}
                  isExpanded={expandedSteps.has(step.step)}
                  onToggle={() => toggleStep(step.step)}
                />
              ))}
            </CardContent>
          </Card>
        </div>
      )}

      {/* Loading State */}
      {loading && !result && (
        <div className="flex items-center justify-center py-20">
          <RefreshCw className="w-8 h-8 text-accent-cyan animate-spin" />
          <span className="ml-3 text-dark-400">Loading calculations...</span>
        </div>
      )}
    </div>
  );
}
