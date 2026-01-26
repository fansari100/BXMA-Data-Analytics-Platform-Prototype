"use client";

import { useState, useEffect, useCallback } from "react";
import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  RefreshCw,
  TrendingUp,
  TrendingDown,
  DollarSign,
  Briefcase,
  PieChart,
  Activity,
  Clock,
  ArrowUpRight,
  ArrowDownRight,
} from "lucide-react";

interface Holding {
  ticker: string;
  name: string;
  asset_class: string;
  sector: string;
  target_weight: number;
  actual_weight: number;
  shares: number;
  cost_basis: number;
  current_price: number;
  market_value: number;
  gain_loss: number;
  gain_loss_pct: number;
  day_change: number;
  day_change_pct: number;
}

interface PortfolioData {
  portfolio_name: string;
  inception_date: string;
  total_value: number;
  total_aum_target: number;
  holdings: Holding[];
  asset_allocation: Record<string, number>;
  benchmark: string;
  risk_free_rate: number;
  timestamp: string;
}

interface MarketIndex {
  ticker: string;
  name: string;
  price: number;
  change: number;
  change_pct: number;
}

export function LivePortfolioPage() {
  const [portfolio, setPortfolio] = useState<PortfolioData | null>(null);
  const [indices, setIndices] = useState<MarketIndex[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const fetchData = useCallback(async () => {
    try {
      const [portfolioRes, indicesRes] = await Promise.all([
        fetch("http://localhost:8000/api/v1/portfolio/live"),
        fetch("http://localhost:8000/api/v1/portfolio/market-indices"),
      ]);

      if (!portfolioRes.ok || !indicesRes.ok) {
        throw new Error("Failed to fetch data");
      }

      const portfolioData = await portfolioRes.json();
      const indicesData = await indicesRes.json();

      setPortfolio(portfolioData);
      setIndices(indicesData.indices);
      setLastUpdate(new Date());
      setError(null);
    } catch (err: any) {
      setError(err.message || "Failed to fetch portfolio data");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();

    // Auto-refresh every 30 seconds
    if (autoRefresh) {
      const interval = setInterval(fetchData, 30000);
      return () => clearInterval(interval);
    }
  }, [fetchData, autoRefresh]);

  const formatCurrency = (value: number) => {
    if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
    if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
    return `$${value.toLocaleString(undefined, { minimumFractionDigits: 2 })}`;
  };

  const formatPercent = (value: number, decimals = 2) => {
    const sign = value >= 0 ? "+" : "";
    return `${sign}${(value * 100).toFixed(decimals)}%`;
  };

  if (loading && !portfolio) {
    return (
      <div className="flex items-center justify-center h-96">
        <RefreshCw className="w-8 h-8 text-accent-cyan animate-spin" />
        <span className="ml-3 text-dark-400">Loading live portfolio data...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight flex items-center gap-3">
            <Briefcase className="w-7 h-7 text-accent-cyan" />
            Live Portfolio
          </h1>
          <p className="text-muted-foreground mt-1">
            Real-time portfolio data with live market prices
          </p>
        </div>
        <div className="flex items-center gap-4">
          {lastUpdate && (
            <div className="flex items-center gap-2 text-sm text-dark-400">
              <Clock className="w-4 h-4" />
              Last update: {lastUpdate.toLocaleTimeString()}
            </div>
          )}
          <Button
            variant={autoRefresh ? "default" : "outline"}
            size="sm"
            onClick={() => setAutoRefresh(!autoRefresh)}
          >
            {autoRefresh ? "Auto-refresh ON" : "Auto-refresh OFF"}
          </Button>
          <Button
            onClick={fetchData}
            disabled={loading}
            className="bg-accent-cyan hover:bg-accent-cyan/80 text-dark-950"
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </Button>
        </div>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-red-400">
          {error}
        </div>
      )}

      {/* Market Indices Ticker */}
      <div className="flex gap-4 overflow-x-auto pb-2">
        {indices.map((index) => (
          <div
            key={index.ticker}
            className="flex items-center gap-3 px-4 py-2 bg-dark-800/50 rounded-lg border border-dark-700/50 min-w-fit"
          >
            <span className="text-dark-400 text-sm">{index.name}</span>
            <span className="font-mono font-semibold text-dark-100">
              ${index.price.toFixed(2)}
            </span>
            <span
              className={`flex items-center text-sm font-medium ${
                index.change_pct >= 0 ? "text-accent-emerald" : "text-accent-rose"
              }`}
            >
              {index.change_pct >= 0 ? (
                <TrendingUp className="w-3 h-3 mr-1" />
              ) : (
                <TrendingDown className="w-3 h-3 mr-1" />
              )}
              {index.change_pct >= 0 ? "+" : ""}
              {index.change_pct.toFixed(2)}%
            </span>
          </div>
        ))}
      </div>

      {portfolio && (
        <>
          {/* Portfolio Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card className="bg-gradient-to-br from-accent-cyan/10 to-dark-800/50 border-accent-cyan/30">
              <CardContent className="pt-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-dark-400">Total Portfolio Value</p>
                    <p className="text-3xl font-bold text-accent-cyan mt-1">
                      {formatCurrency(portfolio.total_value)}
                    </p>
                    <p className="text-xs text-dark-500 mt-1">
                      Target AUM: {formatCurrency(portfolio.total_aum_target)}
                    </p>
                  </div>
                  <DollarSign className="w-10 h-10 text-accent-cyan/50" />
                </div>
              </CardContent>
            </Card>

            <Card className="bg-dark-800/30 border-dark-700/50">
              <CardContent className="pt-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-dark-400">Holdings</p>
                    <p className="text-3xl font-bold text-dark-100 mt-1">
                      {portfolio.holdings.length}
                    </p>
                    <p className="text-xs text-dark-500 mt-1">
                      Active positions
                    </p>
                  </div>
                  <PieChart className="w-10 h-10 text-dark-600" />
                </div>
              </CardContent>
            </Card>

            <Card className="bg-dark-800/30 border-dark-700/50">
              <CardContent className="pt-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-dark-400">Risk-Free Rate</p>
                    <p className="text-3xl font-bold text-dark-100 mt-1">
                      {(portfolio.risk_free_rate * 100).toFixed(2)}%
                    </p>
                    <p className="text-xs text-dark-500 mt-1">
                      10Y Treasury
                    </p>
                  </div>
                  <Activity className="w-10 h-10 text-dark-600" />
                </div>
              </CardContent>
            </Card>

            <Card className="bg-dark-800/30 border-dark-700/50">
              <CardContent className="pt-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-dark-400">Benchmark</p>
                    <p className="text-3xl font-bold text-dark-100 mt-1">
                      {portfolio.benchmark}
                    </p>
                    <p className="text-xs text-dark-500 mt-1">
                      S&P 500 ETF
                    </p>
                  </div>
                  <TrendingUp className="w-10 h-10 text-dark-600" />
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Asset Allocation */}
          <Card className="bg-dark-800/30 border-dark-700/50">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <PieChart className="w-5 h-5 text-accent-cyan" />
                Asset Allocation
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                {Object.entries(portfolio.asset_allocation).map(([assetClass, weight]) => (
                  <div
                    key={assetClass}
                    className="bg-dark-900/50 p-4 rounded-lg border border-dark-700/30"
                  >
                    <p className="text-sm text-dark-400">{assetClass}</p>
                    <p className="text-2xl font-bold text-dark-100 mt-1">
                      {(weight * 100).toFixed(1)}%
                    </p>
                    <div className="w-full h-2 bg-dark-700 rounded-full mt-2 overflow-hidden">
                      <div
                        className="h-full bg-accent-cyan rounded-full"
                        style={{ width: `${weight * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Holdings Table */}
          <Card className="bg-dark-800/30 border-dark-700/50">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Briefcase className="w-5 h-5 text-accent-cyan" />
                Portfolio Holdings (Live Prices)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-dark-700/50">
                      <th className="text-left py-3 px-4 text-xs font-semibold uppercase tracking-wider text-dark-400">
                        Ticker
                      </th>
                      <th className="text-left py-3 px-4 text-xs font-semibold uppercase tracking-wider text-dark-400">
                        Name
                      </th>
                      <th className="text-left py-3 px-4 text-xs font-semibold uppercase tracking-wider text-dark-400">
                        Asset Class
                      </th>
                      <th className="text-right py-3 px-4 text-xs font-semibold uppercase tracking-wider text-dark-400">
                        Price
                      </th>
                      <th className="text-right py-3 px-4 text-xs font-semibold uppercase tracking-wider text-dark-400">
                        Day Chg
                      </th>
                      <th className="text-right py-3 px-4 text-xs font-semibold uppercase tracking-wider text-dark-400">
                        Shares
                      </th>
                      <th className="text-right py-3 px-4 text-xs font-semibold uppercase tracking-wider text-dark-400">
                        Market Value
                      </th>
                      <th className="text-right py-3 px-4 text-xs font-semibold uppercase tracking-wider text-dark-400">
                        Weight
                      </th>
                      <th className="text-right py-3 px-4 text-xs font-semibold uppercase tracking-wider text-dark-400">
                        Total G/L
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {portfolio.holdings.map((holding, idx) => (
                      <motion.tr
                        key={holding.ticker}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: idx * 0.02 }}
                        className="border-b border-dark-700/30 hover:bg-dark-800/30"
                      >
                        <td className="py-3 px-4">
                          <span className="font-mono font-semibold text-accent-cyan">
                            {holding.ticker}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-dark-200 text-sm">
                          {holding.name}
                        </td>
                        <td className="py-3 px-4">
                          <span className="px-2 py-1 text-xs rounded bg-dark-700/50 text-dark-300">
                            {holding.asset_class}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-right font-mono text-dark-100">
                          ${holding.current_price.toFixed(2)}
                        </td>
                        <td className="py-3 px-4 text-right">
                          <span
                            className={`flex items-center justify-end gap-1 font-mono text-sm ${
                              holding.day_change_pct >= 0
                                ? "text-accent-emerald"
                                : "text-accent-rose"
                            }`}
                          >
                            {holding.day_change_pct >= 0 ? (
                              <ArrowUpRight className="w-3 h-3" />
                            ) : (
                              <ArrowDownRight className="w-3 h-3" />
                            )}
                            {holding.day_change_pct >= 0 ? "+" : ""}
                            {holding.day_change_pct.toFixed(2)}%
                          </span>
                        </td>
                        <td className="py-3 px-4 text-right font-mono text-dark-300">
                          {holding.shares.toLocaleString()}
                        </td>
                        <td className="py-3 px-4 text-right font-mono text-dark-100">
                          {formatCurrency(holding.market_value)}
                        </td>
                        <td className="py-3 px-4 text-right font-mono text-dark-300">
                          {(holding.actual_weight * 100).toFixed(2)}%
                        </td>
                        <td className="py-3 px-4 text-right">
                          <span
                            className={`font-mono text-sm ${
                              holding.gain_loss >= 0
                                ? "text-accent-emerald"
                                : "text-accent-rose"
                            }`}
                          >
                            {formatCurrency(holding.gain_loss)}
                            <br />
                            <span className="text-xs opacity-75">
                              ({formatPercent(holding.gain_loss_pct)})
                            </span>
                          </span>
                        </td>
                      </motion.tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
