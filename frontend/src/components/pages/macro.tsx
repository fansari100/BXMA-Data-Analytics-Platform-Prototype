"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  TrendingUp,
  TrendingDown,
  Globe,
  DollarSign,
  Percent,
  BarChart3,
  RefreshCw,
  Calendar,
  AlertCircle,
} from "lucide-react";

// Interfaces for macro data
interface MacroIndicator {
  name: string;
  value: number;
  previous: number;
  unit: string;
  region: string;
  lastUpdate: string;
}

interface YieldCurvePoint {
  tenor: string;
  rate: number;
}

interface FXRate {
  pair: string;
  rate: number;
  change24h: number;
}

interface CommodityPrice {
  name: string;
  price: number;
  unit: string;
  change24h: number;
}

interface EquityIndex {
  name: string;
  value: number;
  change: number;
  changePct: number;
}

interface EconomicEvent {
  date: string;
  time: string;
  indicator: string;
  region: string;
  importance: "low" | "medium" | "high" | "critical";
  consensus: string;
  previous: string;
}

// Sample data
const macroIndicators: MacroIndicator[] = [
  { name: "US GDP (QoQ)", value: 2.5, previous: 2.3, unit: "%", region: "US", lastUpdate: "2026-01-25" },
  { name: "US CPI (YoY)", value: 3.2, previous: 3.4, unit: "%", region: "US", lastUpdate: "2026-01-25" },
  { name: "US Unemployment", value: 4.1, previous: 4.0, unit: "%", region: "US", lastUpdate: "2026-01-25" },
  { name: "Fed Funds Rate", value: 4.25, previous: 4.25, unit: "%", region: "US", lastUpdate: "2026-01-25" },
  { name: "Eurozone CPI", value: 2.8, previous: 2.9, unit: "%", region: "EU", lastUpdate: "2026-01-25" },
  { name: "China GDP (YoY)", value: 5.2, previous: 4.9, unit: "%", region: "CN", lastUpdate: "2026-01-25" },
];

const yieldCurve: YieldCurvePoint[] = [
  { tenor: "1M", rate: 4.30 },
  { tenor: "3M", rate: 4.40 },
  { tenor: "6M", rate: 4.50 },
  { tenor: "1Y", rate: 4.40 },
  { tenor: "2Y", rate: 4.20 },
  { tenor: "5Y", rate: 4.10 },
  { tenor: "10Y", rate: 4.30 },
  { tenor: "30Y", rate: 4.50 },
];

const fxRates: FXRate[] = [
  { pair: "EUR/USD", rate: 1.0850, change24h: 0.15 },
  { pair: "USD/JPY", rate: 148.50, change24h: -0.25 },
  { pair: "GBP/USD", rate: 1.2650, change24h: 0.08 },
  { pair: "USD/CHF", rate: 0.8720, change24h: -0.12 },
];

const commodityPrices: CommodityPrice[] = [
  { name: "WTI Crude", price: 75.50, unit: "$/bbl", change24h: 1.2 },
  { name: "Brent Crude", price: 79.80, unit: "$/bbl", change24h: 0.9 },
  { name: "Gold", price: 2650.0, unit: "$/oz", change24h: 0.5 },
  { name: "Silver", price: 31.50, unit: "$/oz", change24h: 0.8 },
  { name: "Copper", price: 4.15, unit: "$/lb", change24h: -0.3 },
];

const equityIndices: EquityIndex[] = [
  { name: "S&P 500", value: 5950.25, change: 12.50, changePct: 0.21 },
  { name: "Nasdaq 100", value: 21000.75, change: -45.30, changePct: -0.22 },
  { name: "Russell 2000", value: 2250.10, change: 8.20, changePct: 0.37 },
  { name: "VIX", value: 15.50, change: -0.80, changePct: -4.90 },
];

const economicCalendar: EconomicEvent[] = [
  { date: "2026-01-27", time: "08:30 ET", indicator: "Durable Goods Orders", region: "US", importance: "high", consensus: "1.0%", previous: "0.5%" },
  { date: "2026-01-28", time: "10:00 ET", indicator: "Consumer Confidence", region: "US", importance: "high", consensus: "105.0", previous: "104.7" },
  { date: "2026-01-29", time: "14:00 ET", indicator: "FOMC Decision", region: "US", importance: "critical", consensus: "4.25%", previous: "4.25%" },
  { date: "2026-01-30", time: "08:30 ET", indicator: "GDP (Q4 Advance)", region: "US", importance: "high", consensus: "2.5%", previous: "2.3%" },
  { date: "2026-01-31", time: "08:30 ET", indicator: "PCE Deflator", region: "US", importance: "high", consensus: "2.8%", previous: "2.9%" },
];

const importanceColors = {
  low: "bg-dark-600 text-dark-300",
  medium: "bg-cyan-500/20 text-cyan-400",
  high: "bg-amber-500/20 text-amber-400",
  critical: "bg-rose-500/20 text-rose-400",
};

export function MacroDataPage() {
  const [lastRefresh, setLastRefresh] = useState(new Date());
  const [isRefreshing, setIsRefreshing] = useState(false);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await new Promise((resolve) => setTimeout(resolve, 1000));
    setLastRefresh(new Date());
    setIsRefreshing(false);
  };

  // Calculate max rate for yield curve visualization
  const maxRate = Math.max(...yieldCurve.map((p) => p.rate));
  const minRate = Math.min(...yieldCurve.map((p) => p.rate));

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="space-y-6"
    >
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-dark-100">Macro Market Data</h1>
          <p className="text-dark-400 mt-1">
            Real-time macro and asset class specific market data
          </p>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-dark-500 text-sm">
            Last update: {lastRefresh.toLocaleTimeString()}
          </span>
          <Button
            onClick={handleRefresh}
            disabled={isRefreshing}
            className="bg-accent-cyan hover:bg-accent-cyan/80 text-dark-950"
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${isRefreshing ? "animate-spin" : ""}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Equity Indices */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {equityIndices.map((index, i) => (
          <motion.div
            key={index.name}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.05 }}
          >
            <Card className="bg-dark-800/50 border-dark-700">
              <CardContent className="p-4">
                <p className="text-dark-400 text-sm">{index.name}</p>
                <p className="text-2xl font-bold text-dark-100 mt-1">
                  {index.value.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                </p>
                <div className={`flex items-center gap-1 mt-1 ${index.change >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                  {index.change >= 0 ? (
                    <TrendingUp className="w-4 h-4" />
                  ) : (
                    <TrendingDown className="w-4 h-4" />
                  )}
                  <span className="text-sm font-medium">
                    {index.change >= 0 ? "+" : ""}{index.change.toFixed(2)} ({index.changePct >= 0 ? "+" : ""}{index.changePct.toFixed(2)}%)
                  </span>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Yield Curve */}
        <Card className="bg-dark-800/50 border-dark-700">
          <CardHeader>
            <CardTitle className="text-dark-100 flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-accent-cyan" />
              US Treasury Yield Curve
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-48 flex items-end justify-between gap-2">
              {yieldCurve.map((point, i) => {
                const height = ((point.rate - minRate + 0.5) / (maxRate - minRate + 1)) * 100;
                return (
                  <div key={point.tenor} className="flex flex-col items-center flex-1">
                    <span className="text-xs text-dark-300 mb-1">{point.rate.toFixed(2)}%</span>
                    <motion.div
                      initial={{ height: 0 }}
                      animate={{ height: `${height}%` }}
                      transition={{ delay: i * 0.05, duration: 0.5 }}
                      className="w-full bg-gradient-to-t from-accent-cyan/60 to-accent-cyan rounded-t"
                    />
                    <span className="text-xs text-dark-500 mt-2">{point.tenor}</span>
                  </div>
                );
              })}
            </div>
            <div className="flex justify-between mt-4 text-xs text-dark-500">
              <span>2s10s Spread: {(yieldCurve[6].rate - yieldCurve[4].rate).toFixed(2)}%</span>
              <span>Curve: {yieldCurve[6].rate > yieldCurve[4].rate ? "Normal" : "Inverted"}</span>
            </div>
          </CardContent>
        </Card>

        {/* FX Rates */}
        <Card className="bg-dark-800/50 border-dark-700">
          <CardHeader>
            <CardTitle className="text-dark-100 flex items-center gap-2">
              <DollarSign className="w-5 h-5 text-accent-emerald" />
              FX Rates
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {fxRates.map((fx, i) => (
                <motion.div
                  key={fx.pair}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.05 }}
                  className="flex items-center justify-between p-3 bg-dark-700/50 rounded-lg"
                >
                  <span className="text-dark-100 font-medium">{fx.pair}</span>
                  <div className="text-right">
                    <span className="text-dark-100 font-mono">
                      {fx.rate.toFixed(fx.pair.includes("JPY") ? 2 : 4)}
                    </span>
                    <span className={`ml-2 text-sm ${fx.change24h >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                      {fx.change24h >= 0 ? "+" : ""}{fx.change24h.toFixed(2)}%
                    </span>
                  </div>
                </motion.div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Commodities */}
        <Card className="bg-dark-800/50 border-dark-700">
          <CardHeader>
            <CardTitle className="text-dark-100 flex items-center gap-2">
              <Globe className="w-5 h-5 text-accent-amber" />
              Commodities
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {commodityPrices.map((commodity, i) => (
                <motion.div
                  key={commodity.name}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.05 }}
                  className="flex items-center justify-between p-3 bg-dark-700/50 rounded-lg"
                >
                  <span className="text-dark-100 font-medium">{commodity.name}</span>
                  <div className="text-right">
                    <span className="text-dark-100 font-mono">
                      {commodity.price.toLocaleString(undefined, { minimumFractionDigits: 2 })} {commodity.unit}
                    </span>
                    <span className={`ml-2 text-sm ${commodity.change24h >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                      {commodity.change24h >= 0 ? "+" : ""}{commodity.change24h.toFixed(1)}%
                    </span>
                  </div>
                </motion.div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Macro Indicators */}
        <Card className="bg-dark-800/50 border-dark-700">
          <CardHeader>
            <CardTitle className="text-dark-100 flex items-center gap-2">
              <Percent className="w-5 h-5 text-accent-violet" />
              Economic Indicators
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {macroIndicators.map((indicator, i) => (
                <motion.div
                  key={indicator.name}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.05 }}
                  className="flex items-center justify-between p-3 bg-dark-700/50 rounded-lg"
                >
                  <div>
                    <span className="text-dark-100 font-medium">{indicator.name}</span>
                    <span className="ml-2 text-xs text-dark-500 bg-dark-600/50 px-1.5 py-0.5 rounded">
                      {indicator.region}
                    </span>
                  </div>
                  <div className="text-right">
                    <span className="text-dark-100 font-mono">{indicator.value}{indicator.unit}</span>
                    <span className={`ml-2 text-sm ${indicator.value >= indicator.previous ? "text-emerald-400" : "text-rose-400"}`}>
                      (prev: {indicator.previous}{indicator.unit})
                    </span>
                  </div>
                </motion.div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Economic Calendar */}
      <Card className="bg-dark-800/50 border-dark-700">
        <CardHeader>
          <CardTitle className="text-dark-100 flex items-center gap-2">
            <Calendar className="w-5 h-5 text-accent-rose" />
            Economic Calendar
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-dark-400 text-sm border-b border-dark-700">
                  <th className="text-left py-2 px-3">Date</th>
                  <th className="text-left py-2 px-3">Time</th>
                  <th className="text-left py-2 px-3">Event</th>
                  <th className="text-left py-2 px-3">Region</th>
                  <th className="text-left py-2 px-3">Importance</th>
                  <th className="text-right py-2 px-3">Consensus</th>
                  <th className="text-right py-2 px-3">Previous</th>
                </tr>
              </thead>
              <tbody>
                {economicCalendar.map((event, i) => (
                  <motion.tr
                    key={`${event.date}-${event.indicator}`}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.05 }}
                    className="border-b border-dark-700/50 hover:bg-dark-700/30 transition-colors"
                  >
                    <td className="py-3 px-3 text-dark-300">{event.date}</td>
                    <td className="py-3 px-3 text-dark-400">{event.time}</td>
                    <td className="py-3 px-3 text-dark-100 font-medium">{event.indicator}</td>
                    <td className="py-3 px-3">
                      <span className="text-xs bg-dark-600/50 px-2 py-0.5 rounded text-dark-300">
                        {event.region}
                      </span>
                    </td>
                    <td className="py-3 px-3">
                      <span className={`text-xs px-2 py-0.5 rounded ${importanceColors[event.importance]}`}>
                        {event.importance}
                      </span>
                    </td>
                    <td className="py-3 px-3 text-right text-dark-200">{event.consensus}</td>
                    <td className="py-3 px-3 text-right text-dark-400">{event.previous}</td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
