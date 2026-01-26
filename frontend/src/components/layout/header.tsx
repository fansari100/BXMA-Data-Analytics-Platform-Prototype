"use client";

import { useState, useEffect, useCallback } from "react";
import { motion } from "framer-motion";
import {
  Bell,
  Search,
  RefreshCw,
  Clock,
  TrendingUp,
  TrendingDown,
  Activity,
  AlertCircle,
} from "lucide-react";
import { format } from "date-fns";

interface MarketIndicator {
  name: string;
  ticker: string;
  value: number;
  change: number;
  raw_change: number;
}

export function Header() {
  const [currentTime, setCurrentTime] = useState<Date | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [marketIndicators, setMarketIndicators] = useState<MarketIndicator[]>([
    { name: "S&P 500", ticker: "^GSPC", value: 0, change: 0, raw_change: 0 },
    { name: "10Y UST", ticker: "^TNX", value: 0, change: 0, raw_change: 0 },
    { name: "VIX", ticker: "^VIX", value: 0, change: 0, raw_change: 0 },
  ]);
  const [marketDataError, setMarketDataError] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  const fetchMarketData = useCallback(async () => {
    try {
      const response = await fetch("http://localhost:8000/api/v1/market/header-data");
      if (!response.ok) throw new Error("Failed to fetch market data");
      
      const data = await response.json();
      if (data.indicators && data.indicators.length > 0) {
        setMarketIndicators(data.indicators);
        setLastUpdate(new Date(data.timestamp));
        setMarketDataError(false);
      }
    } catch (error) {
      console.error("Error fetching market data:", error);
      setMarketDataError(true);
    }
  }, []);

  useEffect(() => {
    setMounted(true);
    setCurrentTime(new Date());
    
    // Fetch market data immediately
    fetchMarketData();
    
    // Update time every second
    const timeTimer = setInterval(() => setCurrentTime(new Date()), 1000);
    
    // Refresh market data every 2 minutes to respect API rate limits
    const marketTimer = setInterval(fetchMarketData, 120000);
    
    return () => {
      clearInterval(timeTimer);
      clearInterval(marketTimer);
    };
  }, [fetchMarketData]);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await fetchMarketData();
    setTimeout(() => setIsRefreshing(false), 500);
  };

  return (
    <header className="h-16 border-b border-dark-800 bg-dark-900/30 backdrop-blur-xl px-6 flex items-center justify-between">
      {/* Search */}
      <div className="flex items-center gap-4 flex-1">
        <div className="relative w-96">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-dark-500" />
          <input
            type="text"
            placeholder="Search portfolios, securities, analytics..."
            className="w-full pl-10 pr-4 py-2 bg-dark-800/50 border border-dark-700 rounded-lg text-sm text-dark-100 placeholder:text-dark-500 focus:outline-none focus:border-accent-cyan/50 transition-colors"
          />
          <kbd className="absolute right-3 top-1/2 -translate-y-1/2 px-1.5 py-0.5 text-2xs text-dark-500 bg-dark-700/50 rounded border border-dark-600">
            ⌘K
          </kbd>
        </div>
      </div>

      {/* Market indicators */}
      <div className="flex items-center gap-6 mr-8">
        {marketDataError && (
          <div className="flex items-center gap-1 text-xs text-accent-amber">
            <AlertCircle className="w-3 h-3" />
            <span>Market data unavailable</span>
          </div>
        )}
        {marketIndicators.map((indicator) => (
          <div key={indicator.name} className="flex items-center gap-2">
            <span className="text-xs text-dark-500">{indicator.name}</span>
            <span className="text-sm font-mono font-medium text-dark-100">
              {indicator.value === 0 ? (
                <span className="text-dark-500">--</span>
              ) : indicator.name === "10Y UST" ? (
                `${indicator.value.toFixed(2)}%`
              ) : (
                indicator.value.toLocaleString(undefined, {
                  minimumFractionDigits: 2,
                  maximumFractionDigits: 2,
                })
              )}
            </span>
            {indicator.value !== 0 && (
              <div
                className={`flex items-center gap-0.5 text-xs ${
                  indicator.change >= 0 ? "text-accent-emerald" : "text-accent-rose"
                }`}
              >
                {indicator.change >= 0 ? (
                  <TrendingUp className="w-3 h-3" />
                ) : (
                  <TrendingDown className="w-3 h-3" />
                )}
                <span>
                  {indicator.change >= 0 ? "+" : ""}
                  {indicator.change.toFixed(2)}%
                </span>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Right section */}
      <div className="flex items-center gap-4">
        {/* Live indicator */}
        <div className="flex items-center gap-2 px-3 py-1.5 bg-accent-emerald/10 border border-accent-emerald/20 rounded-full">
          <motion.div
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="w-2 h-2 bg-accent-emerald rounded-full shadow-[0_0_8px_rgba(0,255,136,0.5)]"
          />
          <span className="text-xs font-medium text-accent-emerald">LIVE</span>
        </div>

        {/* Time */}
        <div className="flex items-center gap-2 text-dark-400">
          <Clock className="w-4 h-4" />
          <span className="text-sm font-mono">
            {mounted && currentTime ? format(currentTime, "HH:mm:ss") : "--:--:--"}
          </span>
          <span className="text-xs text-dark-500">EST</span>
        </div>

        {/* Refresh */}
        <button
          onClick={handleRefresh}
          className="p-2 text-dark-400 hover:text-dark-100 hover:bg-dark-800 rounded-lg transition-colors"
        >
          <RefreshCw
            className={`w-4 h-4 ${isRefreshing ? "animate-spin" : ""}`}
          />
        </button>

        {/* Notifications */}
        <button className="relative p-2 text-dark-400 hover:text-dark-100 hover:bg-dark-800 rounded-lg transition-colors">
          <Bell className="w-4 h-4" />
          <span className="absolute top-1 right-1 w-2 h-2 bg-accent-rose rounded-full" />
        </button>

        {/* Status */}
        <div className="flex items-center gap-2 px-3 py-1.5 bg-dark-800/50 rounded-lg">
          <Activity className="w-4 h-4 text-accent-cyan" />
          <span className="text-xs text-dark-400">System Healthy</span>
        </div>
      </div>
    </header>
  );
}
