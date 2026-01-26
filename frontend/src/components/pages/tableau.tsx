"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Download,
  Upload,
  RefreshCw,
  FileSpreadsheet,
  Database,
  BarChart3,
  CheckCircle,
  Clock,
  AlertCircle,
} from "lucide-react";

interface ExportConfig {
  id: string;
  name: string;
  description: string;
  format: "csv" | "hyper" | "json";
  schedule: string;
  lastExport: string | null;
  status: "ready" | "pending" | "error";
  rowCount: number;
}

interface PublishedDataSource {
  id: string;
  name: string;
  project: string;
  lastRefresh: string;
  status: "active" | "stale" | "error";
  subscriptions: number;
}

const exportConfigs: ExportConfig[] = [
  {
    id: "daily_risk",
    name: "Daily Risk Report",
    description: "VaR, CVaR, volatility, and factor exposures for all portfolios",
    format: "csv",
    schedule: "Daily @ 06:00 ET",
    lastExport: "2026-01-25T06:00:00Z",
    status: "ready",
    rowCount: 2450,
  },
  {
    id: "attribution",
    name: "Performance Attribution",
    description: "Brinson-Fachler attribution breakdown by sector and asset",
    format: "hyper",
    schedule: "Daily @ 07:00 ET",
    lastExport: "2026-01-25T07:00:00Z",
    status: "ready",
    rowCount: 15680,
  },
  {
    id: "factor_analysis",
    name: "Factor Model Analysis",
    description: "Factor loadings, contributions, and covariance data",
    format: "hyper",
    schedule: "Weekly",
    lastExport: "2026-01-20T08:00:00Z",
    status: "ready",
    rowCount: 45200,
  },
  {
    id: "var_history",
    name: "VaR History",
    description: "Historical VaR calculations with backtesting results",
    format: "csv",
    schedule: "On-demand",
    lastExport: null,
    status: "pending",
    rowCount: 0,
  },
];

const publishedDataSources: PublishedDataSource[] = [
  {
    id: "ds_risk_dashboard",
    name: "BXMA Risk Dashboard",
    project: "BXMA Risk Analytics",
    lastRefresh: "2026-01-25T06:30:00Z",
    status: "active",
    subscriptions: 12,
  },
  {
    id: "ds_portfolio_perf",
    name: "Portfolio Performance",
    project: "BXMA Risk Analytics",
    lastRefresh: "2026-01-25T07:15:00Z",
    status: "active",
    subscriptions: 8,
  },
  {
    id: "ds_factor_model",
    name: "Factor Model Data",
    project: "BXMA Risk Analytics",
    lastRefresh: "2026-01-20T08:30:00Z",
    status: "stale",
    subscriptions: 5,
  },
];

const statusConfig = {
  ready: { icon: CheckCircle, color: "text-emerald-400", bg: "bg-emerald-500/20" },
  pending: { icon: Clock, color: "text-amber-400", bg: "bg-amber-500/20" },
  error: { icon: AlertCircle, color: "text-rose-400", bg: "bg-rose-500/20" },
  active: { icon: CheckCircle, color: "text-emerald-400", bg: "bg-emerald-500/20" },
  stale: { icon: Clock, color: "text-amber-400", bg: "bg-amber-500/20" },
};

export function TableauPage() {
  const [exporting, setExporting] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState<string | null>(null);
  const [selectedFormat, setSelectedFormat] = useState<"csv" | "hyper" | "json">("csv");

  const handleExport = async (configId: string) => {
    setExporting(configId);
    // Simulate export
    await new Promise((resolve) => setTimeout(resolve, 2000));
    setExporting(null);
  };

  const handleRefresh = async (dsId: string) => {
    setRefreshing(dsId);
    // Simulate refresh
    await new Promise((resolve) => setTimeout(resolve, 2000));
    setRefreshing(null);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="space-y-6"
    >
      <div>
        <h1 className="text-3xl font-bold text-dark-100">Tableau Integration</h1>
        <p className="text-dark-400 mt-1">
          Export data and manage Tableau Server data sources for risk analytics visualization
        </p>
      </div>

      {/* Quick Export */}
      <Card className="bg-dark-800/50 border-dark-700">
        <CardHeader>
          <CardTitle className="text-dark-100 flex items-center gap-2">
            <Download className="w-5 h-5 text-accent-cyan" />
            Quick Export
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <Label className="text-dark-300">Data Type</Label>
              <select className="w-full bg-dark-900 border border-dark-600 text-dark-100 rounded-md p-2 mt-1">
                <option value="risk_metrics">Risk Metrics</option>
                <option value="portfolio_holdings">Portfolio Holdings</option>
                <option value="attribution">Attribution</option>
                <option value="factor_exposures">Factor Exposures</option>
                <option value="var_history">VaR History</option>
              </select>
            </div>
            <div>
              <Label className="text-dark-300">Format</Label>
              <div className="flex gap-2 mt-1">
                {(["csv", "hyper", "json"] as const).map((format) => (
                  <Button
                    key={format}
                    variant={selectedFormat === format ? "default" : "outline"}
                    size="sm"
                    onClick={() => setSelectedFormat(format)}
                    className={selectedFormat === format 
                      ? "bg-accent-cyan text-dark-950" 
                      : "border-dark-600 text-dark-300"
                    }
                  >
                    {format.toUpperCase()}
                  </Button>
                ))}
              </div>
            </div>
            <div className="flex items-end">
              <Button className="w-full bg-accent-emerald hover:bg-accent-emerald/80 text-dark-950">
                <Download className="w-4 h-4 mr-2" />
                Export Now
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Export Configurations */}
        <Card className="bg-dark-800/50 border-dark-700">
          <CardHeader>
            <CardTitle className="text-dark-100 flex items-center gap-2">
              <FileSpreadsheet className="w-5 h-5 text-accent-violet" />
              Export Configurations
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {exportConfigs.map((config) => {
              const StatusIcon = statusConfig[config.status].icon;
              return (
                <motion.div
                  key={config.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="bg-dark-700/50 rounded-lg p-4"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-dark-100 font-medium">{config.name}</span>
                        <span className={`px-2 py-0.5 rounded text-xs ${statusConfig[config.status].bg} ${statusConfig[config.status].color}`}>
                          {config.status}
                        </span>
                        <span className="px-2 py-0.5 rounded text-xs bg-dark-600 text-dark-300">
                          {config.format.toUpperCase()}
                        </span>
                      </div>
                      <p className="text-dark-400 text-sm mt-1">{config.description}</p>
                      <div className="flex items-center gap-4 mt-2 text-xs text-dark-500">
                        <span>Schedule: {config.schedule}</span>
                        {config.lastExport && (
                          <span>Last: {new Date(config.lastExport).toLocaleString()}</span>
                        )}
                        {config.rowCount > 0 && (
                          <span>{config.rowCount.toLocaleString()} rows</span>
                        )}
                      </div>
                    </div>
                    <Button
                      size="sm"
                      className="bg-accent-cyan/20 text-accent-cyan hover:bg-accent-cyan/30"
                      onClick={() => handleExport(config.id)}
                      disabled={exporting === config.id}
                    >
                      {exporting === config.id ? (
                        <RefreshCw className="w-4 h-4 animate-spin" />
                      ) : (
                        <Download className="w-4 h-4" />
                      )}
                    </Button>
                  </div>
                </motion.div>
              );
            })}
          </CardContent>
        </Card>

        {/* Published Data Sources */}
        <Card className="bg-dark-800/50 border-dark-700">
          <CardHeader>
            <CardTitle className="text-dark-100 flex items-center gap-2">
              <Database className="w-5 h-5 text-accent-amber" />
              Published Data Sources
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {publishedDataSources.map((ds) => {
              const StatusIcon = statusConfig[ds.status].icon;
              return (
                <motion.div
                  key={ds.id}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="bg-dark-700/50 rounded-lg p-4"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-dark-100 font-medium">{ds.name}</span>
                        <span className={`px-2 py-0.5 rounded text-xs ${statusConfig[ds.status].bg} ${statusConfig[ds.status].color}`}>
                          {ds.status}
                        </span>
                      </div>
                      <p className="text-dark-400 text-sm mt-1">Project: {ds.project}</p>
                      <div className="flex items-center gap-4 mt-2 text-xs text-dark-500">
                        <span>Last refresh: {new Date(ds.lastRefresh).toLocaleString()}</span>
                        <span>{ds.subscriptions} subscribers</span>
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <Button
                        size="sm"
                        className="bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30"
                        onClick={() => handleRefresh(ds.id)}
                        disabled={refreshing === ds.id}
                      >
                        {refreshing === ds.id ? (
                          <RefreshCw className="w-4 h-4 animate-spin" />
                        ) : (
                          <RefreshCw className="w-4 h-4" />
                        )}
                      </Button>
                      <Button
                        size="sm"
                        className="bg-violet-500/20 text-violet-400 hover:bg-violet-500/30"
                      >
                        <Upload className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </CardContent>
        </Card>
      </div>

      {/* Dashboard Embed Preview */}
      <Card className="bg-dark-800/50 border-dark-700">
        <CardHeader>
          <CardTitle className="text-dark-100 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-accent-rose" />
            Tableau Dashboard Embed
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="bg-dark-900 rounded-lg p-8 text-center border-2 border-dashed border-dark-600">
            <BarChart3 className="w-16 h-16 text-dark-500 mx-auto mb-4" />
            <p className="text-dark-300 font-medium">Tableau Dashboard Embed Area</p>
            <p className="text-dark-500 text-sm mt-2">
              Configure Tableau Server URL to embed live dashboards
            </p>
            <div className="flex items-center gap-2 justify-center mt-4">
              <Input
                placeholder="https://tableau-server.blackstone.com"
                className="bg-dark-800 border-dark-600 text-dark-100 max-w-md"
              />
              <Button className="bg-accent-cyan text-dark-950">
                Connect
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Export Format Info */}
      <Card className="bg-dark-800/50 border-dark-700">
        <CardHeader>
          <CardTitle className="text-dark-100">Export Format Guide</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-dark-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <FileSpreadsheet className="w-5 h-5 text-emerald-400" />
                <span className="text-dark-100 font-medium">CSV</span>
              </div>
              <p className="text-dark-400 text-sm">
                Standard comma-separated format. Compatible with any tool. Best for small to medium datasets.
              </p>
            </div>
            <div className="bg-dark-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Database className="w-5 h-5 text-violet-400" />
                <span className="text-dark-100 font-medium">Hyper</span>
              </div>
              <p className="text-dark-400 text-sm">
                Tableau's native extract format. Optimized performance for large datasets and complex analytics.
              </p>
            </div>
            <div className="bg-dark-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <FileSpreadsheet className="w-5 h-5 text-amber-400" />
                <span className="text-dark-100 font-medium">JSON</span>
              </div>
              <p className="text-dark-400 text-sm">
                For Tableau Web Data Connector integration. Ideal for real-time data feeds and API connections.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
