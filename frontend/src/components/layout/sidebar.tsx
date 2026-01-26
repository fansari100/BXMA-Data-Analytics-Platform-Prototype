"use client";

import { motion } from "framer-motion";
import {
  LayoutDashboard,
  Shield,
  Target,
  BarChart3,
  Zap,
  Settings,
  Users,
  HelpCircle,
  ChevronLeft,
  ChevronRight,
  TrendingUp,
  Activity,
  Network,
  Brain,
  Database,
  UsersRound,
  AreaChart,
  LineChart,
  Briefcase,
  Calculator,
} from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";

type PageType = 
  | "dashboard" 
  | "live-portfolio"
  | "risk" 
  | "riskmetrics" 
  | "optimization" 
  | "attribution" 
  | "stress-test" 
  | "volatility" 
  | "regime" 
  | "contagion" 
  | "agents" 
  | "collaboration" 
  | "tableau" 
  | "macro"
  | "calculations";

interface SidebarProps {
  currentPage: PageType;
  onNavigate: (page: PageType) => void;
}

const navigationItems = [
  { id: "dashboard" as PageType, label: "Dashboard", icon: LayoutDashboard },
  { id: "live-portfolio" as PageType, label: "Live Portfolio", icon: Briefcase },
  { id: "calculations" as PageType, label: "Calculations", icon: Calculator },
  { id: "riskmetrics" as PageType, label: "RiskMetrics", icon: Database },
  { id: "risk" as PageType, label: "Risk Analytics", icon: Shield },
  { id: "optimization" as PageType, label: "Optimization", icon: Target },
  { id: "attribution" as PageType, label: "Attribution", icon: BarChart3 },
  { id: "stress-test" as PageType, label: "Stress Testing", icon: Zap },
  { id: "volatility" as PageType, label: "Volatility Surface", icon: TrendingUp },
  { id: "regime" as PageType, label: "Regime Detection", icon: Activity },
  { id: "contagion" as PageType, label: "Contagion Analysis", icon: Network },
  { id: "agents" as PageType, label: "Agentic AI", icon: Brain },
  { id: "macro" as PageType, label: "Macro Data", icon: LineChart },
  { id: "collaboration" as PageType, label: "Collaboration", icon: UsersRound },
  { id: "tableau" as PageType, label: "Tableau", icon: AreaChart },
];

const bottomItems = [
  { id: "settings", label: "Settings", icon: Settings },
  { id: "team", label: "Team", icon: Users },
  { id: "help", label: "Help", icon: HelpCircle },
];

export function Sidebar({ currentPage, onNavigate }: SidebarProps) {
  const [isCollapsed, setIsCollapsed] = useState(false);

  return (
    <motion.aside
      initial={false}
      animate={{ width: isCollapsed ? 80 : 280 }}
      transition={{ duration: 0.3, ease: "easeInOut" }}
      className="relative h-full bg-dark-900/50 border-r border-dark-800 backdrop-blur-xl flex flex-col"
    >
      {/* Logo */}
      <div className="h-16 flex items-center px-6 border-b border-dark-800">
        <motion.div
          initial={false}
          animate={{ opacity: 1 }}
          className="flex items-center gap-3"
        >
          <div className="w-10 h-10 bg-gradient-to-br from-accent-cyan to-accent-emerald rounded-xl flex items-center justify-center shadow-glow-sm">
            <span className="text-dark-950 font-bold text-lg">BX</span>
          </div>
          {!isCollapsed && (
            <motion.div
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -10 }}
              transition={{ duration: 0.2 }}
            >
              <h1 className="font-display font-bold text-lg text-dark-100">BXMA</h1>
              <p className="text-2xs text-dark-500">Risk / Quant Platform</p>
            </motion.div>
          )}
        </motion.div>
      </div>

      {/* Collapse toggle */}
      <button
        onClick={() => setIsCollapsed(!isCollapsed)}
        className="absolute -right-3 top-20 w-6 h-6 bg-dark-800 border border-dark-700 rounded-full flex items-center justify-center text-dark-400 hover:text-dark-100 hover:bg-dark-700 transition-colors z-10"
      >
        {isCollapsed ? (
          <ChevronRight className="w-3.5 h-3.5" />
        ) : (
          <ChevronLeft className="w-3.5 h-3.5" />
        )}
      </button>

      {/* Navigation */}
      <nav className="flex-1 py-6 px-3 space-y-1 overflow-y-auto">
        {!isCollapsed && (
          <p className="px-3 text-2xs font-semibold uppercase tracking-wider text-dark-500 mb-4">
            Analytics
          </p>
        )}

        {navigationItems.map((item) => {
          const isActive = currentPage === item.id;
          const Icon = item.icon;

          return (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className={cn(
                "w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200",
                isActive
                  ? "bg-accent-cyan/10 text-accent-cyan"
                  : "text-dark-400 hover:text-dark-100 hover:bg-dark-800/50"
              )}
            >
              <Icon className={cn("w-5 h-5 flex-shrink-0", isActive && "drop-shadow-[0_0_8px_rgba(0,212,255,0.5)]")} />
              {!isCollapsed && (
                <motion.span
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="text-sm font-medium"
                >
                  {item.label}
                </motion.span>
              )}
              {isActive && !isCollapsed && (
                <motion.div
                  layoutId="activeIndicator"
                  className="ml-auto w-1.5 h-1.5 bg-accent-cyan rounded-full shadow-glow-sm"
                />
              )}
            </button>
          );
        })}
      </nav>

      {/* Bottom section */}
      <div className="py-4 px-3 border-t border-dark-800 space-y-1">
        {bottomItems.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-dark-400 hover:text-dark-100 hover:bg-dark-800/50 transition-all duration-200"
            >
              <Icon className="w-5 h-5 flex-shrink-0" />
              {!isCollapsed && (
                <span className="text-sm font-medium">{item.label}</span>
              )}
            </button>
          );
        })}
      </div>

      {/* User section */}
      {!isCollapsed && (
        <div className="p-4 border-t border-dark-800">
          <div className="flex items-center gap-3 px-3 py-2 rounded-lg bg-dark-800/30">
            <div className="w-9 h-9 bg-gradient-to-br from-accent-violet to-accent-rose rounded-full flex items-center justify-center">
              <span className="text-white font-semibold text-sm">RA</span>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-dark-100 truncate">Ricky Ansari</p>
              <p className="text-2xs text-dark-500 truncate">Risk Analyst</p>
            </div>
          </div>
        </div>
      )}
    </motion.aside>
  );
}
