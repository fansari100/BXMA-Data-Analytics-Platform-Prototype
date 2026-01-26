"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Sidebar } from "@/components/layout/sidebar";
import { Header } from "@/components/layout/header";
import { DashboardPage } from "@/components/pages/dashboard";
import { RiskPage } from "@/components/pages/risk";
import { RiskMetricsPage } from "@/components/pages/riskmetrics";
import { OptimizationPage } from "@/components/pages/optimization";
import { AttributionPage } from "@/components/pages/attribution";
import { StressTestPage } from "@/components/pages/stress-test";
import { VolatilityPage } from "@/components/pages/volatility";
import { RegimePage } from "@/components/pages/regime";
import { ContagionPage } from "@/components/pages/contagion";
import { AgentsPage } from "@/components/pages/agents";
import { CollaborationPage } from "@/components/pages/collaboration";
import { TableauPage } from "@/components/pages/tableau";
import { MacroDataPage } from "@/components/pages/macro";
import { CalculationsPage } from "@/components/pages/calculations";
import { LivePortfolioPage } from "@/components/pages/live-portfolio";

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

export default function Home() {
  const [currentPage, setCurrentPage] = useState<PageType>("dashboard");
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    setIsLoaded(true);
  }, []);

  const pageComponents: Record<PageType, JSX.Element> = {
    dashboard: <DashboardPage />,
    "live-portfolio": <LivePortfolioPage />,
    riskmetrics: <RiskMetricsPage />,
    risk: <RiskPage />,
    optimization: <OptimizationPage />,
    attribution: <AttributionPage />,
    "stress-test": <StressTestPage />,
    volatility: <VolatilityPage />,
    regime: <RegimePage />,
    contagion: <ContagionPage />,
    agents: <AgentsPage />,
    collaboration: <CollaborationPage />,
    tableau: <TableauPage />,
    macro: <MacroDataPage />,
    calculations: <CalculationsPage />,
  };

  return (
    <div className="flex h-screen bg-dark-950 bg-grid overflow-hidden">
      {/* Ambient background effects */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-40 -left-40 w-96 h-96 bg-accent-cyan/5 rounded-full blur-3xl" />
        <div className="absolute top-1/3 -right-40 w-96 h-96 bg-accent-emerald/5 rounded-full blur-3xl" />
        <div className="absolute -bottom-40 left-1/3 w-96 h-96 bg-accent-violet/5 rounded-full blur-3xl" />
      </div>

      {/* Sidebar */}
      <Sidebar currentPage={currentPage} onNavigate={setCurrentPage} />

      {/* Main content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        <Header />
        
        <div className="flex-1 overflow-auto p-6">
          <AnimatePresence mode="wait">
            {isLoaded && (
              <motion.div
                key={currentPage}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
                className="h-full"
              >
                {pageComponents[currentPage]}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </main>
    </div>
  );
}
