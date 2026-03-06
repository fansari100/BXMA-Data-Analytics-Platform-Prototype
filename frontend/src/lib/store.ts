/**
 * BXMA Global State Store
 * 
 * Using Zustand for lightweight, performant state management.
 */

import { create } from "zustand";
import { devtools, persist } from "zustand/middleware";

// Types
interface Portfolio {
  id: string;
  name: string;
  nav: number;
  dailyReturn: number;
  ytdReturn: number;
  weights: Record<string, number>;
}

interface RiskMetrics {
  var95: number;
  cvar95: number;
  volatility: number;
  sharpeRatio: number;
  maxDrawdown: number;
  beta: number;
  trackingError: number;
}

interface User {
  id: string;
  email: string;
  name: string;
  role: string;
  permissions: string[];
}

// Portfolio Store
interface PortfolioState {
  portfolios: Portfolio[];
  selectedPortfolio: Portfolio | null;
  isLoading: boolean;
  error: string | null;
  
  setPortfolios: (portfolios: Portfolio[]) => void;
  selectPortfolio: (portfolio: Portfolio) => void;
  updatePortfolio: (id: string, updates: Partial<Portfolio>) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

export const usePortfolioStore = create<PortfolioState>()(
  devtools(
    persist(
      (set) => ({
        portfolios: [],
        selectedPortfolio: null,
        isLoading: false,
        error: null,
        
        setPortfolios: (portfolios) => set({ portfolios }),
        
        selectPortfolio: (portfolio) => set({ selectedPortfolio: portfolio }),
        
        updatePortfolio: (id, updates) =>
          set((state) => ({
            portfolios: state.portfolios.map((p) =>
              p.id === id ? { ...p, ...updates } : p
            ),
            selectedPortfolio:
              state.selectedPortfolio?.id === id
                ? { ...state.selectedPortfolio, ...updates }
                : state.selectedPortfolio,
          })),
        
        setLoading: (loading) => set({ isLoading: loading }),
        setError: (error) => set({ error }),
      }),
      {
        name: "bxma-portfolio-store",
      }
    )
  )
);

// Risk Store
interface RiskState {
  metrics: RiskMetrics | null;
  historicalVaR: { date: string; var: number; actualReturn: number }[];
  factorExposures: { factor: string; exposure: number }[];
  isCalculating: boolean;
  
  setMetrics: (metrics: RiskMetrics) => void;
  setHistoricalVaR: (data: RiskState["historicalVaR"]) => void;
  setFactorExposures: (exposures: RiskState["factorExposures"]) => void;
  setCalculating: (calculating: boolean) => void;
}

export const useRiskStore = create<RiskState>()(
  devtools((set) => ({
    metrics: null,
    historicalVaR: [],
    factorExposures: [],
    isCalculating: false,
    
    setMetrics: (metrics) => set({ metrics }),
    setHistoricalVaR: (data) => set({ historicalVaR: data }),
    setFactorExposures: (exposures) => set({ factorExposures: exposures }),
    setCalculating: (calculating) => set({ isCalculating: calculating }),
  }))
);

// User Store
interface UserState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  
  setUser: (user: User) => void;
  logout: () => void;
  setLoading: (loading: boolean) => void;
}

export const useUserStore = create<UserState>()(
  devtools(
    persist(
      (set) => ({
        user: null,
        isAuthenticated: false,
        isLoading: true,
        
        setUser: (user) => set({ user, isAuthenticated: true, isLoading: false }),
        logout: () => set({ user: null, isAuthenticated: false }),
        setLoading: (loading) => set({ isLoading: loading }),
      }),
      {
        name: "bxma-user-store",
      }
    )
  )
);

// UI Store
interface UIState {
  sidebarCollapsed: boolean;
  theme: "light" | "dark";
  notifications: { id: string; type: string; message: string; time: Date }[];
  
  toggleSidebar: () => void;
  setTheme: (theme: "light" | "dark") => void;
  addNotification: (notification: Omit<UIState["notifications"][0], "id" | "time">) => void;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
}

export const useUIStore = create<UIState>()(
  devtools(
    persist(
      (set) => ({
        sidebarCollapsed: false,
        theme: "dark",
        notifications: [],
        
        toggleSidebar: () =>
          set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
        
        setTheme: (theme) => set({ theme }),
        
        addNotification: (notification) =>
          set((state) => ({
            notifications: [
              ...state.notifications,
              {
                ...notification,
                id: Math.random().toString(36).substring(7),
                time: new Date(),
              },
            ],
          })),
        
        removeNotification: (id) =>
          set((state) => ({
            notifications: state.notifications.filter((n) => n.id !== id),
          })),
        
        clearNotifications: () => set({ notifications: [] }),
      }),
      {
        name: "bxma-ui-store",
      }
    )
  )
);

// Real-time data store
interface RealtimeState {
  connected: boolean;
  lastUpdate: Date | null;
  streamData: {
    nav: number;
    dailyReturn: number;
    var95: number;
    cvar95: number;
    volatility: number;
    sharpeRatio: number;
    factorExposures: Record<string, number>;
  } | null;
  
  setConnected: (connected: boolean) => void;
  updateStreamData: (data: RealtimeState["streamData"]) => void;
}

export const useRealtimeStore = create<RealtimeState>()(
  devtools((set) => ({
    connected: false,
    lastUpdate: null,
    streamData: null,
    
    setConnected: (connected) => set({ connected }),
    updateStreamData: (data) => set({ streamData: data, lastUpdate: new Date() }),
  }))
);
