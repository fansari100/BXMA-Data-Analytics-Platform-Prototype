/**
 * BXMA API Client
 * 
 * Type-safe API client with automatic token refresh and error handling.
 */

import axios, { AxiosInstance, AxiosError, InternalAxiosRequestConfig } from "axios";

// Types
export interface VaRRequest {
  portfolio: {
    weights: number[];
    asset_ids?: string[];
  };
  returns: {
    returns: number[][];
    asset_ids?: string[];
    dates?: string[];
  };
  confidence_level?: number;
  horizon_days?: number;
  method?: "parametric" | "historical" | "monte_carlo" | "cornish_fisher";
}

export interface VaRResponse {
  var: number;
  cvar: number | null;
  confidence_level: number;
  horizon_days: number;
  method: string;
  component_var: number[] | null;
  solve_time_ms: number;
}

export interface OptimizationRequest {
  expected_returns: number[];
  covariance: number[][];
  method?: "mean_variance" | "min_variance" | "max_sharpe" | "risk_parity" | "hrp";
  risk_aversion?: number;
  constraints?: Record<string, any>;
}

export interface OptimizationResponse {
  weights: number[];
  expected_return: number;
  expected_risk: number;
  sharpe_ratio: number;
  status: string;
  solve_time_ms: number;
  risk_contributions: number[] | null;
}

export interface AttributionRequest {
  portfolio_weights: number[];
  benchmark_weights: number[];
  portfolio_returns: number[];
  benchmark_returns: number[];
  segment_names?: string[];
}

export interface AttributionResponse {
  portfolio_return: number;
  benchmark_return: number;
  active_return: number;
  allocation_effect: number;
  selection_effect: number;
  interaction_effect: number;
  segment_attribution: Record<string, {
    allocation: number;
    selection: number;
    interaction: number;
  }>;
}

export interface StressTestRequest {
  weights: number[];
  scenario_name: string;
  factor_shocks?: Record<string, number>;
}

export interface StressTestResponse {
  scenario_name: string;
  portfolio_return: number;
  position_impacts: Record<string, number>;
}

export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  expires_in: number;
}

// API Client class
class BXMAApiClient {
  private client: AxiosInstance;
  private baseURL: string;
  private accessToken: string | null = null;
  private refreshToken: string | null = null;

  constructor() {
    this.baseURL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    
    this.client = axios.create({
      baseURL: this.baseURL,
      timeout: 30000,
      headers: {
        "Content-Type": "application/json",
      },
    });

    // Request interceptor for auth
    this.client.interceptors.request.use(
      (config: InternalAxiosRequestConfig) => {
        if (this.accessToken) {
          config.headers.Authorization = `Bearer ${this.accessToken}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor for token refresh
    this.client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        const originalRequest = error.config as InternalAxiosRequestConfig & { _retry?: boolean };
        
        if (error.response?.status === 401 && !originalRequest._retry && this.refreshToken) {
          originalRequest._retry = true;
          
          try {
            const tokens = await this.refreshTokens();
            this.setTokens(tokens);
            originalRequest.headers.Authorization = `Bearer ${tokens.access_token}`;
            return this.client(originalRequest);
          } catch (refreshError) {
            this.clearTokens();
            throw refreshError;
          }
        }
        
        return Promise.reject(error);
      }
    );
  }

  // Token management
  setTokens(tokens: AuthTokens) {
    this.accessToken = tokens.access_token;
    this.refreshToken = tokens.refresh_token;
    
    if (typeof window !== "undefined") {
      localStorage.setItem("bxma_access_token", tokens.access_token);
      localStorage.setItem("bxma_refresh_token", tokens.refresh_token);
    }
  }

  loadTokens() {
    if (typeof window !== "undefined") {
      this.accessToken = localStorage.getItem("bxma_access_token");
      this.refreshToken = localStorage.getItem("bxma_refresh_token");
    }
  }

  clearTokens() {
    this.accessToken = null;
    this.refreshToken = null;
    
    if (typeof window !== "undefined") {
      localStorage.removeItem("bxma_access_token");
      localStorage.removeItem("bxma_refresh_token");
    }
  }

  private async refreshTokens(): Promise<AuthTokens> {
    const response = await axios.post(`${this.baseURL}/api/v1/auth/refresh`, {
      refresh_token: this.refreshToken,
    });
    return response.data;
  }

  // Health check
  async health() {
    const response = await this.client.get("/health");
    return response.data;
  }

  // Risk endpoints
  async calculateVaR(request: VaRRequest): Promise<VaRResponse> {
    const response = await this.client.post("/api/v1/risk/var", request);
    return response.data;
  }

  async fitFactorModel(returns: number[][], nFactors: number = 5, method: string = "pca") {
    const response = await this.client.post("/api/v1/risk/factor-model", {
      returns,
      n_factors: nFactors,
      method,
    });
    return response.data;
  }

  async estimateCovariance(returns: number[][], method: string = "ledoit_wolf", halflife: number = 63) {
    const response = await this.client.post("/api/v1/risk/covariance", null, {
      params: { method, halflife },
      data: returns,
    });
    return response.data;
  }

  // Optimization endpoints
  async optimizePortfolio(request: OptimizationRequest): Promise<OptimizationResponse> {
    const response = await this.client.post("/api/v1/optimize", request);
    return response.data;
  }

  async getEfficientFrontier(expectedReturns: number[], covariance: number[][], nPoints: number = 50) {
    const response = await this.client.get("/api/v1/optimize/efficient-frontier", {
      params: {
        expected_returns: expectedReturns.join(","),
        covariance: JSON.stringify(covariance),
        n_points: nPoints,
      },
    });
    return response.data;
  }

  // Attribution endpoints
  async calculateAttribution(request: AttributionRequest): Promise<AttributionResponse> {
    const response = await this.client.post("/api/v1/attribution/brinson", request);
    return response.data;
  }

  // Stress testing endpoints
  async runStressTest(request: StressTestRequest): Promise<StressTestResponse> {
    const response = await this.client.post("/api/v1/stress-test", request);
    return response.data;
  }

  async getStandardScenarios() {
    const response = await this.client.get("/api/v1/stress-test/scenarios");
    return response.data;
  }

  // WebSocket connection for real-time data
  createWebSocket(path: string = "/ws/risk-stream"): WebSocket {
    const wsURL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";
    return new WebSocket(`${wsURL}${path}`);
  }
}

// Singleton instance
export const api = new BXMAApiClient();

// React hooks for API calls
export function useApi() {
  return api;
}
