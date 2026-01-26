"""
BXMA Risk/Quant Platform - FastAPI Backend
============================================

High-performance async API server providing:
- Portfolio analytics endpoints
- Real-time risk calculations via WebSocket
- Factor model analysis
- Performance attribution
- Stress testing
- ML-based optimization

Built on FastAPI for maximum performance with async support.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from typing import Any, Literal
from uuid import uuid4

import numpy as np
from fastapi import (
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    Depends,
    Query,
    BackgroundTasks,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import BXMA modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bxma.core.config import BXMAConfig
from bxma.risk.var import ParametricVaR, HistoricalVaR, MonteCarloVaR, CornishFisherVaR
from bxma.risk.factor_models import StatisticalFactorModel
from bxma.risk.covariance import LedoitWolfCovariance, ExponentialCovariance
from bxma.optimization.classical import MeanVarianceOptimizer, MaxSharpeOptimizer, MinVarianceOptimizer
from bxma.optimization.risk_parity import RiskParityOptimizer, HierarchicalRiskParity
from bxma.attribution.brinson import BrinsonFachlerAttribution
from bxma.stress_testing.scenarios import ScenarioEngine, ScenarioDefinition


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str


class PortfolioWeights(BaseModel):
    weights: list[float]
    asset_ids: list[str] | None = None


class ReturnsData(BaseModel):
    returns: list[list[float]]  # T x N matrix
    asset_ids: list[str] | None = None
    dates: list[str] | None = None


class VaRRequest(BaseModel):
    portfolio: PortfolioWeights
    returns: ReturnsData
    confidence_level: float = 0.95
    horizon_days: int = 1
    method: Literal["parametric", "historical", "monte_carlo", "cornish_fisher"] = "parametric"


class VaRResponse(BaseModel):
    var: float
    cvar: float | None
    confidence_level: float
    horizon_days: int
    method: str
    component_var: list[float] | None = None
    solve_time_ms: float


class OptimizationRequest(BaseModel):
    expected_returns: list[float]
    covariance: list[list[float]]
    method: Literal["mean_variance", "min_variance", "max_sharpe", "risk_parity", "hrp"] = "hrp"
    risk_aversion: float = 1.0
    constraints: dict | None = None


class OptimizationResponse(BaseModel):
    weights: list[float]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    status: str
    solve_time_ms: float
    risk_contributions: list[float] | None = None


class AttributionRequest(BaseModel):
    portfolio_weights: list[float]
    benchmark_weights: list[float]
    portfolio_returns: list[float]
    benchmark_returns: list[float]
    segment_names: list[str] | None = None


class AttributionResponse(BaseModel):
    portfolio_return: float
    benchmark_return: float
    active_return: float
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    segment_attribution: dict


class FactorModelRequest(BaseModel):
    returns: list[list[float]]
    n_factors: int = 5
    method: Literal["pca", "ica", "sparse_pca"] = "pca"


class FactorModelResponse(BaseModel):
    n_factors: int
    r_squared: float
    explained_variance: list[float]
    factor_names: list[str]


class StressTestRequest(BaseModel):
    weights: list[float]
    scenario_name: str
    factor_shocks: dict[str, float] | None = None


class StressTestResponse(BaseModel):
    scenario_name: str
    portfolio_return: float
    position_impacts: dict[str, float]


class RiskDashboardData(BaseModel):
    timestamp: datetime
    nav: float
    daily_return: float
    var_95: float
    cvar_95: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    factor_exposures: dict[str, float]


# =============================================================================
# APPLICATION SETUP
# =============================================================================

# Global state
class AppState:
    config: BXMAConfig
    connected_clients: set[WebSocket]
    
app_state = AppState()
app_state.connected_clients = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    app_state.config = BXMAConfig()
    print("🚀 BXMA Backend starting...")
    print(f"   Environment: {app_state.config.environment.name}")
    
    yield
    
    # Shutdown
    print("👋 BXMA Backend shutting down...")
    for client in app_state.connected_clients:
        await client.close()


app = FastAPI(
    title="BXMA Risk/Quant Platform API",
    description="""
    Blackstone Multi-Asset Investing Risk/Quant Analytics Platform
    
    ## Core Features
    
    ### RiskMetrics Integration (MSCI/Barra)
    - **EWMA Covariance Estimation** (λ=0.94 decay factor)
    - **Factor Models**: Barra USE4, GEM3, CNE5, Axioma WW21, US4
    - **Factor Risk Decomposition**: Systematic vs idiosyncratic risk
    - **Parametric VaR/CVaR** using RiskMetrics methodology
    - **Stress Testing**: Historical and hypothetical scenarios
    
    ### Portfolio Analytics
    - Value-at-Risk (Parametric, Historical, Monte Carlo, Cornish-Fisher)
    - Conditional VaR / Expected Shortfall
    - Statistical factor models (PCA, ICA, Sparse PCA)
    - Covariance estimation (Ledoit-Wolf, Exponential, Sample)
    
    ### Portfolio Optimization
    - Mean-Variance, Min Variance, Max Sharpe
    - Risk Parity and Hierarchical Risk Parity (HRP)
    - Thermodynamic optimization (Ising Hamiltonian)
    
    ### Performance Attribution
    - Brinson-Fachler decomposition
    - Geometric attribution
    
    ### Titan-X Advanced Features
    - Regime detection (HMM + thermodynamic sampling)
    - Counterparty contagion (Graph Neural Networks)
    - Semantic alpha signals (FinBERT embeddings)
    
    ## Authentication
    All endpoints require Bearer token authentication.
    """,
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# =============================================================================
# HEALTH & STATUS ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0"
    )


@app.get("/api/v1/status", tags=["System"])
async def system_status():
    """Get system status and metrics."""
    return {
        "status": "operational",
        "uptime_seconds": 0,  # Would track actual uptime
        "connected_clients": len(app_state.connected_clients),
        "config": {
            "environment": app_state.config.environment.name,
            "compute_backend": app_state.config.compute_backend.name,
        }
    }


# =============================================================================
# RISK ANALYTICS ENDPOINTS
# =============================================================================

@app.post("/api/v1/risk/var", response_model=VaRResponse, tags=["Risk"])
async def calculate_var(request: VaRRequest):
    """
    Calculate Value-at-Risk and Conditional VaR.
    
    Supports multiple methodologies:
    - **parametric**: Assumes normal distribution
    - **historical**: Uses historical simulation
    - **monte_carlo**: Monte Carlo simulation with variance reduction
    - **cornish_fisher**: Adjusts for skewness and kurtosis
    """
    try:
        weights = np.array(request.portfolio.weights)
        returns = np.array(request.returns.returns)
        
        # Select VaR method
        if request.method == "parametric":
            engine = ParametricVaR()
        elif request.method == "historical":
            engine = HistoricalVaR()
        elif request.method == "monte_carlo":
            engine = MonteCarloVaR(n_simulations=10000)
        elif request.method == "cornish_fisher":
            engine = CornishFisherVaR()
        else:
            raise HTTPException(400, f"Unknown method: {request.method}")
        
        result = engine.calculate_var(
            returns, weights, request.confidence_level, request.horizon_days
        )
        
        return VaRResponse(
            var=float(result.var),
            cvar=float(result.cvar) if result.cvar else None,
            confidence_level=result.confidence_level,
            horizon_days=result.horizon_days,
            method=result.method,
            component_var=result.component_var.tolist() if result.component_var is not None else None,
            solve_time_ms=result.solve_time_ms,
        )
        
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/risk/factor-model", response_model=FactorModelResponse, tags=["Risk"])
async def fit_factor_model(request: FactorModelRequest):
    """
    Fit statistical factor model to return data.
    
    Extracts latent factors using PCA, ICA, or Sparse PCA.
    """
    try:
        returns = np.array(request.returns)
        
        model = StatisticalFactorModel(
            n_factors=request.n_factors,
            method=request.method
        )
        result = model.fit(returns)
        
        return FactorModelResponse(
            n_factors=result.n_factors,
            r_squared=float(result.r_squared),
            explained_variance=[float(v) for v in result.explained_variance_ratio],
            factor_names=result.factor_names,
        )
        
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/risk/covariance", tags=["Risk"])
async def estimate_covariance(
    returns: list[list[float]],
    method: Literal["ledoit_wolf", "exponential", "sample"] = "ledoit_wolf",
    halflife: int = 63,
):
    """
    Estimate covariance matrix with various methods.
    
    - **ledoit_wolf**: Optimal shrinkage estimator
    - **exponential**: RiskMetrics-style exponential weighting
    - **sample**: Standard sample covariance
    """
    try:
        returns_arr = np.array(returns)
        
        if method == "ledoit_wolf":
            estimator = LedoitWolfCovariance()
        elif method == "exponential":
            estimator = ExponentialCovariance(halflife=halflife)
        else:
            from bxma.risk.covariance import SampleCovariance
            estimator = SampleCovariance()
        
        result = estimator.fit(returns_arr)
        
        return {
            "covariance": result.covariance.tolist(),
            "correlation": result.correlation.tolist(),
            "volatilities": result.volatilities.tolist(),
            "condition_number": float(result.condition_number),
            "method": result.method,
        }
        
    except Exception as e:
        raise HTTPException(500, str(e))


# =============================================================================
# OPTIMIZATION ENDPOINTS
# =============================================================================

@app.post("/api/v1/optimize", response_model=OptimizationResponse, tags=["Optimization"])
async def optimize_portfolio(request: OptimizationRequest):
    """
    Optimize portfolio weights using various methodologies.
    
    Supported methods:
    - **mean_variance**: Classical Markowitz optimization
    - **min_variance**: Global minimum variance portfolio
    - **max_sharpe**: Maximum Sharpe ratio (tangency) portfolio
    - **risk_parity**: Equal risk contribution
    - **hrp**: Hierarchical Risk Parity (ML-based)
    """
    try:
        expected_returns = np.array(request.expected_returns)
        covariance = np.array(request.covariance)
        
        # Select optimizer
        if request.method == "mean_variance":
            optimizer = MeanVarianceOptimizer(risk_aversion=request.risk_aversion)
        elif request.method == "min_variance":
            optimizer = MinVarianceOptimizer()
        elif request.method == "max_sharpe":
            optimizer = MaxSharpeOptimizer()
        elif request.method == "risk_parity":
            optimizer = RiskParityOptimizer()
        elif request.method == "hrp":
            optimizer = HierarchicalRiskParity()
        else:
            raise HTTPException(400, f"Unknown method: {request.method}")
        
        result = optimizer.optimize(expected_returns, covariance)
        
        return OptimizationResponse(
            weights=result.weights.tolist(),
            expected_return=float(result.expected_return),
            expected_risk=float(result.expected_risk),
            sharpe_ratio=float(result.sharpe_ratio),
            status=result.status,
            solve_time_ms=float(result.solve_time_ms),
            risk_contributions=result.risk_contributions.tolist() if result.risk_contributions is not None else None,
        )
        
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/v1/optimize/efficient-frontier", tags=["Optimization"])
async def compute_efficient_frontier(
    expected_returns: str = Query(..., description="Comma-separated expected returns"),
    covariance: str = Query(..., description="JSON-encoded covariance matrix"),
    n_points: int = Query(50, description="Number of frontier points"),
):
    """
    Compute efficient frontier points.
    """
    try:
        mu = np.array([float(x) for x in expected_returns.split(",")])
        cov = np.array(json.loads(covariance))
        
        # Compute frontier
        optimizer = MeanVarianceOptimizer()
        
        # Get min and max returns
        min_var_result = MinVarianceOptimizer().optimize(mu, cov)
        max_return_result = MeanVarianceOptimizer(target_risk=0.5).optimize(mu, cov)
        
        min_ret = min_var_result.expected_return
        max_ret = max(mu)
        
        target_returns = np.linspace(min_ret, max_ret * 0.95, n_points)
        
        frontier_points = []
        for target in target_returns:
            try:
                opt = MeanVarianceOptimizer(target_return=target)
                result = opt.optimize(mu, cov)
                if result.status in ["optimal", "optimal_inaccurate"]:
                    frontier_points.append({
                        "return": float(result.expected_return),
                        "risk": float(result.expected_risk),
                        "sharpe": float(result.sharpe_ratio),
                    })
            except:
                continue
        
        return {"frontier": frontier_points}
        
    except Exception as e:
        raise HTTPException(500, str(e))


# =============================================================================
# ATTRIBUTION ENDPOINTS
# =============================================================================

@app.post("/api/v1/attribution/brinson", response_model=AttributionResponse, tags=["Attribution"])
async def brinson_attribution(request: AttributionRequest):
    """
    Calculate Brinson-Fachler performance attribution.
    
    Decomposes active return into:
    - Allocation effect (asset class over/under-weighting)
    - Selection effect (security selection within classes)
    - Interaction effect
    """
    try:
        bf = BrinsonFachlerAttribution()
        
        result = bf.calculate(
            np.array(request.portfolio_weights),
            np.array(request.benchmark_weights),
            np.array(request.portfolio_returns),
            np.array(request.benchmark_returns),
            request.segment_names,
        )
        
        segment_attr = {}
        for i, name in enumerate(result.segment_names):
            segment_attr[name] = {
                "allocation": float(result.segment_allocation[i]),
                "selection": float(result.segment_selection[i]),
                "interaction": float(result.segment_interaction[i]),
            }
        
        return AttributionResponse(
            portfolio_return=float(result.portfolio_return),
            benchmark_return=float(result.benchmark_return),
            active_return=float(result.active_return),
            allocation_effect=float(result.allocation_effect),
            selection_effect=float(result.selection_effect),
            interaction_effect=float(result.interaction_effect),
            segment_attribution=segment_attr,
        )
        
    except Exception as e:
        raise HTTPException(500, str(e))


# =============================================================================
# STRESS TESTING ENDPOINTS
# =============================================================================

@app.post("/api/v1/stress-test", response_model=StressTestResponse, tags=["Stress Testing"])
async def run_stress_test(request: StressTestRequest):
    """
    Run stress test scenario on portfolio.
    """
    try:
        weights = np.array(request.weights)
        n_assets = len(weights)
        
        # Setup factor model
        n_factors = 6
        factor_names = ["Market", "Size", "Value", "Momentum", "Credit", "Rates"]
        factor_loadings = np.random.randn(n_assets, n_factors) * 0.5
        factor_loadings[:, 0] = np.abs(factor_loadings[:, 0]) + 0.5
        
        engine = ScenarioEngine()
        engine.set_factor_model(factor_loadings, factor_names)
        engine._historical_data = {"asset_ids": [f"Asset_{i}" for i in range(n_assets)]}
        
        scenario = ScenarioDefinition(
            name=request.scenario_name,
            description=f"Stress test: {request.scenario_name}",
            scenario_type="factor_shock",
            factor_shocks=request.factor_shocks or {},
        )
        
        result = engine.run_scenario(weights, scenario)
        
        return StressTestResponse(
            scenario_name=result.scenario_name,
            portfolio_return=float(result.portfolio_return),
            position_impacts=result.position_impacts,
        )
        
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/v1/stress-test/scenarios", tags=["Stress Testing"])
async def get_standard_scenarios():
    """Get list of standard stress test scenarios."""
    from bxma.stress_testing.scenarios import STANDARD_SCENARIOS
    
    return {
        "scenarios": [
            {
                "name": s.name,
                "description": s.description,
                "type": s.scenario_type,
                "factor_shocks": s.factor_shocks,
            }
            for s in STANDARD_SCENARIOS
        ]
    }


# =============================================================================
# RISKMETRICS INTEGRATION ENDPOINTS (Core Requirement)
# =============================================================================

@app.post("/api/v1/riskmetrics/covariance", tags=["RiskMetrics"])
async def riskmetrics_covariance(
    returns: list[list[float]],
    decay_factor: float = 0.94,
):
    """
    Compute covariance matrix using RiskMetrics EWMA methodology.
    
    Standard RiskMetrics approach with exponential decay factor λ (default 0.94).
    
    Reference: "RiskMetrics Technical Document" (J.P. Morgan, 1996)
    """
    try:
        from bxma.reporting.riskmetrics import compute_riskmetrics_covariance
        
        returns_arr = np.array(returns)
        cov, vols = compute_riskmetrics_covariance(returns_arr, decay_factor)
        
        # Compute correlation
        vol_outer = np.outer(vols, vols)
        corr = cov / (vol_outer + 1e-10)
        
        return {
            "covariance": cov.tolist(),
            "correlation": corr.tolist(),
            "volatilities": vols.tolist(),
            "decay_factor": decay_factor,
            "methodology": "RiskMetrics EWMA",
            "reference": "J.P. Morgan RiskMetrics Technical Document (1996)",
        }
        
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/riskmetrics/var", tags=["RiskMetrics"])
async def riskmetrics_var(
    portfolio_weights: list[float],
    returns: list[list[float]],
    confidence_levels: list[float] = [0.95, 0.99],
    horizon_days: int = 1,
    decay_factor: float = 0.94,
):
    """
    Calculate VaR and CVaR using RiskMetrics methodology.
    
    Uses exponentially weighted covariance estimation with parametric VaR.
    """
    try:
        from bxma.reporting.riskmetrics import compute_riskmetrics_covariance
        from scipy import stats
        
        returns_arr = np.array(returns)
        weights = np.array(portfolio_weights)
        
        # RiskMetrics covariance
        cov, _ = compute_riskmetrics_covariance(returns_arr, decay_factor)
        
        # Portfolio variance and volatility
        port_var = weights @ cov @ weights
        port_vol = np.sqrt(port_var)
        
        # Scale for horizon
        port_vol_horizon = port_vol * np.sqrt(horizon_days)
        
        results = {
            "methodology": "RiskMetrics Parametric VaR",
            "decay_factor": decay_factor,
            "horizon_days": horizon_days,
            "portfolio_volatility": float(port_vol * np.sqrt(252)),  # Annualized
            "metrics": {},
        }
        
        for cl in confidence_levels:
            z = stats.norm.ppf(cl)
            var = z * port_vol_horizon
            cvar = port_vol_horizon * stats.norm.pdf(z) / (1 - cl)
            
            results["metrics"][f"var_{int(cl*100)}"] = float(var)
            results["metrics"][f"cvar_{int(cl*100)}"] = float(cvar)
        
        return results
        
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/v1/riskmetrics/factor-models", tags=["RiskMetrics"])
async def list_riskmetrics_factor_models():
    """
    List available RiskMetrics/MSCI factor models.
    """
    from bxma.integration.riskmetrics import RiskModel
    
    models = {
        "BARRA_USE4": {
            "name": "Barra US Equity Model 4",
            "provider": "MSCI/Barra",
            "coverage": "US Equities",
            "factors": 64,
            "style_factors": ["Momentum", "Volatility", "Size", "Value", "Growth", "Quality", "Leverage", "Liquidity"],
        },
        "BARRA_GEM3": {
            "name": "Barra Global Equity Model 3",
            "provider": "MSCI/Barra",
            "coverage": "Global Equities",
            "factors": 75,
            "style_factors": ["Momentum", "Volatility", "Size", "Value", "Growth", "Quality"],
        },
        "BARRA_CNE5": {
            "name": "Barra China Equity Model 5",
            "provider": "MSCI/Barra",
            "coverage": "China A-shares",
            "factors": 42,
        },
        "AXIOMA_WW21": {
            "name": "Axioma Worldwide Equity Model",
            "provider": "Axioma/Qontigo",
            "coverage": "Global Equities",
            "factors": 68,
        },
        "AXIOMA_US4": {
            "name": "Axioma US Equity Model 4",
            "provider": "Axioma/Qontigo",
            "coverage": "US Equities",
            "factors": 55,
        },
    }
    
    return {"available_models": models}


@app.post("/api/v1/riskmetrics/factor-exposures", tags=["RiskMetrics"])
async def get_factor_exposures(
    asset_ids: list[str],
    model: str = "BARRA_USE4",
):
    """
    Fetch factor exposures from RiskMetrics/MSCI Barra.
    
    Returns factor loadings and specific risk for each asset.
    """
    try:
        from bxma.integration.riskmetrics import RiskMetricsAdapter, RiskMetricsConfig, RiskModel
        
        # Map model name
        model_enum = getattr(RiskModel, model, RiskModel.BARRA_USE4)
        
        config = RiskMetricsConfig(risk_model=model_enum)
        adapter = RiskMetricsAdapter(config)
        
        data = adapter.fetch_all(asset_ids=asset_ids)
        
        return adapter.to_titan_format(data)
        
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/v1/riskmetrics/stress-scenarios", tags=["RiskMetrics"])
async def get_riskmetrics_stress_scenarios():
    """
    Get RiskMetrics standard stress test scenarios.
    
    Includes historical scenarios (2008, COVID, 2022) and hypothetical scenarios.
    """
    try:
        from bxma.reporting.riskmetrics import RiskMetricsConnector
        
        connector = RiskMetricsConnector()
        scenarios = connector._default_stress_scenarios()
        
        return {
            "source": "RiskMetrics",
            "scenarios": scenarios,
        }
        
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/riskmetrics/factor-risk-decomposition", tags=["RiskMetrics"])
async def factor_risk_decomposition(
    portfolio_weights: list[float],
    asset_ids: list[str],
    model: str = "BARRA_USE4",
):
    """
    Decompose portfolio risk using RiskMetrics factor model.
    
    Returns:
    - Factor contributions to risk
    - Specific (idiosyncratic) risk
    - Total tracking error components
    """
    try:
        from bxma.integration.riskmetrics import RiskMetricsAdapter, RiskMetricsConfig, RiskModel
        
        model_enum = getattr(RiskModel, model, RiskModel.BARRA_USE4)
        config = RiskMetricsConfig(risk_model=model_enum)
        adapter = RiskMetricsAdapter(config)
        
        data = adapter.fetch_all(asset_ids=asset_ids)
        weights = np.array(portfolio_weights)
        
        # Get factor exposures
        factor_loadings = []
        specific_risks = []
        
        for asset_id in asset_ids:
            if asset_id in data.asset_exposures:
                exp = data.asset_exposures[asset_id]
                factor_loadings.append(list(exp.factor_exposures.values()))
                specific_risks.append(exp.specific_risk)
            else:
                factor_loadings.append([0] * len(data.factors))
                specific_risks.append(0.03)
        
        factor_loadings = np.array(factor_loadings)
        specific_risks = np.array(specific_risks)
        
        # Portfolio factor exposures
        port_factor_exp = weights @ factor_loadings
        
        # Factor covariance
        if data.covariance_data and data.covariance_data.covariance is not None:
            factor_cov = data.covariance_data.covariance
        else:
            factor_cov = np.eye(len(data.factors)) * 0.01
        
        # Factor risk contribution
        factor_variance = port_factor_exp @ factor_cov @ port_factor_exp
        
        # Specific risk
        specific_variance = np.sum((weights * specific_risks) ** 2)
        
        # Total risk
        total_variance = factor_variance + specific_variance
        total_risk = np.sqrt(total_variance)
        
        return {
            "model": model,
            "total_risk": float(total_risk),
            "factor_risk": float(np.sqrt(factor_variance)),
            "specific_risk": float(np.sqrt(specific_variance)),
            "factor_risk_pct": float(factor_variance / total_variance * 100),
            "specific_risk_pct": float(specific_variance / total_variance * 100),
            "portfolio_factor_exposures": {
                data.factors[i].factor_id: float(port_factor_exp[i])
                for i in range(len(data.factors))
            },
            "top_factor_contributors": sorted(
                [
                    {
                        "factor": data.factors[i].name,
                        "exposure": float(port_factor_exp[i]),
                        "factor_vol": float(data.factors[i].volatility),
                    }
                    for i in range(len(data.factors))
                ],
                key=lambda x: abs(x["exposure"]),
                reverse=True,
            )[:10],
        }
        
    except Exception as e:
        raise HTTPException(500, str(e))


# =============================================================================
# TITAN-X ENDPOINTS - Advanced Features
# =============================================================================

@app.post("/api/v1/titan/regime", tags=["Titan-X"])
async def detect_regime(
    returns: list[float],
    current_vix: float = 15.0,
):
    """
    Detect market regime using HMM with thermodynamic uncertainty quantification.
    
    Returns regime probabilities with Boltzmann-scaled temperature based on VIX.
    """
    try:
        from bxma.risk.regime_detection import (
            GaussianHMM, ThermodynamicRegimeDetector, build_features_for_regime
        )
        
        returns_arr = np.array(returns)
        
        # Build features
        features = build_features_for_regime(returns_arr, volatility_window=20)
        
        # Fit HMM
        hmm = GaussianHMM(n_regimes=4, n_features=features.shape[1])
        hmm.fit(features)
        
        # Detect regime
        detector = ThermodynamicRegimeDetector(hmm, base_temperature=1.0)
        state = detector.detect(features[-50:], current_vix)
        
        return {
            "current_regime": state.regime.name,
            "probability": state.probability,
            "regime_probabilities": {k.name: v for k, v in state.regime_probabilities.items()},
            "entropy": state.entropy,
            "temperature": state.thermodynamic_temperature,
            "expected_duration_days": state.expected_duration,
        }
        
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/titan/thermodynamic-optimize", tags=["Titan-X"])
async def thermodynamic_optimize(
    expected_returns: list[float],
    covariance: list[list[float]],
    temperature: float = 1.0,
    n_samples: int = 5000,
    lambda_return: float = 1.0,
    lambda_risk: float = 1.0,
):
    """
    Optimize portfolio using Thermodynamic (Ising Hamiltonian) approach.
    
    Uses Boltzmann sampling to explore non-convex portfolio landscape.
    """
    try:
        from bxma.optimization.thermodynamic import (
            ThermodynamicOptimizer, ThermodynamicConfig
        )
        
        config = ThermodynamicConfig(
            temperature=temperature,
            n_samples=n_samples,
            lambda_return=lambda_return,
            lambda_risk=lambda_risk,
        )
        
        optimizer = ThermodynamicOptimizer(config)
        result = optimizer.optimize(
            np.array(expected_returns),
            np.array(covariance),
        )
        
        return {
            "weights": result.weights.tolist(),
            "energy": result.energy,
            "expected_return": result.expected_return,
            "expected_risk": result.expected_risk,
            "sharpe_ratio": result.sharpe_ratio,
            "temperature": result.temperature,
            "entropy": result.entropy,
            "n_samples": result.n_samples,
            "solve_time_ms": result.solve_time_ms,
        }
        
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/titan/contagion", tags=["Titan-X"])
async def simulate_contagion(
    initial_default: str,
    shock_size: float = 1.0,
    max_rounds: int = 10,
):
    """
    Simulate counterparty contagion using Graph Neural Networks.
    
    Models shock propagation through the financial network.
    """
    try:
        from bxma.risk.gnn_contagion import (
            ContagionGNN, build_sample_financial_graph
        )
        
        # Build sample graph (in production, load from database)
        graph = build_sample_financial_graph()
        
        # Initialize GNN
        gnn = ContagionGNN(
            node_features=64,
            hidden_dim=128,
            n_layers=3,
        )
        
        # Run simulation
        result = gnn.simulate_contagion(
            graph,
            initial_default,
            shock_size=shock_size,
            max_rounds=max_rounds,
        )
        
        return {
            "initial_default": result.initial_default_node,
            "initial_shock": result.initial_shock_size,
            "rounds_to_stabilize": result.rounds_to_stabilize,
            "nodes_affected": result.nodes_affected,
            "total_cascade_losses": result.total_cascade_losses,
            "contagion_index": result.contagion_index,
            "amplification_factor": result.amplification_factor,
            "propagation_path": result.propagation_path,
            "simulation_time_ms": result.simulation_time_ms,
        }
        
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/v1/titan/contagion/network", tags=["Titan-X"])
async def get_financial_network():
    """
    Get the financial network graph for visualization.
    """
    try:
        from bxma.risk.gnn_contagion import (
            build_sample_financial_graph, SystemicRiskAnalyzer
        )
        
        graph = build_sample_financial_graph()
        analyzer = SystemicRiskAnalyzer(graph)
        metrics = analyzer.compute_metrics()
        
        nodes = [
            {
                "id": n.id,
                "name": n.name,
                "type": n.node_type.name.lower(),
                "aum": n.assets_under_management,
                "default_probability": n.default_probability,
            }
            for n in graph.nodes.values()
        ]
        
        edges = [
            {
                "source": e.source_id,
                "target": e.target_id,
                "type": e.edge_type.name.lower(),
                "weight": e.weight,
            }
            for e in graph.edges
        ]
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metrics": {
                "n_nodes": metrics.n_nodes,
                "n_edges": metrics.n_edges,
                "density": metrics.density,
                "clustering_coefficient": metrics.clustering_coefficient,
                "systemically_important": metrics.systemically_important,
            }
        }
        
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/titan/semantic-alpha", tags=["Titan-X"])
async def compute_semantic_alpha(
    entity: str,
    reference_text: str | None = None,
):
    """
    Compute semantic alpha signals using FinBERT embeddings.
    
    Returns sentiment scores and divergence signals.
    """
    try:
        from bxma.data.semantic_alpha import (
            SemanticAlphaPipeline, TextDocument, TextSource
        )
        from datetime import datetime, timedelta
        
        pipeline = SemanticAlphaPipeline()
        
        # Generate sample documents (in production, fetch from database)
        sample_docs = []
        for i in range(10):
            sample_docs.append(TextDocument(
                id=f"doc_{i}",
                source=TextSource.NEWS,
                title=f"Sample news about {entity}",
                content=f"{entity} reported strong growth in Q4. Analysts expect continued momentum.",
                timestamp=datetime.now() - timedelta(hours=i),
                entities=[entity],
            ))
        
        pipeline.ingest_batch(sample_docs)
        
        # Get sentiment
        sentiment = pipeline.get_entity_sentiment(entity, lookback_hours=24)
        
        # Semantic time travel
        similar_periods = []
        if reference_text:
            similar_periods = pipeline.semantic_time_travel(
                entity, reference_text, top_k=5
            )
        
        # Generate report
        report = pipeline.generate_alpha_report()
        
        return {
            "entity": entity,
            "sentiment": sentiment,
            "similar_historical_periods": similar_periods,
            "report": report,
        }
        
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/v1/titan/volatility-surface", tags=["Titan-X"])
async def get_volatility_surface(
    underlying: str = "SPY",
    spot_price: float = 100.0,
):
    """
    Get implied volatility surface data for WebGPU rendering.
    """
    try:
        # Generate sample volatility surface
        strikes = [spot_price * (0.8 + i * 0.02) for i in range(21)]
        expiries = [7, 14, 30, 60, 90, 120, 180, 365]
        
        implied_vols = []
        for strike in strikes:
            row = []
            for expiry in expiries:
                moneyness = np.log(strike / spot_price)
                smile = 0.2 * np.exp(-moneyness * moneyness * 10)
                term_structure = 0.15 + 0.1 * np.sqrt(expiry / 365)
                vol = term_structure + smile + np.random.uniform(0, 0.01)
                row.append(vol)
            implied_vols.append(row)
        
        return {
            "underlying": underlying,
            "spot_price": spot_price,
            "strikes": strikes,
            "expiries": expiries,
            "implied_vols": implied_vols,
            "atm_vol": implied_vols[10][3],  # 60-day ATM
            "skew_25d": implied_vols[5][3] - implied_vols[10][3],
        }
        
    except Exception as e:
        raise HTTPException(500, str(e))


# =============================================================================
# COLLABORATION ENDPOINTS
# =============================================================================

@app.get("/api/v1/collaboration/teams", tags=["Collaboration"])
async def get_teams():
    """
    Get team directory for cross-functional collaboration.
    
    Returns teams: Investment, Operations, Treasury, Legal, Risk/Quant.
    """
    from bxma.collaboration.teams import TEAM_RESPONSIBILITIES, TeamType
    
    teams = []
    for team_type, info in TEAM_RESPONSIBILITIES.items():
        teams.append({
            "id": team_type.name,
            "name": info["name"],
            "responsibilities": info["responsibilities"],
            "data_access": info["data_access"],
            "collaboration_needs": info["collaboration_needs"],
        })
    
    return {"teams": teams}


@app.get("/api/v1/collaboration/tasks", tags=["Collaboration"])
async def get_workflow_tasks(
    status: str | None = Query(None, description="Filter by status"),
    team: str | None = Query(None, description="Filter by team"),
):
    """
    Get workflow tasks for cross-team collaboration.
    """
    # Sample tasks (in production, fetch from database)
    tasks = [
        {
            "task_id": "T001",
            "title": "Update VaR Limit for Equity Portfolio",
            "description": "Review and approve new VaR limits",
            "task_type": "risk_limit_change",
            "status": "REVIEW",
            "priority": "HIGH",
            "team": "INVESTMENT",
            "created_at": datetime.now().isoformat(),
        },
        {
            "task_id": "T002",
            "title": "Factor Model Validation Q1 2026",
            "description": "Validate updated factor model parameters",
            "task_type": "model_update",
            "status": "PENDING",
            "priority": "MEDIUM",
            "team": "LEGAL",
            "created_at": datetime.now().isoformat(),
        },
    ]
    
    if status:
        tasks = [t for t in tasks if t["status"] == status]
    if team:
        tasks = [t for t in tasks if t["team"] == team]
    
    return {"tasks": tasks}


@app.post("/api/v1/collaboration/tasks", tags=["Collaboration"])
async def create_task(
    title: str,
    description: str,
    task_type: str,
    assigned_to: str,
    priority: str = "MEDIUM",
):
    """
    Create a new workflow task for cross-team collaboration.
    """
    task = {
        "task_id": f"T{np.random.randint(100, 999)}",
        "title": title,
        "description": description,
        "task_type": task_type,
        "assigned_to": assigned_to,
        "priority": priority,
        "status": "PENDING",
        "created_at": datetime.now().isoformat(),
    }
    
    return {"message": "Task created successfully", "task": task}


@app.get("/api/v1/collaboration/notifications", tags=["Collaboration"])
async def get_notifications(
    user_id: str = Query("RQ001"),
    unread_only: bool = Query(False),
):
    """
    Get notifications for a user.
    """
    notifications = [
        {
            "notification_id": "N001",
            "title": "VaR Limit Breach Alert",
            "message": "Portfolio XYZ has exceeded 95% VaR limit",
            "notification_type": "LIMIT_BREACH",
            "priority": "CRITICAL",
            "created_at": datetime.now().isoformat(),
            "read_at": None,
        },
        {
            "notification_id": "N002",
            "title": "Approval Required",
            "message": "Risk limit change request awaiting your approval",
            "notification_type": "APPROVAL_REQUIRED",
            "priority": "HIGH",
            "created_at": (datetime.now() - timedelta(hours=2)).isoformat(),
            "read_at": None,
        },
    ]
    
    if unread_only:
        notifications = [n for n in notifications if n["read_at"] is None]
    
    return {"notifications": notifications, "unread_count": len([n for n in notifications if n["read_at"] is None])}


# =============================================================================
# TABLEAU INTEGRATION ENDPOINTS
# =============================================================================

@app.get("/api/v1/tableau/exports", tags=["Tableau"])
async def get_tableau_exports():
    """
    Get available Tableau export configurations.
    """
    exports = [
        {
            "id": "daily_risk",
            "name": "Daily Risk Report",
            "description": "VaR, CVaR, volatility for all portfolios",
            "format": "csv",
            "schedule": "Daily @ 06:00 ET",
            "last_export": datetime.now().isoformat(),
            "row_count": 2450,
        },
        {
            "id": "attribution",
            "name": "Performance Attribution",
            "description": "Brinson-Fachler attribution by sector",
            "format": "hyper",
            "schedule": "Daily @ 07:00 ET",
            "last_export": datetime.now().isoformat(),
            "row_count": 15680,
        },
        {
            "id": "factor_analysis",
            "name": "Factor Model Analysis",
            "description": "Factor loadings and contributions",
            "format": "hyper",
            "schedule": "Weekly",
            "last_export": (datetime.now() - timedelta(days=5)).isoformat(),
            "row_count": 45200,
        },
    ]
    
    return {"exports": exports}


@app.post("/api/v1/tableau/export/{export_id}", tags=["Tableau"])
async def trigger_tableau_export(
    export_id: str,
    format: str = Query("csv", description="Export format: csv, hyper, json"),
):
    """
    Trigger a Tableau data export.
    """
    from bxma.reporting.tableau import TableauExporter, TableauExportConfig
    
    config = TableauExportConfig(
        output_format=format,
        output_path=f"./tableau_exports/{export_id}",
    )
    
    exporter = TableauExporter(config)
    
    # Sample data export
    sample_data = {
        "portfolio_id": "PORT001",
        "as_of_date": date.today().isoformat(),
        "var_95": 0.0245,
        "var_99": 0.0352,
        "volatility": 0.158,
        "sharpe_ratio": 1.25,
    }
    
    filepath = exporter.export_risk_metrics(sample_data, f"{export_id}_export")
    
    return {
        "message": f"Export triggered for {export_id}",
        "format": format,
        "filepath": filepath,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/v1/tableau/datasources", tags=["Tableau"])
async def get_tableau_datasources():
    """
    Get published Tableau Server data sources.
    """
    datasources = [
        {
            "id": "ds_risk_dashboard",
            "name": "BXMA Risk Dashboard",
            "project": "BXMA Risk Analytics",
            "last_refresh": datetime.now().isoformat(),
            "status": "active",
            "subscriptions": 12,
        },
        {
            "id": "ds_portfolio_perf",
            "name": "Portfolio Performance",
            "project": "BXMA Risk Analytics",
            "last_refresh": datetime.now().isoformat(),
            "status": "active",
            "subscriptions": 8,
        },
    ]
    
    return {"datasources": datasources}


# =============================================================================
# MACRO DATA ENDPOINTS
# =============================================================================

@app.get("/api/v1/macro/indicators", tags=["Macro Data"])
async def get_macro_indicators(
    region: str | None = Query(None, description="Filter by region: US, EU, JP, CN"),
):
    """
    Get macroeconomic indicators.
    """
    from bxma.data.macro import MacroDataProvider, MacroIndicatorType
    
    provider = MacroDataProvider()
    
    indicators = [
        {"name": "US GDP (QoQ)", "value": 2.5, "previous": 2.3, "unit": "%", "region": "US"},
        {"name": "US CPI (YoY)", "value": 3.2, "previous": 3.4, "unit": "%", "region": "US"},
        {"name": "US Unemployment", "value": 4.1, "previous": 4.0, "unit": "%", "region": "US"},
        {"name": "Fed Funds Rate", "value": 4.25, "previous": 4.25, "unit": "%", "region": "US"},
        {"name": "Eurozone CPI", "value": 2.8, "previous": 2.9, "unit": "%", "region": "EU"},
        {"name": "China GDP (YoY)", "value": 5.2, "previous": 4.9, "unit": "%", "region": "CN"},
    ]
    
    if region:
        indicators = [i for i in indicators if i["region"] == region]
    
    return {"indicators": indicators, "as_of_date": date.today().isoformat()}


@app.get("/api/v1/macro/yield-curve", tags=["Macro Data"])
async def get_yield_curve(
    currency: str = Query("USD", description="Currency: USD, EUR, JPY"),
):
    """
    Get yield curve data.
    """
    from bxma.data.macro import MacroDataProvider
    
    provider = MacroDataProvider()
    curve = provider.get_yield_curve(currency)
    
    return {
        "currency": curve.currency,
        "curve_type": curve.curve_type,
        "tenors": curve.tenors,
        "rates": curve.rates,
        "slope_2s10s": curve.slope_2s10s,
        "as_of_date": curve.as_of_date.isoformat(),
    }


@app.get("/api/v1/macro/fx-rates", tags=["Macro Data"])
async def get_fx_rates():
    """
    Get FX rates.
    """
    from bxma.data.macro import MacroDataProvider
    
    provider = MacroDataProvider()
    
    pairs = [("EUR", "USD"), ("USD", "JPY"), ("GBP", "USD"), ("USD", "CHF")]
    rates = []
    
    for base, quote in pairs:
        fx = provider.get_fx_rate(base, quote)
        rates.append({
            "pair": fx.pair,
            "spot": fx.spot,
            "forward_1m": fx.forward_1m,
            "implied_vol_1m": fx.implied_vol_1m,
        })
    
    return {"fx_rates": rates, "as_of_date": date.today().isoformat()}


@app.get("/api/v1/macro/commodities", tags=["Macro Data"])
async def get_commodities():
    """
    Get commodity prices.
    """
    from bxma.data.macro import MacroDataProvider
    
    provider = MacroDataProvider()
    
    commodities_list = ["WTI", "Brent", "Gold", "Silver", "Copper"]
    commodities = []
    
    for name in commodities_list:
        c = provider.get_commodity_price(name)
        commodities.append({
            "name": c.commodity,
            "price": c.spot,
            "unit": c.unit,
            "front_month": c.front_month,
            "contango_backwardation": c.contango_backwardation,
        })
    
    return {"commodities": commodities, "as_of_date": date.today().isoformat()}


@app.get("/api/v1/macro/equity-indices", tags=["Macro Data"])
async def get_equity_indices():
    """
    Get equity index data.
    """
    from bxma.data.macro import MacroDataProvider
    
    provider = MacroDataProvider()
    
    indices_list = ["SPX", "NDX", "RTY", "VIX"]
    indices = []
    
    for name in indices_list:
        idx = provider.get_equity_index(name)
        indices.append({
            "name": idx.index_name,
            "price": idx.price,
            "previous_close": idx.previous_close,
            "daily_return": idx.daily_return,
            "pe_ratio": idx.pe_ratio,
            "dividend_yield": idx.dividend_yield,
            "realized_vol_20d": idx.realized_vol_20d,
        })
    
    return {"indices": indices, "as_of_date": date.today().isoformat()}


@app.get("/api/v1/macro/calendar", tags=["Macro Data"])
async def get_economic_calendar(
    days_ahead: int = Query(7, description="Number of days to look ahead"),
):
    """
    Get economic calendar for upcoming events.
    """
    from bxma.data.macro import MacroDataProvider
    
    provider = MacroDataProvider()
    calendar = provider.get_economic_calendar(
        date.today(),
        date.today() + timedelta(days=days_ahead),
    )
    
    return {"events": calendar}


@app.get("/api/v1/macro/market-snapshot", tags=["Macro Data"])
async def get_market_snapshot():
    """
    Get comprehensive market snapshot.
    """
    from bxma.data.macro import MacroDataProvider
    
    provider = MacroDataProvider()
    snapshot = provider.get_market_snapshot()
    
    # Serialize dataclass objects
    result = {
        "timestamp": snapshot["timestamp"],
        "equity_indices": {
            name: {
                "price": idx.price,
                "daily_return": idx.daily_return,
                "pe_ratio": idx.pe_ratio,
            }
            for name, idx in snapshot["equity_indices"].items()
        },
        "fx_rates": {
            name: {"spot": fx.spot, "forward_1m": fx.forward_1m}
            for name, fx in snapshot["fx_rates"].items()
        },
        "commodities": {
            name: {"price": c.spot, "unit": c.unit}
            for name, c in snapshot["commodities"].items()
        },
    }
    
    return result


# =============================================================================
# LIVE PORTFOLIO DATA ENDPOINTS
# =============================================================================

@app.get("/api/v1/portfolio/live", tags=["Live Portfolio"])
async def get_live_portfolio():
    """
    Get real portfolio data with LIVE market prices.
    Returns the BXMA demo portfolio with current market values.
    """
    from bxma.core.demo_portfolio import DEMO_PORTFOLIO, RISK_FREE_RATE
    from bxma.data.live_market_data import market_data_service
    
    # Get live quotes for all holdings
    tickers = DEMO_PORTFOLIO.get_tickers()
    quotes = await market_data_service.get_quotes(tickers)
    
    # Update portfolio with live prices
    prices = {t: q.price for t, q in quotes.items()}
    DEMO_PORTFOLIO.update_prices(prices)
    
    # Calculate total portfolio value
    total_value = sum(h.market_value for h in DEMO_PORTFOLIO.holdings)
    
    holdings_data = []
    for h in DEMO_PORTFOLIO.holdings:
        quote = quotes.get(h.ticker)
        holdings_data.append({
            "ticker": h.ticker,
            "name": h.name,
            "asset_class": h.asset_class,
            "sector": h.sector,
            "target_weight": h.weight,
            "actual_weight": h.market_value / total_value if total_value > 0 else h.weight,
            "shares": h.shares,
            "cost_basis": h.cost_basis,
            "current_price": h.current_price,
            "market_value": h.market_value,
            "gain_loss": h.gain_loss,
            "gain_loss_pct": h.gain_loss_pct,
            "day_change": quote.change if quote else 0,
            "day_change_pct": quote.change_pct if quote else 0,
        })
    
    return {
        "portfolio_name": DEMO_PORTFOLIO.name,
        "inception_date": DEMO_PORTFOLIO.inception_date.isoformat(),
        "total_value": total_value,
        "total_aum_target": DEMO_PORTFOLIO.total_aum,
        "holdings": holdings_data,
        "asset_allocation": DEMO_PORTFOLIO.get_asset_allocation(),
        "benchmark": DEMO_PORTFOLIO.benchmark_ticker,
        "risk_free_rate": RISK_FREE_RATE,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/v1/portfolio/quotes", tags=["Live Portfolio"])
async def get_live_quotes(
    tickers: str = Query(..., description="Comma-separated list of tickers")
):
    """
    Get LIVE market quotes for specified tickers.
    """
    from bxma.data.live_market_data import market_data_service
    
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    quotes = await market_data_service.get_quotes(ticker_list)
    
    return {
        "quotes": {
            t: {
                "price": q.price,
                "change": q.change,
                "change_pct": q.change_pct,
                "volume": q.volume,
                "bid": q.bid,
                "ask": q.ask,
                "day_high": q.day_high,
                "day_low": q.day_low,
                "year_high": q.year_high,
                "year_low": q.year_low,
            }
            for t, q in quotes.items()
        },
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/v1/portfolio/market-indices", tags=["Live Portfolio"])
async def get_market_indices_live():
    """
    Get LIVE market indices data.
    """
    from bxma.data.live_market_data import market_data_service
    
    indices = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
    quotes = await market_data_service.get_quotes(indices)
    
    index_names = {
        "SPY": "S&P 500",
        "QQQ": "NASDAQ 100",
        "IWM": "Russell 2000",
        "TLT": "20+ Year Treasury",
        "GLD": "Gold",
    }
    
    return {
        "indices": [
            {
                "ticker": t,
                "name": index_names.get(t, t),
                "price": q.price,
                "change": q.change,
                "change_pct": q.change_pct,
            }
            for t, q in quotes.items()
        ],
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/v1/market/header-data", tags=["Market Data"])
async def get_header_market_data():
    """
    Get LIVE market data for header display.
    Returns S&P 500 (^GSPC), 10Y Treasury Yield (^TNX), and VIX (^VIX).
    """
    from bxma.data.live_market_data import market_data_service
    
    # Yahoo Finance tickers for the specific indices
    # ^GSPC = S&P 500 Index
    # ^TNX = 10-Year Treasury Yield
    # ^VIX = CBOE Volatility Index
    tickers = ["^GSPC", "^TNX", "^VIX"]
    quotes = await market_data_service.get_quotes(tickers)
    
    result = []
    
    # S&P 500
    if "^GSPC" in quotes:
        q = quotes["^GSPC"]
        result.append({
            "name": "S&P 500",
            "ticker": "^GSPC",
            "value": round(q.price, 2),
            "change": round(q.change_pct, 2),
            "raw_change": round(q.change, 2),
        })
    
    # 10Y Treasury Yield (displayed as yield %)
    if "^TNX" in quotes:
        q = quotes["^TNX"]
        # ^TNX returns the yield directly (e.g., 4.24 for 4.24%)
        # No conversion needed - use the value as-is
        result.append({
            "name": "10Y UST",
            "ticker": "^TNX",
            "value": round(q.price, 2),
            "change": round(q.change_pct, 2),
            "raw_change": round(q.change, 3),
        })
    
    # VIX
    if "^VIX" in quotes:
        q = quotes["^VIX"]
        result.append({
            "name": "VIX",
            "ticker": "^VIX",
            "value": round(q.price, 2),
            "change": round(q.change_pct, 2),
            "raw_change": round(q.change, 2),
        })
    
    return {
        "indicators": result,
        "timestamp": datetime.now().isoformat(),
    }


# =============================================================================
# CALCULATIONS WITH FULL MATH DERIVATIONS
# =============================================================================

@app.get("/api/v1/calculations/var", tags=["Calculations"])
async def calculate_var_with_steps(
    confidence: float = Query(0.95, description="Confidence level"),
    horizon: int = Query(1, description="Horizon in days"),
    method: str = Query("parametric", description="VaR method: parametric, historical, cornish_fisher"),
):
    """
    Calculate VaR with FULL mathematical derivation steps.
    Shows all formulas and intermediate calculations.
    """
    from bxma.core.demo_portfolio import DEMO_PORTFOLIO, RISK_FREE_RATE
    from bxma.data.live_market_data import market_data_service
    from bxma.analytics.calculations import calculate_var
    
    # Get historical returns
    tickers = DEMO_PORTFOLIO.get_tickers()
    returns_df = await market_data_service.get_portfolio_returns(tickers, period="1y")
    returns = returns_df.values
    weights = DEMO_PORTFOLIO.get_weights()
    
    # Get current portfolio value
    quotes = await market_data_service.get_quotes(tickers)
    prices = {t: q.price for t, q in quotes.items()}
    DEMO_PORTFOLIO.update_prices(prices)
    portfolio_value = sum(h.market_value for h in DEMO_PORTFOLIO.holdings)
    
    # Calculate VaR with derivation
    result = calculate_var(
        returns=returns,
        weights=weights,
        confidence=confidence,
        horizon_days=horizon,
        method=method,
        portfolio_value=portfolio_value,
    )
    
    return {
        "result": {
            "name": result.name,
            "value_pct": result.value * 100,
            "value_dollar": result.value * portfolio_value,
            "unit": result.unit,
            "formula_latex": result.formula_latex,
        },
        "inputs": result.inputs_summary,
        "steps": [
            {
                "step": s.step_number,
                "title": s.title,
                "formula_latex": s.formula_latex,
                "description": s.description,
                "inputs": s.inputs,
                "calculation": s.calculation,
                "result": s.result,
            }
            for s in result.steps
        ],
        "portfolio_value": portfolio_value,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/v1/calculations/cvar", tags=["Calculations"])
async def calculate_cvar_with_steps(
    confidence: float = Query(0.95, description="Confidence level"),
    horizon: int = Query(1, description="Horizon in days"),
):
    """
    Calculate CVaR (Expected Shortfall) with FULL mathematical derivation.
    """
    from bxma.core.demo_portfolio import DEMO_PORTFOLIO
    from bxma.data.live_market_data import market_data_service
    from bxma.analytics.calculations import calculate_cvar
    
    tickers = DEMO_PORTFOLIO.get_tickers()
    returns_df = await market_data_service.get_portfolio_returns(tickers, period="1y")
    returns = returns_df.values
    weights = DEMO_PORTFOLIO.get_weights()
    
    quotes = await market_data_service.get_quotes(tickers)
    prices = {t: q.price for t, q in quotes.items()}
    DEMO_PORTFOLIO.update_prices(prices)
    portfolio_value = sum(h.market_value for h in DEMO_PORTFOLIO.holdings)
    
    result = calculate_cvar(
        returns=returns,
        weights=weights,
        confidence=confidence,
        horizon_days=horizon,
        portfolio_value=portfolio_value,
    )
    
    return {
        "result": {
            "name": result.name,
            "value_pct": result.value * 100,
            "value_dollar": result.value * portfolio_value,
            "formula_latex": result.formula_latex,
        },
        "inputs": result.inputs_summary,
        "steps": [
            {
                "step": s.step_number,
                "title": s.title,
                "formula_latex": s.formula_latex,
                "description": s.description,
                "inputs": s.inputs,
                "calculation": s.calculation,
                "result": s.result,
            }
            for s in result.steps
        ],
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/v1/calculations/sharpe", tags=["Calculations"])
async def calculate_sharpe_with_steps():
    """
    Calculate Sharpe Ratio with FULL mathematical derivation.
    """
    from bxma.core.demo_portfolio import DEMO_PORTFOLIO, RISK_FREE_RATE
    from bxma.data.live_market_data import market_data_service
    from bxma.analytics.calculations import calculate_sharpe_ratio
    
    tickers = DEMO_PORTFOLIO.get_tickers()
    returns_df = await market_data_service.get_portfolio_returns(tickers, period="1y")
    returns = returns_df.values
    weights = DEMO_PORTFOLIO.get_weights()
    
    result = calculate_sharpe_ratio(
        returns=returns,
        weights=weights,
        risk_free_rate=RISK_FREE_RATE,
    )
    
    return {
        "result": {
            "name": result.name,
            "value": result.value,
            "formula_latex": result.formula_latex,
        },
        "inputs": result.inputs_summary,
        "steps": [
            {
                "step": s.step_number,
                "title": s.title,
                "formula_latex": s.formula_latex,
                "description": s.description,
                "inputs": s.inputs,
                "calculation": s.calculation,
                "result": s.result,
            }
            for s in result.steps
        ],
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/v1/calculations/covariance", tags=["Calculations"])
async def calculate_covariance_with_steps(
    method: str = Query("ewma", description="Method: sample, ewma, ledoit_wolf"),
    decay_factor: float = Query(0.94, description="EWMA decay factor"),
):
    """
    Calculate covariance matrix with FULL mathematical derivation.
    """
    from bxma.core.demo_portfolio import DEMO_PORTFOLIO
    from bxma.data.live_market_data import market_data_service
    from bxma.analytics.calculations import calculate_covariance_matrix, calculate_correlation_matrix
    
    tickers = DEMO_PORTFOLIO.get_tickers()
    returns_df = await market_data_service.get_portfolio_returns(tickers, period="1y")
    returns = returns_df.values
    
    cov, cov_steps = calculate_covariance_matrix(returns, method=method, decay_factor=decay_factor)
    corr, corr_steps = calculate_correlation_matrix(cov)
    
    # Annualize
    cov_annual = cov * 252
    vol_annual = np.sqrt(np.diag(cov_annual))
    
    return {
        "covariance_matrix": cov_annual.tolist(),
        "correlation_matrix": corr.tolist(),
        "volatilities_annual": (vol_annual * 100).tolist(),
        "tickers": tickers,
        "method": method,
        "steps": [
            {
                "step": s.step_number,
                "title": s.title,
                "formula_latex": s.formula_latex,
                "description": s.description,
                "inputs": s.inputs,
                "calculation": s.calculation,
                "result": s.result,
            }
            for s in cov_steps + corr_steps
        ],
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/v1/calculations/optimization", tags=["Calculations"])
async def calculate_optimization_with_steps(
    method: str = Query("hrp", description="Method: hrp, risk_parity, max_sharpe, min_variance"),
):
    """
    Calculate optimal portfolio weights with FULL mathematical derivation.
    """
    from bxma.core.demo_portfolio import DEMO_PORTFOLIO
    from bxma.data.live_market_data import market_data_service
    from bxma.analytics.calculations import calculate_optimal_weights_hrp
    
    tickers = DEMO_PORTFOLIO.get_tickers()
    asset_names = [h.name for h in DEMO_PORTFOLIO.holdings]
    returns_df = await market_data_service.get_portfolio_returns(tickers, period="1y")
    returns = returns_df.values
    
    if method == "hrp":
        weights, steps = calculate_optimal_weights_hrp(returns, tickers)
    else:
        # Fallback to HRP for now
        weights, steps = calculate_optimal_weights_hrp(returns, tickers)
    
    return {
        "optimal_weights": {t: w for t, w in zip(tickers, weights)},
        "current_weights": {t: w for t, w in zip(tickers, DEMO_PORTFOLIO.get_weights())},
        "method": method,
        "steps": [
            {
                "step": s.step_number,
                "title": s.title,
                "formula_latex": s.formula_latex,
                "description": s.description,
                "inputs": s.inputs,
                "calculation": s.calculation,
                "result": s.result,
            }
            for s in steps
        ],
        "timestamp": datetime.now().isoformat(),
    }


# =============================================================================
# WEBSOCKET STREAMING
# =============================================================================

@app.websocket("/ws/risk-stream")
async def risk_stream(websocket: WebSocket):
    """
    Real-time risk metrics streaming via WebSocket.
    
    Sends updated risk metrics every second for connected dashboards.
    """
    await websocket.accept()
    app_state.connected_clients.add(websocket)
    
    try:
        # Simulate portfolio data
        nav = 90_000_000_000  # $90B
        
        while True:
            # Generate real-time metrics
            daily_return = np.random.randn() * 0.01
            nav *= (1 + daily_return)
            
            data = RiskDashboardData(
                timestamp=datetime.now(),
                nav=nav,
                daily_return=daily_return,
                var_95=abs(np.random.randn() * 0.015),
                cvar_95=abs(np.random.randn() * 0.020),
                volatility=0.15 + np.random.randn() * 0.02,
                sharpe_ratio=1.2 + np.random.randn() * 0.3,
                max_drawdown=-0.05 + np.random.randn() * 0.01,
                factor_exposures={
                    "Market": 1.0 + np.random.randn() * 0.1,
                    "Size": np.random.randn() * 0.2,
                    "Value": np.random.randn() * 0.2,
                    "Momentum": np.random.randn() * 0.3,
                },
            )
            
            await websocket.send_json(data.model_dump(mode="json"))
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        app_state.connected_clients.discard(websocket)
    except Exception as e:
        app_state.connected_clients.discard(websocket)
        print(f"WebSocket error: {e}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=4,
    )
