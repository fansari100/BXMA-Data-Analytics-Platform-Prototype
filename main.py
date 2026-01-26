#!/usr/bin/env python3
"""
BXMA Risk/Quant Platform - Main Entry Point
============================================

Blackstone Multi-Asset Investing Risk & Quantitative Analytics Platform

A comprehensive, production-ready platform for:
- Portfolio risk analytics (VaR, CVaR, factor models)
- Portfolio optimization (HRP, Risk Parity, Mean-Variance)
- Performance attribution (Brinson, Geometric)
- Stress testing and scenario analysis
- Real-time streaming analytics

Usage:
    python main.py                  # Run demo
    python main.py --server         # Start API server
    python main.py --test           # Run test suite

Author: BXMA Risk/Quant Team
Version: 1.0.0
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

import numpy as np


def run_demo():
    """Run comprehensive platform demonstration."""
    print("=" * 70)
    print("  BXMA Risk/Quant Platform - Demonstration")
    print("  Blackstone Multi-Asset Investing")
    print("=" * 70)
    print()

    # =========================================================================
    # 1. Portfolio Setup
    # =========================================================================
    print("📊 1. PORTFOLIO CONFIGURATION")
    print("-" * 50)
    
    from bxma.core.portfolio import Portfolio, Position, SecurityIdentifier, Strategy
    from bxma.core.types import AssetClass
    
    # Create sample portfolio
    assets = [
        ("SPY", "US Large Cap Equity", AssetClass.EQUITY_LARGE_CAP, 0.25),
        ("IWM", "US Small Cap Equity", AssetClass.EQUITY_SMALL_CAP, 0.10),
        ("EFA", "Int'l Developed Equity", AssetClass.EQUITY_DEVELOPED, 0.15),
        ("EEM", "Emerging Market Equity", AssetClass.EQUITY_EMERGING, 0.08),
        ("AGG", "US Aggregate Bond", AssetClass.FIXED_INCOME_SOVEREIGN, 0.20),
        ("LQD", "Investment Grade Credit", AssetClass.FIXED_INCOME_CORPORATE_IG, 0.10),
        ("GLD", "Gold", AssetClass.ALTERNATIVE_COMMODITIES, 0.05),
        ("VNQ", "Real Estate", AssetClass.ALTERNATIVE_REAL_ESTATE, 0.05),
        ("CASH", "Cash", AssetClass.CASH, 0.02),
    ]
    
    print(f"  Portfolio: Multi-Asset Balanced Strategy")
    print(f"  AUM: $90,000,000,000")
    print(f"  Holdings: {len(assets)} asset classes")
    print()
    
    for ticker, name, asset_class, weight in assets:
        print(f"    {ticker:6} | {name:25} | {weight*100:5.1f}%")
    print()

    # =========================================================================
    # 2. Risk Analytics
    # =========================================================================
    print("🛡️  2. RISK ANALYTICS")
    print("-" * 50)
    
    from bxma.risk.var import ParametricVaR, HistoricalVaR, MonteCarloVaR, CornishFisherVaR
    from bxma.risk.covariance import LedoitWolfCovariance
    from bxma.risk.factor_models import StatisticalFactorModel
    
    # Generate realistic return data
    np.random.seed(42)
    n_assets = len(assets)
    n_days = 252 * 3  # 3 years
    
    # Simulate correlated returns
    vols = np.array([0.16, 0.22, 0.18, 0.25, 0.05, 0.08, 0.15, 0.18, 0.01])
    correlation = np.array([
        [1.00, 0.85, 0.75, 0.65, 0.10, 0.20, 0.05, 0.60, 0.00],
        [0.85, 1.00, 0.70, 0.60, 0.08, 0.18, 0.03, 0.55, 0.00],
        [0.75, 0.70, 1.00, 0.70, 0.12, 0.22, 0.08, 0.50, 0.00],
        [0.65, 0.60, 0.70, 1.00, 0.05, 0.15, 0.10, 0.45, 0.00],
        [0.10, 0.08, 0.12, 0.05, 1.00, 0.75, 0.15, 0.20, 0.00],
        [0.20, 0.18, 0.22, 0.15, 0.75, 1.00, 0.10, 0.25, 0.00],
        [0.05, 0.03, 0.08, 0.10, 0.15, 0.10, 1.00, 0.10, 0.00],
        [0.60, 0.55, 0.50, 0.45, 0.20, 0.25, 0.10, 1.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
    ])
    
    cov_true = np.outer(vols, vols) * correlation
    L = np.linalg.cholesky(cov_true)
    returns = (L @ np.random.randn(n_assets, n_days)).T / np.sqrt(252)
    
    weights = np.array([w for _, _, _, w in assets])
    
    # VaR Calculations
    print("  Value-at-Risk Analysis (1-day horizon):")
    print()
    
    methods = [
        ("Parametric", ParametricVaR()),
        ("Historical", HistoricalVaR()),
        ("Monte Carlo", MonteCarloVaR(n_simulations=10000)),
        ("Cornish-Fisher", CornishFisherVaR()),
    ]
    
    import time
    for name, engine in methods:
        t0 = time.perf_counter()
        result = engine.calculate_var(returns, weights, 0.95, 1)
        t1 = time.perf_counter()
        cvar_str = f"{result.cvar*100:6.3f}%" if result.cvar else "N/A"
        print(f"    {name:15} | VaR: {result.var*100:6.3f}% | CVaR: {cvar_str:>7} | Time: {(t1-t0)*1000:.1f}ms")
    
    print()
    
    # Covariance estimation
    cov_estimator = LedoitWolfCovariance()
    cov_result = cov_estimator.fit(returns)
    print(f"  Covariance Estimation (Ledoit-Wolf):")
    print(f"    Condition Number: {cov_result.condition_number:.1f}")
    print(f"    Effective Rank: {cov_result.effective_rank:.1f}")
    print()
    
    # Factor model
    factor_model = StatisticalFactorModel(n_factors=5, method="pca")
    factor_result = factor_model.fit(returns)
    print(f"  Factor Model (PCA, 5 factors):")
    print(f"    R-squared: {factor_result.r_squared*100:.1f}%")
    print(f"    Explained Variance: {[f'{v*100:.1f}%' for v in factor_result.explained_variance_ratio[:3]]}")
    print()

    # =========================================================================
    # 3. Portfolio Optimization
    # =========================================================================
    print("🎯 3. PORTFOLIO OPTIMIZATION")
    print("-" * 50)
    
    from bxma.optimization.classical import MeanVarianceOptimizer, MinVarianceOptimizer, MaxSharpeOptimizer
    from bxma.optimization.risk_parity import RiskParityOptimizer, HierarchicalRiskParity
    
    # Expected returns
    expected_returns = np.array([0.08, 0.10, 0.07, 0.09, 0.03, 0.04, 0.02, 0.06, 0.01])
    
    print("  Optimization Results:")
    print()
    
    optimizers = [
        ("Min Variance", MinVarianceOptimizer()),
        ("Max Sharpe", MaxSharpeOptimizer()),
        ("Risk Parity", RiskParityOptimizer()),
        ("HRP", HierarchicalRiskParity()),
    ]
    
    print(f"    {'Method':15} | {'Return':8} | {'Risk':8} | {'Sharpe':8} | {'Time':8}")
    print("    " + "-" * 55)
    
    for name, optimizer in optimizers:
        result = optimizer.optimize(expected_returns, cov_result.covariance)
        print(f"    {name:15} | {result.expected_return*100:6.2f}%  | {result.expected_risk*100:6.2f}%  | {result.sharpe_ratio:6.2f}   | {result.solve_time_ms:6.1f}ms")
    
    print()

    # =========================================================================
    # 4. Performance Attribution
    # =========================================================================
    print("📈 4. PERFORMANCE ATTRIBUTION")
    print("-" * 50)
    
    from bxma.attribution.brinson import BrinsonFachlerAttribution
    from bxma.attribution.geometric import CarinoAttribution
    
    # Attribution analysis
    bf = BrinsonFachlerAttribution()
    
    # Simulated benchmark (60/40 proxy)
    benchmark_weights = np.array([0.20, 0.08, 0.12, 0.08, 0.30, 0.12, 0.02, 0.05, 0.03])
    
    # Period returns
    portfolio_returns = np.array([0.08, 0.12, 0.06, 0.10, 0.02, 0.03, 0.04, 0.07, 0.005])
    benchmark_returns = np.array([0.07, 0.10, 0.05, 0.08, 0.025, 0.035, 0.03, 0.06, 0.005])
    
    segment_names = [a[1] for a in assets]
    
    attr_result = bf.calculate(
        weights, benchmark_weights,
        portfolio_returns, benchmark_returns,
        segment_names
    )
    
    print(f"  Brinson-Fachler Attribution:")
    print(f"    Portfolio Return:   {attr_result.portfolio_return*100:6.2f}%")
    print(f"    Benchmark Return:   {attr_result.benchmark_return*100:6.2f}%")
    print(f"    Active Return:      {attr_result.active_return*100:6.2f}%")
    print()
    print(f"    Allocation Effect:  {attr_result.allocation_effect*10000:6.1f} bps")
    print(f"    Selection Effect:   {attr_result.selection_effect*10000:6.1f} bps")
    print(f"    Interaction Effect: {attr_result.interaction_effect*10000:6.1f} bps")
    print()

    # =========================================================================
    # 5. Stress Testing
    # =========================================================================
    print("⚡ 5. STRESS TESTING")
    print("-" * 50)
    
    from bxma.stress_testing.scenarios import ScenarioEngine, ScenarioDefinition
    
    engine = ScenarioEngine()
    
    # Setup factor model for scenarios
    factor_names = ["Market", "Size", "Value", "Momentum", "Credit", "Rates"]
    factor_loadings = np.random.randn(n_assets, 6) * 0.5
    factor_loadings[:, 0] = np.abs(factor_loadings[:, 0]) + 0.5  # Positive market exposure
    engine.set_factor_model(factor_loadings, factor_names)
    
    # Define stress scenarios
    stress_scenarios = [
        ScenarioDefinition(
            name="2008 Financial Crisis",
            description="Global credit freeze, equity collapse",
            scenario_type="factor_shock",
            factor_shocks={"Market": -0.40, "Credit": -0.25, "Rates": -0.15}
        ),
        ScenarioDefinition(
            name="COVID-19 Crash",
            description="March 2020 pandemic shock",
            scenario_type="factor_shock",
            factor_shocks={"Market": -0.35, "Credit": -0.15, "Momentum": 0.20}
        ),
        ScenarioDefinition(
            name="Rates +200bps",
            description="Aggressive Fed tightening",
            scenario_type="factor_shock",
            factor_shocks={"Rates": 0.20, "Credit": -0.05, "Market": -0.08}
        ),
        ScenarioDefinition(
            name="EM Currency Crisis",
            description="Emerging market contagion",
            scenario_type="factor_shock",
            factor_shocks={"Market": -0.20, "Credit": -0.10, "Value": -0.15}
        ),
        ScenarioDefinition(
            name="Equity Rally",
            description="Risk-on environment",
            scenario_type="factor_shock",
            factor_shocks={"Market": 0.20, "Size": 0.10, "Momentum": 0.15}
        ),
    ]
    
    print("  Scenario Analysis:")
    print()
    print(f"    {'Scenario':25} | {'Impact':10} | {'P&L ($B)':12}")
    print("    " + "-" * 50)
    
    aum = 90_000_000_000  # $90B
    
    for scenario in stress_scenarios:
        result = engine.run_scenario(weights, scenario)
        pnl = result.portfolio_return * aum
        print(f"    {scenario.name:25} | {result.portfolio_return*100:8.2f}% | ${pnl/1e9:8.2f}B")
    
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("  BXMA Platform Demonstration Complete")
    print("=" * 70)
    print()
    print("  Full-Stack Components:")
    print("    ✓ Risk Analytics (VaR, CVaR, Factor Models)")
    print("    ✓ Portfolio Optimization (HRP, Risk Parity, MVO)")
    print("    ✓ Performance Attribution (Brinson, Geometric)")
    print("    ✓ Stress Testing (Historical & Hypothetical Scenarios)")
    print("    ✓ FastAPI Backend with WebSocket Streaming")
    print("    ✓ Next.js 14 Frontend with Real-time Dashboards")
    print("    ✓ PostgreSQL/TimescaleDB for Time-Series Data")
    print("    ✓ Redis Caching & Kafka Event Streaming")
    print("    ✓ Docker/Kubernetes Deployment Ready")
    print("    ✓ OAuth2/JWT Authentication with RBAC")
    print()
    print("  Start the platform:")
    print("    docker-compose up -d           # Start all services")
    print("    cd frontend && npm run dev     # Start frontend (dev)")
    print("    python -m backend.main         # Start backend (dev)")
    print()


def run_server():
    """Start the FastAPI server."""
    import uvicorn
    
    print("Starting BXMA API Server...")
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,
    )


def run_tests():
    """Run the test suite."""
    import subprocess
    
    print("Running BXMA Test Suite...")
    subprocess.run([
        "python", "-m", "pytest",
        "tests/",
        "-v",
        "--cov=bxma",
        "--cov-report=term-missing"
    ])


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="BXMA Risk/Quant Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py              # Run demonstration
    python main.py --server     # Start API server
    python main.py --test       # Run tests
        """
    )
    
    parser.add_argument(
        "--server",
        action="store_true",
        help="Start the FastAPI server"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the test suite"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="BXMA Platform v1.0.0"
    )
    
    args = parser.parse_args()
    
    if args.server:
        run_server()
    elif args.test:
        run_tests()
    else:
        run_demo()


if __name__ == "__main__":
    main()
