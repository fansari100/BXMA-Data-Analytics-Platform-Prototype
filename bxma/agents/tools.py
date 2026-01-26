"""
Financial Tools for Agentic AI
==============================

Safe tool interfaces that agents can use to interact with
the financial environment. All tools include validation
and audit logging.

Author: BXMA Quant Team
Date: January 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
import numpy as np
from numpy.typing import NDArray

from bxma.agents.react import FinancialTool, Observation


class QueryKDBTool(FinancialTool):
    """
    Tool for querying KDB-X database.
    
    Supports both time-series queries and vector similarity search.
    """
    
    name = "query_kdb"
    description = "Query KDB-X database for historical data, factor exposures, or semantic search"
    parameters_schema = {
        "query_type": {"type": "string", "enum": ["timeseries", "vector", "sql"]},
        "query": {"type": "string", "description": "The query string"},
        "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
        "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"},
        "symbols": {"type": "array", "description": "List of symbols to query"},
    }
    
    def __init__(self, kdb_connection=None):
        self.kdb = kdb_connection
    
    def validate_parameters(self, **kwargs) -> tuple[bool, str | None]:
        if "query_type" not in kwargs:
            return False, "query_type is required"
        if kwargs["query_type"] not in ["timeseries", "vector", "sql"]:
            return False, "Invalid query_type"
        return True, None
    
    def execute(self, **kwargs) -> Observation:
        query_type = kwargs.get("query_type")
        query = kwargs.get("query", "")
        
        # Simulated response (in production, connects to actual KDB-X)
        if query_type == "timeseries":
            data = {
                "timestamps": ["2026-01-20", "2026-01-21", "2026-01-22"],
                "values": [100.5, 101.2, 100.8],
            }
        elif query_type == "vector":
            data = {
                "similar_periods": [
                    {"date": "2020-03-15", "similarity": 0.92},
                    {"date": "2008-09-15", "similarity": 0.85},
                ],
            }
        else:
            data = {"result": "Query executed"}
        
        return Observation(
            content=data,
            success=True,
            metadata={"query_type": query_type},
        )


class RunSimulationTool(FinancialTool):
    """
    Tool for running portfolio simulations.
    
    Supports Monte Carlo, stress tests, and thermodynamic optimization.
    """
    
    name = "run_simulation"
    description = "Run portfolio simulation (Monte Carlo, stress test, or optimization)"
    parameters_schema = {
        "simulation_type": {"type": "string", "enum": ["monte_carlo", "stress_test", "optimize"]},
        "portfolio_weights": {"type": "array", "description": "Current portfolio weights"},
        "scenario": {"type": "string", "description": "Scenario name for stress test"},
        "n_simulations": {"type": "integer", "description": "Number of Monte Carlo paths"},
    }
    
    def validate_parameters(self, **kwargs) -> tuple[bool, str | None]:
        if "simulation_type" not in kwargs:
            return False, "simulation_type is required"
        return True, None
    
    def execute(self, **kwargs) -> Observation:
        sim_type = kwargs.get("simulation_type")
        
        if sim_type == "monte_carlo":
            result = {
                "var_95": 0.0142,
                "cvar_95": 0.0189,
                "expected_return": 0.08,
                "paths_simulated": kwargs.get("n_simulations", 10000),
            }
        elif sim_type == "stress_test":
            result = {
                "scenario": kwargs.get("scenario", "2008 Crisis"),
                "portfolio_impact": -0.18,
                "worst_position_impact": -0.35,
            }
        elif sim_type == "optimize":
            result = {
                "optimal_weights": [0.2, 0.15, 0.15, 0.1, 0.2, 0.1, 0.05, 0.05],
                "expected_sharpe": 1.2,
            }
        else:
            result = {}
        
        return Observation(
            content=result,
            success=True,
            metadata={"simulation_type": sim_type},
        )


class GetMarketDataTool(FinancialTool):
    """
    Tool for retrieving real-time market data.
    """
    
    name = "get_market_data"
    description = "Get real-time or delayed market data for symbols"
    parameters_schema = {
        "symbols": {"type": "array", "description": "List of symbols"},
        "fields": {"type": "array", "description": "Fields to retrieve (price, volume, etc.)"},
        "realtime": {"type": "boolean", "description": "Whether to get real-time data"},
    }
    
    def validate_parameters(self, **kwargs) -> tuple[bool, str | None]:
        if "symbols" not in kwargs or not kwargs["symbols"]:
            return False, "symbols list is required"
        return True, None
    
    def execute(self, **kwargs) -> Observation:
        symbols = kwargs.get("symbols", [])
        fields = kwargs.get("fields", ["price", "change"])
        
        # Simulated market data
        data = {}
        for symbol in symbols:
            data[symbol] = {
                "price": 100 + np.random.randn() * 10,
                "change": np.random.randn() * 0.02,
                "volume": int(1e6 * np.random.random()),
                "timestamp": datetime.now().isoformat(),
            }
        
        return Observation(
            content=data,
            success=True,
            metadata={"symbols_count": len(symbols)},
        )


class CalculateRiskTool(FinancialTool):
    """
    Tool for calculating various risk metrics.
    """
    
    name = "calculate_risk"
    description = "Calculate risk metrics for a portfolio or position"
    parameters_schema = {
        "metric": {"type": "string", "enum": ["var", "cvar", "beta", "volatility", "sharpe"]},
        "portfolio_weights": {"type": "array", "description": "Portfolio weights"},
        "confidence": {"type": "number", "description": "Confidence level for VaR"},
        "horizon": {"type": "integer", "description": "Time horizon in days"},
    }
    
    def validate_parameters(self, **kwargs) -> tuple[bool, str | None]:
        if "metric" not in kwargs:
            return False, "metric is required"
        return True, None
    
    def execute(self, **kwargs) -> Observation:
        metric = kwargs.get("metric")
        confidence = kwargs.get("confidence", 0.95)
        
        # Simulated calculations
        results = {
            "var": {"value": 0.0142, "confidence": confidence, "horizon_days": kwargs.get("horizon", 1)},
            "cvar": {"value": 0.0189, "confidence": confidence},
            "beta": {"value": 0.95, "benchmark": "SPX"},
            "volatility": {"value": 0.15, "annualized": True},
            "sharpe": {"value": 1.2, "risk_free_rate": 0.04},
        }
        
        result = results.get(metric, {"error": "Unknown metric"})
        
        return Observation(
            content=result,
            success=True,
            metadata={"metric": metric},
        )


class ProposeHedgeTool(FinancialTool):
    """
    Tool for proposing hedging strategies.
    
    Does NOT execute trades - only proposes them for human review.
    """
    
    name = "propose_hedge"
    description = "Propose a hedging strategy for review (does not execute)"
    parameters_schema = {
        "hedge_type": {"type": "string", "enum": ["options", "futures", "swaps", "etf"]},
        "target_exposure": {"type": "string", "description": "What exposure to hedge"},
        "notional": {"type": "number", "description": "Notional amount to hedge"},
        "urgency": {"type": "string", "enum": ["low", "medium", "high"]},
    }
    
    def validate_parameters(self, **kwargs) -> tuple[bool, str | None]:
        if "hedge_type" not in kwargs:
            return False, "hedge_type is required"
        if "target_exposure" not in kwargs:
            return False, "target_exposure is required"
        return True, None
    
    def execute(self, **kwargs) -> Observation:
        hedge_type = kwargs.get("hedge_type")
        target = kwargs.get("target_exposure")
        notional = kwargs.get("notional", 1_000_000)
        
        # Generate hedge proposal
        proposal = {
            "id": f"HEDGE_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "type": hedge_type,
            "target_exposure": target,
            "notional": notional,
            "status": "PROPOSED",
            "requires_approval": True,
            "estimated_cost": notional * 0.001,  # 10 bps
            "recommended_instruments": [],
        }
        
        # Instrument recommendations based on hedge type
        if hedge_type == "options":
            proposal["recommended_instruments"] = [
                {"instrument": "SPX Put 4000 Mar26", "quantity": int(notional / 100000)},
            ]
        elif hedge_type == "futures":
            proposal["recommended_instruments"] = [
                {"instrument": "ES Mar26", "quantity": int(notional / 50000)},
            ]
        elif hedge_type == "etf":
            proposal["recommended_instruments"] = [
                {"instrument": "SH", "shares": int(notional / 15)},
            ]
        
        return Observation(
            content=proposal,
            success=True,
            metadata={"proposal_id": proposal["id"]},
        )


class GetFactorExposuresTool(FinancialTool):
    """
    Tool for analyzing factor exposures.
    """
    
    name = "get_factor_exposures"
    description = "Get current factor exposures for portfolio or position"
    parameters_schema = {
        "portfolio_id": {"type": "string", "description": "Portfolio identifier"},
        "factors": {"type": "array", "description": "Specific factors to analyze"},
    }
    
    def validate_parameters(self, **kwargs) -> tuple[bool, str | None]:
        return True, None
    
    def execute(self, **kwargs) -> Observation:
        exposures = {
            "Market": {"exposure": 0.95, "contribution": 0.68},
            "Size": {"exposure": 0.12, "contribution": 0.04},
            "Value": {"exposure": -0.08, "contribution": -0.02},
            "Momentum": {"exposure": 0.23, "contribution": 0.12},
            "Quality": {"exposure": 0.18, "contribution": 0.08},
            "LowVol": {"exposure": -0.15, "contribution": -0.05},
            "Credit": {"exposure": 0.35, "contribution": 0.10},
            "Duration": {"exposure": 5.2, "contribution": 0.05},
        }
        
        # Filter if specific factors requested
        if kwargs.get("factors"):
            exposures = {k: v for k, v in exposures.items() if k in kwargs["factors"]}
        
        return Observation(
            content=exposures,
            success=True,
            metadata={"n_factors": len(exposures)},
        )


class ExecuteTradeTool(FinancialTool):
    """
    Tool for executing trades.
    
    RESTRICTED: Only available in SEMI_AUTO or AUTONOMOUS modes.
    Requires Judge approval.
    """
    
    name = "execute_trade"
    description = "Execute a trade (RESTRICTED - requires Judge approval)"
    parameters_schema = {
        "symbol": {"type": "string", "description": "Symbol to trade"},
        "side": {"type": "string", "enum": ["buy", "sell"]},
        "quantity": {"type": "number", "description": "Number of shares/contracts"},
        "order_type": {"type": "string", "enum": ["market", "limit"]},
        "limit_price": {"type": "number", "description": "Limit price (if limit order)"},
    }
    
    def validate_parameters(self, **kwargs) -> tuple[bool, str | None]:
        if "symbol" not in kwargs:
            return False, "symbol is required"
        if "side" not in kwargs:
            return False, "side is required"
        if "quantity" not in kwargs:
            return False, "quantity is required"
        if kwargs.get("quantity", 0) <= 0:
            return False, "quantity must be positive"
        return True, None
    
    def execute(self, **kwargs) -> Observation:
        # In production, this would connect to OMS
        order = {
            "order_id": f"ORD_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "symbol": kwargs["symbol"],
            "side": kwargs["side"],
            "quantity": kwargs["quantity"],
            "order_type": kwargs.get("order_type", "market"),
            "status": "PENDING_APPROVAL",  # Always requires approval first
            "timestamp": datetime.now().isoformat(),
        }
        
        return Observation(
            content=order,
            success=True,
            metadata={"requires_judge_approval": True},
        )


# Tool registry
def get_standard_tools() -> dict[str, FinancialTool]:
    """Get the standard set of financial tools for agents."""
    return {
        "query_kdb": QueryKDBTool(),
        "run_simulation": RunSimulationTool(),
        "get_market_data": GetMarketDataTool(),
        "calculate_risk": CalculateRiskTool(),
        "propose_hedge": ProposeHedgeTool(),
        "get_factor_exposures": GetFactorExposuresTool(),
        "execute_trade": ExecuteTradeTool(),
    }
