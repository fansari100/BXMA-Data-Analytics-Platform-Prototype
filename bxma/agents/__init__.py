"""
Agentic AI Module for BXMA Risk/Quant Platform
===============================================

Implements autonomous risk management agents using the ReAct
(Reason + Act) pattern with Dual LLM security for prompt injection protection.

Components:
- RiskSwarm: Hierarchical multi-agent system
- ReActAgent: Base agent with thought-action-observation loop
- FinancialTools: Safe tool interfaces for market operations
- DualLLMGateway: Security layer for prompt injection protection
- JudgeAgent: Compliance firewall (non-LLM)

Author: BXMA Quant Team
Date: January 2026
"""

from bxma.agents.react import (
    ReActAgent,
    Thought,
    Action,
    Observation,
    AgentState,
    AgentConfig,
)
from bxma.agents.swarm import (
    RiskSwarm,
    ArchitectAgent,
    AnalystAgent,
    JudgeAgent,
    SwarmConfig,
)
from bxma.agents.tools import (
    FinancialTool,
    QueryKDBTool,
    RunSimulationTool,
    GetMarketDataTool,
    CalculateRiskTool,
    ProposeHedgeTool,
)
from bxma.agents.security import (
    DualLLMGateway,
    ReaderLLM,
    ThinkerLLM,
    SanitizedInput,
    SecurityConfig,
)

__all__ = [
    # ReAct
    "ReActAgent",
    "Thought",
    "Action",
    "Observation",
    "AgentState",
    "AgentConfig",
    # Swarm
    "RiskSwarm",
    "ArchitectAgent",
    "AnalystAgent",
    "JudgeAgent",
    "SwarmConfig",
    # Tools
    "FinancialTool",
    "QueryKDBTool",
    "RunSimulationTool",
    "GetMarketDataTool",
    "CalculateRiskTool",
    "ProposeHedgeTool",
    # Security
    "DualLLMGateway",
    "ReaderLLM",
    "ThinkerLLM",
    "SanitizedInput",
    "SecurityConfig",
]
