"""
Stress Testing and Scenario Analysis for BXMA Data Analytics Platform.

Implements comprehensive stress testing capabilities:
- Historical scenario replay
- Hypothetical scenario construction
- Factor shock analysis
- Monte Carlo simulation
- Reverse stress testing

Designed for regulatory compliance and risk management.
"""

from bxma.stress_testing.scenarios import (
    ScenarioEngine,
    ScenarioDefinition,
    ScenarioResult,
    STANDARD_SCENARIOS,
)

__all__ = [
    "ScenarioEngine",
    "ScenarioDefinition",
    "ScenarioResult",
    "STANDARD_SCENARIOS",
]
