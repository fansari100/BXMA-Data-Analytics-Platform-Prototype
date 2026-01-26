"""
Risk Reporting Stack for BXMA Risk/Quant Platform.

Integrates multiple data sources and provides comprehensive reporting:
- RiskMetrics Integration (MSCI/BARRA models, EWMA covariance)
- Tableau Integration (exports, data sources, dashboards)
- Third-party data connectors (Bloomberg, Reuters)
- Proprietary data pipelines
- Real-time dashboards
- Automated report generation

Designed for cross-functional collaboration with Investment,
Operations, Treasury, and Legal teams.

CRITICAL: Fulfills requirement for building risk reporting stack
integrating RiskMetrics, other third-party, and proprietary sources.
"""

from bxma.reporting.riskmetrics import (
    RiskMetricsConnector,
    RiskMetricsData,
    compute_riskmetrics_covariance,
)
from bxma.reporting.dashboard import RiskDashboard, DashboardConfig
from bxma.reporting.tableau import (
    TableauExporter,
    TableauExportConfig,
    TableauDataSource,
    TableauServerPublisher,
    BXMA_TABLEAU_EXPORTS,
)

__all__ = [
    # RiskMetrics Integration
    "RiskMetricsConnector",
    "RiskMetricsData",
    "compute_riskmetrics_covariance",
    # Dashboard
    "RiskDashboard",
    "DashboardConfig",
    # Tableau Integration
    "TableauExporter",
    "TableauExportConfig",
    "TableauDataSource",
    "TableauServerPublisher",
    "BXMA_TABLEAU_EXPORTS",
]
