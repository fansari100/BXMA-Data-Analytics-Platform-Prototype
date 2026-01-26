"""
BXMA Integration Module
=======================

Anti-Corruption Layer (ACL) for legacy system integration.

Components:
- RiskMetrics integration
- Third-party data feeds
- Legacy protocol adapters
- Message format converters

Author: BXMA Quant Team
Date: January 2026
"""

from bxma.integration.acl import (
    AntiCorruptionLayer,
    LegacyAdapter,
    MessageConverter,
    SchemaTransformer,
)

from bxma.integration.riskmetrics import (
    RiskMetricsAdapter,
    RiskMetricsConfig,
    RiskMetricsData,
)

from bxma.integration.data_feeds import (
    DataFeedManager,
    BloombergAdapter,
    ReutersAdapter,
    FeedConfig,
)

__all__ = [
    "AntiCorruptionLayer",
    "LegacyAdapter",
    "MessageConverter",
    "SchemaTransformer",
    "RiskMetricsAdapter",
    "RiskMetricsConfig",
    "RiskMetricsData",
    "DataFeedManager",
    "BloombergAdapter",
    "ReutersAdapter",
    "FeedConfig",
]
