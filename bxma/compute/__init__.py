"""
BXMA Compute Module
===================

Distributed computing infrastructure for the Titan-X platform.

Components:
- Ray-based distributed actors for risk computation
- Zero-copy Arrow data transfer
- GPU acceleration abstractions
- Elastic scaling with KubeRay integration

Author: BXMA Quant Team
Date: January 2026
"""

from bxma.compute.ray_fabric import (
    RiskActor,
    RiskActorConfig,
    DistributedRiskEngine,
    ComputeCluster,
    ComputeTask,
    ComputeResult,
)

from bxma.compute.arrow_io import (
    ArrowBuffer,
    ZeroCopyTransfer,
    SharedMemoryManager,
    ArrowSerializer,
    RecordBatchBuilder,
)

__all__ = [
    "RiskActor",
    "RiskActorConfig", 
    "DistributedRiskEngine",
    "ComputeCluster",
    "ComputeTask",
    "ComputeResult",
    "ArrowBuffer",
    "ZeroCopyTransfer",
    "SharedMemoryManager",
    "ArrowSerializer",
    "RecordBatchBuilder",
]
