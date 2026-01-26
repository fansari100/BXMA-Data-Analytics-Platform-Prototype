"""
Performance Attribution Engine for BXMA Risk/Quant Platform.

Implements comprehensive attribution methodologies:
- Brinson-Fachler (arithmetic)
- Brinson-Hood-Beebower
- Geometric attribution (Cariño, Menchero, GRAP)
- Multi-period linking (Frongello, Davies)
- Factor-based attribution
- Risk attribution
- Fixed income attribution (Campisi)

References:
- "Determinants of Portfolio Performance" (Brinson, Hood, Beebower, 1986)
- "Geometric Attribution" (Cariño, 1999)
- "A Fully Geometric Approach to Attribution" (Menchero, 2000)
"""

from bxma.attribution.brinson import (
    BrinsonFachlerAttribution,
    BrinsonHoodBeebowerAttribution,
)
from bxma.attribution.geometric import (
    GeometricAttribution,
    CarinoAttribution,
    MencheroAttribution,
)
from bxma.attribution.linking import (
    FrongelloLinking,
    DaviesLinking,
    GeometricLinking,
)
from bxma.attribution.factor_attribution import (
    FactorAttribution,
    RiskAttribution,
)

__all__ = [
    "BrinsonFachlerAttribution",
    "BrinsonHoodBeebowerAttribution",
    "GeometricAttribution",
    "CarinoAttribution",
    "MencheroAttribution",
    "FrongelloLinking",
    "DaviesLinking",
    "GeometricLinking",
    "FactorAttribution",
    "RiskAttribution",
]
