"""
Cross-Functional Team Management
================================

Manages collaboration across BXMA groups:
- Investment Teams
- Operations
- Treasury
- Legal
- Risk/Quant

CRITICAL REQUIREMENT: Ensure seamless integration, alignment,
and effective collaboration across all BXMA groups.

Author: BXMA Quant Team
Date: January 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
from enum import Enum, auto


class TeamType(Enum):
    """BXMA team types for cross-functional collaboration."""
    INVESTMENT = auto()      # Investment decision makers
    OPERATIONS = auto()      # Trade operations, settlements
    TREASURY = auto()        # Cash management, funding
    LEGAL = auto()           # Compliance, regulatory
    RISK_QUANT = auto()      # Risk analytics, quant research
    TECHNOLOGY = auto()      # IT, infrastructure
    EXECUTIVE = auto()       # Senior leadership


class NotificationPreference(Enum):
    """Notification delivery preferences."""
    EMAIL = auto()
    SLACK = auto()
    TEAMS = auto()
    SMS = auto()
    IN_APP = auto()
    ALL = auto()


@dataclass
class TeamMember:
    """A team member in the BXMA organization."""
    
    user_id: str
    name: str
    email: str
    team: TeamType
    
    # Role
    title: str = ""
    is_manager: bool = False
    is_approver: bool = False
    
    # Notifications
    notification_pref: NotificationPreference = NotificationPreference.EMAIL
    
    # Permissions
    can_view_all_portfolios: bool = False
    can_execute_trades: bool = False
    can_approve_risk_limits: bool = False
    can_modify_models: bool = False
    
    # Status
    is_active: bool = True
    last_login: datetime | None = None


@dataclass
class TeamDirectory:
    """
    Directory of all BXMA teams and members.
    
    Enables cross-functional collaboration by providing:
    - Team contact lookup
    - Escalation paths
    - Approval routing
    """
    
    members: dict[str, TeamMember] = field(default_factory=dict)
    
    # Team leads for escalation
    team_leads: dict[TeamType, str] = field(default_factory=dict)
    
    # Approval chains
    approval_chains: dict[str, list[str]] = field(default_factory=dict)
    
    def add_member(self, member: TeamMember):
        """Add a team member."""
        self.members[member.user_id] = member
        if member.is_manager:
            self.team_leads[member.team] = member.user_id
    
    def get_team_members(self, team: TeamType) -> list[TeamMember]:
        """Get all members of a team."""
        return [m for m in self.members.values() if m.team == team]
    
    def get_approvers(self, team: TeamType) -> list[TeamMember]:
        """Get approvers for a team."""
        return [m for m in self.members.values() if m.team == team and m.is_approver]
    
    def get_escalation_path(self, team: TeamType) -> list[TeamMember]:
        """Get escalation path for a team."""
        path = []
        
        # Team lead
        if team in self.team_leads:
            lead_id = self.team_leads[team]
            if lead_id in self.members:
                path.append(self.members[lead_id])
        
        # Executive escalation
        exec_members = self.get_team_members(TeamType.EXECUTIVE)
        path.extend(exec_members)
        
        return path
    
    def find_by_email(self, email: str) -> TeamMember | None:
        """Find member by email."""
        for member in self.members.values():
            if member.email.lower() == email.lower():
                return member
        return None


# Standard BXMA team responsibilities
TEAM_RESPONSIBILITIES = {
    TeamType.INVESTMENT: {
        "name": "Investment Teams",
        "responsibilities": [
            "Portfolio strategy and allocation decisions",
            "Security selection and trade ideas",
            "Market analysis and research",
            "Investment performance review",
        ],
        "data_access": [
            "Portfolio holdings and NAV",
            "Performance attribution",
            "Market data and analytics",
            "Risk reports and limits",
        ],
        "collaboration_needs": [
            "Real-time risk metrics",
            "Optimization recommendations",
            "What-if scenario analysis",
            "Trade impact analysis",
        ],
    },
    TeamType.OPERATIONS: {
        "name": "Operations",
        "responsibilities": [
            "Trade execution and settlement",
            "Position reconciliation",
            "Corporate actions processing",
            "NAV calculation support",
        ],
        "data_access": [
            "Trade blotter",
            "Position data",
            "Settlement status",
            "Reconciliation breaks",
        ],
        "collaboration_needs": [
            "Trade fail alerts",
            "Position discrepancy flags",
            "Settlement projections",
            "Cash flow forecasts",
        ],
    },
    TeamType.TREASURY: {
        "name": "Treasury",
        "responsibilities": [
            "Cash and liquidity management",
            "Funding and financing",
            "Currency hedging",
            "Collateral management",
        ],
        "data_access": [
            "Cash positions",
            "Margin requirements",
            "Funding rates",
            "Collateral balances",
        ],
        "collaboration_needs": [
            "Cash flow projections",
            "Liquidity stress tests",
            "Margin call alerts",
            "Funding cost analysis",
        ],
    },
    TeamType.LEGAL: {
        "name": "Legal & Compliance",
        "responsibilities": [
            "Regulatory compliance monitoring",
            "Investment guideline enforcement",
            "Restricted list management",
            "Regulatory reporting",
        ],
        "data_access": [
            "Compliance reports",
            "Guideline breaches",
            "Regulatory filings",
            "Audit trails",
        ],
        "collaboration_needs": [
            "Pre-trade compliance checks",
            "Breach notifications",
            "Regulatory deadline tracking",
            "Policy change alerts",
        ],
    },
    TeamType.RISK_QUANT: {
        "name": "Risk/Quant",
        "responsibilities": [
            "Risk model development and validation",
            "Performance and attribution analytics",
            "Portfolio optimization",
            "Data infrastructure management",
        ],
        "data_access": [
            "All portfolio data",
            "Market data",
            "Risk models",
            "Historical data",
        ],
        "collaboration_needs": [
            "Cross-team data requests",
            "Model validation support",
            "Ad-hoc analytics requests",
            "System integration support",
        ],
    },
}


def create_sample_directory() -> TeamDirectory:
    """Create a sample team directory for testing."""
    directory = TeamDirectory()
    
    # Sample members
    sample_members = [
        TeamMember(
            user_id="INV001",
            name="John Smith",
            email="john.smith@blackstone.com",
            team=TeamType.INVESTMENT,
            title="Senior Portfolio Manager",
            is_manager=True,
            is_approver=True,
            can_view_all_portfolios=True,
        ),
        TeamMember(
            user_id="OPS001",
            name="Jane Doe",
            email="jane.doe@blackstone.com",
            team=TeamType.OPERATIONS,
            title="Operations Manager",
            is_manager=True,
            is_approver=True,
        ),
        TeamMember(
            user_id="TRE001",
            name="Bob Johnson",
            email="bob.johnson@blackstone.com",
            team=TeamType.TREASURY,
            title="Treasury Director",
            is_manager=True,
            is_approver=True,
        ),
        TeamMember(
            user_id="LEG001",
            name="Alice Williams",
            email="alice.williams@blackstone.com",
            team=TeamType.LEGAL,
            title="Chief Compliance Officer",
            is_manager=True,
            is_approver=True,
        ),
        TeamMember(
            user_id="RQ001",
            name="Farooq Ansari",
            email="farooq.ansari@blackstone.com",
            team=TeamType.RISK_QUANT,
            title="Risk Analyst",
            can_modify_models=True,
            can_view_all_portfolios=True,
        ),
    ]
    
    for member in sample_members:
        directory.add_member(member)
    
    return directory
