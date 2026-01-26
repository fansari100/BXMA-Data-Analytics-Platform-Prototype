"""
BXMA Collaboration Module
=========================

Cross-functional collaboration tools for seamless integration with:
- Investment Teams
- Operations
- Treasury
- Legal

Components:
- Workflow management (JIRA integration)
- Documentation (Confluence integration)
- Notification system
- Approval workflows
- Audit trails

Author: BXMA Quant Team
Date: January 2026
"""

from bxma.collaboration.workflow import (
    WorkflowManager,
    WorkflowTask,
    TaskStatus,
    TaskPriority,
    ApprovalRequest,
    ApprovalStatus,
)

from bxma.collaboration.teams import (
    TeamType,
    TeamMember,
    TeamDirectory,
    NotificationPreference,
)

from bxma.collaboration.notifications import (
    NotificationService,
    NotificationType,
    NotificationChannel,
    Notification,
)

from bxma.collaboration.audit import (
    AuditLogger,
    AuditEvent,
    AuditEventType,
)

__all__ = [
    "WorkflowManager",
    "WorkflowTask",
    "TaskStatus",
    "TaskPriority",
    "ApprovalRequest",
    "ApprovalStatus",
    "TeamType",
    "TeamMember",
    "TeamDirectory",
    "NotificationPreference",
    "NotificationService",
    "NotificationType",
    "NotificationChannel",
    "Notification",
    "AuditLogger",
    "AuditEvent",
    "AuditEventType",
]
