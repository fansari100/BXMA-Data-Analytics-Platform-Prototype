"""
Audit Trail System
==================

Comprehensive audit logging for compliance and regulatory requirements.

Tracks all significant actions across the BXMA platform:
- User actions
- System changes
- Data modifications
- Approval workflows
- Risk limit changes

Author: BXMA Quant Team
Date: January 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from enum import Enum, auto
import uuid
import json


class AuditEventType(Enum):
    """Types of auditable events."""
    # User actions
    USER_LOGIN = auto()
    USER_LOGOUT = auto()
    USER_ACTION = auto()
    
    # Data operations
    DATA_CREATE = auto()
    DATA_READ = auto()
    DATA_UPDATE = auto()
    DATA_DELETE = auto()
    DATA_EXPORT = auto()
    
    # Portfolio operations
    PORTFOLIO_VIEW = auto()
    PORTFOLIO_MODIFY = auto()
    TRADE_EXECUTE = auto()
    TRADE_CANCEL = auto()
    
    # Risk operations
    RISK_CALCULATE = auto()
    RISK_LIMIT_CHANGE = auto()
    RISK_BREACH = auto()
    
    # Model operations
    MODEL_CREATE = auto()
    MODEL_UPDATE = auto()
    MODEL_VALIDATE = auto()
    MODEL_DEPLOY = auto()
    
    # Report operations
    REPORT_GENERATE = auto()
    REPORT_DISTRIBUTE = auto()
    
    # Workflow operations
    WORKFLOW_CREATE = auto()
    WORKFLOW_APPROVE = auto()
    WORKFLOW_REJECT = auto()
    WORKFLOW_COMPLETE = auto()
    
    # System operations
    SYSTEM_CONFIG_CHANGE = auto()
    SYSTEM_ACCESS_GRANT = auto()
    SYSTEM_ACCESS_REVOKE = auto()


@dataclass
class AuditEvent:
    """An auditable event."""
    
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Event classification
    event_type: AuditEventType = AuditEventType.USER_ACTION
    category: str = ""
    action: str = ""
    
    # Actor
    user_id: str = ""
    user_name: str = ""
    user_team: str = ""
    ip_address: str = ""
    
    # Target
    resource_type: str = ""
    resource_id: str = ""
    resource_name: str = ""
    
    # Details
    description: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    
    # State changes
    before_state: dict[str, Any] | None = None
    after_state: dict[str, Any] | None = None
    
    # Timing
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float | None = None
    
    # Outcome
    success: bool = True
    error_message: str | None = None
    
    # Compliance
    requires_review: bool = False
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.name,
            "category": self.category,
            "action": self.action,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "user_team": self.user_team,
            "ip_address": self.ip_address,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "resource_name": self.resource_name,
            "description": self.description,
            "details": self.details,
            "before_state": self.before_state,
            "after_state": self.after_state,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error_message": self.error_message,
            "requires_review": self.requires_review,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
        }


class AuditLogger:
    """
    Comprehensive audit logging system.
    
    Provides:
    - Immutable audit trail
    - Compliance reporting
    - Search and analysis
    - Retention management
    """
    
    def __init__(self, storage_backend: str = "memory"):
        self.storage_backend = storage_backend
        self._events: list[AuditEvent] = []
        self._event_handlers: list[callable] = []
    
    def log(self, event: AuditEvent):
        """Log an audit event."""
        self._events.append(event)
        
        # Notify handlers
        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"Audit handler error: {e}")
        
        # In production, persist to database
        self._persist(event)
    
    def log_action(
        self,
        event_type: AuditEventType,
        user_id: str,
        action: str,
        resource_type: str = "",
        resource_id: str = "",
        details: dict | None = None,
        success: bool = True,
    ) -> AuditEvent:
        """Log a user action."""
        event = AuditEvent(
            event_type=event_type,
            action=action,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            success=success,
        )
        
        self.log(event)
        return event
    
    def log_data_change(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        before_state: dict | None,
        after_state: dict | None,
        action: str = "update",
    ) -> AuditEvent:
        """Log a data change with before/after state."""
        event_type = {
            "create": AuditEventType.DATA_CREATE,
            "update": AuditEventType.DATA_UPDATE,
            "delete": AuditEventType.DATA_DELETE,
        }.get(action, AuditEventType.DATA_UPDATE)
        
        event = AuditEvent(
            event_type=event_type,
            action=action,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            before_state=before_state,
            after_state=after_state,
        )
        
        self.log(event)
        return event
    
    def log_risk_limit_change(
        self,
        user_id: str,
        portfolio_id: str,
        limit_type: str,
        old_value: float,
        new_value: float,
        reason: str,
        approved_by: str | None = None,
    ) -> AuditEvent:
        """Log a risk limit change."""
        event = AuditEvent(
            event_type=AuditEventType.RISK_LIMIT_CHANGE,
            action="risk_limit_change",
            user_id=user_id,
            resource_type="risk_limit",
            resource_id=f"{portfolio_id}_{limit_type}",
            description=f"Changed {limit_type} limit for {portfolio_id}",
            details={
                "portfolio_id": portfolio_id,
                "limit_type": limit_type,
                "reason": reason,
                "approved_by": approved_by,
            },
            before_state={"value": old_value},
            after_state={"value": new_value},
            requires_review=True,
        )
        
        self.log(event)
        return event
    
    def log_model_deployment(
        self,
        user_id: str,
        model_name: str,
        model_version: str,
        validation_status: str,
    ) -> AuditEvent:
        """Log a model deployment."""
        event = AuditEvent(
            event_type=AuditEventType.MODEL_DEPLOY,
            action="model_deploy",
            user_id=user_id,
            resource_type="risk_model",
            resource_id=f"{model_name}_v{model_version}",
            description=f"Deployed {model_name} version {model_version}",
            details={
                "model_name": model_name,
                "version": model_version,
                "validation_status": validation_status,
            },
            requires_review=True,
        )
        
        self.log(event)
        return event
    
    def search(
        self,
        event_type: AuditEventType | None = None,
        user_id: str | None = None,
        resource_type: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Search audit events."""
        results = []
        
        for event in reversed(self._events):
            if event_type and event.event_type != event_type:
                continue
            if user_id and event.user_id != user_id:
                continue
            if resource_type and event.resource_type != resource_type:
                continue
            if start_date and event.timestamp < start_date:
                continue
            if end_date and event.timestamp > end_date:
                continue
            
            results.append(event)
            
            if len(results) >= limit:
                break
        
        return results
    
    def get_events_requiring_review(self) -> list[AuditEvent]:
        """Get events that require compliance review."""
        return [e for e in self._events if e.requires_review and not e.reviewed_by]
    
    def mark_reviewed(self, event_id: str, reviewer_id: str):
        """Mark an event as reviewed."""
        for event in self._events:
            if event.event_id == event_id:
                event.reviewed_by = reviewer_id
                event.reviewed_at = datetime.now()
                break
    
    def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> dict:
        """Generate a compliance report for a date range."""
        events = self.search(start_date=start_date, end_date=end_date, limit=10000)
        
        # Categorize events
        by_type = {}
        by_user = {}
        high_risk = []
        
        for event in events:
            # By type
            type_name = event.event_type.name
            if type_name not in by_type:
                by_type[type_name] = 0
            by_type[type_name] += 1
            
            # By user
            if event.user_id not in by_user:
                by_user[event.user_id] = 0
            by_user[event.user_id] += 1
            
            # High risk events
            if event.requires_review:
                high_risk.append(event.to_dict())
        
        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "summary": {
                "total_events": len(events),
                "events_by_type": by_type,
                "events_by_user": by_user,
                "high_risk_events": len(high_risk),
            },
            "high_risk_events": high_risk,
            "generated_at": datetime.now().isoformat(),
        }
    
    def _persist(self, event: AuditEvent):
        """Persist event to storage."""
        if self.storage_backend == "memory":
            return
        
        # In production, write to database/log aggregator
        # Examples: PostgreSQL, Elasticsearch, Splunk
        pass
    
    def register_handler(self, handler: callable):
        """Register an event handler."""
        self._event_handlers.append(handler)
