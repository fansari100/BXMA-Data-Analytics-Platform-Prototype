"""
Workflow Management System
==========================

Task and approval workflow management with JIRA integration.

Enables seamless collaboration across BXMA groups for:
- Risk limit changes
- Model updates
- Report requests
- Data corrections

Author: BXMA Quant Team
Date: January 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Any
from enum import Enum, auto
import uuid


class TaskStatus(Enum):
    """Workflow task status."""
    DRAFT = auto()
    PENDING = auto()
    IN_PROGRESS = auto()
    REVIEW = auto()
    APPROVED = auto()
    REJECTED = auto()
    COMPLETED = auto()
    CANCELLED = auto()


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class ApprovalStatus(Enum):
    """Approval request status."""
    PENDING = auto()
    APPROVED = auto()
    REJECTED = auto()
    ESCALATED = auto()


@dataclass
class WorkflowTask:
    """A workflow task for cross-team collaboration."""
    
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Task details
    title: str = ""
    description: str = ""
    task_type: str = ""  # risk_limit_change, model_update, data_request, etc.
    
    # Ownership
    created_by: str = ""
    assigned_to: str = ""
    team: str = ""
    
    # Status
    status: TaskStatus = TaskStatus.DRAFT
    priority: TaskPriority = TaskPriority.MEDIUM
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    due_date: datetime | None = None
    completed_at: datetime | None = None
    
    # Dependencies
    depends_on: list[str] = field(default_factory=list)
    blocks: list[str] = field(default_factory=list)
    
    # Approvals required
    requires_approval: bool = False
    approvers: list[str] = field(default_factory=list)
    
    # Attachments and context
    attachments: list[str] = field(default_factory=list)
    context_data: dict[str, Any] = field(default_factory=dict)
    
    # JIRA integration
    jira_key: str | None = None
    jira_url: str | None = None
    
    # Audit
    history: list[dict] = field(default_factory=list)
    
    def add_history(self, action: str, user: str, details: str = ""):
        """Add history entry."""
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "user": user,
            "details": details,
        })


@dataclass
class ApprovalRequest:
    """An approval request within a workflow."""
    
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Related task
    task_id: str = ""
    
    # Request details
    request_type: str = ""
    description: str = ""
    
    # Requester
    requested_by: str = ""
    requested_at: datetime = field(default_factory=datetime.now)
    
    # Approver
    approver: str = ""
    status: ApprovalStatus = ApprovalStatus.PENDING
    
    # Response
    responded_at: datetime | None = None
    response_notes: str = ""
    
    # Escalation
    escalated: bool = False
    escalated_to: str | None = None
    escalation_reason: str = ""


class WorkflowManager:
    """
    Manages workflows and approvals across BXMA teams.
    
    Integrates with:
    - JIRA for task tracking
    - Email/Slack for notifications
    - Audit system for compliance
    """
    
    def __init__(self):
        self.tasks: dict[str, WorkflowTask] = {}
        self.approvals: dict[str, ApprovalRequest] = {}
        self._jira_client = None
        self._notification_handlers: list[Callable] = []
    
    def create_task(
        self,
        title: str,
        description: str,
        task_type: str,
        created_by: str,
        team: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        due_date: datetime | None = None,
        requires_approval: bool = False,
        approvers: list[str] | None = None,
    ) -> WorkflowTask:
        """Create a new workflow task."""
        task = WorkflowTask(
            title=title,
            description=description,
            task_type=task_type,
            created_by=created_by,
            team=team,
            priority=priority,
            due_date=due_date,
            requires_approval=requires_approval,
            approvers=approvers or [],
        )
        
        task.add_history("created", created_by)
        self.tasks[task.task_id] = task
        
        # Sync to JIRA if configured
        if self._jira_client:
            self._sync_to_jira(task)
        
        return task
    
    def assign_task(self, task_id: str, assignee: str, assigned_by: str):
        """Assign a task to a team member."""
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        
        task.assigned_to = assignee
        task.status = TaskStatus.PENDING
        task.add_history("assigned", assigned_by, f"Assigned to {assignee}")
        
        self._notify(task, "assigned", assignee)
    
    def start_task(self, task_id: str, user: str):
        """Mark a task as in progress."""
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        
        task.status = TaskStatus.IN_PROGRESS
        task.add_history("started", user)
    
    def submit_for_review(self, task_id: str, user: str):
        """Submit a task for review/approval."""
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        
        task.status = TaskStatus.REVIEW
        task.add_history("submitted_for_review", user)
        
        # Create approval requests
        if task.requires_approval:
            for approver in task.approvers:
                self._create_approval_request(task, user, approver)
    
    def _create_approval_request(
        self,
        task: WorkflowTask,
        requested_by: str,
        approver: str,
    ) -> ApprovalRequest:
        """Create an approval request."""
        request = ApprovalRequest(
            task_id=task.task_id,
            request_type=task.task_type,
            description=f"Approval required for: {task.title}",
            requested_by=requested_by,
            approver=approver,
        )
        
        self.approvals[request.request_id] = request
        self._notify(task, "approval_required", approver)
        
        return request
    
    def approve(self, request_id: str, approver: str, notes: str = ""):
        """Approve an approval request."""
        request = self.approvals.get(request_id)
        if not request:
            raise ValueError(f"Approval request not found: {request_id}")
        
        if request.approver != approver:
            raise PermissionError(f"User {approver} is not the designated approver")
        
        request.status = ApprovalStatus.APPROVED
        request.responded_at = datetime.now()
        request.response_notes = notes
        
        task = self.tasks.get(request.task_id)
        if task:
            task.add_history("approved", approver, notes)
            
            # Check if all approvals are complete
            pending_approvals = [
                a for a in self.approvals.values()
                if a.task_id == task.task_id and a.status == ApprovalStatus.PENDING
            ]
            
            if not pending_approvals:
                task.status = TaskStatus.APPROVED
                self._notify(task, "all_approved", task.created_by)
    
    def reject(self, request_id: str, approver: str, reason: str):
        """Reject an approval request."""
        request = self.approvals.get(request_id)
        if not request:
            raise ValueError(f"Approval request not found: {request_id}")
        
        request.status = ApprovalStatus.REJECTED
        request.responded_at = datetime.now()
        request.response_notes = reason
        
        task = self.tasks.get(request.task_id)
        if task:
            task.status = TaskStatus.REJECTED
            task.add_history("rejected", approver, reason)
            self._notify(task, "rejected", task.created_by)
    
    def complete_task(self, task_id: str, user: str):
        """Mark a task as completed."""
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now()
        task.add_history("completed", user)
        
        self._notify(task, "completed", task.created_by)
    
    def get_tasks_for_user(self, user_id: str) -> list[WorkflowTask]:
        """Get all tasks assigned to a user."""
        return [t for t in self.tasks.values() if t.assigned_to == user_id]
    
    def get_pending_approvals(self, approver_id: str) -> list[ApprovalRequest]:
        """Get pending approvals for a user."""
        return [
            a for a in self.approvals.values()
            if a.approver == approver_id and a.status == ApprovalStatus.PENDING
        ]
    
    def get_overdue_tasks(self) -> list[WorkflowTask]:
        """Get all overdue tasks."""
        now = datetime.now()
        return [
            t for t in self.tasks.values()
            if t.due_date and t.due_date < now and t.status not in [
                TaskStatus.COMPLETED, TaskStatus.CANCELLED
            ]
        ]
    
    def _notify(self, task: WorkflowTask, event: str, recipient: str):
        """Send notification."""
        for handler in self._notification_handlers:
            try:
                handler(task, event, recipient)
            except Exception as e:
                print(f"Notification handler error: {e}")
    
    def _sync_to_jira(self, task: WorkflowTask):
        """Sync task to JIRA."""
        # In production, this would use the JIRA API
        pass
    
    def register_notification_handler(self, handler: Callable):
        """Register a notification handler."""
        self._notification_handlers.append(handler)


# Standard workflow types
WORKFLOW_TYPES = {
    "risk_limit_change": {
        "name": "Risk Limit Change",
        "requires_approval": True,
        "approvers": ["risk_manager", "investment_head"],
        "teams": ["RISK_QUANT", "INVESTMENT"],
    },
    "model_update": {
        "name": "Risk Model Update",
        "requires_approval": True,
        "approvers": ["model_validator", "risk_manager"],
        "teams": ["RISK_QUANT"],
    },
    "data_request": {
        "name": "Ad-Hoc Data Request",
        "requires_approval": False,
        "teams": ["RISK_QUANT", "OPERATIONS", "INVESTMENT"],
    },
    "report_request": {
        "name": "Custom Report Request",
        "requires_approval": False,
        "teams": ["RISK_QUANT", "INVESTMENT", "LEGAL"],
    },
    "trade_exception": {
        "name": "Trade Exception Request",
        "requires_approval": True,
        "approvers": ["compliance_officer", "investment_head"],
        "teams": ["LEGAL", "INVESTMENT", "OPERATIONS"],
    },
    "guideline_breach": {
        "name": "Investment Guideline Breach",
        "requires_approval": True,
        "approvers": ["compliance_officer", "risk_manager"],
        "teams": ["LEGAL", "RISK_QUANT", "INVESTMENT"],
    },
}
