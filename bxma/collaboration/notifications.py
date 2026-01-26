"""
Notification System
===================

Multi-channel notification delivery for cross-team alerts:
- Email
- Slack
- Microsoft Teams
- SMS (critical alerts)
- In-app notifications

Author: BXMA Quant Team
Date: January 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Any
from enum import Enum, auto
import uuid


class NotificationType(Enum):
    """Types of notifications."""
    TASK_ASSIGNED = auto()
    TASK_COMPLETED = auto()
    APPROVAL_REQUIRED = auto()
    APPROVAL_GRANTED = auto()
    APPROVAL_REJECTED = auto()
    RISK_ALERT = auto()
    LIMIT_BREACH = auto()
    SYSTEM_ALERT = auto()
    REPORT_READY = auto()
    DATA_UPDATE = auto()


class NotificationChannel(Enum):
    """Notification delivery channels."""
    EMAIL = auto()
    SLACK = auto()
    TEAMS = auto()
    SMS = auto()
    IN_APP = auto()


class NotificationPriority(Enum):
    """Notification priority levels."""
    CRITICAL = 1    # Immediate delivery on all channels
    HIGH = 2        # Quick delivery on primary channel
    MEDIUM = 3      # Normal delivery
    LOW = 4         # Batched delivery


@dataclass
class Notification:
    """A notification to be delivered."""
    
    notification_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Content
    title: str = ""
    message: str = ""
    notification_type: NotificationType = NotificationType.SYSTEM_ALERT
    
    # Recipient
    recipient_id: str = ""
    recipient_email: str = ""
    
    # Delivery
    channels: list[NotificationChannel] = field(default_factory=list)
    priority: NotificationPriority = NotificationPriority.MEDIUM
    
    # Status
    created_at: datetime = field(default_factory=datetime.now)
    delivered_at: dict[NotificationChannel, datetime] = field(default_factory=dict)
    read_at: datetime | None = None
    
    # Context
    context_data: dict[str, Any] = field(default_factory=dict)
    action_url: str | None = None


class NotificationService:
    """
    Multi-channel notification delivery service.
    
    Enables real-time alerts across BXMA teams for:
    - Risk limit breaches
    - Approval requests
    - Task updates
    - System alerts
    """
    
    def __init__(self):
        self._pending: list[Notification] = []
        self._sent: list[Notification] = []
        self._channel_handlers: dict[NotificationChannel, Callable] = {}
        
        # Register default handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default channel handlers."""
        self._channel_handlers[NotificationChannel.EMAIL] = self._send_email
        self._channel_handlers[NotificationChannel.SLACK] = self._send_slack
        self._channel_handlers[NotificationChannel.TEAMS] = self._send_teams
        self._channel_handlers[NotificationChannel.SMS] = self._send_sms
        self._channel_handlers[NotificationChannel.IN_APP] = self._send_in_app
    
    def send(self, notification: Notification) -> bool:
        """Send a notification."""
        success = True
        
        for channel in notification.channels:
            handler = self._channel_handlers.get(channel)
            if handler:
                try:
                    handler(notification)
                    notification.delivered_at[channel] = datetime.now()
                except Exception as e:
                    print(f"Failed to send via {channel.name}: {e}")
                    success = False
        
        self._sent.append(notification)
        return success
    
    def send_risk_alert(
        self,
        recipient_id: str,
        recipient_email: str,
        title: str,
        message: str,
        severity: str = "high",
        context: dict | None = None,
    ) -> Notification:
        """Send a risk alert notification."""
        channels = [NotificationChannel.EMAIL, NotificationChannel.IN_APP]
        priority = NotificationPriority.MEDIUM
        
        if severity == "critical":
            channels.extend([NotificationChannel.SLACK, NotificationChannel.SMS])
            priority = NotificationPriority.CRITICAL
        elif severity == "high":
            channels.append(NotificationChannel.SLACK)
            priority = NotificationPriority.HIGH
        
        notification = Notification(
            title=title,
            message=message,
            notification_type=NotificationType.RISK_ALERT,
            recipient_id=recipient_id,
            recipient_email=recipient_email,
            channels=channels,
            priority=priority,
            context_data=context or {},
        )
        
        self.send(notification)
        return notification
    
    def send_limit_breach(
        self,
        recipient_id: str,
        recipient_email: str,
        limit_type: str,
        current_value: float,
        limit_value: float,
        portfolio_name: str,
    ) -> Notification:
        """Send a limit breach notification."""
        pct_breach = ((current_value - limit_value) / limit_value) * 100
        
        title = f"⚠️ Limit Breach: {limit_type}"
        message = (
            f"Portfolio '{portfolio_name}' has breached the {limit_type} limit.\n\n"
            f"Current: {current_value:.2f}\n"
            f"Limit: {limit_value:.2f}\n"
            f"Breach: {pct_breach:+.1f}%"
        )
        
        notification = Notification(
            title=title,
            message=message,
            notification_type=NotificationType.LIMIT_BREACH,
            recipient_id=recipient_id,
            recipient_email=recipient_email,
            channels=[
                NotificationChannel.EMAIL,
                NotificationChannel.SLACK,
                NotificationChannel.IN_APP,
            ],
            priority=NotificationPriority.HIGH,
            context_data={
                "limit_type": limit_type,
                "current_value": current_value,
                "limit_value": limit_value,
                "portfolio": portfolio_name,
            },
        )
        
        self.send(notification)
        return notification
    
    def send_approval_request(
        self,
        approver_id: str,
        approver_email: str,
        request_type: str,
        requester_name: str,
        description: str,
        action_url: str,
    ) -> Notification:
        """Send an approval request notification."""
        notification = Notification(
            title=f"Approval Required: {request_type}",
            message=f"{requester_name} has requested your approval.\n\n{description}",
            notification_type=NotificationType.APPROVAL_REQUIRED,
            recipient_id=approver_id,
            recipient_email=approver_email,
            channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
            priority=NotificationPriority.HIGH,
            action_url=action_url,
        )
        
        self.send(notification)
        return notification
    
    def send_report_ready(
        self,
        recipient_id: str,
        recipient_email: str,
        report_name: str,
        report_url: str,
    ) -> Notification:
        """Send a report ready notification."""
        notification = Notification(
            title=f"Report Ready: {report_name}",
            message=f"Your requested report '{report_name}' is ready for download.",
            notification_type=NotificationType.REPORT_READY,
            recipient_id=recipient_id,
            recipient_email=recipient_email,
            channels=[NotificationChannel.EMAIL, NotificationChannel.IN_APP],
            priority=NotificationPriority.LOW,
            action_url=report_url,
        )
        
        self.send(notification)
        return notification
    
    def _send_email(self, notification: Notification):
        """Send email notification."""
        # In production, use SMTP or email service
        print(f"[EMAIL] To: {notification.recipient_email}")
        print(f"        Subject: {notification.title}")
        print(f"        Body: {notification.message[:100]}...")
    
    def _send_slack(self, notification: Notification):
        """Send Slack notification."""
        # In production, use Slack API
        print(f"[SLACK] To: {notification.recipient_id}")
        print(f"        Message: {notification.title}")
    
    def _send_teams(self, notification: Notification):
        """Send Microsoft Teams notification."""
        # In production, use Teams webhook
        print(f"[TEAMS] To: {notification.recipient_id}")
        print(f"        Message: {notification.title}")
    
    def _send_sms(self, notification: Notification):
        """Send SMS notification."""
        # In production, use Twilio or similar
        print(f"[SMS] To: {notification.recipient_id}")
        print(f"      Message: {notification.title[:160]}")
    
    def _send_in_app(self, notification: Notification):
        """Send in-app notification."""
        # Store for in-app display
        print(f"[IN_APP] To: {notification.recipient_id}")
        print(f"         Message: {notification.title}")
    
    def get_unread_for_user(self, user_id: str) -> list[Notification]:
        """Get unread notifications for a user."""
        return [
            n for n in self._sent
            if n.recipient_id == user_id and n.read_at is None
        ]
    
    def mark_read(self, notification_id: str):
        """Mark a notification as read."""
        for notification in self._sent:
            if notification.notification_id == notification_id:
                notification.read_at = datetime.now()
                break


# Notification templates for common scenarios
NOTIFICATION_TEMPLATES = {
    "var_breach": {
        "title": "⚠️ VaR Limit Breach Alert",
        "template": (
            "Portfolio {portfolio_name} has exceeded its VaR limit.\n\n"
            "Current VaR (95%): {current_var:.2%}\n"
            "VaR Limit: {var_limit:.2%}\n"
            "Action Required: Review positions and reduce risk exposure."
        ),
        "priority": NotificationPriority.CRITICAL,
        "channels": [NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.SMS],
    },
    "guideline_breach": {
        "title": "🚨 Investment Guideline Breach",
        "template": (
            "A guideline breach has been detected for {portfolio_name}.\n\n"
            "Guideline: {guideline_name}\n"
            "Current: {current_value}\n"
            "Limit: {limit_value}\n"
            "Status: {status}"
        ),
        "priority": NotificationPriority.HIGH,
        "channels": [NotificationChannel.EMAIL, NotificationChannel.SLACK],
    },
    "model_update": {
        "title": "📊 Risk Model Updated",
        "template": (
            "The {model_name} risk model has been updated.\n\n"
            "Version: {version}\n"
            "Changes: {changes}\n"
            "Effective: {effective_date}"
        ),
        "priority": NotificationPriority.MEDIUM,
        "channels": [NotificationChannel.EMAIL, NotificationChannel.IN_APP],
    },
}
