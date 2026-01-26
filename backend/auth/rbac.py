"""
Role-Based Access Control (RBAC) for BXMA Platform.

Implements a hierarchical permission system with roles and fine-grained permissions.
"""

from __future__ import annotations

from enum import Enum
from functools import wraps
from typing import Callable, Set

from fastapi import HTTPException, status


class Permission(str, Enum):
    """Fine-grained permissions for the platform."""
    
    # Portfolio permissions
    PORTFOLIO_READ = "portfolio:read"
    PORTFOLIO_WRITE = "portfolio:write"
    PORTFOLIO_DELETE = "portfolio:delete"
    PORTFOLIO_EXECUTE = "portfolio:execute"
    
    # Risk permissions
    RISK_READ = "risk:read"
    RISK_WRITE = "risk:write"
    RISK_CALCULATE = "risk:calculate"
    
    # Optimization permissions
    OPTIMIZE_READ = "optimize:read"
    OPTIMIZE_EXECUTE = "optimize:execute"
    OPTIMIZE_APPROVE = "optimize:approve"
    
    # Attribution permissions
    ATTRIBUTION_READ = "attribution:read"
    ATTRIBUTION_CALCULATE = "attribution:calculate"
    
    # Stress testing permissions
    STRESS_TEST_READ = "stress:read"
    STRESS_TEST_EXECUTE = "stress:execute"
    STRESS_TEST_CREATE = "stress:create"
    
    # Data permissions
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    DATA_UPLOAD = "data:upload"
    DATA_DELETE = "data:delete"
    
    # Report permissions
    REPORT_READ = "report:read"
    REPORT_CREATE = "report:create"
    REPORT_EXPORT = "report:export"
    
    # Admin permissions
    ADMIN_USERS = "admin:users"
    ADMIN_SYSTEM = "admin:system"
    ADMIN_AUDIT = "admin:audit"
    
    # API permissions
    API_ACCESS = "api:access"
    API_RATE_UNLIMITED = "api:rate_unlimited"


class Role(str, Enum):
    """User roles with associated permissions."""
    
    VIEWER = "viewer"
    ANALYST = "analyst"
    PORTFOLIO_MANAGER = "portfolio_manager"
    RISK_MANAGER = "risk_manager"
    QUANT = "quant"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


# Role to permissions mapping
ROLE_PERMISSIONS: dict[Role, Set[Permission]] = {
    Role.VIEWER: {
        Permission.PORTFOLIO_READ,
        Permission.RISK_READ,
        Permission.OPTIMIZE_READ,
        Permission.ATTRIBUTION_READ,
        Permission.STRESS_TEST_READ,
        Permission.DATA_READ,
        Permission.REPORT_READ,
        Permission.API_ACCESS,
    },
    
    Role.ANALYST: {
        Permission.PORTFOLIO_READ,
        Permission.PORTFOLIO_WRITE,
        Permission.RISK_READ,
        Permission.RISK_CALCULATE,
        Permission.OPTIMIZE_READ,
        Permission.OPTIMIZE_EXECUTE,
        Permission.ATTRIBUTION_READ,
        Permission.ATTRIBUTION_CALCULATE,
        Permission.STRESS_TEST_READ,
        Permission.STRESS_TEST_EXECUTE,
        Permission.DATA_READ,
        Permission.DATA_WRITE,
        Permission.REPORT_READ,
        Permission.REPORT_CREATE,
        Permission.API_ACCESS,
    },
    
    Role.PORTFOLIO_MANAGER: {
        Permission.PORTFOLIO_READ,
        Permission.PORTFOLIO_WRITE,
        Permission.PORTFOLIO_EXECUTE,
        Permission.RISK_READ,
        Permission.RISK_CALCULATE,
        Permission.OPTIMIZE_READ,
        Permission.OPTIMIZE_EXECUTE,
        Permission.OPTIMIZE_APPROVE,
        Permission.ATTRIBUTION_READ,
        Permission.ATTRIBUTION_CALCULATE,
        Permission.STRESS_TEST_READ,
        Permission.STRESS_TEST_EXECUTE,
        Permission.DATA_READ,
        Permission.REPORT_READ,
        Permission.REPORT_CREATE,
        Permission.REPORT_EXPORT,
        Permission.API_ACCESS,
    },
    
    Role.RISK_MANAGER: {
        Permission.PORTFOLIO_READ,
        Permission.RISK_READ,
        Permission.RISK_WRITE,
        Permission.RISK_CALCULATE,
        Permission.OPTIMIZE_READ,
        Permission.ATTRIBUTION_READ,
        Permission.ATTRIBUTION_CALCULATE,
        Permission.STRESS_TEST_READ,
        Permission.STRESS_TEST_EXECUTE,
        Permission.STRESS_TEST_CREATE,
        Permission.DATA_READ,
        Permission.DATA_WRITE,
        Permission.REPORT_READ,
        Permission.REPORT_CREATE,
        Permission.REPORT_EXPORT,
        Permission.API_ACCESS,
    },
    
    Role.QUANT: {
        Permission.PORTFOLIO_READ,
        Permission.PORTFOLIO_WRITE,
        Permission.RISK_READ,
        Permission.RISK_WRITE,
        Permission.RISK_CALCULATE,
        Permission.OPTIMIZE_READ,
        Permission.OPTIMIZE_EXECUTE,
        Permission.ATTRIBUTION_READ,
        Permission.ATTRIBUTION_CALCULATE,
        Permission.STRESS_TEST_READ,
        Permission.STRESS_TEST_EXECUTE,
        Permission.STRESS_TEST_CREATE,
        Permission.DATA_READ,
        Permission.DATA_WRITE,
        Permission.DATA_UPLOAD,
        Permission.REPORT_READ,
        Permission.REPORT_CREATE,
        Permission.REPORT_EXPORT,
        Permission.API_ACCESS,
        Permission.API_RATE_UNLIMITED,
    },
    
    Role.ADMIN: {
        Permission.PORTFOLIO_READ,
        Permission.PORTFOLIO_WRITE,
        Permission.PORTFOLIO_DELETE,
        Permission.PORTFOLIO_EXECUTE,
        Permission.RISK_READ,
        Permission.RISK_WRITE,
        Permission.RISK_CALCULATE,
        Permission.OPTIMIZE_READ,
        Permission.OPTIMIZE_EXECUTE,
        Permission.OPTIMIZE_APPROVE,
        Permission.ATTRIBUTION_READ,
        Permission.ATTRIBUTION_CALCULATE,
        Permission.STRESS_TEST_READ,
        Permission.STRESS_TEST_EXECUTE,
        Permission.STRESS_TEST_CREATE,
        Permission.DATA_READ,
        Permission.DATA_WRITE,
        Permission.DATA_UPLOAD,
        Permission.DATA_DELETE,
        Permission.REPORT_READ,
        Permission.REPORT_CREATE,
        Permission.REPORT_EXPORT,
        Permission.ADMIN_USERS,
        Permission.ADMIN_AUDIT,
        Permission.API_ACCESS,
        Permission.API_RATE_UNLIMITED,
    },
    
    Role.SUPER_ADMIN: set(Permission),  # All permissions
}


def get_role_permissions(role: Role) -> Set[Permission]:
    """
    Get all permissions for a given role.
    
    Args:
        role: User role
        
    Returns:
        Set of permissions for the role
    """
    return ROLE_PERMISSIONS.get(role, set())


def get_user_permissions(roles: list[str]) -> Set[Permission]:
    """
    Get combined permissions from multiple roles.
    
    Args:
        roles: List of role names
        
    Returns:
        Combined set of permissions
    """
    permissions: Set[Permission] = set()
    
    for role_name in roles:
        try:
            role = Role(role_name)
            permissions |= get_role_permissions(role)
        except ValueError:
            continue
    
    return permissions


def check_permission(
    user_roles: list[str],
    user_permissions: list[str],
    required_permission: Permission,
) -> bool:
    """
    Check if user has a specific permission.
    
    Args:
        user_roles: List of user role names
        user_permissions: List of explicit permissions
        required_permission: Permission to check
        
    Returns:
        True if user has permission
    """
    # Get permissions from roles
    role_permissions = get_user_permissions(user_roles)
    
    # Add explicit permissions
    explicit_permissions = {
        Permission(p) for p in user_permissions 
        if p in [perm.value for perm in Permission]
    }
    
    all_permissions = role_permissions | explicit_permissions
    
    return required_permission in all_permissions


def require_permission(permission: Permission) -> Callable:
    """
    Decorator to require a specific permission.
    
    Usage:
        @app.get("/protected")
        @require_permission(Permission.PORTFOLIO_READ)
        async def protected_endpoint(user: UserInfo = Depends(get_current_user)):
            ...
    
    Args:
        permission: Required permission
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get user from kwargs (injected by Depends)
            user = kwargs.get("user") or kwargs.get("current_user")
            
            if user is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )
            
            if not check_permission(user.roles, user.permissions, permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied: {permission.value}",
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_role(role: Role) -> Callable:
    """
    Decorator to require a specific role.
    
    Args:
        role: Required role
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user = kwargs.get("user") or kwargs.get("current_user")
            
            if user is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )
            
            if role.value not in user.roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role required: {role.value}",
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator
