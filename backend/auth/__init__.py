"""Authentication module for BXMA Platform."""

from backend.auth.jwt import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
)
from backend.auth.password import hash_password, verify_password
from backend.auth.rbac import Permission, Role, check_permission

__all__ = [
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "get_current_user",
    "hash_password",
    "verify_password",
    "Permission",
    "Role",
    "check_permission",
]
