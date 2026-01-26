"""
JWT Authentication for BXMA Platform.

Implements OAuth2 with JWT tokens including access and refresh tokens.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel

# Configuration
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "bxma-super-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7


class TokenPayload(BaseModel):
    """JWT token payload."""
    sub: str
    exp: datetime
    iat: datetime
    type: str  # "access" or "refresh"
    roles: list[str] = []
    permissions: list[str] = []


class TokenResponse(BaseModel):
    """Response containing access and refresh tokens."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class UserInfo(BaseModel):
    """Authenticated user information."""
    user_id: str
    email: str
    roles: list[str]
    permissions: list[str]


security = HTTPBearer()


def create_access_token(
    user_id: str,
    email: str,
    roles: list[str] | None = None,
    permissions: list[str] | None = None,
    expires_delta: timedelta | None = None,
) -> str:
    """
    Create a new access token.
    
    Args:
        user_id: Unique user identifier
        email: User email address
        roles: List of user roles
        permissions: List of specific permissions
        expires_delta: Custom expiration time
        
    Returns:
        Encoded JWT access token
    """
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    
    payload = {
        "sub": user_id,
        "email": email,
        "exp": expire,
        "iat": now,
        "type": "access",
        "roles": roles or [],
        "permissions": permissions or [],
    }
    
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(
    user_id: str,
    expires_delta: timedelta | None = None,
) -> str:
    """
    Create a new refresh token.
    
    Args:
        user_id: Unique user identifier
        expires_delta: Custom expiration time
        
    Returns:
        Encoded JWT refresh token
    """
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))
    
    payload = {
        "sub": user_id,
        "exp": expire,
        "iat": now,
        "type": "refresh",
    }
    
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict[str, Any]:
    """
    Decode and validate a JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded token payload
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> UserInfo:
    """
    Get current authenticated user from JWT token.
    
    This is a FastAPI dependency that extracts and validates the user
    from the Authorization header.
    
    Args:
        credentials: HTTP Authorization credentials
        
    Returns:
        UserInfo object with user details
        
    Raises:
        HTTPException: If authentication fails
    """
    token = credentials.credentials
    payload = decode_token(token)
    
    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return UserInfo(
        user_id=payload["sub"],
        email=payload.get("email", ""),
        roles=payload.get("roles", []),
        permissions=payload.get("permissions", []),
    )


def create_token_response(
    user_id: str,
    email: str,
    roles: list[str] | None = None,
    permissions: list[str] | None = None,
) -> TokenResponse:
    """
    Create a complete token response with access and refresh tokens.
    
    Args:
        user_id: Unique user identifier
        email: User email address
        roles: List of user roles
        permissions: List of specific permissions
        
    Returns:
        TokenResponse with both tokens
    """
    access_token = create_access_token(user_id, email, roles, permissions)
    refresh_token = create_refresh_token(user_id)
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )
