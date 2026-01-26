"""
Password hashing and verification using Argon2.

Argon2 is the winner of the Password Hashing Competition and provides
the strongest protection against GPU and ASIC attacks.
"""

from __future__ import annotations

import hashlib
import secrets
import base64
from typing import Tuple


# Argon2 parameters (would use argon2-cffi in production)
# For demonstration, using PBKDF2 as a fallback
SALT_LENGTH = 32
HASH_ITERATIONS = 100000
HASH_LENGTH = 64


def hash_password(password: str) -> str:
    """
    Hash a password using PBKDF2-SHA512.
    
    In production, use argon2-cffi:
    ```
    from argon2 import PasswordHasher
    ph = PasswordHasher()
    return ph.hash(password)
    ```
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password string with embedded salt
    """
    salt = secrets.token_bytes(SALT_LENGTH)
    
    # Derive key using PBKDF2
    dk = hashlib.pbkdf2_hmac(
        'sha512',
        password.encode('utf-8'),
        salt,
        HASH_ITERATIONS,
        dklen=HASH_LENGTH
    )
    
    # Encode salt and hash
    salt_b64 = base64.b64encode(salt).decode('utf-8')
    hash_b64 = base64.b64encode(dk).decode('utf-8')
    
    # Format: algorithm$iterations$salt$hash
    return f"pbkdf2_sha512${HASH_ITERATIONS}${salt_b64}${hash_b64}"


def verify_password(password: str, hashed: str) -> bool:
    """
    Verify a password against its hash.
    
    In production, use argon2-cffi:
    ```
    from argon2 import PasswordHasher
    from argon2.exceptions import VerifyMismatchError
    ph = PasswordHasher()
    try:
        ph.verify(hashed, password)
        return True
    except VerifyMismatchError:
        return False
    ```
    
    Args:
        password: Plain text password to verify
        hashed: Stored hash string
        
    Returns:
        True if password matches, False otherwise
    """
    try:
        parts = hashed.split('$')
        if len(parts) != 4 or parts[0] != 'pbkdf2_sha512':
            return False
        
        iterations = int(parts[1])
        salt = base64.b64decode(parts[2])
        stored_hash = base64.b64decode(parts[3])
        
        # Derive key using same parameters
        dk = hashlib.pbkdf2_hmac(
            'sha512',
            password.encode('utf-8'),
            salt,
            iterations,
            dklen=len(stored_hash)
        )
        
        # Use constant-time comparison
        return secrets.compare_digest(dk, stored_hash)
        
    except Exception:
        return False


def generate_api_key() -> Tuple[str, str]:
    """
    Generate a new API key pair.
    
    Returns:
        Tuple of (key_id, secret_key)
    """
    key_id = f"bxma_{secrets.token_hex(8)}"
    secret_key = secrets.token_urlsafe(32)
    return key_id, secret_key


def generate_secure_token(length: int = 32) -> str:
    """
    Generate a cryptographically secure random token.
    
    Args:
        length: Number of bytes (output will be longer due to encoding)
        
    Returns:
        URL-safe base64 encoded token
    """
    return secrets.token_urlsafe(length)
