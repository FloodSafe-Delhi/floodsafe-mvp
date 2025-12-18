"""
JWT Security utilities for FloodSafe authentication.
Handles token creation, verification, and hashing.
"""
from datetime import datetime, timedelta
from typing import Optional
import hashlib
import secrets

from jose import jwt, JWTError

from src.core.config import settings


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.

    Args:
        data: Payload data (typically contains 'sub' with user_id)
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })

    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )
    return encoded_jwt


def create_refresh_token(user_id: str) -> tuple[str, str]:
    """
    Create a refresh token and its hash for storage.

    Args:
        user_id: The user's ID to include in the token

    Returns:
        Tuple of (raw_token, token_hash) - store the hash, return the raw token to client
    """
    # Generate a secure random token
    random_bytes = secrets.token_bytes(32)
    raw_token = secrets.token_urlsafe(32)

    # Create JWT with the random token as identifier
    expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    token_data = {
        "sub": user_id,
        "jti": raw_token,  # JWT ID for uniqueness
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    }

    encoded_jwt = jwt.encode(
        token_data,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )

    # Hash the token for secure storage
    token_hash = hash_token(encoded_jwt)

    return encoded_jwt, token_hash


def verify_token(token: str, token_type: str = "access") -> Optional[dict]:
    """
    Verify and decode a JWT token.

    Args:
        token: The JWT token to verify
        token_type: Expected token type ('access' or 'refresh')

    Returns:
        Decoded payload if valid, None if invalid
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )

        # Verify token type
        if payload.get("type") != token_type:
            return None

        # Check expiration (jose does this, but be explicit)
        exp = payload.get("exp")
        if exp and datetime.utcnow() > datetime.fromtimestamp(exp):
            return None

        return payload

    except JWTError:
        return None


def hash_token(token: str) -> str:
    """
    Create a SHA-256 hash of a token for secure storage.

    Args:
        token: The token to hash

    Returns:
        Hex-encoded hash string
    """
    return hashlib.sha256(token.encode()).hexdigest()


def get_token_expiry(token: str) -> Optional[datetime]:
    """
    Extract expiration time from a token without full verification.

    Args:
        token: The JWT token

    Returns:
        Expiration datetime or None if invalid
    """
    try:
        # Decode without verification to get expiry
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            options={"verify_exp": False}
        )
        exp = payload.get("exp")
        if exp:
            return datetime.fromtimestamp(exp)
        return None
    except JWTError:
        return None


# =============================================================================
# Password Hashing (bcrypt)
# =============================================================================
import bcrypt


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt with salt.

    Args:
        password: The plaintext password to hash

    Returns:
        The bcrypt hash string (includes salt)
    """
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')


def verify_password(password: str, password_hash: str) -> bool:
    """
    Verify a password against its bcrypt hash.

    Args:
        password: The plaintext password to verify
        password_hash: The bcrypt hash to check against

    Returns:
        True if password matches, False otherwise
    """
    try:
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    except Exception:
        return False
