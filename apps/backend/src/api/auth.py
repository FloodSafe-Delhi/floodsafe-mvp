"""
Authentication API endpoints for FloodSafe.
Handles Google OAuth, Phone Auth, and token management.
"""
from typing import Optional
from pydantic import BaseModel, Field

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.infrastructure.database import get_db
from src.infrastructure.models import User
from src.domain.services.auth_service import auth_service
from .deps import get_current_user


router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================

class GoogleAuthRequest(BaseModel):
    """Request for Google OAuth authentication"""
    id_token: str = Field(..., description="Google ID token from client-side sign-in")


class PhoneAuthRequest(BaseModel):
    """Request for Firebase Phone authentication"""
    id_token: str = Field(..., description="Firebase ID token from phone auth")


class RefreshTokenRequest(BaseModel):
    """Request to refresh access token"""
    refresh_token: str = Field(..., description="Refresh token")


class LogoutRequest(BaseModel):
    """Request to logout (revoke refresh token)"""
    refresh_token: str = Field(..., description="Refresh token to revoke")


class TokenResponse(BaseModel):
    """Response containing auth tokens"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class UserResponse(BaseModel):
    """User profile response"""
    id: str
    username: str
    email: Optional[str] = None
    phone: Optional[str] = None
    role: str
    auth_provider: str
    profile_photo_url: Optional[str] = None
    points: int
    level: int
    reputation_score: int

    # Onboarding & City Preference
    city_preference: Optional[str] = None
    profile_complete: bool = False
    onboarding_step: Optional[int] = None

    class Config:
        from_attributes = True


class MessageResponse(BaseModel):
    """Simple message response"""
    message: str


# =============================================================================
# Google OAuth Endpoints
# =============================================================================

@router.post("/google", response_model=TokenResponse, tags=["authentication"])
async def google_auth(
    request: GoogleAuthRequest,
    db: Session = Depends(get_db)
):
    """
    Authenticate with Google OAuth.

    Exchange a Google ID token (from client-side Google Sign-In) for
    FloodSafe JWT tokens.

    Returns access and refresh tokens for authenticated requests.
    """
    # Verify Google token
    google_data = await auth_service.verify_google_token(request.id_token)

    if not google_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google token"
        )

    # Get or create user
    user = auth_service.get_or_create_google_user(google_data, db)

    # Create tokens
    tokens = auth_service.create_tokens(user, db)

    return TokenResponse(**tokens)


# =============================================================================
# Phone Authentication Endpoints
# =============================================================================

@router.post("/phone/verify", response_model=TokenResponse, tags=["authentication"])
async def phone_auth(
    request: PhoneAuthRequest,
    db: Session = Depends(get_db)
):
    """
    Authenticate with Firebase Phone Auth.

    Exchange a Firebase ID token (from phone OTP verification) for
    FloodSafe JWT tokens.

    The client should:
    1. Use Firebase SDK to send OTP to phone
    2. Verify the OTP with Firebase
    3. Get the ID token from Firebase
    4. Send the ID token to this endpoint

    Returns access and refresh tokens for authenticated requests.
    """
    # Verify Firebase token
    phone_data = await auth_service.verify_firebase_phone_token(request.id_token)

    if not phone_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid phone verification token"
        )

    # Get or create user
    user = auth_service.get_or_create_phone_user(phone_data["phone"], db)

    # Create tokens
    tokens = auth_service.create_tokens(user, db)

    return TokenResponse(**tokens)


# =============================================================================
# Email/Password Authentication Endpoints
# =============================================================================

class EmailRegisterRequest(BaseModel):
    """Request for email/password registration"""
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=8, max_length=128, description="Password (min 8 chars)")
    username: Optional[str] = Field(None, min_length=3, max_length=50, description="Optional username")


class EmailLoginRequest(BaseModel):
    """Request for email/password login"""
    email: str = Field(..., description="User email address")
    password: str = Field(..., description="Password")


@router.post("/register/email", response_model=TokenResponse, tags=["authentication"])
async def register_email(
    request: EmailRegisterRequest,
    db: Session = Depends(get_db)
):
    """
    Register a new user with email and password.

    Creates a new user account and returns JWT tokens.
    Password must be at least 8 characters.

    Returns access and refresh tokens for authenticated requests.
    """
    # Basic email validation
    email = request.email.lower().strip()
    if '@' not in email or '.' not in email.split('@')[1]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email format"
        )

    try:
        user = auth_service.register_email_user(
            email=email,
            password=request.password,
            username=request.username,
            db=db
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    # Create tokens
    tokens = auth_service.create_tokens(user, db)

    return TokenResponse(**tokens)


@router.post("/login/email", response_model=TokenResponse, tags=["authentication"])
async def login_email(
    request: EmailLoginRequest,
    db: Session = Depends(get_db)
):
    """
    Authenticate with email and password.

    Exchange email/password credentials for JWT tokens.

    Returns access and refresh tokens for authenticated requests.
    """
    user = auth_service.authenticate_email_user(
        email=request.email,
        password=request.password,
        db=db
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    # Create tokens
    tokens = auth_service.create_tokens(user, db)

    return TokenResponse(**tokens)


# =============================================================================
# Token Management Endpoints
# =============================================================================

@router.post("/refresh", response_model=TokenResponse, tags=["authentication"])
async def refresh_token(
    request: RefreshTokenRequest,
    db: Session = Depends(get_db)
):
    """
    Refresh access token.

    Exchange a valid refresh token for new access and refresh tokens.
    The old refresh token is revoked (token rotation for security).

    Call this endpoint when the access token expires.
    """
    tokens = auth_service.refresh_tokens(request.refresh_token, db)

    if not tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token"
        )

    return TokenResponse(**tokens)


@router.post("/logout", response_model=MessageResponse, tags=["authentication"])
async def logout(
    request: LogoutRequest,
    db: Session = Depends(get_db)
):
    """
    Logout and revoke refresh token.

    Revokes the provided refresh token so it can no longer be used.
    The client should also clear local token storage.
    """
    success = auth_service.revoke_refresh_token(request.refresh_token, db)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token not found or already revoked"
        )

    return MessageResponse(message="Successfully logged out")


@router.post("/logout-all", response_model=MessageResponse, tags=["authentication"])
async def logout_all(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Logout from all devices.

    Revokes all refresh tokens for the current user.
    Requires authentication.
    """
    count = auth_service.revoke_all_user_tokens(str(current_user.id), db)

    return MessageResponse(message=f"Logged out from {count} session(s)")


# =============================================================================
# User Profile Endpoints
# =============================================================================

@router.get("/me", response_model=UserResponse, tags=["authentication"])
async def get_current_user_profile(
    current_user: User = Depends(get_current_user)
):
    """
    Get current authenticated user's profile.

    Returns the profile of the currently authenticated user.
    Requires a valid access token.
    """
    return UserResponse(
        id=str(current_user.id),
        username=current_user.username,
        email=current_user.email,
        phone=current_user.phone,
        role=current_user.role,
        auth_provider=current_user.auth_provider or "local",
        profile_photo_url=current_user.profile_photo_url,
        points=current_user.points,
        level=current_user.level,
        reputation_score=current_user.reputation_score,
        city_preference=current_user.city_preference,
        profile_complete=current_user.profile_complete,
        onboarding_step=current_user.onboarding_step,
    )


@router.get("/check", tags=["authentication"])
async def check_auth(
    current_user: User = Depends(get_current_user)
):
    """
    Check if the current token is valid.

    Simple endpoint to verify authentication status.
    Returns 200 if authenticated, 401 if not.
    """
    return {"authenticated": True, "user_id": str(current_user.id)}
