"""
OTP API endpoints for phone verification
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from ..domain.services.otp_service import get_otp_service


router = APIRouter(prefix="/otp", tags=["otp"])


class SendOTPRequest(BaseModel):
    """Request to send OTP to phone number"""
    phone_number: str = Field(..., min_length=10, max_length=20,
                              description="Phone number with country code")


class SendOTPResponse(BaseModel):
    """Response after sending OTP"""
    success: bool
    message: str
    expires_in: int  # seconds


class VerifyOTPRequest(BaseModel):
    """Request to verify OTP code"""
    phone_number: str = Field(..., min_length=10, max_length=20)
    otp_code: str = Field(..., min_length=6, max_length=6, pattern="^[0-9]{6}$")


class VerifyOTPResponse(BaseModel):
    """Response after verifying OTP"""
    verified: bool
    message: str
    token: Optional[str] = None  # Verification token if successful


@router.post("/send", response_model=SendOTPResponse)
async def send_otp(request: SendOTPRequest):
    """
    Send OTP to phone number.

    Rate limit: 3 OTPs per hour per phone number.
    OTP expires in 5 minutes.
    """
    otp_service = get_otp_service()

    success, error = otp_service.send_otp(request.phone_number)

    if not success:
        raise HTTPException(status_code=429, detail=error)

    return SendOTPResponse(
        success=True,
        message="OTP sent successfully",
        expires_in=300  # 5 minutes
    )


@router.post("/verify", response_model=VerifyOTPResponse)
async def verify_otp(request: VerifyOTPRequest):
    """
    Verify OTP code.

    Returns verification token if successful.
    Token must be used during report submission.
    """
    otp_service = get_otp_service()

    verified, result = otp_service.verify_otp(request.phone_number, request.otp_code)

    if verified:
        return VerifyOTPResponse(
            verified=True,
            message="Phone number verified successfully",
            token=result  # result contains the token
        )
    else:
        # result contains the error message
        return VerifyOTPResponse(
            verified=False,
            message=result,
            token=None
        )
