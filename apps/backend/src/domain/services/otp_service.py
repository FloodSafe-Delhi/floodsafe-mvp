"""
OTP Service for phone verification
Handles OTP generation, storage, and verification
"""

import random
import string
from datetime import datetime, timedelta
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class OTPRecord:
    """OTP verification record"""
    phone_number: str
    otp_code: str
    created_at: datetime
    expires_at: datetime
    attempts: int = 0
    verified: bool = False


class OTPService:
    """
    OTP Service with in-memory storage.
    TODO: Replace with Redis for production.
    """

    def __init__(self, expiry_minutes: int = 5, max_attempts: int = 3):
        self.expiry_minutes = expiry_minutes
        self.max_attempts = max_attempts
        self._storage: Dict[str, OTPRecord] = {}  # phone_number -> OTPRecord
        self._rate_limits: Dict[str, list] = {}  # phone_number -> [timestamps]

    def generate_otp(self) -> str:
        """Generate a 6-digit OTP code"""
        return ''.join(random.choices(string.digits, k=6))

    def _check_rate_limit(self, phone_number: str, limit: int = 3, window_minutes: int = 60) -> bool:
        """
        Check if phone number has exceeded rate limit.
        Returns True if within limit, False if exceeded.
        """
        now = datetime.utcnow()
        cutoff = now - timedelta(minutes=window_minutes)

        # Clean old entries
        if phone_number in self._rate_limits:
            self._rate_limits[phone_number] = [
                ts for ts in self._rate_limits[phone_number] if ts > cutoff
            ]
        else:
            self._rate_limits[phone_number] = []

        # Check limit
        if len(self._rate_limits[phone_number]) >= limit:
            return False

        return True

    def send_otp(self, phone_number: str) -> tuple[bool, Optional[str]]:
        """
        Generate and "send" OTP to phone number.
        Returns (success, error_message)

        For MVP: Just stores OTP in memory.
        TODO: Integrate Twilio/AWS SNS for actual SMS sending.
        """
        # Check rate limit (3 OTPs per hour)
        if not self._check_rate_limit(phone_number, limit=3, window_minutes=60):
            return False, "Too many OTP requests. Please try again later."

        # Generate new OTP
        otp_code = self.generate_otp()
        now = datetime.utcnow()
        expires_at = now + timedelta(minutes=self.expiry_minutes)

        # Store OTP record
        self._storage[phone_number] = OTPRecord(
            phone_number=phone_number,
            otp_code=otp_code,
            created_at=now,
            expires_at=expires_at,
            attempts=0,
            verified=False
        )

        # Track rate limit
        self._rate_limits[phone_number].append(now)

        # TODO: Send actual SMS via Twilio
        # For MVP: Log to console
        print(f"[OTP] Phone: {phone_number}, Code: {otp_code}, Expires: {expires_at}")

        return True, None

    def verify_otp(self, phone_number: str, otp_code: str) -> tuple[bool, Optional[str]]:
        """
        Verify OTP code for phone number.
        Returns (verified, error_message)
        """
        # Check if OTP exists
        if phone_number not in self._storage:
            return False, "No OTP found for this phone number. Please request a new one."

        record = self._storage[phone_number]

        # Check if already verified
        if record.verified:
            return False, "OTP already verified. Please request a new one."

        # Check expiry
        if datetime.utcnow() > record.expires_at:
            del self._storage[phone_number]
            return False, "OTP expired. Please request a new one."

        # Check attempts
        if record.attempts >= self.max_attempts:
            del self._storage[phone_number]
            return False, "Too many failed attempts. Please request a new OTP."

        # Verify OTP
        if otp_code == record.otp_code:
            record.verified = True
            # Generate verification token (simple UUID for MVP)
            import uuid
            token = str(uuid.uuid4())
            # Store token mapping (for verification during report submission)
            self._storage[f"token:{token}"] = record
            return True, token
        else:
            record.attempts += 1
            return False, f"Invalid OTP. {self.max_attempts - record.attempts} attempts remaining."

    def verify_token(self, phone_number: str, token: str) -> bool:
        """
        Verify that token is valid for the given phone number.
        Used during report submission.
        """
        token_key = f"token:{token}"
        if token_key in self._storage:
            record = self._storage[token_key]
            return record.phone_number == phone_number and record.verified
        return False

    def cleanup_expired(self):
        """Remove expired OTP records (call periodically)"""
        now = datetime.utcnow()
        expired_phones = [
            phone for phone, record in self._storage.items()
            if not phone.startswith("token:") and record.expires_at < now
        ]
        for phone in expired_phones:
            del self._storage[phone]


# Global singleton instance
_otp_service_instance = None


def get_otp_service() -> OTPService:
    """Get global OTP service instance"""
    global _otp_service_instance
    if _otp_service_instance is None:
        _otp_service_instance = OTPService()
    return _otp_service_instance
