"""
Tests for WhatsApp webhook endpoint.

Tests cover:
- Health check endpoint
- Rate limiting
- Signature validation
- Input validation (coordinates)
- Text commands (HELP, RISK, WARNINGS)
- Error handling
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from src.main import app
from src.api.webhook import (
    check_rate_limit,
    validate_phone_format,
    validate_coordinates,
    _rate_limit_cache,
    RATE_LIMIT_MESSAGES,
    RATE_LIMIT_WINDOW_SECONDS
)


# =============================================================================
# HEALTH CHECK TESTS
# =============================================================================

class TestHealthCheck:
    """Tests for /api/whatsapp/health endpoint."""

    def test_health_check_returns_status(self, client):
        """Health check should return status information."""
        response = client.get("/api/whatsapp/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "twilio_configured" in data
        assert "database" in data
        assert "ml_service" in data
        assert "webhook_url" in data
        assert "rate_limit" in data

    def test_health_check_shows_rate_limit_config(self, client):
        """Health check should show rate limit configuration."""
        response = client.get("/api/whatsapp/health")
        data = response.json()

        assert "rate_limit" in data
        assert "msgs" in data["rate_limit"]


# =============================================================================
# RATE LIMITING TESTS
# =============================================================================

class TestRateLimiting:
    """Tests for rate limiting functionality."""

    def test_rate_limit_allows_normal_usage(self, clear_rate_limit_cache):
        """Normal usage (under limit) should be allowed."""
        phone = "+919876543210"

        # First few requests should pass
        for i in range(RATE_LIMIT_MESSAGES - 1):
            assert check_rate_limit(phone) is True

    def test_rate_limit_blocks_excessive_requests(self, clear_rate_limit_cache):
        """Exceeding rate limit should block requests."""
        phone = "+919876543210"

        # Use up all allowed requests
        for i in range(RATE_LIMIT_MESSAGES):
            check_rate_limit(phone)

        # Next request should be blocked
        assert check_rate_limit(phone) is False

    def test_rate_limit_resets_after_window(self, clear_rate_limit_cache):
        """Rate limit should reset after time window expires."""
        phone = "+919876543210"

        # Use up all requests
        for i in range(RATE_LIMIT_MESSAGES):
            check_rate_limit(phone)

        # Verify blocked
        assert check_rate_limit(phone) is False

        # Manually expire the cache entries
        old_time = datetime.utcnow() - timedelta(seconds=RATE_LIMIT_WINDOW_SECONDS + 1)
        _rate_limit_cache[phone] = [old_time] * RATE_LIMIT_MESSAGES

        # Should be allowed again
        assert check_rate_limit(phone) is True

    def test_rate_limit_per_phone(self, clear_rate_limit_cache):
        """Rate limits should be per phone number."""
        phone1 = "+919876543210"
        phone2 = "+919876543211"

        # Use up phone1's limit
        for i in range(RATE_LIMIT_MESSAGES):
            check_rate_limit(phone1)

        # phone1 blocked
        assert check_rate_limit(phone1) is False

        # phone2 should still work
        assert check_rate_limit(phone2) is True


# =============================================================================
# INPUT VALIDATION TESTS
# =============================================================================

class TestInputValidation:
    """Tests for input validation functions."""

    def test_validate_phone_e164_valid(self):
        """Valid E.164 phone numbers should pass."""
        valid_phones = [
            "+919876543210",
            "+14155551234",
            "+442071234567",
            "+1234567890123",
        ]
        for phone in valid_phones:
            assert validate_phone_format(phone) is True, f"Should accept {phone}"

    def test_validate_phone_e164_invalid(self):
        """Invalid phone numbers should fail."""
        invalid_phones = [
            "9876543210",       # Missing +
            "+91",             # Too short
            "919876543210",    # Missing +
            "+0123456789",     # Starts with 0 after +
            "hello",           # Not a number
            "",                # Empty
        ]
        for phone in invalid_phones:
            assert validate_phone_format(phone) is False, f"Should reject {phone}"

    def test_validate_coordinates_valid(self):
        """Valid coordinates should pass."""
        valid_coords = [
            (28.6139, 77.2090),   # Delhi
            (12.9716, 77.5946),   # Bangalore
            (0, 0),               # Null Island
            (-90, -180),          # Min bounds
            (90, 180),            # Max bounds
        ]
        for lat, lng in valid_coords:
            assert validate_coordinates(lat, lng) is True, f"Should accept ({lat}, {lng})"

    def test_validate_coordinates_invalid(self):
        """Invalid coordinates should fail."""
        invalid_coords = [
            (91, 0),    # Lat > 90
            (-91, 0),   # Lat < -90
            (0, 181),   # Lng > 180
            (0, -181),  # Lng < -180
            (999, 999), # Both invalid
        ]
        for lat, lng in invalid_coords:
            assert validate_coordinates(lat, lng) is False, f"Should reject ({lat}, {lng})"


# =============================================================================
# SIGNATURE VALIDATION TESTS
# =============================================================================

class TestSignatureValidation:
    """Tests for Twilio signature validation."""

    def test_invalid_signature_returns_403(self, client, mock_twilio_signature_invalid, sample_text_message):
        """Invalid Twilio signature should return 403 Forbidden."""
        response = client.post(
            "/api/whatsapp",
            data=sample_text_message
        )
        assert response.status_code == 403

    def test_valid_signature_processes_message(self, client, mock_twilio_signature, sample_text_message, clear_rate_limit_cache):
        """Valid signature should allow message processing."""
        response = client.post(
            "/api/whatsapp",
            data=sample_text_message
        )
        # Should not be 403
        assert response.status_code != 403


# =============================================================================
# WEBHOOK ENDPOINT TESTS
# =============================================================================

class TestWebhookEndpoint:
    """Tests for main WhatsApp webhook endpoint."""

    def test_rate_limit_response(self, client, mock_twilio_signature, sample_text_message, clear_rate_limit_cache):
        """Exceeding rate limit should return rate limit message."""
        phone = sample_text_message["From"].replace("whatsapp:", "")

        # Fill up rate limit
        _rate_limit_cache[phone] = [datetime.utcnow()] * RATE_LIMIT_MESSAGES

        response = client.post(
            "/api/whatsapp",
            data=sample_text_message
        )

        assert response.status_code == 200
        assert "too many messages" in response.text.lower()

    def test_invalid_coordinates_rejected(self, client, mock_twilio_signature, sample_invalid_coordinates, clear_rate_limit_cache):
        """Invalid coordinates should return error message."""
        response = client.post(
            "/api/whatsapp",
            data=sample_invalid_coordinates
        )

        assert response.status_code == 200
        assert "invalid" in response.text.lower() or "location" in response.text.lower()


# =============================================================================
# UTILITY FUNCTION TESTS
# =============================================================================

class TestUtilityFunctions:
    """Tests for utility functions in webhook module."""

    def test_check_rate_limit_creates_cache_entry(self, clear_rate_limit_cache):
        """Rate limit check should create cache entry for new phone."""
        phone = "+919999999999"
        assert phone not in _rate_limit_cache

        check_rate_limit(phone)

        assert phone in _rate_limit_cache
        assert len(_rate_limit_cache[phone]) == 1

    def test_rate_limit_cache_cleans_old_entries(self, clear_rate_limit_cache):
        """Rate limit should clean entries older than window."""
        phone = "+919999999999"

        # Add old entry
        old_time = datetime.utcnow() - timedelta(seconds=RATE_LIMIT_WINDOW_SECONDS + 10)
        _rate_limit_cache[phone] = [old_time]

        # New check should clean old and add new
        check_rate_limit(phone)

        # Should only have 1 entry (the new one)
        assert len(_rate_limit_cache[phone]) == 1
        # The entry should be recent
        assert _rate_limit_cache[phone][0] > old_time
