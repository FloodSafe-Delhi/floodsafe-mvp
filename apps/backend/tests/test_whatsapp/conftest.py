"""
Pytest fixtures for WhatsApp integration tests.

Provides mock fixtures for Twilio, database sessions, and sample messages.
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_twilio_signature():
    """
    Patch Twilio signature validation to always return True.
    Use this fixture for tests that don't specifically test signature validation.
    """
    with patch('src.api.webhook.validate_twilio_signature', return_value=True):
        yield


@pytest.fixture
def mock_twilio_signature_invalid():
    """
    Patch Twilio signature validation to always return False.
    Use this for testing signature validation failure.
    """
    with patch('src.api.webhook.validate_twilio_signature', return_value=False):
        yield


@pytest.fixture
def mock_ml_service():
    """
    Mock ML service for flood classification.
    Returns a successful flood detection response.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "is_flood": True,
        "confidence": 0.85,
        "classification": "flood"
    }

    with patch('httpx.AsyncClient') as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
        yield mock_client


@pytest.fixture
def mock_ml_service_unavailable():
    """
    Mock ML service that returns 503 (unavailable).
    """
    mock_response = MagicMock()
    mock_response.status_code = 503

    with patch('httpx.AsyncClient') as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        yield mock_client


@pytest.fixture
def sample_text_message():
    """Sample WhatsApp text message (HELP command)."""
    # Only include fields that have values - form data doesn't support None
    return {
        "From": "whatsapp:+919876543210",
        "Body": "HELP",
        "MessageSid": "SM12345abcdef",
        "NumMedia": "0"
    }


@pytest.fixture
def sample_location_message():
    """Sample WhatsApp location pin message (Delhi coordinates)."""
    return {
        "From": "whatsapp:+919876543210",
        "Body": "",
        "MessageSid": "SM12346abcdef",
        "NumMedia": "0",
        "Latitude": "28.6139",
        "Longitude": "77.2090"
    }


@pytest.fixture
def sample_photo_message():
    """Sample WhatsApp photo message (no location)."""
    return {
        "From": "whatsapp:+919876543210",
        "Body": "",
        "MessageSid": "SM12347abcdef",
        "NumMedia": "1",
        "MediaUrl0": "https://api.twilio.com/2010-04-01/Accounts/AC123/Messages/SM123/Media/ME123",
        "MediaContentType0": "image/jpeg"
    }


@pytest.fixture
def sample_photo_with_location_message():
    """Sample WhatsApp photo + location message (ideal SOS report)."""
    return {
        "From": "whatsapp:+919876543210",
        "Body": "",
        "MessageSid": "SM12348abcdef",
        "NumMedia": "1",
        "Latitude": "28.6139",
        "Longitude": "77.2090",
        "MediaUrl0": "https://api.twilio.com/2010-04-01/Accounts/AC123/Messages/SM123/Media/ME123",
        "MediaContentType0": "image/jpeg"
    }


@pytest.fixture
def sample_button_tap():
    """Sample button tap response (Quick Reply button)."""
    return {
        "From": "whatsapp:+919876543210",
        "Body": "",
        "MessageSid": "SM12349abcdef",
        "NumMedia": "0",
        "ButtonPayload": "report_flood",
        "ButtonText": "Report Flood"
    }


@pytest.fixture
def sample_risk_command():
    """Sample RISK command message."""
    return {
        "From": "whatsapp:+919876543210",
        "Body": "RISK",
        "MessageSid": "SM12350abcdef",
        "NumMedia": "0"
    }


@pytest.fixture
def sample_invalid_coordinates():
    """Sample message with invalid coordinates (outside valid range)."""
    return {
        "From": "whatsapp:+919876543210",
        "Body": "",
        "MessageSid": "SM12351abcdef",
        "NumMedia": "0",
        "Latitude": "999.999",  # Invalid latitude
        "Longitude": "999.999"   # Invalid longitude
    }


@pytest.fixture
def clear_rate_limit_cache():
    """Clear the rate limit cache before/after tests."""
    from src.api.webhook import _rate_limit_cache
    _rate_limit_cache.clear()
    yield
    _rate_limit_cache.clear()
