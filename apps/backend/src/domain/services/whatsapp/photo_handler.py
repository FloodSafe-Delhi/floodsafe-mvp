"""
WhatsApp Photo Handler

Downloads photos from Twilio media URLs and classifies them
using the ML service for flood detection.
"""
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

import httpx

from ....core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class FloodClassification:
    """Result of ML flood classification."""
    is_flood: bool
    confidence: float  # 0.0-1.0
    classification: str  # "flood" or "no_flood"
    needs_review: bool
    raw_response: dict


async def download_twilio_media(media_url: str) -> Optional[bytes]:
    """
    Download media from Twilio's media URL.

    Twilio media URLs require HTTP Basic Auth with account credentials.

    Args:
        media_url: Full Twilio media URL (e.g., https://api.twilio.com/2010-04-01/...)

    Returns:
        Image bytes if successful, None if failed
    """
    if not media_url:
        logger.warning("Empty media URL provided")
        return None

    if not settings.TWILIO_ACCOUNT_SID or not settings.TWILIO_AUTH_TOKEN:
        logger.warning("Twilio credentials not configured - cannot download media")
        return None

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                media_url,
                auth=(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN),
                follow_redirects=True,
                timeout=30.0
            )

            if response.status_code != 200:
                logger.error(f"Failed to download Twilio media: {response.status_code}")
                return None

            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                logger.warning(f"Unexpected content type: {content_type}")
                return None

            return response.content

    except httpx.TimeoutException:
        logger.error("Timeout downloading Twilio media")
        return None
    except Exception as e:
        logger.error(f"Error downloading Twilio media: {e}")
        return None


async def classify_flood_image(
    image_bytes: bytes,
    content_type: str = "image/jpeg"
) -> Optional[FloodClassification]:
    """
    Send image to ML service for flood classification.

    Args:
        image_bytes: Raw image bytes
        content_type: MIME type of the image

    Returns:
        FloodClassification if successful, None if ML service unavailable
    """
    if not image_bytes:
        return None

    # Use internal ML service URL for server-side calls
    ml_url = settings.ML_SERVICE_URL
    if not ml_url:
        logger.warning("ML_SERVICE_URL not configured")
        return None

    try:
        async with httpx.AsyncClient() as client:
            files = {
                "image": ("whatsapp_photo.jpg", image_bytes, content_type)
            }

            response = await client.post(
                f"{ml_url}/api/v1/classify-flood",
                files=files,
                timeout=15.0
            )

            if response.status_code == 503:
                logger.warning("ML classifier not loaded")
                return None

            if response.status_code != 200:
                logger.error(f"ML classification failed: {response.status_code}")
                return None

            data = response.json()
            return FloodClassification(
                is_flood=data.get("is_flood", False),
                confidence=data.get("confidence", 0.0),
                classification=data.get("classification", "no_flood"),
                needs_review=data.get("needs_review", True),
                raw_response=data
            )

    except httpx.TimeoutException:
        logger.warning("ML classification timed out")
        return None
    except httpx.ConnectError:
        logger.warning(f"Cannot connect to ML service at {ml_url}")
        return None
    except Exception as e:
        logger.error(f"ML classification error: {e}")
        return None


async def process_sos_with_photo(
    media_url: str,
    content_type: str = "image/jpeg"
) -> Tuple[Optional[bytes], Optional[FloodClassification]]:
    """
    Full photo processing pipeline for WhatsApp SOS.

    Downloads the photo from Twilio and runs ML classification.

    Args:
        media_url: Twilio media URL
        content_type: MIME type from Twilio

    Returns:
        Tuple of (image_bytes, classification) - either can be None if step failed
    """
    # Step 1: Download from Twilio
    image_bytes = await download_twilio_media(media_url)
    if not image_bytes:
        logger.warning("Failed to download photo from Twilio")
        return None, None

    logger.info(f"Downloaded {len(image_bytes)} bytes from Twilio")

    # Step 2: Classify with ML
    classification = await classify_flood_image(image_bytes, content_type)
    if classification:
        logger.info(
            f"ML classification: {classification.classification} "
            f"(confidence: {classification.confidence:.1%})"
        )
    else:
        logger.warning("ML classification unavailable")

    return image_bytes, classification


def get_severity_from_classification(
    classification: Optional[FloodClassification]
) -> str:
    """
    Map ML classification to human-readable severity.

    Used for WhatsApp response messages.
    """
    if not classification:
        return "Unknown (AI unavailable)"

    if not classification.is_flood:
        return "No flooding detected"

    confidence = classification.confidence
    if confidence >= 0.8:
        return "Waterlogging likely impassable"
    elif confidence >= 0.6:
        return "Significant waterlogging"
    elif confidence >= 0.4:
        return "Moderate waterlogging"
    else:
        return "Possible waterlogging (needs verification)"


def get_confidence_text(
    classification: Optional[FloodClassification],
    language: str = 'en'
) -> str:
    """
    Format confidence as human-readable text.
    """
    if not classification:
        if language == 'hi':
            return "AI सत्यापन उपलब्ध नहीं"
        return "AI verification unavailable"

    confidence_pct = int(classification.confidence * 100)

    if classification.is_flood:
        if language == 'hi':
            return f"बाढ़ का पता चला ({confidence_pct}% confidence)"
        return f"FLOODING DETECTED ({confidence_pct}% confidence)"
    else:
        if language == 'hi':
            return f"छवि में बाढ़ नहीं मिली ({confidence_pct}% confidence)"
        return f"No flooding detected ({confidence_pct}% confidence)"
