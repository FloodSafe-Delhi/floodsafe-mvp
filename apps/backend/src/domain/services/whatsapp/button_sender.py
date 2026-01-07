"""
WhatsApp Quick Reply Button Sender

Uses Twilio Content API to send messages with tappable buttons.
No more typing commands - users just tap!

Architecture:
1. Templates must be created first (via API or Console)
2. Messages are sent using ContentSid (template ID)
3. User taps button -> webhook receives ButtonPayload

Usage:
    await send_welcome_with_buttons(phone)
    await send_after_location_buttons(phone)
    await send_message_with_buttons(phone, "Custom text", BUTTON_SETS["menu"])
"""
import logging
import json
from typing import List, Tuple, Optional, Dict
from functools import lru_cache

import httpx

# Twilio is optional - may not be installed in all environments
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TwilioClient = None
    TWILIO_AVAILABLE = False

from ....core.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# BUTTON DEFINITIONS
# =============================================================================

# Button sets: (button_id, button_title)
# button_id is what we receive in ButtonPayload
# button_title is what the user sees
BUTTON_SETS: Dict[str, List[Tuple[str, str]]] = {
    "welcome": [
        ("report_flood", "📸 Report Flood"),
        ("check_risk", "🔍 Check Risk"),
        ("view_alerts", "⚠️ Alerts"),
    ],
    "after_location": [
        ("add_photo", "📸 Add Photo"),
        ("submit_anyway", "✅ Submit Anyway"),
        ("cancel", "❌ Cancel"),
    ],
    "after_report": [
        ("check_risk", "🔍 Check Nearby"),
        ("report_another", "📸 Report Another"),
        ("menu", "🏠 Menu"),
    ],
    "after_report_no_flood": [
        ("report_another", "📸 Report Another"),
        ("check_risk", "🔍 Check Risk"),
        ("menu", "🏠 Menu"),
    ],
    "risk_result": [
        ("check_my_location", "📍 My Location"),
        ("view_alerts", "⚠️ Alerts"),
        ("menu", "🏠 Menu"),
    ],
    "alerts_result": [
        ("check_risk", "🔍 Check Risk"),
        ("report_flood", "📸 Report"),
        ("menu", "🏠 Menu"),
    ],
    "menu": [
        ("report_flood", "📸 Report Flood"),
        ("check_risk", "🔍 Check Risk"),
        ("view_alerts", "⚠️ Alerts"),
    ],
}

# Hindi button labels (for bilingual support)
BUTTON_SETS_HI: Dict[str, List[Tuple[str, str]]] = {
    "welcome": [
        ("report_flood", "📸 बाढ़ रिपोर्ट"),
        ("check_risk", "🔍 जोखिम जांचें"),
        ("view_alerts", "⚠️ अलर्ट"),
    ],
    "after_location": [
        ("add_photo", "📸 फोटो जोड़ें"),
        ("submit_anyway", "✅ बिना फोटो भेजें"),
        ("cancel", "❌ रद्द करें"),
    ],
    "after_report": [
        ("check_risk", "🔍 आसपास जांचें"),
        ("report_another", "📸 और रिपोर्ट"),
        ("menu", "🏠 मेनू"),
    ],
    "menu": [
        ("report_flood", "📸 बाढ़ रिपोर्ट"),
        ("check_risk", "🔍 जोखिम जांचें"),
        ("view_alerts", "⚠️ अलर्ट"),
    ],
}


# =============================================================================
# TEMPLATE MANAGEMENT
# =============================================================================

# Cache for template SIDs (template_name -> SID)
_template_sid_cache: Dict[str, str] = {}


def _get_twilio_client():
    """Get Twilio client if configured and available."""
    if not TWILIO_AVAILABLE:
        logger.warning("Twilio package not installed")
        return None
    if not settings.TWILIO_ACCOUNT_SID or not settings.TWILIO_AUTH_TOKEN:
        logger.warning("Twilio credentials not configured")
        return None
    return TwilioClient(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)


async def create_content_template(
    friendly_name: str,
    body: str,
    buttons: List[Tuple[str, str]],
    language: str = "en"
) -> Optional[str]:
    """
    Create a Content Template in Twilio.

    Args:
        friendly_name: Unique template name (e.g., "floodsafe_welcome_en")
        body: Message text
        buttons: List of (id, title) tuples (max 3)
        language: Language code (en, hi)

    Returns:
        Content SID (HX...) if successful, None otherwise
    """
    if not settings.TWILIO_ACCOUNT_SID or not settings.TWILIO_AUTH_TOKEN:
        logger.warning("Cannot create template - Twilio not configured")
        return None

    # Check cache first
    if friendly_name in _template_sid_cache:
        return _template_sid_cache[friendly_name]

    # Build button actions
    actions = [{"id": btn_id, "title": btn_title} for btn_id, btn_title in buttons[:3]]

    payload = {
        "friendly_name": friendly_name,
        "language": language,
        "types": {
            "twilio/quick-reply": {
                "body": body,
                "actions": actions
            },
            # Fallback for non-WhatsApp channels
            "twilio/text": {
                "body": body
            }
        }
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://content.twilio.com/v1/Content",
                auth=(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN),
                json=payload,
                timeout=15.0
            )

            if response.status_code == 201:
                data = response.json()
                sid = data.get("sid")
                logger.info(f"Created Content Template: {friendly_name} -> {sid}")
                _template_sid_cache[friendly_name] = sid
                return sid
            elif response.status_code == 409:
                # Template already exists - try to fetch it
                logger.info(f"Template {friendly_name} already exists, fetching SID")
                return await get_template_by_name(friendly_name)
            else:
                logger.error(f"Failed to create template: {response.status_code} - {response.text}")
                return None

    except Exception as e:
        logger.error(f"Error creating Content Template: {e}")
        return None


async def get_template_by_name(friendly_name: str) -> Optional[str]:
    """
    Fetch template SID by friendly name.

    Note: Twilio doesn't have a direct lookup by name, so we list and filter.
    """
    if friendly_name in _template_sid_cache:
        return _template_sid_cache[friendly_name]

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://content.twilio.com/v1/Content",
                auth=(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN),
                timeout=15.0
            )

            if response.status_code == 200:
                data = response.json()
                for content in data.get("contents", []):
                    if content.get("friendly_name") == friendly_name:
                        sid = content.get("sid")
                        _template_sid_cache[friendly_name] = sid
                        return sid

            logger.warning(f"Template {friendly_name} not found")
            return None

    except Exception as e:
        logger.error(f"Error fetching templates: {e}")
        return None


# =============================================================================
# MESSAGE SENDING
# =============================================================================

async def send_message_with_buttons(
    phone: str,
    body: str,
    button_set_name: str,
    language: str = "en"
) -> bool:
    """
    Send a WhatsApp message with Quick Reply buttons.

    Args:
        phone: Phone number in E.164 format (e.g., +919876543210)
        body: Message text
        button_set_name: Key from BUTTON_SETS (e.g., "welcome", "menu")
        language: "en" or "hi"

    Returns:
        True if sent successfully
    """
    # Get buttons for the language
    button_sets = BUTTON_SETS_HI if language == "hi" else BUTTON_SETS
    buttons = button_sets.get(button_set_name, BUTTON_SETS.get(button_set_name, []))

    if not buttons:
        logger.warning(f"Unknown button set: {button_set_name}")
        return await send_text_message(phone, body)

    # Generate template name
    template_name = f"floodsafe_{button_set_name}_{language}"

    # Get or create template
    content_sid = await get_template_by_name(template_name)
    if not content_sid:
        content_sid = await create_content_template(
            friendly_name=template_name,
            body=body,
            buttons=buttons,
            language=language
        )

    if not content_sid:
        # Fallback to text-only message
        logger.warning("No template available, falling back to text")
        return await send_text_message(phone, body)

    # Send using Content SID
    return await _send_with_content_sid(phone, content_sid)


async def _send_with_content_sid(phone: str, content_sid: str) -> bool:
    """Send message using a Content Template SID."""
    client = _get_twilio_client()
    if not client:
        return False

    try:
        # Normalize phone number
        if not phone.startswith("+"):
            phone = f"+91{phone}" if len(phone) == 10 else f"+{phone}"

        message = client.messages.create(
            from_=settings.TWILIO_WHATSAPP_NUMBER,
            to=f"whatsapp:{phone}",
            content_sid=content_sid
        )

        logger.info(f"Sent button message to {phone}: {message.sid}")
        return True

    except Exception as e:
        logger.error(f"Failed to send button message: {e}")
        return False


async def send_text_message(phone: str, body: str) -> bool:
    """Send a plain text message (fallback when buttons unavailable)."""
    client = _get_twilio_client()
    if not client:
        return False

    try:
        if not phone.startswith("+"):
            phone = f"+91{phone}" if len(phone) == 10 else f"+{phone}"

        message = client.messages.create(
            from_=settings.TWILIO_WHATSAPP_NUMBER,
            to=f"whatsapp:{phone}",
            body=body
        )

        logger.info(f"Sent text message to {phone}: {message.sid}")
        return True

    except Exception as e:
        logger.error(f"Failed to send text message: {e}")
        return False


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def send_welcome_with_buttons(phone: str, language: str = "en") -> bool:
    """Send welcome message with main action buttons."""
    body = (
        "🌊 Welcome to FloodSafe!\n\n"
        "Report floods happening around you.\n"
        "Your reports alert nearby residents."
    ) if language == "en" else (
        "🌊 FloodSafe में आपका स्वागत है!\n\n"
        "अपने आसपास की बाढ़ की रिपोर्ट करें।\n"
        "आपकी रिपोर्ट पास के लोगों को अलर्ट करती है।"
    )
    return await send_message_with_buttons(phone, body, "welcome", language)


async def send_after_location_buttons(phone: str, language: str = "en") -> bool:
    """Send buttons after user sends location without photo."""
    body = (
        "📍 Location received!\n\n"
        "Add a photo for faster verification.\n"
        "Photos help our AI confirm flooding."
    ) if language == "en" else (
        "📍 स्थान प्राप्त हुआ!\n\n"
        "तेज़ verification के लिए फोटो जोड़ें।\n"
        "फोटो हमारे AI को बाढ़ confirm करने में मदद करती है।"
    )
    return await send_message_with_buttons(phone, body, "after_location", language)


async def send_after_report_buttons(
    phone: str,
    location_name: str,
    confidence: int,
    alerts_count: int,
    language: str = "en"
) -> bool:
    """Send confirmation with buttons after report submitted."""
    body = (
        f"✅ FLOOD REPORT SUBMITTED\n\n"
        f"📍 {location_name}\n"
        f"🤖 AI: FLOODING DETECTED ({confidence}%)\n"
        f"🔔 {alerts_count} people alerted"
    ) if language == "en" else (
        f"✅ बाढ़ रिपोर्ट सबमिट हो गई\n\n"
        f"📍 {location_name}\n"
        f"🤖 AI: बाढ़ का पता चला ({confidence}%)\n"
        f"🔔 {alerts_count} लोगों को अलर्ट किया गया"
    )
    return await send_message_with_buttons(phone, body, "after_report", language)


async def send_risk_result_buttons(
    phone: str,
    location_name: str,
    risk_level: str,
    risk_emoji: str,
    language: str = "en"
) -> bool:
    """Send risk check result with buttons."""
    body = (
        f"📊 FLOOD RISK: {location_name}\n\n"
        f"Current Risk: {risk_emoji} {risk_level.upper()}\n"
    ) if language == "en" else (
        f"📊 बाढ़ जोखिम: {location_name}\n\n"
        f"वर्तमान जोखिम: {risk_emoji} {risk_level.upper()}\n"
    )
    return await send_message_with_buttons(phone, body, "risk_result", language)


async def send_menu_buttons(phone: str, language: str = "en") -> bool:
    """Send main menu with action buttons."""
    body = "What would you like to do?" if language == "en" else "आप क्या करना चाहेंगे?"
    return await send_message_with_buttons(phone, body, "menu", language)


# =============================================================================
# INITIALIZATION
# =============================================================================

async def ensure_templates_exist() -> Dict[str, str]:
    """
    Ensure all required Content Templates exist in Twilio.

    Call this on startup to pre-create templates.
    Returns dict of template_name -> SID.
    """
    templates = {}

    # Define all templates we need
    template_definitions = [
        ("floodsafe_welcome_en", "🌊 Welcome to FloodSafe!\n\nReport floods happening around you.", BUTTON_SETS["welcome"], "en"),
        ("floodsafe_welcome_hi", "🌊 FloodSafe में आपका स्वागत है!\n\nअपने आसपास की बाढ़ की रिपोर्ट करें।", BUTTON_SETS_HI["welcome"], "hi"),
        ("floodsafe_after_location_en", "📍 Location received!\n\nAdd a photo for faster verification.", BUTTON_SETS["after_location"], "en"),
        ("floodsafe_after_location_hi", "📍 स्थान प्राप्त हुआ!\n\nतेज़ verification के लिए फोटो जोड़ें।", BUTTON_SETS_HI["after_location"], "hi"),
        ("floodsafe_menu_en", "What would you like to do?", BUTTON_SETS["menu"], "en"),
        ("floodsafe_menu_hi", "आप क्या करना चाहेंगे?", BUTTON_SETS_HI["menu"], "hi"),
    ]

    for name, body, buttons, lang in template_definitions:
        sid = await get_template_by_name(name)
        if not sid:
            sid = await create_content_template(name, body, buttons, lang)
        if sid:
            templates[name] = sid

    logger.info(f"Initialized {len(templates)} Content Templates")
    return templates
