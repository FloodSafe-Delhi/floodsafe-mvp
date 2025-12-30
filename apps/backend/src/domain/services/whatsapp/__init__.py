"""
WhatsApp Integration Services

User-centric WhatsApp bot for flood reporting and queries.
Primary action: Photo + Location = Flood Report

Modules:
- message_templates: Bilingual message templates (English/Hindi)
- photo_handler: Twilio media download + ML classification
- command_handlers: RISK, WARNINGS, MY AREAS handlers
"""

from .message_templates import (
    TemplateKey,
    get_message,
    get_user_language,
    format_risk_factors,
    format_alerts_list,
    format_watch_areas,
)

from .photo_handler import (
    FloodClassification,
    download_twilio_media,
    classify_flood_image,
    process_sos_with_photo,
    get_severity_from_classification,
    get_confidence_text,
)

from .command_handlers import (
    geocode_location,
    handle_risk_command,
    handle_warnings_command,
    handle_my_areas_command,
    handle_help_command,
    handle_status_command,
    get_readable_location,
)

__all__ = [
    # Templates
    "TemplateKey",
    "get_message",
    "get_user_language",
    "format_risk_factors",
    "format_alerts_list",
    "format_watch_areas",
    # Photo handling
    "FloodClassification",
    "download_twilio_media",
    "classify_flood_image",
    "process_sos_with_photo",
    "get_severity_from_classification",
    "get_confidence_text",
    # Command handlers
    "geocode_location",
    "handle_risk_command",
    "handle_warnings_command",
    "handle_my_areas_command",
    "handle_help_command",
    "handle_status_command",
    "get_readable_location",
]
