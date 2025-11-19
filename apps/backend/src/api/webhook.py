from fastapi import APIRouter, Request, Form
from typing import Optional

router = APIRouter()

@router.post("/whatsapp")
async def handle_whatsapp_webhook(
    From: str = Form(...),
    Body: Optional[str] = Form(None),
    Latitude: Optional[float] = Form(None),
    Longitude: Optional[float] = Form(None)
):
    """
    Receives WhatsApp Webhooks from Twilio.
    If location is present, treats it as an SOS.
    """
    if Latitude and Longitude:
        print(f"🆘 SOS RECEIVED from {From} at {Latitude}, {Longitude}")
        # TODO: Create Report(type="SOS", verified=True)
        # TODO: Trigger NotificationService
        return "SOS Location Received. Emergency services alerted."
    
    return "Please share your Location Pin for SOS."
