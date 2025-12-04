from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/webhook", tags=["webhook"])

@router.post("/whatsapp-sos")
async def whatsapp_sos(req: Request):
    """
    Minimal webhook endpoint for receiving WhatsApp SOS payloads.
    In production you should verify signatures and authenticate the sender.
    """
    try:
        payload = await req.json()
    except Exception:
        payload = {"raw_body": (await req.body()).decode(errors="ignore")}
    # Here you could enqueue the payload, notify a human, or create a report automatically.
    return JSONResponse({"ok": True, "received": True, "payload_summary": {"keys": list(payload.keys())}})
