# WhatsApp Bot Setup Guide

A step-by-step guide to set up the FloodSafe WhatsApp bot with Twilio sandbox.

**Total Setup Time: ~10 minutes**

---

## Prerequisites

- [ ] Twilio account (free trial works: https://www.twilio.com/try-twilio)
- [ ] ngrok installed for local development (`npm install -g ngrok` or download from https://ngrok.com)
- [ ] Backend running locally on port 8000
- [ ] Database migration applied (see Step 0)

---

## Step 0: Database Migration (First Time Only)

Ensure the WhatsApp sessions table exists:

```bash
cd apps/backend
python -m src.scripts.migrate_add_whatsapp_sessions
```

---

## Step 1: Get Twilio Sandbox Credentials (5 min)

1. Go to https://console.twilio.com/
2. Sign up or log in
3. From the dashboard, copy your credentials:
   - **Account SID**: Starts with `AC...` (visible on dashboard)
   - **Auth Token**: Click "Show" to reveal

4. Navigate to: **Messaging → Try it Out → Send a WhatsApp message**
5. Note your sandbox code (e.g., "join hungry-crocodile")

---

## Step 2: Configure Environment Variables (2 min)

Add these to your `apps/backend/.env` file:

```env
# Twilio Configuration
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
# TWILIO_WEBHOOK_URL will be set after starting ngrok (Step 3)
```

---

## Step 3: Start ngrok Tunnel (1 min)

ngrok creates a public URL that forwards to your local server.

```bash
ngrok http 8000
```

You'll see output like:
```
Forwarding  https://abc123.ngrok-free.app -> http://localhost:8000
```

**Copy the HTTPS URL** (e.g., `https://abc123.ngrok-free.app`)

Add to your `.env` file:
```env
TWILIO_WEBHOOK_URL=https://abc123.ngrok-free.app/api/whatsapp
```

> **Important**: The ngrok URL changes each time you restart. Update TWILIO_WEBHOOK_URL and the Twilio Console (Step 4) whenever it changes.

---

## Step 4: Configure Twilio Webhook (2 min)

1. Go to Twilio Console: https://console.twilio.com/us1/develop/sms/try-it-out/whatsapp-learn
2. Under **Sandbox Configuration**, find "When a message comes in"
3. Set the webhook URL to:
   ```
   https://abc123.ngrok-free.app/api/whatsapp
   ```
   (Replace with your actual ngrok URL)
4. Ensure method is **POST**
5. Click **Save**

---

## Step 5: Join the Sandbox (1 min)

1. Open WhatsApp on your phone
2. Add the Twilio sandbox number to your contacts: **+1-415-523-8886**
3. Send the join code message:
   ```
   join <your-sandbox-code>
   ```
   Example: `join hungry-crocodile`
4. You should receive a confirmation: "You're connected to the sandbox..."

---

## Step 6: Start Backend & Test (1 min)

1. Start your backend server:
   ```bash
   cd apps/backend
   python -m uvicorn src.main:app --reload --port 8000
   ```

2. Verify health check:
   ```bash
   curl http://localhost:8000/api/whatsapp/health
   ```
   Should return:
   ```json
   {"twilio_configured": true, "database": "ok", "ml_service": "ok", "webhook_url": "https://..."}
   ```

3. Send a test message to the sandbox:
   - Send: `HELP`
   - Expected: Bot returns command menu

---

## Available Commands

Once connected, you can use these commands:

| Command | Description |
|---------|-------------|
| `HELP` | Show all available commands |
| `RISK` | Check flood risk (send location after) |
| `WARNINGS` | Get official flood alerts |
| `MY AREAS` | View your watch areas (requires linked account) |
| `STATUS` | Check your account status |
| `LINK` | Link your WhatsApp to a FloodSafe account |

### Primary Action: Report Flooding

1. **Send a photo** of flooding
2. **Send your location** (pin)
3. Bot creates a verified flood report with AI analysis

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No response from bot | Check ngrok is running, check backend logs |
| "Invalid Twilio signature" | Verify `TWILIO_WEBHOOK_URL` matches ngrok URL exactly |
| "Technical difficulties" | Check database is running and migration applied |
| 403 Forbidden | Twilio signature validation failed - check auth token |
| Messages not received | Ensure you've joined the sandbox (Step 5) |
| ngrok session expired | Restart ngrok, update webhook URL in .env AND Twilio Console |

### Debug Tips

1. **Check backend logs**: Look for "WhatsApp message from..." entries
2. **Check ngrok dashboard**: http://127.0.0.1:4040 shows all requests
3. **Verify webhook URL**: Must end with `/api/whatsapp` (not `/api/webhooks/whatsapp`)
4. **Test health endpoint**: `curl http://localhost:8000/api/whatsapp/health`

---

## Production Deployment

For production (not sandbox), you'll need:

1. **WhatsApp Business API** account (via Twilio or Meta)
2. **Verified business profile** on WhatsApp
3. **Approved message templates** for proactive notifications
4. Update webhook URL to production backend:
   ```
   https://floodsafe-backend-floodsafe-dda84554.koyeb.app/api/whatsapp
   ```

See [Twilio WhatsApp API docs](https://www.twilio.com/docs/whatsapp/api) for production setup.

---

## Environment Variables Reference

| Variable | Description | Example |
|----------|-------------|---------|
| `TWILIO_ACCOUNT_SID` | Your Twilio Account SID | `ACxxxxxxxx...` |
| `TWILIO_AUTH_TOKEN` | Your Twilio Auth Token | `xxxxxxxx...` |
| `TWILIO_WHATSAPP_NUMBER` | Twilio WhatsApp number | `whatsapp:+14155238886` |
| `TWILIO_WEBHOOK_URL` | Public URL for webhook | `https://xxx.ngrok.io/api/whatsapp` |
| `TWILIO_SMS_NUMBER` | (Optional) SMS fallback | `+1234567890` |
