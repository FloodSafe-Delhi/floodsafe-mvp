# FloodSafe FREE Stack Deployment Guide

> Deployed services: Supabase + Vercel + Render + HuggingFace Spaces
> Branch: `deployment/free-stack`

---

## Deployed Services Status

| Service | Status | URL |
|---------|--------|-----|
| **Supabase DB** | ACTIVE | `https://udblirsscaghsepuxxqv.supabase.co` |
| **ML Service** | RUNNING | `https://aniru8-floodsafe-ml.hf.space` |
| **Backend API** | PENDING | Deploy to Render |
| **Frontend** | PENDING | Deploy to Vercel |

---

## 1. Supabase Database (COMPLETE)

**Project Details:**
- Project ID: `udblirsscaghsepuxxqv`
- Region: `ap-southeast-1` (Singapore)
- Status: `ACTIVE_HEALTHY`

**Tables Created (19 total):**
```
users, badges, sensors, flood_zones, reports, readings,
watch_areas, daily_routes, alerts, reputation_history,
role_history, user_badges, refresh_tokens,
email_verification_tokens, saved_routes, report_votes,
comments, external_alerts, whatsapp_sessions
```

**Extensions Enabled:**
- `postgis` (v3.3.7 in tiger schema)
- `uuid-ossp` (UUID generation)

**Connection String (get password from Supabase Dashboard):**
```
postgresql://postgres.[ref]:[YOUR_PASSWORD]@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres
```

---

## 2. ML Service - HuggingFace Spaces (COMPLETE)

**Space URL:** https://huggingface.co/spaces/aniru8/floodsafe-ml

**API Endpoints:**
```
GET  /health                        -> {"status": "healthy"}
GET  /api/v1/hotspots/all           -> 90 Delhi hotspots (GeoJSON)
GET  /api/v1/hotspots/{id}          -> Single hotspot risk
POST /api/v1/classify-flood         -> Image classification
GET  /api/v1/docs                   -> Swagger UI
```

**Models Deployed:**
- XGBoost Hotspot Model (TRAINED - 90 locations)
- MobileNet Flood Classifier (TRAINED - sohail_flood_model.h5)

**Environment Variables Set:**
```env
FASTAPI_PORT=7860
GEE_ENABLED=false
DATA_CACHE_DIR=/tmp/ml-cache
```

---

## 3. Backend API - Render (NEXT STEP)

### 3.1 Create Render Web Service

1. Go to https://dashboard.render.com
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name:** `floodsafe-api`
   - **Root Directory:** `apps/backend`
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn src.main:app --host 0.0.0.0 --port $PORT`
   - **Plan:** Free

### 3.2 Environment Variables (Required)

```env
# Database (REQUIRED - get from Supabase Dashboard > Settings > Database)
DATABASE_URL=postgresql://postgres.[ref]:[password]@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres

# JWT Secret (REQUIRED - use this or generate new: openssl rand -hex 32)
JWT_SECRET_KEY=77cff92b97f9aab1fc33b7179d6fe7437af671d50033b97c1700d41d338c7123

# ML Service (HuggingFace Spaces)
ML_SERVICE_URL=https://aniru8-floodsafe-ml.hf.space

# URLs (set after deploy)
FRONTEND_URL=https://floodsafe.vercel.app
BACKEND_URL=https://floodsafe-api.onrender.com

# CORS (JSON array)
BACKEND_CORS_ORIGINS=["https://floodsafe.vercel.app"]
```

### 3.3 Optional Environment Variables

```env
# Firebase (for phone auth)
FIREBASE_PROJECT_ID=your-firebase-project

# Google OAuth
GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-secret

# Mapbox (for routing)
MAPBOX_ACCESS_TOKEN=pk.your-mapbox-token

# SendGrid (for email verification)
SENDGRID_API_KEY=SG.your-sendgrid-key

# Twilio (for WhatsApp/SMS)
TWILIO_ACCOUNT_SID=AC...
TWILIO_AUTH_TOKEN=your-token
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
```

---

## 4. Frontend - Vercel (AFTER BACKEND)

### 4.1 Deploy to Vercel

1. Go to https://vercel.com/new
2. Import your GitHub repository
3. Configure:
   - **Framework:** Vite
   - **Root Directory:** `apps/frontend`
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`

### 4.2 Environment Variables

```env
# Backend API URL (set after Render deploy)
VITE_API_URL=https://floodsafe-api.onrender.com

# Firebase (for auth)
VITE_FIREBASE_API_KEY=your-api-key
VITE_FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
VITE_FIREBASE_PROJECT_ID=your-project-id
VITE_FIREBASE_STORAGE_BUCKET=your-project.appspot.com
VITE_FIREBASE_MESSAGING_SENDER_ID=123456789
VITE_FIREBASE_APP_ID=1:123456789:web:abc123

# Google OAuth
VITE_GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com

# Mapbox
VITE_MAPBOX_ACCESS_TOKEN=pk.your-mapbox-token
```

---

## 5. OAuth & Firebase Configuration

After deploying to Vercel, update authorized domains:

### 5.1 Google Cloud Console
1. Go to https://console.cloud.google.com/apis/credentials
2. Edit your OAuth 2.0 Client ID
3. Add to "Authorized JavaScript origins":
   - `https://floodsafe.vercel.app`
4. Add to "Authorized redirect URIs":
   - `https://floodsafe.vercel.app`

### 5.2 Firebase Console
1. Go to https://console.firebase.google.com
2. Authentication → Settings → Authorized domains
3. Add: `floodsafe.vercel.app`

---

## 6. Verification Checklist

After all deployments, verify:

- [ ] ML Service: `curl https://aniru8-floodsafe-ml.hf.space/health`
- [ ] Backend: `curl https://floodsafe-api.onrender.com/health`
- [ ] Frontend: Open https://floodsafe.vercel.app
- [ ] Login: Test Google/Email/Phone auth
- [ ] Map: Hotspots display correctly
- [ ] Reports: Can submit new report
- [ ] Mobile: Responsive on phone

---

## Quick Commands

```bash
# Check ML Service
curl https://aniru8-floodsafe-ml.hf.space/health
curl https://aniru8-floodsafe-ml.hf.space/api/v1/hotspots/all | jq '.features | length'

# Check Backend (after deploy)
curl https://floodsafe-api.onrender.com/health

# Local Development
docker-compose up
```

---

## Cost Summary (FREE Tier)

| Service | Plan | Limits |
|---------|------|--------|
| Supabase | Free | 500MB DB, 1GB storage |
| HuggingFace | Free | 16GB RAM, CPU only |
| Render | Free | 512MB RAM, sleeps after 15min |
| Vercel | Hobby | 100GB bandwidth/month |

**Total Monthly Cost: $0**
