# FloodSafe Setup Guide

Complete guide for setting up FloodSafe on a new machine.

---

## Prerequisites

### Option A: Docker (Recommended)
| Requirement | Version | Notes |
|-------------|---------|-------|
| Docker Desktop | Latest | Windows/Mac/Linux |
| Git | 2.x+ | For cloning repo |
| RAM | 8GB+ | For running all containers |

### Option B: Local Development (Without Docker)
| Requirement | Version | Notes |
|-------------|---------|-------|
| Node.js | 18+ | Frontend development |
| Python | 3.11+ | Backend & ML services |
| PostgreSQL | 15+ | With PostGIS extension |
| Git | 2.x+ | For cloning repo |

---

## Quick Start with Docker

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-org/FloodSafe.git
cd FloodSafe
```

### Step 2: Copy Environment Files
```bash
# Windows (PowerShell)
copy apps\backend\.env.example apps\backend\.env
copy apps\frontend\.env.example apps\frontend\.env
copy apps\ml-service\.env.example apps\ml-service\.env

# Linux/Mac
cp apps/backend/.env.example apps/backend/.env
cp apps/frontend/.env.example apps/frontend/.env
cp apps/ml-service/.env.example apps/ml-service/.env
```

### Step 3: Configure Environment Variables
Edit each `.env` file with your credentials (see [Environment Variables](#environment-variables) below).

### Step 4: Start All Services
```bash
docker-compose up
```

### Step 5: Verify Installation
| Service | URL | Expected |
|---------|-----|----------|
| Frontend | http://localhost:5175 | React app loads |
| Backend API | http://localhost:8000/docs | Swagger UI |
| ML Service | http://localhost:8002/docs | Swagger UI |

---

## Environment Variables

### Backend (`apps/backend/.env`)

**Required:**
```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/floodsafe

# Authentication (CHANGE IN PRODUCTION)
JWT_SECRET_KEY=your-secret-key-min-32-characters-long

# Google OAuth (from Google Cloud Console)
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret

# Firebase (for phone auth)
FIREBASE_PROJECT_ID=your-firebase-project-id
```

**Optional:**
```env
# Mapbox (for safe routing feature)
MAPBOX_ACCESS_TOKEN=your-mapbox-token

# External alerts
RSS_FEEDS_ENABLED=true
IMD_API_ENABLED=true
CWC_SCRAPER_ENABLED=true
```

### Frontend (`apps/frontend/.env`)

```env
# Backend API endpoint
VITE_API_URL=http://localhost:8000

# Google OAuth (Client ID only - no secret on frontend)
VITE_GOOGLE_CLIENT_ID=your-google-client-id

# Firebase Configuration (from Firebase Console)
VITE_FIREBASE_API_KEY=your-firebase-api-key
VITE_FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
VITE_FIREBASE_PROJECT_ID=your-firebase-project-id
VITE_FIREBASE_STORAGE_BUCKET=your-project.appspot.com
VITE_FIREBASE_MESSAGING_SENDER_ID=your-sender-id
VITE_FIREBASE_APP_ID=your-app-id
```

### ML Service (`apps/ml-service/.env`)

```env
# Google Cloud
GCP_PROJECT_ID=your-gcp-project-id

# Google Earth Engine Service Account
GEE_SERVICE_ACCOUNT_KEY=./credentials/gee-service-account.json

# Database (same as backend)
DATABASE_URL=postgresql://user:password@localhost:5432/floodsafe

# Cache directories
MODEL_CACHE_DIR=./models
DATA_CACHE_DIR=./cache
```

---

## External Services Setup

### 1. Google Cloud Console (OAuth)

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project or select existing
3. Navigate to **APIs & Services > Credentials**
4. Click **Create Credentials > OAuth 2.0 Client IDs**
5. Choose **Web application**
6. Add authorized JavaScript origins:
   - `http://localhost:5175` (development)
   - Your production domain
7. Add authorized redirect URIs:
   - `http://localhost:5175/auth/callback`
8. Copy **Client ID** and **Client Secret** to your `.env` files

### 2. Firebase Console (Phone Auth)

1. Go to [Firebase Console](https://console.firebase.google.com)
2. Create a new project or select existing
3. Navigate to **Authentication > Sign-in method**
4. Enable **Phone** authentication
5. Add `localhost` to authorized domains
6. Go to **Project Settings > General**
7. Scroll to **Your apps** and click **Web** (`</>`)
8. Copy the Firebase config values to `apps/frontend/.env`

### 3. GCP Service Account (ML Service)

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Navigate to **IAM & Admin > Service Accounts**
3. Create a new service account
4. Grant roles:
   - **Earth Engine Resource Admin** (for GEE access)
   - **Storage Object Viewer** (if using GCS)
5. Click **Keys > Add Key > Create new key > JSON**
6. Save the downloaded file to `apps/ml-service/credentials/gee-service-account.json`
7. Update `GEE_SERVICE_ACCOUNT_KEY` path in `.env`

### 4. Mapbox (Optional - for Routing)

1. Go to [Mapbox](https://www.mapbox.com) and create an account
2. Navigate to **Account > Access tokens**
3. Copy your default public token or create a new one
4. Add to `apps/backend/.env` as `MAPBOX_ACCESS_TOKEN`

---

## Local Development (Without Docker)

### Database Setup

```bash
# Install PostgreSQL 15 with PostGIS extension
# Windows: Use installer from postgresql.org
# Mac: brew install postgresql@15 postgis
# Linux: apt install postgresql-15 postgresql-15-postgis-3

# Create database
psql -U postgres
CREATE DATABASE floodsafe;
\c floodsafe
CREATE EXTENSION postgis;
\q
```

### Backend Setup

```bash
cd apps/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run migrations (if any)
# python -m alembic upgrade head

# Start server
python -m uvicorn src.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd apps/frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### ML Service Setup

```bash
cd apps/ml-service

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Start server
python -m uvicorn src.main:app --reload --port 8002
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Unauthorized JavaScript origin" | Add `http://localhost:5175` to Google Cloud Console OAuth credentials |
| "Firebase not configured" | Verify all `VITE_FIREBASE_*` variables in frontend `.env` |
| "SMS not received" | Check Firebase phone auth is enabled; use `+91` format for India |
| "ML service fails to start" | Check GEE service account key path; allow 60s startup time |
| "DATABASE_URL invalid" | Verify format: `postgresql://user:pass@host:5432/db` |
| "Port already in use" | Change port in docker-compose.yml or kill existing process |
| "PostGIS extension missing" | Run `CREATE EXTENSION postgis;` in psql |

---

## Service Ports

| Service | Port | Description |
|---------|------|-------------|
| Frontend | 5175 | React + Vite development server |
| Backend API | 8000 | FastAPI with Swagger UI |
| ML Service | 8002 | ML predictions API |
| IoT Ingestion | 8001 | Sensor data ingestion |
| PostgreSQL | 5432 | Database with PostGIS |

---

## Detailed Setup Guides

For more detailed setup instructions:
- **Authentication**: See `AUTHENTICATION_SETUP_GUIDE.md`
- **ML Service**: See `apps/ml-service/README.md`
- **DEM Processing**: See `apps/frontend/data/delhi/dem/README.md`

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review existing documentation in the repo
3. Open an issue on GitHub
