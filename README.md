# FloodSafe

> Nonprofit flood monitoring platform for social good. AI-powered flood prediction, community reporting, and safe routing for Indian cities.

---

## System Status

**All services operational for development**:
- **Backend API**: http://localhost:8000 (FastAPI with Clean Architecture)
- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **Frontend**: http://localhost:5175 (React + Vite + MapLibre GL)
- **Database**: PostgreSQL + PostGIS on port 5432
- **ML Service**: http://localhost:8002 (PyTorch, GEE, Prophet)
- **IoT Service**: http://localhost:8001 (High-throughput ingestion)

---

## Quick Start

```bash
# Start all services
docker-compose up -d

# Frontend development (local)
cd apps/frontend && npm run dev

# Backend development (local - change .env to use localhost:5432)
cd apps/backend && python -m uvicorn src.main:app --reload

# Run tests
cd apps/frontend && npm run build && npx tsc --noEmit
cd apps/backend && pytest
```

---

## Architecture

```
FloodSafe/
├── apps/
│   ├── backend/          # FastAPI, SQLAlchemy, PostGIS (Clean Architecture)
│   ├── frontend/         # React 18, TypeScript, TanStack Query, MapLibre GL
│   ├── ml-service/       # PyTorch, Google Earth Engine, Prophet
│   └── iot-ingestion/    # FastAPI (high-throughput sensor data)
├── docker-compose.yml
└── CLAUDE.md             # Development guide for AI assistants
```

### Backend (Python)
- **Pattern**: Clean Architecture with layers: `api/` → `domain/services/` → `infrastructure/`
- **Database**: PostgreSQL with PostGIS (SRID 4326 for GeoJSON)
- **Auth**: Email/Password (bcrypt), Google OAuth, Phone OTP (Firebase)

### Frontend (React/TypeScript)
- **State Management**: React Context (global) + TanStack Query (server state)
- **Maps**: MapLibre GL with PMTiles for offline-capable flood visualization
- **Styling**: Tailwind CSS + Radix UI components

---

## Features (Implemented)

### Core Features
- **User Authentication**: Email/Password, Google OAuth, Phone OTP
- **Onboarding Wizard**: 5-step flow for city preference and watch areas
- **Flood Reports**: Photo upload with GPS verification and EXIF extraction
- **Community Voting**: Upvote/downvote with deduplication, comments
- **Watch Areas**: User-defined alert zones with PostGIS spatial queries

### Flood Intelligence
- **FHI (Flood Hazard Index)**: Live weather-based risk calculation
  - Formula: `FHI = (0.35×P + 0.18×I + 0.12×S + 0.12×A + 0.08×R + 0.15×E) × T`
  - Components: Precipitation, Intensity, Soil saturation, Antecedent rain, Runoff, Elevation
  - Monsoon modifier: 1.2x during June-September
- **Waterlogging Hotspots**: 62 Delhi locations with live FHI coloring
- **Historical Floods**: Delhi NCR 1969-2023 (45 events from IFI-Impacts dataset)
- **ML Predictions**: Heatmap grid with cached fallback data

### Safe Routing
- **Route Comparison**: Normal vs FloodSafe routes
- **Hotspot Avoidance**: HARD AVOID for HIGH/EXTREME FHI zones (300m buffer)
- **Metro Integration**: Nearby station suggestions
- **Live Navigation**: Turn-by-turn with voice guidance

### Gamification
- **Points System**: Earn points for verified reports
- **Levels & Badges**: Progressive leveling ("First Reporter", "Flood Guardian")
- **Community Leaderboard**: Top contributors by area

---

## City Coverage

| City | Status | Features |
|------|--------|----------|
| Delhi | Full | Hotspots (62), Historical Floods (45), ML Predictions |
| Bangalore | Basic | Map only, no hotspot/historical data yet |

---

## API Endpoints

### Authentication
- `POST /api/auth/register/email` - Email registration
- `POST /api/auth/login/email` - Email login
- `POST /api/auth/google` - Google OAuth
- `POST /api/auth/phone/send-otp` - Phone OTP

### Reports
- `GET /api/reports` - List all reports
- `POST /api/reports` - Submit flood report (with photo)
- `POST /api/reports/{id}/upvote` - Toggle upvote
- `POST /api/reports/{id}/downvote` - Toggle downvote

### Comments
- `GET /api/reports/{id}/comments` - List comments
- `POST /api/reports/{id}/comments` - Add comment
- `DELETE /api/comments/{id}` - Delete comment

### Hotspots & Routing
- `GET /api/hotspots` - Get waterlogging hotspots with FHI
- `POST /api/routes/compare` - Compare normal vs safe routes

### Watch Areas & Alerts
- `GET /api/watch-areas` - User's watch areas
- `POST /api/watch-areas` - Create watch area
- `GET /api/alerts` - Get alerts for user's areas

---

## Development

### Prerequisites
- Docker Desktop
- Node.js 18+
- Python 3.11+

### Environment Setup
```bash
# Docker (uses internal hostname)
DATABASE_URL=postgresql://user:password@db:5432/floodsafe

# Local development (change to localhost)
DATABASE_URL=postgresql://user:password@localhost:5432/floodsafe
```

### Quality Gates
```bash
# Frontend
cd apps/frontend
npx tsc --noEmit      # Type checking
npm run build         # Production build
npm run lint          # ESLint

# Backend
cd apps/backend
pytest                # Unit tests

# E2E Testing
cd apps/frontend
npx tsx scripts/e2e-full-test.ts  # Full flow test
```

---

## Roadmap

### Completed (Tier 1 & 2)
- [x] Authentication (Email/Google/Phone)
- [x] Report submission with GPS verification
- [x] Community voting and comments
- [x] Watch areas and alerts
- [x] FHI-based hotspot coloring
- [x] Safe routing with hotspot avoidance
- [x] Historical floods panel (Delhi)
- [x] Gamification (points, levels, badges)
- [x] E2E testing suite

### In Progress (ML Foundation)
- [ ] Fix feature dimension alignment (40-dim extractor vs 33-dim data)
- [ ] Train XGBoost/RF on 18-dim hotspot data
- [ ] Address data imbalance (only 4 high-risk events in 5 years)

### Next Priorities (Tier 3)
- [ ] RF + XGBoost ensemble for predictions
- [ ] Photo ML: Flood detection + severity estimation
- [ ] IoT integration: Sensor registration, auth
- [ ] Real photo storage (S3/Blob)
- [ ] Multi-language (i18n)

---

## Contributing

See `CLAUDE.md` for development patterns and domain contexts.

**Key Patterns**:
1. Use explore agents before implementing
2. Full E2E verification for every feature
3. Type safety: `npx tsc --noEmit` must pass
4. Clean console: No warnings in browser

---

## License

This is a nonprofit project for social good. Contact for licensing inquiries.
