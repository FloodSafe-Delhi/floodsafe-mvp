# FloodSafe Development Guide

> Nonprofit flood monitoring platform for social good.
> AI assistants: Read this FIRST before any work.

---

## TOP PRIORITY RULES (NON-NEGOTIABLE)

**These rules override all other instructions. Violation is unacceptable.**

1. **NEVER TAKE SHORTCUTS** - Every feature requires full systems-level thinking. No "quick fixes" that skip proper architecture.

2. **NEVER MAKE ASSUMPTIONS** - Ask questions when unclear. Verify requirements before implementing. Don't guess user intent.

3. **ALWAYS EXPLORE FIRST** - Use explore agents before any implementation. Understand existing code patterns before writing new code.

4. **COMPLETE END-TO-END** - No partial implementations. No "TODO later" comments. Every feature must work fully when marked complete.

5. **TEST EVERYTHING** - Type safety (`npx tsc --noEmit`), build (`npm run build`), console clean, E2E verification. All gates must pass.

6. **NO JARGON-LOADED UI** - Keep user interfaces simple and clear. Avoid technical terms. The close button should be a simple X, not buried in complex patterns.

---

## Quick Reference

| Component | Location | Tech |
|-----------|----------|------|
| Backend API | `apps/backend/` | FastAPI, SQLAlchemy, PostGIS |
| Frontend | `apps/frontend/` | React 18, TypeScript, Vite |
| ML Service | `apps/ml-service/` | PyTorch, GEE, Prophet |
| IoT Ingestion | `apps/iot-ingestion/` | FastAPI (high-throughput) |

## Commands
```bash
# Full stack (Docker)
docker-compose up

# Local dev (requires DB running)
docker-compose up -d db  # Start only database

# Frontend dev
cd apps/frontend && npm run dev

# Backend dev (LOCAL - use localhost in .env)
cd apps/backend && python -m uvicorn src.main:app --reload

# Tests
cd apps/frontend && npm run build && npx tsc --noEmit
cd apps/backend && pytest

# Visual Testing (Playwright)
cd apps/frontend && npm run screenshot
```

### Environment Setup
- **Docker**: Uses `DATABASE_URL=postgresql://user:password@db:5432/floodsafe`
- **Local dev**: Change `.env` to `DATABASE_URL=postgresql://user:password@localhost:5432/floodsafe`

---

## Subagent Strategy

### When to Use Subagents
**ALWAYS prefer subagents for:**
- Exploring unfamiliar code areas (3+ files to check)
- Parallel verification tasks
- Multi-domain features (frontend + backend)

**Use direct tools for:**
- Single file edits
- Quick lookups (known file paths)
- Running build/test commands

### Subagent Types

| Agent | Scope | When to Use |
|-------|-------|-------------|
| `explore` | Codebase research | FIRST step for any task |
| `frontend-ui` | React/TypeScript | UI features, component fixes, visual testing |
| `backend-api` | FastAPI/Python | API endpoints, services |
| `maps-geo` | MapLibre, PostGIS | Map features, spatial queries |
| `ml-data` | ML pipelines | GEE, predictions, model training |
| `verifier` | E2E verification | After implementation, test flows |
| `code-reviewer` | Quality check | After completing code changes |
| `planner` | Architecture | Complex multi-file features |

---

## Architecture Rules

### Backend (Python)
- **Layers**: `api/` → `domain/services/` → `infrastructure/`
- **Models**: Pydantic v2 with `from_attributes=True`
- **Database**: SQLAlchemy 2.0, UUID PKs, PostGIS (SRID 4326)
- **Never**: DB queries in routers, business logic in models

### Frontend (React/TS)
- **State**: Context (global) + TanStack Query (server)
- **API**: Use `fetchJson`/`uploadFile` from `lib/api/client.ts`
- **Components**: `screens/` for pages, `ui/` for primitives
- **Styling**: Tailwind CSS + Radix UI

### Common Gotchas (IMPORTANT)

#### Timestamps (UTC)
Backend stores UTC timestamps WITHOUT 'Z' suffix. Frontend must parse as UTC:
```typescript
const parseUTCTimestamp = (timestamp: string) => {
    if (!timestamp.endsWith('Z') && !timestamp.includes('+')) {
        return new Date(timestamp + 'Z');
    }
    return new Date(timestamp);
};
```

#### Query Invalidation
After mutations, invalidate queries to refresh data:
```typescript
queryClient.invalidateQueries({ queryKey: ['reports'] });
```

#### GeoJSON Coordinates
Always `[longitude, latitude]` order (not lat/lng).

---

## Development Philosophy

### Core Principle: NEVER TAKE SHORTCUTS

Every feature must be approached with systems-level thinking:

1. **UNDERSTAND** - What components are involved? How do they interact?
2. **PLAN** - Identify ALL affected files, consider edge cases
3. **IMPLEMENT** - Handle errors, use proper TypeScript types (never `any`)
4. **VERIFY** - Test E2E, check console for warnings

### Anti-Patterns (FORBIDDEN)

| Don't | Do Instead |
|-------|------------|
| Fix only the symptom | Trace root cause through system |
| Skip planning for "simple" tasks | Plan even small changes |
| Use `any` TypeScript type | Define proper interfaces |
| Ignore console warnings | Fix all warnings |
| Test only happy path | Test edge cases and errors |

### Quality Gates (NON-NEGOTIABLE)

| Gate | Command |
|------|---------|
| Type Safety | `npx tsc --noEmit` |
| Build | `npm run build` |
| Console Clean | Check browser console |

---

## Domain Contexts

### @reports
```yaml
files: [ReportScreen.tsx, ReportModal.tsx, reports.py, hooks.ts]
patterns: FormData upload, EXIF extraction, PostGIS POINT
status: COMPLETE
```

### @alerts
```yaml
files: [alerts.py, alert_service.py, Placeholders.tsx, TopNav.tsx]
patterns: Watch areas, PostGIS ST_DWithin, notification badges
status: COMPLETE
```

### @auth
```yaml
files: [AuthContext.tsx, token-storage.ts, auth.py, auth_service.py]
patterns: JWT, Firebase, refresh tokens
warning: HIGH-RISK - extra review required
```

### @onboarding (COMPLETE)
```yaml
files:
  - apps/backend/src/scripts/migrate_add_onboarding_fields.py
  - apps/backend/src/api/daily_routes.py
  - apps/frontend/src/components/screens/OnboardingScreen.tsx
patterns: 5-step wizard, resumable flow, city preference
flow: Login → profile_complete check → OnboardingScreen → HomeScreen
migration: python -m apps.backend.src.scripts.migrate_add_onboarding_fields
```

### @historical-floods (COMPLETE)
```yaml
files:
  - apps/frontend/src/components/HistoricalFloodsPanel.tsx
  - apps/frontend/src/lib/api/historical-floods.ts
  - apps/backend/src/api/historical_floods.py
  - apps/ml-service/data/delhi_historical_floods.json
data_source: IFI-Impacts (IIT-Delhi Hydrosense Lab, Zenodo)
coverage: Delhi NCR 1969-2023 (45 events)
features:
  - Decade-grouped timeline view
  - Severity color coding (minor/moderate/severe)
  - Stats: events, fatalities, severe count
  - City-specific: Delhi shows data, Bangalore shows "Coming Soon"
  - Custom scrollbar with purple theme
patterns:
  - GeoJSON FeatureCollection response
  - useHistoricalFloods hook (24hr cache)
  - Panel overlay with click-outside-to-close
```

### @hotspots (COMPLETE)
```yaml
files:
  ML Service:
  - apps/ml-service/src/api/hotspots.py - 62 Delhi waterlogging hotspots
  - apps/ml-service/src/data/fhi_calculator.py - FHI calculation
  - apps/ml-service/data/delhi_waterlogging_hotspots.json - Location data

  Backend:
  - apps/backend/src/api/hotspots.py - API proxy with caching
  - apps/backend/verify_hotspot_spatial.py - Spatial differentiation test

  Frontend:
  - apps/frontend/src/lib/api/hooks.ts - useHotspots hook (30min cache)
  - apps/frontend/src/components/MapComponent.tsx:413-554 - Layer rendering

architecture:
  FHI (Flood Hazard Index) - Multi-component live risk:
  - Formula: FHI = (0.35×P + 0.18×I + 0.12×S + 0.12×A + 0.08×R + 0.15×E) × T
  - P: Precipitation forecast with probability correction (1.5-2.25x)
  - I: Hourly max intensity
  - S: Soil saturation (urban hybrid: 70% drainage + 30% soil)
  - A: Antecedent conditions (3-day rainfall)
  - R: Runoff potential (pressure-based)
  - E: Elevation risk (inverted: low elev = high risk)
  - T: Monsoon modifier (1.2x June-Sept, 1.0x otherwise)
  - Rain-gate: If <5mm/3d, FHI capped at 0.15 (prevents false alarms)

color_priority: |
  FHI color PRIMARY (live weather), fallback to ML risk (static terrain)
  - MapLibre expression: ['coalesce', ['get', 'fhi_color'], ['get', 'risk_color']]
  - Dry weather: All green (0.15 FHI) - correct behavior
  - During rain: Colors vary by location (spatial differentiation)

verification:
  - Run: python apps/backend/verify_hotspot_spatial.py
  - Tests: 62 unique coordinates, elevation variation, FHI distribution
  - Expected: PASS with 32+ unique elevations across 70m+ range

status: COMPLETE - FHI-first coloring, spatial differentiation verified
```

### @ml-predictions
```yaml
files:
  - apps/ml-service/src/features/extractor.py - 81-dim feature extraction
  - apps/ml-service/src/api/predictions.py - /forecast-grid endpoint
  - apps/ml-service/src/models/lstm_model.py - Attention LSTM
  - apps/frontend/src/components/MapComponent.tsx - Heatmap layer

risk_levels:
  - 0.0-0.2: Low (green)
  - 0.2-0.4: Moderate (yellow)
  - 0.4-0.7: High (orange)
  - 0.7-1.0: Extreme (red)

status: |
  COMPLETE: Attention LSTM (96.2%), heatmap UI
  MISSING: LightGBM ensemble, Wavelet preprocessing
```

### @routing (COMPLETE)
```yaml
files:
  Backend:
  - apps/backend/src/domain/services/routing_service.py - Safe route calculation
  - apps/backend/src/domain/services/hotspot_routing.py - Hotspot avoidance module
  - apps/backend/src/api/routes_api.py - POST /routes/compare endpoint
  - apps/backend/src/domain/models.py - HotspotAnalysis, NearbyHotspot Pydantic models

  Frontend:
  - apps/frontend/src/components/NavigationPanel.tsx - Route planning UI (Sheet)
  - apps/frontend/src/components/RouteComparisonCard.tsx - Normal vs FloodSafe comparison
  - apps/frontend/src/types.ts - HotspotAnalysis, NearbyHotspot TypeScript interfaces

architecture:
  HARD AVOID Strategy (300m proximity threshold):
  - LOW FHI: Allow - no penalty
  - MODERATE FHI: Allow - show warning only
  - HIGH FHI: AVOID - route must reroute around
  - EXTREME FHI: AVOID - route must reroute around

  Route Comparison Flow:
  1. User enters origin/destination in NavigationPanel
  2. POST /routes/compare with city code
  3. Backend fetches hotspots for Delhi (not Bangalore)
  4. Analyzes both normal and FloodSafe routes
  5. Returns hotspot_analysis with nearby_hotspots
  6. Frontend shows checkmarks on selected route + hotspot warnings

features:
  - Visible X close button on NavigationPanel
  - Loading spinner during route calculation
  - Checkmark badges on selected route cards
  - Hotspot analysis section with FHI color dots
  - "Must reroute" warnings for HIGH/EXTREME hotspots
  - Flood zone opacity increased to 0.5

status: COMPLETE - Hotspot-aware routing with HARD AVOID strategy
```

---

## Safety Rules

### Never Modify Without Reading First
- `infrastructure/models.py` - Database schema
- `auth_service.py`, `AuthContext.tsx` - Auth flows
- `token-storage.ts` - Token handling
- `core/config.py` - Environment config

### Database Changes Require
1. Migration script in `scripts/migrate_*.py`
2. Test on dev database
3. Rollback procedure documented

### No Secrets in Code
- Use `.env` files (check `.env.example`)
- Never commit credentials

---

## Testing Requirements

### Quality Gates
```bash
# Type safety & build
cd apps/frontend && npx tsc --noEmit
cd apps/frontend && npm run build

# Backend tests
cd apps/backend && pytest

# Hotspots spatial differentiation
python apps/backend/verify_hotspot_spatial.py  # Verify 62 unique locations

# Visual testing (Playwright)
cd apps/frontend && npm run screenshot  # Requires auth
```

### Before Marking Complete
- [ ] `npx tsc --noEmit` passes (no type errors)
- [ ] `npm run build` passes (frontend)
- [ ] No new TypeScript `any` types
- [ ] Error handling present
- [ ] Console clean (no warnings)

---

## Current Priorities (Roadmap)

### Tier 1: Community Intelligence (COMPLETE)
- [x] Report submission E2E
- [x] Reports on map + auto-update
- [x] User profiles - My Reports
- [x] Alert mechanism MVP
- [x] Onboarding system
- [x] HomeScreen live data
- [x] Authentication

### Tier 2: ML/AI Foundation (IN PROGRESS)
- [x] Attention LSTM model (96.2% accuracy)
- [x] 81-dim feature vector (AlphaEarth + GloFAS)
- [x] Heatmap visualization
- [x] Historical Floods Panel (Delhi NCR 1969-2023)
- [x] Hotspots FHI (62 Delhi locations, live weather-based colors)
- [x] Spatial differentiation verification
- [ ] **LightGBM ensemble** - MISSING
- [ ] **Wavelet preprocessing** - MISSING
- [ ] **GNN spatial modeling** - Phase 2

### Tier 3: Scale (LATER)
- Flood simulation
- Multi-language (i18n)
