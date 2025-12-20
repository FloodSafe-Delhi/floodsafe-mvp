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

7. **FRONTEND DEV SERVER IS PORT 5175** - The frontend runs on `http://localhost:5175`, NOT 5173. Never confuse this.

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

# Local dev (requires DB + ML service)
docker-compose up -d db ml-service  # Start database and ML service

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

#### Frontend Layout Rules (MANDATORY)

**1. Dynamic Sizing Over Hardcoded Values**
- NEVER use hardcoded pixel values for heights/widths that depend on viewport or content
- USE: `h-full`, `min-h-screen`, `flex-1`, `calc()`, CSS Grid with `fr` units
- AVOID: `h-[500px]`, `w-[800px]` unless truly fixed design requirement
- Components must adapt to their container, not assume fixed dimensions

**2. Relative Positioning Awareness**
- Before adding/modifying positioned elements, MAP the existing positioning context:
  - What elements use `relative`, `absolute`, `fixed`, `sticky`?
  - What are their parent containers?
  - What z-index values exist in the hierarchy?
- Document positioning decisions in comments when non-obvious

**3. Overlap Prevention Checklist**
Before any frontend change, verify:
- [ ] New element doesn't overlap existing fixed/absolute elements
- [ ] Z-index doesn't conflict with modals, navbars, or overlays
- [ ] Mobile viewport doesn't cause content to overflow or hide
- [ ] Scroll behavior remains correct (no double scrollbars)

**4. Systematic Layout Thinking**
- Think in layout hierarchy: Viewport → Page → Section → Component → Element
- Each level should handle its own spacing/positioning
- Parent components control layout flow, children fill allocated space
- Test at multiple viewport sizes (mobile 375px, tablet 768px, desktop 1280px)

**Anti-Pattern Examples:**
```tsx
// BAD: Hardcoded height breaks on different screens
<div className="h-[calc(100vh-200px)]">  // Magic number 200px

// GOOD: Flex layout adapts to available space
<div className="flex flex-col h-full">
  <header className="flex-shrink-0">...</header>
  <main className="flex-1 overflow-auto">...</main>
</div>

// BAD: Absolute without understanding context
<div className="absolute top-0">  // Where does this land?

// GOOD: Clear positioning context
<div className="relative">  {/* Positioning anchor */}
  <div className="absolute top-0 right-0">  {/* Relative to parent */}
```

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

#### CSS Stacking Context & Fixed Positioning
**CRITICAL**: When using `position: fixed` for modals/overlays:
1. **Transform creates new stacking context**: If ANY parent has `transform`, `filter`, or `perspective`, fixed positioning becomes relative to that parent, not viewport
2. **Z-index is relative within stacking contexts**: A z-index of 9999 inside a stacking context can still appear behind elements outside it
3. **Radix UI components (Sheet, Dialog)**: Use Portal to render at document root, avoiding stacking issues. If removing Portal, ensure no parent transforms exist
4. **Debug tip**: Use JS to check `element.getBoundingClientRect()` - if `top` is way off viewport (e.g., 1178px when viewport is 739px), a parent transform is likely the cause

**Solution**: Render fixed overlays via Portal to document root, or use custom divs without Radix context dependencies.

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

### @community (COMPLETE)
```yaml
files:
  Backend:
  - apps/backend/src/api/comments.py - Comments CRUD API
  - apps/backend/src/api/reports.py - Vote endpoints with deduplication
  - apps/backend/src/infrastructure/models.py - ReportVote, Comment models
  - apps/backend/src/domain/models.py - CommentCreate, CommentResponse, VoteResponse DTOs

  Frontend:
  - apps/frontend/src/components/screens/AlertsScreen.tsx - Community filter shows ReportCard
  - apps/frontend/src/components/ReportCard.tsx - Reusable report card with voting/comments
  - apps/frontend/src/lib/api/hooks.ts - useUpvoteReport, useDownvoteReport, useComments, useAddComment

ui_architecture: |
  Community reports are integrated INTO the Alerts screen via filter tabs.
  NOT a separate screen/tab. Filter tabs: All, Official, News, Social, Community.
  When "Community" filter selected, shows ReportCard components with voting/comments.

database_models:
  ReportVote:
    - Tracks user votes for deduplication
    - Unique constraint on (user_id, report_id)
    - vote_type: 'upvote' or 'downvote'
  Comment:
    - report_id, user_id, content (max 500 chars), created_at
    - Index on (report_id, created_at) for efficient fetching

api_endpoints:
  Voting:
    - POST /api/reports/{id}/upvote - Toggle upvote (auth required)
    - POST /api/reports/{id}/downvote - Toggle downvote (auth required)
    - Deduplication: Clicking same vote removes it, opposite vote switches
  Comments:
    - GET /api/reports/{id}/comments - List comments (oldest first)
    - POST /api/reports/{id}/comments - Add comment (auth, rate limited 5/min)
    - DELETE /api/comments/{id} - Delete own comment or admin
    - GET /api/reports/{id}/comments/count - Lightweight count endpoint

features:
  - Vote deduplication via ReportVote table
  - Vote toggle (click again to remove)
  - Comment count displayed on report cards
  - Real-time UI updates via query invalidation
  - Rate limiting: max 5 comments per minute per user
  - Community filter in AlertsScreen (NOT separate tab)

comment_count_implementation: |
  Comment counts are calculated dynamically in list_reports() using
  get_comment_counts() helper which fetches counts for all reports in
  a single efficient query. This avoids data consistency issues vs
  storing count on Report model.

migration: python -m apps.backend.src.scripts.migrate_add_community_features
status: COMPLETE
```

### @alerts
```yaml
files: [alerts.py, alert_service.py, Placeholders.tsx, TopNav.tsx]
patterns: Watch areas, PostGIS ST_DWithin, notification badges
status: COMPLETE
```

### @auth (COMPLETE)
```yaml
files:
  Backend:
  - apps/backend/src/api/auth.py - /register/email, /login/email endpoints
  - apps/backend/src/domain/services/auth_service.py - register_email_user, authenticate_email_user
  - apps/backend/src/domain/services/security.py - hash_password, verify_password (bcrypt)
  - apps/backend/src/infrastructure/models.py - password_hash column

  Frontend:
  - apps/frontend/src/contexts/AuthContext.tsx - registerWithEmail, loginWithEmail
  - apps/frontend/src/components/screens/LoginScreen.tsx - Email/Google/Phone tabs

auth_methods:
  - Email/Password: bcrypt hashing, 8+ char minimum
  - Google OAuth: Firebase integration
  - Phone OTP: Firebase SMS

migration: python -m apps.backend.src.scripts.migrate_add_password_auth
status: COMPLETE - All 3 auth methods working
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
  - apps/frontend/src/components/MapComponent.tsx:710-857 - Layer rendering

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

### @ml-predictions (NEEDS FIXING)
```yaml
files:
  ML Service:
  - apps/ml-service/src/features/extractor.py - 40-dim feature extraction
  - apps/ml-service/src/features/hotspot_features.py - 18-dim hotspot features
  - apps/ml-service/src/api/predictions.py - /forecast-grid endpoint
  - apps/ml-service/src/models/lstm_model.py - LSTM architecture (untrained)
  - apps/ml-service/src/models/lightgbm_model.py - LightGBM code (untrained)
  - apps/ml-service/data/grid_predictions_cache.json - Pre-computed fallback

  Frontend:
  - apps/frontend/src/components/MapComponent.tsx - Heatmap layer

feature_extractors:
  Hotspot (18-dim): elevation, slope, TPI, TRI, TWI, SPI, rainfall stats, land cover, SAR, monsoon
  General (40-dim): Dynamic World (9), WorldCover (6), Sentinel-2 (5), Terrain (6), Precip (8), Temporal (4), GloFAS (2)

training_data:
  - hotspot_training_data.npz: 486 samples × 18-dim (for XGBoost/RF)
  - delhi_monsoon_5years_*.npz: 605 samples × 33-dim (dimension mismatch with 40-dim extractor)

risk_levels:
  - 0.0-0.2: Low (green)
  - 0.2-0.4: Moderate (yellow)
  - 0.4-0.7: High (orange)
  - 0.7-1.0: Extreme (red)

what_works:
  - FHI formula (rule-based, not ML)
  - Pre-computed heatmap cache
  - Feature extraction pipelines

what_needs_fixing:
  - Dimension mismatch (extractor: 40-dim vs data: 33-dim)
  - All models in _archive/ trained on wrong dimensions
  - Data highly imbalanced (583 low / 18 moderate / 4 high-risk)
  - No trained model weights exist

status: |
  NEEDS FIXING - Architecture exists, models not trained
  BLOCKER: Feature dimension alignment + retraining required
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

### @e2e-testing (COMPLETE)
```yaml
files:
  - apps/frontend/scripts/e2e-full-test.ts - Playwright E2E test suite

test_coverage:
  - Account creation via API + database verification
  - Login flow via UI (email/password)
  - Onboarding wizard (5 steps)
  - HomeScreen features (risk banner, cards, map)
  - Report submission (4-step wizard)
  - Profile and Watch Areas
  - Flood Atlas navigation

run: cd apps/frontend && npx tsx scripts/e2e-full-test.ts
output: 21 screenshots (e2e-1-*.png to e2e-21-*.png)
status: COMPLETE
```

### @iot-ingestion (MVP)
```yaml
files:
  - apps/iot-ingestion/src/main.py - High-throughput ingestion endpoint

endpoints:
  POST /ingest:
    - Accepts sensor_id, water_level, timestamp
    - Stores to readings table
    - Updates sensor.last_ping
    - NO FK validation (accepts any sensor_id for speed)

architecture: |
  Intentionally isolated from main backend for performance.
  Raw SQL for speed, no ORM overhead.
  No authentication on ingestion endpoint (MVP).

limitations:
  - Accepts readings from unregistered sensors
  - No sensor pairing/registration workflow
  - No authentication on ingest endpoint
  - No rate limiting

status: |
  MVP COMPLETE - Works for high-throughput ingestion
  MISSING: Sensor registration, auth, rate limiting
```

### @profiles (BASIC)
```yaml
files:
  Backend:
  - apps/backend/src/infrastructure/models.py:16 - User.role column
  - apps/backend/src/api/deps.py:109-128 - Admin role check
  - apps/backend/src/api/users.py - User management

current_roles:
  - user: Default role, can submit reports
  - admin: Can access admin endpoints

needed_roles:
  - verified_reporter: Trusted users with higher credibility scores
  - moderator: Can verify/reject reports

limitations:
  - No permission granularity
  - No audit trail for role changes
  - No moderator queue

status: |
  BASIC - Only admin/user implemented
  NEXT: Add verified_reporter role with auto-elevated trust scores
```

### @photo-verification (GPS Only)
```yaml
files:
  - apps/backend/src/api/reports.py:241-288 - EXIF GPS extraction
  - apps/backend/src/domain/services/validation_service.py - IoT cross-validation

current_verification:
  1. Extract GPS from photo EXIF metadata (PIL)
  2. Compare photo GPS to reported location
  3. If distance > 100m, set location_verified=False
  4. Cross-validate with nearby IoT sensors (1km radius)
  5. Calculate iot_validation_score (0-100)

what_does_NOT_exist:
  - ML classification of flood vs non-flood images
  - Water depth estimation from photos
  - Fake/manipulated image detection
  - Actual photo storage (uses mock URLs)

photo_storage: |
  MOCKED - reports.py:267: media_url = f"https://mock-storage.com/{image.filename}"
  Photos are processed but NOT stored to S3/Blob

status: |
  PARTIAL - GPS verification works
  MISSING: ML flood detection, severity estimation, real storage
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

# E2E Full Test (creates account, tests all flows)
cd apps/frontend && npx tsx scripts/e2e-full-test.ts
```

### Test Accounts
```yaml
# E2E Test Account (auto-created by e2e-full-test.ts)
email: e2e_test_<timestamp>@floodsafe.test
password: TestPassword123!
city: Delhi
watch_area: Connaught Place
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
- [x] Authentication (Email/Password + Google + Phone)
- [x] E2E Testing Suite (Playwright)

### Tier 2: ML/AI Foundation (NEEDS FIXING)

**What Works:**
- [x] FHI rule-based risk calculation (not ML, but functional)
- [x] Hotspot feature extractor (18-dim)
- [x] General feature extractor (40-dim)
- [x] Pre-computed heatmap cache
- [x] Historical Floods Panel (Delhi NCR 1969-2023)
- [x] Hotspots with live weather (FHI formula)
- [x] Spatial differentiation verification

**Training Data Exists:**
- [x] hotspot_training_data.npz: 486 samples × 18-dim
- [x] delhi_monsoon_5years_*.npz: 605 samples × 33-dim

**What Needs Fixing:**
- [ ] Feature dimension alignment (extractor: 40-dim vs data: 33-dim)
- [ ] Train XGBoost/RF on 18-dim hotspot data
- [ ] Train LSTM on aligned feature data
- [ ] Data imbalance handling (only 4 high-risk events in 5 years)
- [ ] Replace broken models in _archive/

### Tier 3: Next Priorities
- [ ] RF + XGBoost ensemble for hotspot predictions
- [ ] Profile backend: Admin + Verified Reporter + User roles
- [ ] Photo ML: Flood detection + severity estimation
- [ ] IoT integration: Sensor registration, auth
- [ ] Photo storage: Replace mock URLs with S3/Blob
- [ ] UI cleanup

### Tier 4: Scale (LATER)
- Flood simulation
- Multi-language (i18n)
- GNN spatial modeling
