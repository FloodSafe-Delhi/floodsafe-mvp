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

8. **CHECK DEPENDENCIES BEFORE CREATING** - Before creating new functions/files, search if similar functionality exists. Reuse existing code. Never duplicate.

9. **ASK QUESTIONS WHENEVER NEEDED** - Don't proceed with ambiguity. Clarify scope, requirements, and acceptance criteria. Better to ask than assume wrong.

10. **BE PATIENT - DON'T RUSH TO FINISH** - Don't jump to conclusions. Don't mark things complete prematurely. Take time to do it right.

11. **VERIFY BEFORE CLAIMING COMPLETE** - "It should work" is not verification. TEST IT. PROVE IT. Be skeptical of your own work.

12. **USE SUBAGENTS PRODUCTIVELY** - Use explore agents for unfamiliar code (3+ files). Use specialized agents for their domains. Use verifier/code-reviewer after implementation.

13. **DOCUMENT IMPORTANT FINDINGS** - Add significant discoveries to REALISATIONS.md. Record gotchas, edge cases, and non-obvious behaviors.

---

## THE 10 COMMANDMENTS (Quick Reference)

1. **EXPLORE FIRST** - Understand before you change
2. **ASK QUESTIONS** - Clarify before you assume
3. **CHECK EXISTING CODE** - Reuse before you create
4. **PLAN PROPERLY** - Think before you code
5. **NO SHORTCUTS** - Do it right, not fast
6. **TYPE EVERYTHING** - No `any`, ever
7. **FIX ROOT CAUSES** - Not symptoms
8. **VERIFY THOROUGHLY** - Test, don't assume
9. **BE PATIENT** - Don't rush to "done"
10. **DOCUMENT LEARNINGS** - Record in REALISATIONS.md

---

## LARGE DATA FILE HANDLING (MANDATORY)

**CRITICAL: NEVER attempt to read entire training data files. This WILL exhaust context and cause failure.**

### Dangerous File Types
| Type | Example | Trap |
|------|---------|------|
| `.npz` | `hotspot_training_data.npz` | metadata field can be 90K+ chars |
| `.csv` | `India_Flood_Inventory_v3.csv` | 1000+ rows |
| `.json` | Large GeoJSON files | Feature arrays with 100+ elements |

### Safe Inspection Patterns

**1. NPZ files** - Use Python one-liner (NEVER use Read tool):
```bash
python -c "import numpy as np; d = np.load('file.npz'); print([(k, d[k].shape, d[k].dtype) for k in d.keys()])"
```

**2. CSV files** - Read first 10-20 lines only:
```
Use Read tool with limit: 15
```

**3. JSON files** - Get count first, then sample ONE element:
```bash
python -c "import json; d=json.load(open('file.json')); print(f'Type: {type(d).__name__}, Len: {len(d)}')"
```

### Red Flags (STOP IMMEDIATELY)
- About to read a data file WITHOUT `limit` parameter
- Planning to read entire .npz file (binary - won't work anyway)
- Second read of same large file "to see more examples"
- Accessing `np.load()['metadata']` directly (often 90K+ chars!)

### Use @data Skill
For data inspection, invoke: `@data path/to/file.npz`
This automatically applies safe patterns. See `.claude/commands/data.md`.

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

#### Docker Named Volumes vs Local Files
**CRITICAL**: ML models trained locally won't appear in Docker containers using named volumes!

The `docker-compose.yml` uses named volumes for ML models:
```yaml
volumes:
  - ml_models:/app/models  # Named volume - isolated from local filesystem!
```

**Problem**: Training saves to `apps/ml-service/models/` locally, but Docker uses `/app/models/` from the named volume (separate storage).

**Solutions**:
1. **Copy manually**: `docker cp local/model.pt container:/app/models/`
2. **Restart service**: `docker-compose restart ml-service`
3. **For development**: Consider bind mount instead:
   ```yaml
   volumes:
     - ./apps/ml-service/models:/app/models  # Bind mount - shares local files
   ```

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
  - apps/ml-service/src/api/hotspots.py - 90 Delhi waterlogging hotspots (62 MCD + 28 OSM)
  - apps/ml-service/src/data/fhi_calculator.py - FHI calculation
  - apps/ml-service/data/delhi_waterlogging_hotspots.json - Location data with source field

  Backend:
  - apps/backend/src/api/hotspots.py - API proxy with caching, source/verified fields
  - apps/backend/verify_hotspot_spatial.py - Spatial differentiation test

  Frontend:
  - apps/frontend/src/lib/api/hooks.ts - useHotspots hook (30min cache), HotspotFeature types
  - apps/frontend/src/components/MapComponent.tsx:710-857 - Layer rendering

hotspot_composition:
  total: 90
  sources:
    - mcd_reports: 62 (verified - original MCD Delhi validated)
    - osm_underpass: 28 (unverified - ML-predicted high-risk underpasses)

architecture:
  FHI (Flood Hazard Index) - CUSTOM HEURISTIC (NOT from published research):
  - Formula: FHI = (0.35×P + 0.18×I + 0.12×S + 0.12×A + 0.08×R + 0.15×E) × T
  - NOTE: Weights (0.35, 0.18, etc.) are empirically tuned, not from academic papers
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
  - Tests: 90 unique coordinates, elevation variation, FHI distribution
  - Expected: PASS with 40+ unique elevations across 70m+ range

status: COMPLETE - 90 hotspots with source/verified fields, FHI-first coloring
```

### @ml-predictions (PARTIAL - XGBoost works, Ensemble broken)
```yaml
files:
  ML Service:
  - apps/ml-service/src/features/extractor.py - 37-dim feature extraction (was 40, reverted)
  - apps/ml-service/src/features/hotspot_features.py - 18-dim hotspot features
  - apps/ml-service/src/api/predictions.py - /forecast-grid endpoint
  - apps/ml-service/src/models/xgboost_hotspot.py - XGBoost hotspot model (TRAINED)
  - apps/ml-service/src/models/lstm_model.py - LSTM architecture (NOT TRAINED)
  - apps/ml-service/src/models/lightgbm_model.py - LightGBM code (NOT TRAINED)
  - apps/ml-service/data/grid_predictions_cache.json - Pre-computed fallback

  Frontend:
  - apps/frontend/src/components/MapComponent.tsx - Heatmap layer

trained_models:
  XGBoost Hotspot Model v2.1 (90 HOTSPOTS):
    - Location: apps/ml-service/models/xgboost_hotspot/xgboost_model.json
    - Trained: 2025-12-26 (retrained with 90 hotspots)
    - Training Data: 570 samples (90 hotspots + 100 negatives × 3 monsoon dates)
    - PURPOSE 1 - Dynamic Risk at Known Hotspots: WORKS
      - Standard CV AUC: 0.9868 ± 0.017 (exceeds 0.85 target)
      - Precision: 0.917, Recall: 0.974, F1: 0.945
      - Model responds to weather changes for same location
    - PURPOSE 2 - Generalize to New Locations: LIMITED
      - Location-Aware CV AUC: 0.7535 ± 0.033 (below 0.85 target)
      - NOTE: AUC varies 0.70-0.82 depending on random negative samples
      - Temporal consistency gap: 0.041 (PASSES <0.10)
    - Features: 18-dim (elevation, slope, TPI, TRI, TWI, SPI, rainfall, land cover, SAR)
    - Top predictors: built_up_pct (15.7%), sar_vv_mean (10.3%), sar_vh_mean (8.8%)
    - Next target: 300+ locations for AUC ≥0.85

broken_models:
  ConvLSTM/GNN/LightGBM Ensemble:
    - Status: Architecture exists, NEVER TRAINED
    - /forecast endpoint returns hardcoded 0.1 probability (fallback)
    - No trained weights in models/ensemble/ directory
    - Silently fails - users see low risk for all predictions

feature_extractors:
  Hotspot (18-dim): elevation, slope, TPI, TRI, TWI, SPI, rainfall stats, land cover, SAR, monsoon
  General (37-dim): Dynamic World (9), WorldCover (6), Sentinel-2 (5), Terrain (6), Precip (5), Temporal (4), GloFAS (2)

training_data:
  - hotspot_training_data.npz: 570 samples × 18-dim (90 hotspots + 100 negatives × 3 dates)
  - delhi_monsoon_5years_*.npz: 605 samples × 37-dim (for future ensemble training)

verification_results:
  Run: python apps/ml-service/scripts/verify_xgboost_model.py
  Date: 2025-12-26 (retrained with 90 hotspots)

  Standard CV:
    - AUC: 0.9868 ± 0.017
    - Precision: 0.917, Recall: 0.974, F1: 0.945
    - All folds PASS 0.85 target

  Location-Aware CV (HONEST - GroupKFold):
    - AUC: 0.7535 ± 0.033 (below 0.85 target)
    - Same location NEVER in train+test
    - Per-fold: 0.726, 0.714, 0.755, 0.766, 0.807
    - NOTE: AUC varies 0.70-0.82 due to random negative sampling

  Temporal Split (PASSES):
    - 2022 -> 2023: AUC 0.956
    - 2023 -> 2022: AUC 0.997
    - Gap: 0.041 (< 0.10 threshold)

  Why Still Below 0.85:
    - 190 unique locations is limited (need 300+)
    - Model memorizes terrain patterns, struggles to generalize
    - Fold 1 shows 91 location overlap explains inflated Standard CV
    - No drainage network data (infrastructure failures invisible)

  Next Phase: Collect 150+ more diverse flood-prone locations

data_imbalance:
  - Low risk (0.0-0.2): 583 samples (96.4%)
  - Moderate (0.2-0.5): 18 samples (3.0%)
  - High (0.5-0.9): 4 samples (0.7%)
  - Very High (0.9-1.0): 0 samples
  - NOTE: Only 4 high-risk samples in 5 years - severe imbalance

risk_levels:
  - 0.0-0.2: Low (green)
  - 0.2-0.4: Moderate (yellow)
  - 0.4-0.7: High (orange)
  - 0.7-1.0: Extreme (red)

what_works:
  - XGBoost for KNOWN 62 HOTSPOTS - VERIFIED WORKING (weather-sensitive)
    - Model responds to weather changes (MODERATE sensitivity)
    - SAR features (cloud-penetrating) drive predictions during monsoon
    - Predictions vary 0.22-0.89 for same location based on weather
  - FHI formula (rule-based, custom heuristic) - WORKS for known locations
  - Pre-computed heatmap cache
  - Feature extraction pipelines (18-dim and 37-dim)

what_does_not_work:
  - XGBoost for NEW LOCATIONS - LIMITED (AUC 0.70-0.82, avg ~0.75)
    - Model memorizes terrain of 90 known hotspots
    - Cannot reliably generalize to discover new flood-prone areas
    - AUC varies with random negative sample selection
    - Solution: Collect 300+ diverse training locations

  - Ensemble Models (LSTM/GNN/LightGBM) - BROKEN
    - Architecture exists but never trained
    - /forecast endpoint returns fallback 0.1

what_needs_fixing:
  Medium Priority:
    - Train ConvLSTM/GNN/LightGBM ensemble on 37-dim data
    - Address data imbalance with cost-weighted training
    - Collect training data from more diverse locations

  Low Priority:
    - is_monsoon feature has 0% importance (all samples from monsoon)
    - Consider removing or replacing with month encoding

status: |
  XGBoost Hotspot Model - DUAL PURPOSE VERDICT:
  - PURPOSE 1 (Known Hotspots): WORKS - model responds to weather
  - PURPOSE 2 (New Locations): FAILS - AUC 0.71, cannot generalize
  USE FOR: Dynamic risk calculation at 62 known Delhi waterlogging hotspots
  DO NOT USE FOR: Discovering new flood-prone locations
  Ensemble models (LSTM/GNN/LightGBM) are BROKEN - return fallback 0.1
  NOTE: FHI is a CUSTOM HEURISTIC, not from published research
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

### @photo-verification (GPS + ML Classification COMPLETE)
```yaml
files:
  GPS Verification:
  - apps/backend/src/api/reports.py:241-288 - EXIF GPS extraction
  - apps/backend/src/domain/services/validation_service.py - IoT cross-validation

  ML Classification:
  - apps/ml-service/src/models/mobilenet_flood_classifier.py - MobileNet architecture
  - apps/ml-service/src/api/image_classification.py - /classify-flood endpoint
  - apps/ml-service/models/sohail_flood_model.h5 - Pre-trained weights (28MB, 55 layers)
  - apps/backend/src/api/ml.py - Backend proxy endpoint
  - apps/frontend/src/lib/api/hooks.ts - useClassifyFloodImage() hook

current_verification:
  GPS:
    1. Extract GPS from photo EXIF metadata (PIL)
    2. Compare photo GPS to reported location
    3. If distance > 100m, set location_verified=False
    4. Cross-validate with nearby IoT sensors (1km radius)
    5. Calculate iot_validation_score (0-100)

  ML Classification:
    1. Frontend sends image to POST /api/ml/classify-flood
    2. Backend proxies to ML service at :8002/api/v1/classify-flood
    3. MobileNet model classifies flood vs no_flood
    4. Returns: {is_flood, confidence, flood_probability, class_name}
    5. Threshold: 0.3 (safety-first to minimize false negatives)

ml_model_details:
  architecture: MobileNet with custom binary classification head
  input: 224x224 RGB images
  output: flood probability (0-1)
  threshold: 0.3 (low to catch more potential floods)
  training: apps/ml-service/scripts/train_flood_binary.py
  weights: models/sohail_flood_model.h5

what_does_NOT_exist:
  - Water depth estimation from photos
  - Fake/manipulated image detection
  - Actual photo storage (uses mock URLs)

photo_storage: |
  MOCKED - reports.py:267: media_url = f"https://mock-storage.com/{image.filename}"
  Photos are processed but NOT stored to S3/Blob

status: |
  GPS verification: COMPLETE
  ML flood classification: COMPLETE
  MISSING: Depth estimation, fake detection, real storage
```

### @whatsapp (MVP)
```yaml
files:
  Backend:
  - apps/backend/src/api/webhook.py - WhatsApp webhook handler with conversation flow
  - apps/backend/src/domain/services/notification_service.py - TwilioNotificationService
  - apps/backend/src/infrastructure/models.py - WhatsAppSession model
  - apps/backend/src/scripts/migrate_add_whatsapp_sessions.py - Migration script
  - apps/backend/src/core/config.py - Twilio settings

features:
  Inbound (SOS):
    - Receive location pins from WhatsApp → Create SOS reports
    - Signature validation (X-Twilio-Signature)
    - User account linking flow (create or link to existing)
    - Conversation state tracking (WhatsAppSession table)
    - TwiML XML responses

  Outbound (Alerts):
    - TwilioNotificationService implements INotificationService
    - Send WhatsApp alerts to users in watch areas
    - Respects user.notification_whatsapp preference
    - Graceful fallback when Twilio not configured

conversation_flow: |
  1. User sends location pin
  2. SOS report created immediately
  3. AlertService.check_watch_areas_for_report() runs
  4. Users in watch areas get WhatsApp notification
  5. If user not linked, prompted: "1. Create account / 2. Stay anonymous"
  6. Email input → Creates/links FloodSafe account

commands:
  - Send location = Submit SOS
  - LINK = Connect FloodSafe account
  - STATUS = Check account status
  - START/STOP = Subscribe/unsubscribe alerts

setup:
  1. Create Twilio account at https://www.twilio.com/try-twilio
  2. Go to Messaging → Try it out → Send a WhatsApp message (sandbox)
  3. Copy TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN to .env
  4. Set TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
  5. Use ngrok for local testing: ngrok http 8000
  6. Configure sandbox webhook URL in Twilio Console
  7. Run migration: python -m apps.backend.src.scripts.migrate_add_whatsapp_sessions

status: MVP COMPLETE
  - Inbound SOS reports: WORKING
  - User account linking: WORKING
  - Outbound alerts: WORKING (requires Twilio config)
  - Conversation state: WORKING (30min session timeout)
```

### @mobile (NOT STARTED)
```yaml
current: Web-responsive only via Tailwind CSS
missing: Capacitor config, PWA manifest, service worker, offline mode
next: Add Capacitor wrapper for Android/iOS builds
```

### @edge-ai (PLANNED)
```yaml
concept: ANN model running on IoT devices (ESP32/Raspberry Pi)
goal: Local flood prediction without cloud dependency
research: Based on research papers for edge computing in flood monitoring
next: Design lightweight neural network architecture
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

### Tier 2: ML/AI Foundation (PARTIAL - See Dual Purpose Analysis)

**What Works (Verified 2025-12-24):**
- [x] XGBoost for KNOWN 62 HOTSPOTS - WEATHER SENSITIVE
  - Model responds to weather changes (MODERATE sensitivity)
  - SAR VV/VH features drive predictions (r=0.35-0.36)
  - Predictions vary 0.22-0.89 based on rainfall/SAR
  - Verification: python apps/ml-service/scripts/test_xgboost_weather_sensitivity.py
- [x] FHI rule-based risk calculation (CUSTOM HEURISTIC, not from published research)
- [x] Hotspot feature extractor (18-dim)
- [x] General feature extractor (37-dim)
- [x] Pre-computed heatmap cache
- [x] Historical Floods Panel (Delhi NCR 1969-2023)
- [x] Hotspots with live weather (FHI formula)

**XGBoost Dual Purpose Verdict:**
- PURPOSE 1 (Known Hotspots): WORKS - responds to weather
- PURPOSE 2 (New Locations): LIMITED - AUC 0.70-0.82 (needs 0.85)
- USE FOR: Dynamic risk at 90 known Delhi waterlogging hotspots
- DO NOT USE FOR: Discovering new flood-prone locations

**Training Data:**
- [x] hotspot_training_data.npz: 570 samples × 18-dim (47% positive)
- 190 unique locations (90 hotspots + 100 negatives × 3 dates)
- is_monsoon feature has 0% importance (all samples from monsoon)

**What's Broken:**
- [ ] XGBoost spatial generalization - Need 300+ diverse locations to discover new hotspots
- [ ] ConvLSTM/GNN/LightGBM Ensemble - Architecture exists but NEVER TRAINED
- [ ] /forecast endpoint silently returns hardcoded 0.1 probability (fallback)
- [ ] Data imbalance (only 4 high-risk events in 5 years) - needs cost-weighted training

### Tier 3: Smart Sensors & Edge AI
- [ ] IoT sensor registration + pairing workflow
- [ ] IoT authentication on /ingest endpoint
- [ ] Edge ANN model for local flood prediction (ESP32/RPi)
- [ ] Sensor-to-cloud sync with offline buffering
- [ ] Real-time sensor status dashboard

### Tier 4: Photo Intelligence
- [x] Photo ML: Flood vs non-flood classification (MobileNet, threshold 0.3)
- [ ] Water depth estimation from images
- [ ] Fake/manipulated image detection
- [ ] Real photo storage (S3/Blob, replace mock URLs)
- [ ] Photo verification confidence score

### Tier 5: Messaging & Reach
- [ ] WhatsApp Bot: Onboarding, preferences, watch areas
- [ ] WhatsApp Bot: Alert delivery
- [ ] SMS fallback for non-WhatsApp users
- [ ] Notification service (implement INotificationService)

### Tier 6: Mobile Native
- [ ] Capacitor wrapper for Android/iOS
- [ ] PWA manifest + service worker
- [ ] Offline mode with sync
- [ ] Push notifications (Firebase Cloud Messaging)
- [ ] Background location tracking

### Tier 7: Scale & Intelligence
- [ ] Multi-language (i18n) - Hindi, Kannada
- [ ] GNN spatial modeling
- [ ] Flood simulation engine
- [ ] City expansion (Mumbai, Chennai, Kolkata)
