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

## MCP Servers (Model Context Protocol)

MCP servers extend Claude's capabilities with external tools and integrations. These are configured in `~/.claude/plugins/` and enabled in `~/.claude/settings.json`.

### Active MCPs (Use These)

| MCP | Purpose | When to Use |
|-----|---------|-------------|
| **Supabase** | Database management, SQL execution, migrations | Production database setup, schema changes, troubleshooting |
| **Context7** | Library documentation lookup | Looking up API docs without web search (MapLibre, TanStack, FastAPI) |
| **Figma** | Design-to-code conversion | Implementing UI from Figma designs |
| **Firebase** | Firebase Auth/Config management | Auth configuration, SDK config |
| **Vercel** | Frontend deployment | Deploying frontend to production |
| **Serena** | Code intelligence, symbol analysis | Deep refactoring, symbol navigation |
| **Claude-in-Chrome** | Browser automation | E2E testing, visual verification |

### MCP Tool Reference

#### Supabase (CRITICAL for Deployment)
```
mcp__plugin_supabase_supabase__list_projects    # List all projects
mcp__plugin_supabase_supabase__execute_sql      # Run raw SQL
mcp__plugin_supabase_supabase__apply_migration  # Apply tracked migrations
mcp__plugin_supabase_supabase__list_tables      # Verify tables
mcp__plugin_supabase_supabase__get_project_url  # Get API URL
mcp__plugin_supabase_supabase__get_publishable_keys  # Get API keys
mcp__plugin_supabase_supabase__get_advisors     # Security/performance checks
```

#### Context7 (Documentation Lookup)
```
# Step 1: Find library ID
mcp__plugin_context7_context7__resolve-library-id
# Step 2: Query docs
mcp__plugin_context7_context7__query-docs
```
**Example**: Looking up MapLibre GL JS API
```
1. resolve-library-id: query="MapLibre GL JS", libraryName="maplibre-gl"
2. query-docs: libraryId="/maplibre/maplibre-gl-js", query="add GeoJSON layer"
```

#### Serena (Code Intelligence)
```
find_symbol           # Find symbol definitions
find_referencing_symbols  # Find all references
get_symbols_overview  # Get file/project symbols
rename_symbol         # Safe refactoring
replace_symbol_body   # Replace implementation
```

#### Claude-in-Chrome (Browser Automation)
```
mcp__claude-in-chrome__tabs_context_mcp  # Get current tabs
mcp__claude-in-chrome__computer          # Screenshots, clicks, typing
mcp__claude-in-chrome__read_page         # Accessibility tree
mcp__claude-in-chrome__navigate          # URL navigation
mcp__claude-in-chrome__javascript_tool   # Execute JS
```

### Disabled MCPs (Not Needed)

| MCP | Reason |
|-----|--------|
| **GitHub MCP** | Requires Copilot subscription; use `gh` CLI instead |
| **jdtls-lsp** | No Java code in FloodSafe |

### MCP Configuration Notes

1. **Serena**: Uses pip-installed `serena` command (fixed from uvx)
   - Config: `~/.claude/plugins/cache/.../serena/.mcp.json`
   - Command: `serena start-mcp-server`

2. **LSP Plugins** (typescript-lsp, pyright-lsp): May have race condition on startup
   - Config: `.lsp.json` in project root
   - Restart Claude if LSP not connecting

### Best Practices

1. **Database Work**: Use Supabase MCP for production, direct SQL for local dev
2. **Docs Lookup**: Use Context7 instead of web search for library APIs
3. **UI Implementation**: Use Figma MCP when designs are available
4. **E2E Testing**: Use Claude-in-Chrome for browser automation
5. **Refactoring**: Use Serena for safe symbol renaming across codebase

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

#### Pydantic-Settings v2 and List Types
**CRITICAL**: `List[str]` fields fail to parse non-JSON env vars, even with `field_validator(mode="before")`.

**Problem**: Setting `BACKEND_CORS_ORIGINS=https://example.com` fails with:
```
pydantic_settings.exceptions.SettingsError: error parsing value for field "BACKEND_CORS_ORIGINS"
```

**Root Cause**: Pydantic-settings JSON-parses `List[str]` fields BEFORE validators run. The validator never gets a chance to handle comma-separated or single URLs.

**Wrong Fix**: Changing type to `Union[str, List[str]]` and adding a property wrapper. This is a shortcut that doesn't address the root cause.

**Proper Fix**: Use `Annotated[List[str], NoDecode]` from `pydantic_settings` to disable pre-parsing:
```python
from pydantic_settings import BaseSettings, NoDecode
from typing_extensions import Annotated

class Settings(BaseSettings):
    BACKEND_CORS_ORIGINS: Annotated[List[str], NoDecode] = ["http://localhost:5175"]

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        # Now this validator actually runs!
        if isinstance(v, str):
            if "," in v:
                return [url.strip() for url in v.split(",")]
            return [v.strip()]
        return v
```

**Reference**: [GitHub Issue #7749](https://github.com/pydantic/pydantic/issues/7749)

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
  - apps/backend/src/api/comments.py, reports.py - Voting + comments API
  - apps/frontend/src/components/ReportCard.tsx - Card with voting/comments
  - apps/frontend/src/components/screens/AlertsScreen.tsx - Community filter

key_points:
  - Community tab is a FILTER in AlertsScreen (not separate screen)
  - Vote deduplication via ReportVote table (unique user_id + report_id)
  - Rate limiting: max 5 comments/minute/user

migration: python -m apps.backend.src.scripts.migrate_add_community_features
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
  - apps/ml-service/src/api/hotspots.py - 90 Delhi hotspots (62 MCD + 28 OSM)
  - apps/ml-service/src/data/fhi_calculator.py - FHI calculation
  - apps/ml-service/data/delhi_waterlogging_hotspots.json - Location data
  - apps/backend/src/api/hotspots.py - API proxy with caching

FHI_formula: |
  FHI = (0.35×P + 0.18×I + 0.12×S + 0.12×A + 0.08×R + 0.15×E) × T
  CUSTOM HEURISTIC - weights empirically tuned, not from research
  Rain-gate: If <5mm/3d, FHI capped at 0.15 (prevents false alarms)

color_priority: FHI first (live weather), fallback to ML risk (static)
verification: python apps/backend/verify_hotspot_spatial.py
```

### @ml-predictions (PARTIAL)
```yaml
files:
  - apps/ml-service/src/models/xgboost_hotspot.py - XGBoost model (TRAINED)
  - apps/ml-service/src/features/hotspot_features.py - 18-dim features
  - apps/ml-service/src/features/extractor.py - 37-dim features
  - apps/ml-service/models/xgboost_hotspot/xgboost_model.json - Trained weights

model_status:
  XGBoost (90 hotspots):
    - Known Hotspots: WORKS (AUC 0.98, weather-sensitive)
    - New Locations: LIMITED (AUC 0.70-0.82, needs 0.85)
    - USE FOR: Dynamic risk at 90 known Delhi hotspots
    - DO NOT USE FOR: Discovering new locations

  Ensemble (LSTM/GNN/LightGBM): BROKEN - never trained, returns fallback 0.1

features:
  Hotspot (18-dim): elevation, slope, TPI, TRI, TWI, SPI, rainfall, land cover, SAR
  General (37-dim): Dynamic World, WorldCover, Sentinel-2, Terrain, Precip, Temporal, GloFAS

verification: python apps/ml-service/scripts/verify_xgboost_model.py
next: Collect 300+ diverse locations for better generalization
```

### @routing (COMPLETE)
```yaml
files:
  - apps/backend/src/domain/services/routing_service.py - Safe route calculation
  - apps/backend/src/domain/services/hotspot_routing.py - Hotspot avoidance
  - apps/frontend/src/components/NavigationPanel.tsx - Route planning UI

strategy: |
  HARD AVOID (300m threshold):
  - LOW/MODERATE FHI: Allow (warning only)
  - HIGH/EXTREME FHI: Must reroute around

flow: POST /routes/compare → analyze hotspots → normal vs FloodSafe comparison
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

### @photo-verification (GPS + ML COMPLETE)
```yaml
files:
  - apps/backend/src/api/reports.py:241-288 - EXIF GPS extraction
  - apps/ml-service/src/models/mobilenet_flood_classifier.py - MobileNet
  - apps/ml-service/models/sohail_flood_model.h5 - Trained weights

gps: Extract EXIF → compare to location → set location_verified if >100m
ml: MobileNet (224x224) → flood/no_flood, threshold 0.3 (safety-first)
storage: MOCKED - uses mock URLs, no real S3/Blob storage
missing: Depth estimation, fake detection, real storage
```

### @whatsapp (MVP COMPLETE)
```yaml
files:
  - apps/backend/src/api/webhook.py - WhatsApp webhook handler
  - apps/backend/src/domain/services/notification_service.py - TwilioNotificationService

inbound: Location pin → SOS report, account linking, conversation state
outbound: Alerts to watch areas via Twilio
commands: Send location (SOS), LINK, STATUS, START/STOP

setup: |
  1. Get Twilio sandbox creds → .env (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
  2. Set TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
  3. ngrok http 8000 → configure webhook URL in Twilio Console
  4. Run migration: python -m apps.backend.src.scripts.migrate_add_whatsapp_sessions
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

### Tier 1: Community Intelligence ✅ COMPLETE
Reports, map, alerts, onboarding, auth (Email/Google/Phone), E2E tests

### Tier 2: ML/AI Foundation (PARTIAL)
- [x] XGBoost for 90 known hotspots (weather-sensitive)
- [x] FHI formula, feature extractors, heatmap cache
- [x] Historical Floods Panel, Photo classification (MobileNet)
- [ ] Ensemble models (LSTM/GNN) - NOT TRAINED
- [ ] Better generalization (need 300+ locations)

### Tier 3: Smart Sensors & Edge AI
IoT registration, authentication, edge ANN, offline sync

### Tier 4: Photo Intelligence
Water depth estimation, fake detection, real storage (S3)

### Tier 5: Messaging
WhatsApp bot (onboarding, alerts), SMS fallback

### Tier 6: Mobile Native
Capacitor, PWA, offline mode, push notifications

### Tier 7: Scale
Multi-language (Hindi, Kannada), GNN, city expansion
