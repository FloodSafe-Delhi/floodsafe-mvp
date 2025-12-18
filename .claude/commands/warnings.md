# @warnings Domain Context

Load the External Warnings domain and work on: $ARGUMENTS

## Files to Create/Modify
- `apps/backend/src/api/external_warnings.py` - fetch & cache
- `apps/backend/src/domain/services/warning_aggregator.py` - source adapters
- `apps/frontend/src/components/WarningsHub.tsx` - display

## Data Sources
| Source | Status | Access |
|--------|--------|--------|
| IMD APIs | Requires IP whitelist | Apply at mausam.imd.gov.in |
| GloFAS | Public API | Copernicus CDS (free registration) |
| FloodList | RSS feed | Public |
| CWC | No API | Static links only |

## Implementation Phases
1. **Phase 1 (Now)**: Static links + FloodList RSS
2. **Phase 2**: IMD API (after approval) + GloFAS
3. **Phase 3**: ML prediction overlay

## Warning Card Format
```
Source: IMD/GloFAS/FloodList
Region: Delhi/Bangalore
Severity: LOW/MEDIUM/HIGH/CRITICAL
Issued: timestamp
Link: [View Full Bulletin]
```

## Now proceed to work on the task specified.
