# @reports Domain Context

Load the Reports domain files and work on: $ARGUMENTS

## Files to Read First
- `apps/frontend/src/components/screens/ReportScreen.tsx`
- `apps/frontend/src/components/ReportModal.tsx`
- `apps/backend/src/api/reports.py`
- `apps/backend/src/domain/services/validation_service.py`
- `apps/frontend/src/types.ts`

## Patterns
- FormData upload with multipart/form-data
- EXIF GPS extraction from photos
- PostGIS POINT geometry for locations
- IoT sensor cross-validation

## Key Functions
- Frontend: `useReportMutation()` in hooks.ts
- Backend: `create_report()` in reports.py
- Validation: `validate_report()` in validation_service.py

## Now proceed to work on the task specified.
