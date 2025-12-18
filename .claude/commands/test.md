# Test Agent

You are in TEST mode. Verify the implementation with tests.

## Your Task
Test: $ARGUMENTS

## Commands to Run
```bash
# Backend
cd apps/backend && pytest -v

# Frontend
cd apps/frontend && npm test

# TypeScript types
cd apps/frontend && npx tsc --noEmit
```

## Process
1. Run existing tests first - ensure no regressions
2. Write new tests if needed for changed functionality
3. Fix any failing tests before proceeding

## Test Locations
- Backend: `apps/backend/tests/`
- Frontend: `apps/frontend/src/**/*.test.tsx`

## Output Format
```
## TEST RESULTS

### Backend (pytest)
- Status: PASS/FAIL
- Tests run: X
- Failures: [list if any]

### Frontend (npm test)
- Status: PASS/FAIL
- Tests run: X
- Failures: [list if any]

### TypeScript (tsc)
- Status: PASS/FAIL
- Errors: [list if any]

### New Tests Written
- `test_file.py::test_name` - tests [functionality]

### Ready for E2E
[YES/NO]
```

## Gate
All tests must pass before proceeding to E2E verification.
