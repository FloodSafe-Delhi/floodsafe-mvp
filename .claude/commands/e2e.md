# E2E Verification Agent

You are in E2E mode. Verify the full flow works end-to-end.

## Your Task
Verify E2E: $ARGUMENTS

## Checklist
Go through each item and verify:

- [ ] **Happy path works** - Main flow completes successfully
- [ ] **Error states handled** - Invalid inputs show appropriate errors
- [ ] **Loading states show** - UI indicates when operations are in progress
- [ ] **Mobile layout correct** - Responsive design works
- [ ] **Data persists correctly** - Database stores data as expected

## Verification Methods
1. **API Testing**: Use curl or the app to test endpoints
2. **Database Check**: Verify data stored correctly in PostGIS
3. **UI Flow**: Trace the user journey through the interface
4. **Error Scenarios**: Test with invalid/missing data

## Output Format
```
## E2E VERIFICATION

### Flow Tested
[Description of the user flow tested]

### Results
| Check | Status | Notes |
|-------|--------|-------|
| Happy path | PASS/FAIL | [details] |
| Error states | PASS/FAIL | [details] |
| Loading states | PASS/FAIL | [details] |
| Mobile layout | PASS/FAIL | [details] |
| Data persistence | PASS/FAIL | [details] |

### Issues Found
- Issue 1: [description] → [fix needed]

### Feature Status
[COMPLETE / NEEDS FIXES]
```

## If Issues Found
Return to EXPLORE with debug focus:
- What specific step fails?
- What's the expected vs actual behavior?
- Which file likely contains the bug?

## Gate
All checklist items must pass for feature to be marked complete.
