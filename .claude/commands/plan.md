# Plan Agent

You are in PLAN mode. Design the implementation approach.

## Your Task
Plan implementation for: $ARGUMENTS

## Instructions
1. Use findings from previous explore (don't re-read files unnecessarily)
2. Break down into specific, atomic steps
3. Reference exact file paths and line numbers
4. Consider edge cases
5. Keep changes minimal - don't over-engineer

## Output Format
Provide your plan in this format:

```
## IMPLEMENTATION PLAN

### Checklist
- [ ] Step 1 - `file.py:20-45` - description
- [ ] Step 2 - `component.tsx:100-120` - description
- [ ] Step 3 - Create `new_file.py` - description

### Expected Behavior
When [action] happens → [result] should occur

### Edge Cases
- Case 1: [scenario] → handled by [approach]
- Case 2: [scenario] → handled by [approach]

### Files to Modify
| File | Action | Lines |
|------|--------|-------|
| path/file.py | MODIFY | 20-45 |
| path/new.py | CREATE | - |

### Testing Strategy
- Unit test: [description]
- Integration: [description]
```

## Gate
Get user approval before proceeding to code phase (auto-approve if <3 files modified).
