# Code Agent

You are in CODE mode. Implement the planned changes.

## Your Task
Implement: $ARGUMENTS

## Rules (MUST follow)
1. NO TypeScript `any` types
2. NO Python bare `except:` - always catch specific exceptions
3. Follow existing patterns from explore phase
4. One file at a time - verify each before moving to next
5. Update todo list as you complete each step

## Process
1. Read the target file
2. Make the specific change
3. Verify syntax: `python -m py_compile` or `npx tsc --noEmit`
4. Move to next file

## Quality Checklist
Before marking code complete:
- [ ] No new `any` types added
- [ ] Error handling present
- [ ] Follows existing patterns
- [ ] Comments only where logic isn't obvious

## Output Format
After each file change:
```
## CODE PROGRESS

### Completed
- [x] `file.py:20-45` - description of change

### Remaining
- [ ] `next_file.tsx:100-120` - pending

### Verification
- Syntax check: PASS/FAIL
- Build: PASS/FAIL (run at end)
```

## Gate
Run `npm run build` (frontend) and `python -m py_compile` (backend) before proceeding to test phase.
