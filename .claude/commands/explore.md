# Explore Agent

You are in EXPLORE mode. Research the codebase before implementing.

## Your Task
Explore: $ARGUMENTS

## Instructions
1. Read CLAUDE.md first to understand architecture
2. Identify the relevant domain context (@reports, @auth, @maps, @profile, @ml, @warnings)
3. Use Glob and Grep to find related files (max 8 files)
4. Read the key files to understand patterns
5. Map the data flow: Input → Process → Storage → Response

## Output Format
Provide your findings in this format:

```
## EXPLORE RESULTS

### Files Found
- `path/file.py` - purpose
- `path/file.tsx` - purpose

### Data Flow
[Describe: Input → Process → Storage → Response]

### Patterns Used
- Pattern 1: description
- Pattern 2: description

### Issues/Risks Identified
- Issue 1: description + impact
- Issue 2: description + impact

### Ready for Planning
[YES/NO + summary of what needs to be done]
```

## Gate
You MUST list files and data flow before proceeding to plan phase.
