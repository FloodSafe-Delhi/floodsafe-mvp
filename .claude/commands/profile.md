# @profile Domain Context

Load the User Profile & Gamification domain files and work on: $ARGUMENTS

## Files to Read First
- `apps/frontend/src/components/screens/ProfileScreen.tsx`
- `apps/backend/src/api/users.py`
- `apps/backend/src/domain/services/reputation_service.py`

## Patterns
- Points system: reports give 5-15 points
- Levels: (points // 100) + 1
- Streaks: consecutive days reporting
- Badges: earned for milestones

## Gamification Rules
- Base report: +5 points
- IoT-verified report: +15 points
- Streak bonus: varies by streak length
- Level up every 100 points

## Now proceed to work on the task specified.
