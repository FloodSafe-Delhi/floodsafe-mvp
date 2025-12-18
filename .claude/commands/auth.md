# @auth Domain Context

Load the Authentication domain files and work on: $ARGUMENTS

## WARNING: HIGH-RISK DOMAIN - Extra review required

## Files to Read First
- `apps/frontend/src/contexts/AuthContext.tsx`
- `apps/frontend/src/lib/auth/token-storage.ts`
- `apps/backend/src/api/auth.py`
- `apps/backend/src/domain/services/auth_service.py`
- `apps/backend/src/domain/services/security.py`

## Patterns
- JWT tokens with refresh mechanism
- Firebase authentication
- Token storage in secure storage
- Protected route handling

## Security Rules
- NEVER log tokens or credentials
- ALWAYS validate tokens server-side
- Use httpOnly cookies where possible
- Refresh tokens before expiry

## Now proceed to work on the task specified.
