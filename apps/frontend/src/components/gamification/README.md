# Gamification Components

These components enhance user engagement by displaying reputation, badges, streaks, and level progress in the FloodSafe Profile section.

## Components

### 1. StreakWidget

Shows user's current report streak with fire animation and 7-day activity indicator.

**Features:**
- Current streak count with fire icon animation
- 7-day activity visualization
- Milestone badges (7, 14, 30, 60, 90 days)
- Next milestone countdown
- CTA to maintain streak

**Usage:**
```tsx
import { StreakWidget } from '@/components/gamification';

<StreakWidget className="mb-4" />
```

**Data Source:** `useMyReputation()` hook → `streak_days`

---

### 2. ReputationDashboard

Displays reputation score (0-100), accuracy rate, and recent point changes.

**Features:**
- Color-coded reputation score (green >70, yellow 40-70, red <40)
- Accuracy percentage
- Current streak display
- Recent activity history (last 5 changes)
- Collapsible "How is this calculated?" explainer

**Usage:**
```tsx
import { ReputationDashboard } from '@/components/gamification';

<ReputationDashboard className="mb-4" />
```

**Data Sources:**
- `useMyReputation()` → `reputation_score`, `accuracy_rate`, `streak_days`
- `useMyReputationHistory(5)` → recent point changes

---

### 3. LevelProgressCard

Shows current level, progress to next level, and milestone achievements.

**Features:**
- Current level with gradient badge
- Progress bar to next level (100 points per level)
- Milestone visualization (levels 5, 10, 15, 20, 25)
- Estimated reports needed to level up
- Next milestone indicator

**Usage:**
```tsx
import { LevelProgressCard } from '@/components/gamification';

<LevelProgressCard user={user} className="mb-4" />
```

**Props:**
- `user`: User object with `level` and `points` fields
- `className?`: Optional Tailwind classes

**Levels & Titles:**
- Level 5: Reporter 🌱
- Level 10: Guardian 🛡️
- Level 15: Sentinel ⚔️
- Level 20: Hero 🦸
- Level 25: Legend 👑

---

### 4. BadgeGrid

Displays earned badges and in-progress badges with unlock progress.

**Features:**
- Grid layout (2-3 columns responsive)
- Earned badges with full color
- Locked badges with grayscale + lock icon
- Progress bars for in-progress badges
- Category color coding
- Optional limit parameter

**Usage:**
```tsx
import { BadgeGrid } from '@/components/gamification';

// Show all badges
<BadgeGrid />

// Show first 6 badges
<BadgeGrid limit={6} />
```

**Data Source:** `useMyBadges()` → `earned`, `in_progress`

**Badge Categories:**
- `reporting` → Blue
- `verification` → Green
- `community` → Purple
- `streak` → Orange
- `special` → Yellow

---

## API Hooks Used

All components rely on hooks from `@/lib/api/hooks`:

```typescript
import {
  useMyReputation,      // Reputation summary
  useMyReputationHistory, // Point change history
  useMyBadges           // Badges with progress
} from '@/lib/api/hooks';
```

## Integration Example (ProfileScreen)

```tsx
import {
  StreakWidget,
  ReputationDashboard,
  LevelProgressCard,
  BadgeGrid
} from '@/components/gamification';

export function ProfileScreen() {
  const { user } = useAuth();

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold text-gray-800 px-1">Your Progress</h2>

      <StreakWidget />
      <ReputationDashboard />
      {user && <LevelProgressCard user={user} />}
      <BadgeGrid limit={6} />
    </div>
  );
}
```

## Design Patterns

- **Loading States:** Skeleton UI with pulsing animation
- **Error States:** Red-bordered card with error message
- **Colors:** Purple accent (`purple-500`, `purple-600`) for primary actions
- **Icons:** Lucide React icons throughout
- **Responsive:** Mobile-first with responsive grids
- **No `any` types:** Full TypeScript type safety

## Backend Requirements

These components require the following backend endpoints to be functional:

- `GET /api/gamification/me/reputation` - Reputation summary
- `GET /api/gamification/me/reputation/history` - Point change history
- `GET /api/gamification/me/badges` - Earned and in-progress badges

All endpoints require authentication via Bearer token.
