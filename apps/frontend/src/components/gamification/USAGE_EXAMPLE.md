# Gamification Components Usage Guide

This folder contains 7 gamification components for the FloodSafe platform. Here's how to use them:

## Components Overview

### 1. **StreakWidget** - Daily Report Streak Tracker
Shows user's consecutive days with verified reports, milestone badges, and last 7 days activity.

```tsx
import { StreakWidget } from '@/components/gamification';

<StreakWidget className="mb-4" />
```

### 2. **ReputationDashboard** - Detailed Reputation View
Displays reputation score (0-100), accuracy rate, recent activity history, and explainer.

```tsx
import { ReputationDashboard } from '@/components/gamification';

<ReputationDashboard className="mb-4" />
```

### 3. **LevelProgressCard** - Level and XP Progress
Shows current level, points progress bar, milestone visualization, and next level estimate.

```tsx
import { LevelProgressCard } from '@/components/gamification';
import { User } from '@/types';

const user: User = { /* ... */ };

<LevelProgressCard user={user} className="mb-4" />
```

### 4. **BadgeGrid** - Earned and In-Progress Badges
Grid of badges with earned/locked states, progress bars for locked badges, and category colors.

```tsx
import { BadgeGrid } from '@/components/gamification';

// Show all badges
<BadgeGrid />

// Show limited number (e.g., 6 for preview)
<BadgeGrid limit={6} />
```

### 5. **BadgeCatalogModal** - Full Badge Catalog with Search
Modal showing all available badges, searchable, filterable by category, with progress tracking.

```tsx
import { useState } from 'react';
import { BadgeCatalogModal } from '@/components/gamification';

function MyComponent() {
  const [catalogOpen, setCatalogOpen] = useState(false);

  return (
    <>
      <button onClick={() => setCatalogOpen(true)}>View All Badges</button>
      <BadgeCatalogModal open={catalogOpen} onOpenChange={setCatalogOpen} />
    </>
  );
}
```

### 6. **LeaderboardSection** - Collapsible Top 10 Leaderboard
Collapsible accordion showing top 10 users with Global/Weekly/Monthly tabs.

```tsx
import { useState } from 'react';
import { LeaderboardSection } from '@/components/gamification';

function MyComponent() {
  const [fullLeaderboardOpen, setFullLeaderboardOpen] = useState(false);
  const userId = "current-user-id";

  return (
    <LeaderboardSection
      userId={userId}
      onViewFull={() => setFullLeaderboardOpen(true)}
      className="mb-4"
    />
  );
}
```

### 7. **LeaderboardModal** - Full Top 100 Leaderboard
Modal with scrollable table showing top 100 users, tabs for different time periods.

```tsx
import { useState } from 'react';
import { LeaderboardModal } from '@/components/gamification';

function MyComponent() {
  const [modalOpen, setModalOpen] = useState(false);
  const userId = "current-user-id";

  return (
    <>
      <button onClick={() => setModalOpen(true)}>View Leaderboard</button>
      <LeaderboardModal
        open={modalOpen}
        onOpenChange={setModalOpen}
        userId={userId}
      />
    </>
  );
}
```

## Complete ProfileScreen Example

Here's how to integrate all components in a Profile screen:

```tsx
import { useState } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import {
  StreakWidget,
  ReputationDashboard,
  LevelProgressCard,
  BadgeGrid,
  BadgeCatalogModal,
  LeaderboardSection,
  LeaderboardModal
} from '@/components/gamification';

export function ProfileScreen() {
  const { user } = useAuth();
  const [badgeCatalogOpen, setBadgeCatalogOpen] = useState(false);
  const [leaderboardOpen, setLeaderboardOpen] = useState(false);

  if (!user) return null;

  return (
    <div className="p-4 space-y-4">
      {/* User Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold">{user.username}</h1>
        <p className="text-gray-600">Level {user.level} Reporter</p>
      </div>

      {/* Gamification Widgets */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Streak */}
        <StreakWidget />

        {/* Reputation */}
        <ReputationDashboard />

        {/* Level Progress */}
        <LevelProgressCard user={user} className="md:col-span-2" />
      </div>

      {/* Badge Grid with "View All" Button */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold">Badges</h2>
          <button
            onClick={() => setBadgeCatalogOpen(true)}
            className="text-sm text-purple-600 hover:text-purple-700"
          >
            View All →
          </button>
        </div>
        <BadgeGrid limit={6} />
      </div>

      {/* Leaderboard Section */}
      <LeaderboardSection
        userId={user.id}
        onViewFull={() => setLeaderboardOpen(true)}
      />

      {/* Modals */}
      <BadgeCatalogModal
        open={badgeCatalogOpen}
        onOpenChange={setBadgeCatalogOpen}
      />

      <LeaderboardModal
        open={leaderboardOpen}
        onOpenChange={setLeaderboardOpen}
        userId={user.id}
      />
    </div>
  );
}
```

## API Hooks Used

All components use TanStack Query hooks from `@/lib/api/hooks`:

- `useMyBadges()` - Fetch current user's earned and in-progress badges
- `useBadgesCatalog()` - Fetch all available badges (public endpoint)
- `useMyReputation()` - Fetch current user's reputation summary
- `useMyReputationHistory()` - Fetch reputation point change history
- `useLeaderboard(type, limit, userId)` - Fetch leaderboard with optional user rank

## Styling

All components use:
- Tailwind CSS for styling
- Radix UI primitives for accessibility
- Lucide React icons
- Purple accent color (`bg-purple-500`, `text-purple-600`)
- Consistent card layouts with shadows and borders

## Loading and Error States

All components handle:
- Loading states with skeleton loaders
- Error states with red error cards
- Empty states with helpful messages

## Mobile Responsive

All components are mobile-responsive:
- Grid columns adapt to screen size
- Modals are scrollable and fit mobile viewports
- Text sizes and spacing adjust for small screens
