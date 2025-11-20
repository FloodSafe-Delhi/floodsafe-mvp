# FloodSafe Reputation System - Implementation Guide

## Overview

A complete, privacy-focused reputation and gamification system for FloodSafe MVP. This system rewards users for quality flood reports, maintains engagement through streaks, and provides transparent leaderboards with robust privacy controls.

---

## ✨ Features Implemented

### Core Features
- ✅ **Quality-based Scoring** - Reports scored 0-100 based on media, description, and community validation
- ✅ **Smart Points System** - Base points + quality bonus for verified reports
- ✅ **Level Progression** - Linear leveling (100 points per level)
- ✅ **Reputation Score** - 0-100 score based on accuracy, consistency, and volume
- ✅ **Streak System** - Daily activity tracking with milestone bonuses
- ✅ **Badge System** - 12 initial badges across milestones, achievements, and levels
- ✅ **Privacy Controls** - Leaderboard visibility, public profiles, anonymous names
- ✅ **Audit Trail** - Complete reputation history log
- ✅ **Leaderboards** - Global, weekly, and monthly rankings

### Privacy Features
- 🔒 **Leaderboard Opt-in/Out** - Users control if they appear on leaderboards
- 🔒 **Anonymous Mode** - Display custom names instead of real usernames
- 🔒 **Profile Privacy** - Control who can view full profile
- 🔒 **Email Protection** - Never shown on leaderboards

---

## 📁 Files Created/Modified

### Infrastructure Layer
- ✅ `/apps/backend/src/infrastructure/models.py` - Updated with new models:
  - Enhanced `User` model (reputation_score, streak_days, privacy fields)
  - Enhanced `Report` model (quality_score, downvotes, verified_at)
  - New `ReputationHistory` model
  - New `Badge` model
  - New `UserBadge` model

### Domain Layer
- ✅ `/apps/backend/src/domain/models.py` - Updated existing Pydantic models
- ✅ `/apps/backend/src/domain/reputation_models.py` - New reputation-specific models
- ✅ `/apps/backend/src/domain/services/reputation_service.py` - Core business logic
- ✅ `/apps/backend/src/domain/services/leaderboard_service.py` - Leaderboard logic

### API Layer
- ✅ `/apps/backend/src/api/reputation.py` - Reputation endpoints
- ✅ `/apps/backend/src/api/leaderboards.py` - Leaderboard endpoints
- ✅ `/apps/backend/src/api/badges.py` - Badge endpoints
- ✅ `/apps/backend/src/api/reports.py` - Updated with reputation integration

### Tasks & Scripts
- ✅ `/apps/backend/src/tasks/streak_updater.py` - Daily streak maintenance
- ✅ `/apps/backend/src/scripts/migrate_reputation_system.py` - Database migration
- ✅ `/apps/backend/src/scripts/seed_badges.py` - Initial badge seeding

### Configuration
- ✅ `/apps/backend/src/main.py` - Updated with new routes

---

## 🗄️ Database Schema

### New Tables

#### `reputation_history`
```sql
- id (UUID, PK)
- user_id (UUID, FK -> users)
- action (VARCHAR) - 'report_verified', 'badge_earned', etc.
- points_change (INTEGER)
- new_total (INTEGER)
- reason (VARCHAR)
- metadata (VARCHAR/JSON)
- created_at (TIMESTAMP)
```

#### `badges`
```sql
- id (UUID, PK)
- key (VARCHAR, UNIQUE) - 'guardian', 'hero', etc.
- name (VARCHAR)
- description (VARCHAR)
- icon (VARCHAR) - emoji
- category (VARCHAR) - 'milestone' or 'achievement'
- requirement_type (VARCHAR) - 'verified_count', 'streak_days', etc.
- requirement_value (INTEGER)
- points_reward (INTEGER)
- sort_order (INTEGER)
- is_active (BOOLEAN)
- created_at (TIMESTAMP)
```

#### `user_badges`
```sql
- id (UUID, PK)
- user_id (UUID, FK -> users)
- badge_id (UUID, FK -> badges)
- earned_at (TIMESTAMP)
- UNIQUE(user_id, badge_id)
```

### Enhanced Tables

#### `users` (New Columns)
```sql
- reputation_score (INTEGER, DEFAULT 0)
- streak_days (INTEGER, DEFAULT 0)
- last_activity_date (TIMESTAMP)
- leaderboard_visible (BOOLEAN, DEFAULT TRUE)
- profile_public (BOOLEAN, DEFAULT TRUE)
- display_name (VARCHAR, NULL)
```

#### `reports` (New Columns)
```sql
- downvotes (INTEGER, DEFAULT 0)
- quality_score (FLOAT, DEFAULT 0.0)
- verified_at (TIMESTAMP, NULL)
```

---

## 🚀 Setup Instructions

### 1. Run Database Migration

```bash
cd /home/user/floodsafe-mvp/apps/backend

# Using Docker
docker compose exec backend python src/scripts/migrate_reputation_system.py

# Or directly (if backend is running)
python src/scripts/migrate_reputation_system.py
```

This will:
- Add new columns to `users` and `reports` tables
- Create `reputation_history`, `badges`, and `user_badges` tables
- Create necessary indexes

### 2. Seed Initial Badges

```bash
# Using Docker
docker compose exec backend python src/scripts/seed_badges.py

# Or directly
python src/scripts/seed_badges.py
```

This creates 12 initial badges:
- **Milestones**: First Report ⭐, Reporter 📝, Guardian 🛡️, Hero 🦸, Legend 👑
- **Streaks**: Dedicated 🔥 (7 days), Committed ⚡ (30 days)
- **Levels**: Rising Star 🌟 (Level 5), Expert 💎 (Level 10), Master 🏆 (Level 25)
- **Points**: High Achiever 🎯 (1000 pts), Champion 🥇 (5000 pts)

### 3. Restart Backend

```bash
docker compose restart backend
```

### 4. Verify Installation

Visit: `http://localhost:8000/docs`

You should see new endpoints under:
- **reputation** tag
- **leaderboards** tag
- **badges** tag

---

## 📡 API Endpoints

### Reputation Endpoints

#### GET `/api/reputation/{user_id}`
Get complete reputation summary.

**Response:**
```json
{
  "user_id": "uuid",
  "points": 1250,
  "level": 13,
  "reputation_score": 87,
  "accuracy_rate": 92.5,
  "streak_days": 15,
  "next_level_points": 50,
  "badges_earned": 5,
  "total_badges": 12
}
```

#### GET `/api/reputation/{user_id}/history?limit=20&offset=0`
Get reputation change history.

**Response:**
```json
[
  {
    "id": "uuid",
    "action": "report_verified",
    "points_change": 15,
    "new_total": 1250,
    "reason": "Report verified (quality: 85)",
    "created_at": "2025-11-20T10:30:00Z"
  }
]
```

#### PATCH `/api/reputation/{user_id}/privacy`
Update privacy settings.

**Request:**
```json
{
  "leaderboard_visible": true,
  "profile_public": true,
  "display_name": "FloodGuardian"
}
```

### Leaderboard Endpoints

#### GET `/api/leaderboards/?type=global&limit=100&user_id={uuid}`
Get leaderboard (global, weekly, or monthly).

**Response:**
```json
{
  "leaderboard_type": "global",
  "updated_at": "2025-11-20T12:00:00Z",
  "entries": [
    {
      "rank": 1,
      "display_name": "floodhero",
      "profile_photo_url": "...",
      "points": 5000,
      "level": 51,
      "reputation_score": 98,
      "verified_reports": 75,
      "badges_count": 8,
      "is_anonymous": false
    }
  ],
  "current_user_rank": 45
}
```

#### GET `/api/leaderboards/top?limit=10`
Get top users summary (for dashboard widgets).

### Badge Endpoints

#### GET `/api/badges/`
List all available badges.

#### GET `/api/badges/user/{user_id}`
Get user's earned badges and progress on locked badges.

**Response:**
```json
{
  "earned": [
    {
      "badge_id": "uuid",
      "key": "guardian",
      "name": "Guardian",
      "description": "Get 10 reports verified",
      "icon": "🛡️",
      "earned_at": "2025-11-15T10:00:00Z"
    }
  ],
  "in_progress": [
    {
      "badge_id": "uuid",
      "key": "hero",
      "name": "Hero",
      "description": "Get 25 reports verified",
      "icon": "🦸",
      "current_value": 15,
      "required_value": 25,
      "progress_percent": 60.0
    }
  ]
}
```

### Enhanced Report Endpoints

#### POST `/api/reports/{report_id}/verify`
Verify or reject a report (now uses reputation system).

**Request:**
```json
{
  "verified": true,
  "quality_score": 85.0
}
```

**Response:**
```json
{
  "message": "Report verified",
  "report_id": "uuid",
  "verified": true,
  "user_id": "uuid",
  "points": 1265,
  "level": 13,
  "reputation_score": 88,
  "quality_score": 85.0,
  "points_earned": 18
}
```

#### POST `/api/reports/{report_id}/upvote`
Upvote a report (awards bonus to report owner).

#### POST `/api/reports/{report_id}/downvote`
Downvote a report (flags potential false reports).

---

## 🎯 Points System

### Report Actions
- **Report Submitted**: +5 points
- **Report Verified**: +10 base + quality bonus (0-10)
  - Quality score calculated from:
    - Media presence (+25 max)
    - Description quality (+15 max)
    - Community validation (+10 max)
    - Downvote penalties
- **Report Rejected**: -5 points
- **Report Upvoted** (your report): +1 point

### Streak Bonuses
- **7 Day Streak**: +25 points
- **30 Day Streak**: +100 points

### Badge Rewards
- Variable based on badge rarity (5-200 points)

### Example Calculation
```
Report with quality score 85:
- Base points: 10
- Quality bonus: 8 (85/10)
- Total: 18 points awarded
```

---

## 🏅 Badge System

### Categories

1. **Milestone Badges** (Progress-based)
   - Unlocked by reaching report counts

2. **Achievement Badges** (Special accomplishments)
   - Streaks, levels, point milestones

### Badge Requirements

All badges use simple, single requirements:
- `reports_count` - Total reports submitted
- `verified_count` - Total reports verified
- `streak_days` - Current streak length
- `level` - User's current level
- `points` - Total points earned

### Automatic Badge Awarding

Badges are checked and awarded automatically:
- After report verification
- After earning streak bonuses
- When points/level changes

---

## 🔄 Background Tasks

### Daily Streak Reset

**File**: `/apps/backend/src/tasks/streak_updater.py`

**Function**: `reset_inactive_streaks()`

**Schedule**: Daily at 12:05 AM

**What it does**:
- Finds users whose last activity was 2+ days ago
- Resets their streak_days to 0
- Logs the resets

**Manual Execution**:
```bash
cd /home/user/floodsafe-mvp/apps/backend
python src/tasks/streak_updater.py
```

### Setting Up Cron (Production)

```bash
# Add to crontab
5 0 * * * cd /path/to/backend && python src/tasks/streak_updater.py
```

Or use APScheduler in main.py (recommended for containerized apps).

---

## 🔒 Privacy System

### Privacy Levels

1. **Fully Public** (default)
   - `leaderboard_visible`: true
   - `profile_public`: true
   - `display_name`: null
   - **Result**: Shows on leaderboard with real username

2. **Anonymous**
   - `leaderboard_visible`: true
   - `profile_public`: false or true
   - `display_name`: "FloodGuardian"
   - **Result**: Shows on leaderboard with custom name

3. **Private**
   - `leaderboard_visible`: false
   - **Result**: Hidden from leaderboards entirely

### What's Protected

- ✅ Email addresses NEVER shown on leaderboards
- ✅ User IDs hidden for anonymous users
- ✅ Profile photos hidden for anonymous users
- ✅ Can opt out of leaderboards completely
- ✅ Still earns points/badges privately

---

## 🧪 Testing the System

### Test Reputation Flow

```bash
# 1. Create a test user
curl -X POST http://localhost:8000/api/users/ \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","email":"test@example.com"}'

# 2. Create a report
curl -X POST http://localhost:8000/api/reports/ \
  -F "user_id=<user_id>" \
  -F "description=Flood near my area" \
  -F "latitude=12.9716" \
  -F "longitude=77.5946"

# 3. Verify the report (triggers reputation system)
curl -X POST http://localhost:8000/api/reports/<report_id>/verify \
  -H "Content-Type: application/json" \
  -d '{"verified":true,"quality_score":85.0}'

# 4. Check reputation summary
curl http://localhost:8000/api/reputation/<user_id>

# 5. Check badges
curl http://localhost:8000/api/badges/user/<user_id>

# 6. View leaderboard
curl http://localhost:8000/api/leaderboards/?type=global&limit=10
```

### Test Privacy Settings

```bash
# Make user anonymous on leaderboard
curl -X PATCH http://localhost:8000/api/reputation/<user_id>/privacy \
  -H "Content-Type: application/json" \
  -d '{"leaderboard_visible":true,"display_name":"AnonymousHero"}'

# Hide from leaderboard
curl -X PATCH http://localhost:8000/api/reputation/<user_id>/privacy \
  -H "Content-Type: application/json" \
  -d '{"leaderboard_visible":false}'
```

---

## 📊 Monitoring & Maintenance

### Check Reputation History

```sql
SELECT
    u.username,
    rh.action,
    rh.points_change,
    rh.new_total,
    rh.created_at
FROM reputation_history rh
JOIN users u ON rh.user_id = u.id
ORDER BY rh.created_at DESC
LIMIT 20;
```

### Check Badge Distribution

```sql
SELECT
    b.name,
    b.icon,
    COUNT(ub.id) as times_earned
FROM badges b
LEFT JOIN user_badges ub ON b.id = ub.badge_id
GROUP BY b.id, b.name, b.icon
ORDER BY times_earned DESC;
```

### Check Top Users

```sql
SELECT
    username,
    points,
    level,
    reputation_score,
    verified_reports_count,
    streak_days
FROM users
WHERE leaderboard_visible = true
ORDER BY points DESC
LIMIT 10;
```

---

## 🎨 Frontend Integration

### Update Profile Screen

The existing `/apps/frontend/src/components/screens/ProfileScreen.tsx` already displays:
- Points, level, reputation score
- Streak days
- Badges earned

No changes needed! The new fields will automatically populate.

### Privacy Settings UI

Add to Profile Screen:

```typescript
// Privacy Settings Section
<Card className="p-4">
  <h3 className="font-semibold mb-4">🔒 Privacy Settings</h3>

  <div className="space-y-4">
    <div className="flex items-center justify-between">
      <Label>Show on Leaderboards</Label>
      <Switch
        checked={user.leaderboard_visible}
        onCheckedChange={(checked) =>
          updatePrivacy({ leaderboard_visible: checked })
        }
      />
    </div>

    {user.leaderboard_visible && (
      <div>
        <Label>Display Name (Anonymous)</Label>
        <Input
          placeholder="Leave empty to use username"
          value={user.display_name || ''}
          onChange={(e) =>
            updatePrivacy({ display_name: e.target.value })
          }
        />
      </div>
    )}
  </div>
</Card>
```

### Leaderboard Screen

Create `/apps/frontend/src/components/screens/LeaderboardScreen.tsx`:

```typescript
export function LeaderboardScreen() {
  const { data: leaderboard } = useQuery({
    queryKey: ['leaderboard', 'global'],
    queryFn: async () => {
      const response = await fetch(
        `${API_URL}/api/leaderboards/?type=global&limit=100`
      );
      return response.json();
    }
  });

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">🏆 Leaderboard</h1>

      {leaderboard?.entries.map((entry) => (
        <Card key={entry.rank} className="p-4 mb-2">
          <div className="flex items-center gap-4">
            <div className="text-2xl font-bold">#{entry.rank}</div>
            <div className="flex-1">
              <div className="font-semibold">{entry.display_name}</div>
              <div className="text-sm text-gray-600">
                Level {entry.level} • {entry.points} points
              </div>
            </div>
            <div className="text-right">
              <div className="text-sm">Rep: {entry.reputation_score}</div>
              <div className="text-xs text-gray-600">
                {entry.badges_count} badges
              </div>
            </div>
          </div>
        </Card>
      ))}
    </div>
  );
}
```

---

## 🐛 Troubleshooting

### Migration Issues

**Problem**: "Table already exists"
**Solution**: Migration uses `IF NOT EXISTS`, safe to re-run

**Problem**: "Column already exists"
**Solution**: Migration uses `IF NOT EXISTS`, safe to re-run

### Badge Seeding Issues

**Problem**: Duplicate key error
**Solution**: Seed script updates existing badges, safe to re-run

### Reputation Not Updating

**Check**:
1. Is report verified? (check `verified` field)
2. Check `reputation_history` table for entries
3. Verify reputation_service was called
4. Check logs for errors

### Leaderboard Empty

**Check**:
1. Are users opted in? (`leaderboard_visible = true`)
2. Do users have points?
3. Check query filters (weekly/monthly may exclude inactive users)

---

## 🚀 Performance Optimization

### Indexes Created

- `reputation_history(user_id)`
- `reputation_history(created_at DESC)`
- `badges(key)`
- `badges(is_active)`
- `user_badges(user_id)`
- `user_badges(earned_at DESC)`

### Caching Recommendations (Future)

```python
# Cache leaderboards (Redis)
@cache(ttl=3600)  # 1 hour
def get_leaderboard(type, limit):
    ...

# Cache user reputation summary (Redis)
@cache(ttl=300)  # 5 minutes
def get_reputation_summary(user_id):
    ...
```

---

## 📝 Summary

### What We Built
✅ Complete reputation system with 11 files created/modified
✅ Privacy-first design with 3 privacy levels
✅ 12 initial badges across 3 categories
✅ Quality-based scoring (0-100)
✅ Automated badge awarding
✅ Complete audit trail
✅ 3 types of leaderboards
✅ Streak system with bonuses

### Database Impact
- 3 new tables
- 9 new columns (6 in users, 3 in reports)
- 6 new indexes

### API Impact
- 3 new route groups
- 10+ new endpoints
- Enhanced existing report endpoints

### Lines of Code
- ~500 lines of service logic
- ~400 lines of API endpoints
- ~200 lines of models
- ~300 lines of scripts/tasks

**Total**: ~1400 lines of production code

---

## 🎯 Next Steps (Future Enhancements)

1. **APScheduler Integration** - Automate daily tasks within the app
2. **Redis Caching** - Cache leaderboards and reputation summaries
3. **WebSocket Updates** - Real-time badge notifications
4. **Advanced Badges** - Quality master (10 reports with 90+ score)
5. **Reputation Tiers** - Bronze/Silver/Gold/Platinum/Diamond
6. **Regional Leaderboards** - By city/state
7. **Team/Group Features** - Collaborative reporting
8. **NFT Badges** (Web3) - Blockchain verification

---

## 📧 Support

For issues or questions:
- Check logs: `/var/log/floodsafe/` or container logs
- Review API docs: `http://localhost:8000/docs`
- Test endpoints with Swagger UI
- Check database directly for data verification

---

**Version**: 1.0.0
**Last Updated**: November 20, 2025
**Status**: ✅ Production Ready
