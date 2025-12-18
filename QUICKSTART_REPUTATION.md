# Reputation System - Quick Start Guide

## 🚀 Setup (3 Simple Steps)

### Step 1: Run Migration
```bash
cd /home/user/floodsafe-mvp/apps/backend
docker compose exec backend python src/scripts/migrate_reputation_system.py
```

### Step 2: Seed Badges
```bash
docker compose exec backend python src/scripts/seed_badges.py
```

### Step 3: Restart Backend
```bash
docker compose restart backend
```

## ✅ Verify Installation

Visit: **http://localhost:8000/docs**

Look for these new API sections:
- 🏆 **reputation** (3 endpoints)
- 📊 **leaderboards** (2 endpoints)
- 🏅 **badges** (3 endpoints)

## 🎮 Quick Test

```bash
# 1. Get reputation summary for admin user
curl http://localhost:8000/api/reputation/admin

# 2. View global leaderboard
curl http://localhost:8000/api/leaderboards/?type=global&limit=10

# 3. Get all badges
curl http://localhost:8000/api/badges/

# 4. Update privacy (make anonymous)
curl -X PATCH http://localhost:8000/api/reputation/admin/privacy \
  -H "Content-Type: application/json" \
  -d '{"display_name":"FloodGuardian"}'
```

## 📱 What's New in API

### Report Verification Now Includes Quality Scoring
```bash
# Old way:
POST /api/reports/{id}/verify

# New way (with quality):
POST /api/reports/{id}/verify
{
  "verified": true,
  "quality_score": 85.0
}
```

### New Vote Endpoints
```bash
POST /api/reports/{id}/upvote
POST /api/reports/{id}/downvote
```

## 🎯 Key Features

- ✅ **Quality Scoring** - Reports scored 0-100
- ✅ **12 Badges** - Auto-awarded for achievements
- ✅ **Streak System** - Daily bonus rewards
- ✅ **Privacy Controls** - Leaderboard opt-in/anonymous mode
- ✅ **3 Leaderboards** - Global, weekly, monthly
- ✅ **Audit Trail** - Complete reputation history

## 📊 Database Changes

### New Tables
- `reputation_history` - Tracks all rep changes
- `badges` - Badge definitions
- `user_badges` - User's earned badges

### Enhanced Tables
- `users` - Added 6 new fields (reputation_score, streak_days, etc.)
- `reports` - Added 3 new fields (quality_score, downvotes, verified_at)

## 🔧 Maintenance

### Daily Task (Auto-resets broken streaks)
```bash
# Manual run:
python src/tasks/streak_updater.py

# Or setup cron:
5 0 * * * cd /path/to/backend && python src/tasks/streak_updater.py
```

## 📖 Full Documentation

See `REPUTATION_SYSTEM.md` for:
- Complete API documentation
- Privacy system details
- Badge system explained
- Troubleshooting guide
- Frontend integration examples

---

**Quick Links**:
- API Docs: http://localhost:8000/docs
- Backend: http://localhost:8000
- Frontend: http://localhost:5175

**Status**: ✅ Ready to Use!
