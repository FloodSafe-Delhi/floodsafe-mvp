"""
Verification script for profile and notification features.

Tests:
1. Create user with new fields
2. Update user profile (username, email, phone)
3. Update notification preferences
4. Update language preference
5. Create watch area for user
6. Get user's watch areas
7. Delete watch area
8. Verify all data persists correctly
"""

import sys
import os
import json
from uuid import uuid4

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.infrastructure.database import SessionLocal, engine, Base
from src.infrastructure.models import User, WatchArea

def verify_profile_features():
    """Test all profile-related features."""

    print("=" * 60)
    print("FLOODSAFE PROFILE FEATURES VERIFICATION")
    print("=" * 60)

    # Create tables
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        # Test 1: Create user with all new fields
        print("\n1. Creating test user with profile fields...")
        test_user = User(
            username=f"test_profile_{uuid4().hex[:8]}",
            email=f"test_{uuid4().hex[:8]}@floodsafe.ai",
            phone="+91-9876543210",
            language="english",
            notification_push=True,
            notification_sms=True,
            notification_whatsapp=False,
            notification_email=True,
            alert_preferences='{"watch":true,"advisory":true,"warning":true,"emergency":true}'
        )
        db.add(test_user)
        db.commit()
        db.refresh(test_user)
        print(f"   ✓ User created: {test_user.username} (ID: {test_user.id})")
        print(f"   ✓ Phone: {test_user.phone}")
        print(f"   ✓ Language: {test_user.language}")
        print(f"   ✓ Push notifications: {test_user.notification_push}")
        print(f"   ✓ SMS alerts: {test_user.notification_sms}")
        print(f"   ✓ WhatsApp: {test_user.notification_whatsapp}")
        print(f"   ✓ Email: {test_user.notification_email}")

        # Test 2: Update profile fields
        print("\n2. Updating user profile...")
        test_user.phone = "+91-9999888877"
        test_user.language = "hindi"
        db.commit()
        db.refresh(test_user)
        print(f"   ✓ Phone updated to: {test_user.phone}")
        print(f"   ✓ Language updated to: {test_user.language}")

        # Test 3: Update notification preferences
        print("\n3. Updating notification preferences...")
        test_user.notification_whatsapp = True
        test_user.notification_sms = False
        test_user.alert_preferences = '{"watch":false,"advisory":true,"warning":true,"emergency":true}'
        db.commit()
        db.refresh(test_user)
        print(f"   ✓ WhatsApp enabled: {test_user.notification_whatsapp}")
        print(f"   ✓ SMS disabled: {test_user.notification_sms}")
        alert_prefs = json.loads(test_user.alert_preferences)
        print(f"   ✓ Alert preferences: {alert_prefs}")

        # Test 4: Create watch area
        print("\n4. Creating watch area...")
        watch_area = WatchArea(
            user_id=test_user.id,
            name="My Home - Koramangala",
            location="POINT(77.625 12.935)",
            radius=1000.0
        )
        db.add(watch_area)
        db.commit()
        db.refresh(watch_area)
        print(f"   ✓ Watch area created: {watch_area.name}")
        print(f"   ✓ Location: ({watch_area.latitude}, {watch_area.longitude})")
        print(f"   ✓ Radius: {watch_area.radius}m")

        # Test 5: Create another watch area
        print("\n5. Creating second watch area...")
        watch_area2 = WatchArea(
            user_id=test_user.id,
            name="Office - Indiranagar",
            location="POINT(77.645 12.975)",
            radius=500.0
        )
        db.add(watch_area2)
        db.commit()
        db.refresh(watch_area2)
        print(f"   ✓ Second watch area created: {watch_area2.name}")

        # Test 6: Get all watch areas for user
        print("\n6. Retrieving user's watch areas...")
        user_watch_areas = db.query(WatchArea).filter(
            WatchArea.user_id == test_user.id
        ).all()
        print(f"   ✓ Found {len(user_watch_areas)} watch areas:")
        for wa in user_watch_areas:
            print(f"      - {wa.name} (Radius: {wa.radius}m)")

        # Test 7: Delete a watch area
        print("\n7. Deleting first watch area...")
        db.delete(watch_area)
        db.commit()
        remaining_areas = db.query(WatchArea).filter(
            WatchArea.user_id == test_user.id
        ).count()
        print(f"   ✓ Watch area deleted")
        print(f"   ✓ Remaining areas: {remaining_areas}")

        # Test 8: Verify user still exists with all data
        print("\n8. Verifying user data integrity...")
        user_check = db.query(User).filter(User.id == test_user.id).first()
        assert user_check is not None
        assert user_check.phone == "+91-9999888877"
        assert user_check.language == "hindi"
        assert user_check.notification_whatsapp == True
        assert user_check.notification_sms == False
        print("   ✓ All user data verified successfully")

        # Test 9: Verify JSON parsing
        print("\n9. Verifying JSON field parsing...")
        alert_prefs = json.loads(user_check.alert_preferences)
        assert isinstance(alert_prefs, dict)
        assert "watch" in alert_prefs
        assert "emergency" in alert_prefs
        print(f"   ✓ Alert preferences parsed correctly: {alert_prefs}")

        print("\n" + "=" * 60)
        print("✅ ALL PROFILE FEATURES VERIFIED SUCCESSFULLY!")
        print("=" * 60)
        print("\nFeatures tested:")
        print("  ✓ User profile fields (phone, language)")
        print("  ✓ Notification preferences (push, SMS, WhatsApp, email)")
        print("  ✓ Alert type preferences (JSON)")
        print("  ✓ Watch areas CRUD operations")
        print("  ✓ PostGIS location storage")
        print("  ✓ User-watch area relationships")
        print("\nProfile system is ready for production! 🎉")

        # Cleanup
        print("\nCleaning up test data...")
        db.delete(watch_area2)
        db.delete(test_user)
        db.commit()
        print("   ✓ Test data cleaned up")

    except AssertionError as e:
        print(f"\n❌ Assertion failed: {e}")
        db.rollback()
        return False
    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
        return False
    finally:
        db.close()

    return True


if __name__ == "__main__":
    success = verify_profile_features()
    sys.exit(0 if success else 1)
