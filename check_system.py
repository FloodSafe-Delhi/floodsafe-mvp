import urllib.request
import json

BASE_URL = "http://localhost:8000/api"

def get_json(url):
    try:
        with urllib.request.urlopen(url) as response:
            if response.status == 200:
                return json.loads(response.read().decode())
    except Exception as e:
        print(f"Error fetching {url}: {e}")
    return None

def check_system():
    print("[INFO] Checking System State...\n")

    # 1. Check Users
    users = get_json(f"{BASE_URL}/users/leaderboard/top")
    if users is not None:
        print(f"[OK] Users System Working: Found {len(users)} users in leaderboard.")
        for u in users[:3]:
            print(f"   - {u['username']} (Points: {u['points']})")
    else:
        print("[FAIL] Users System Failed")

    # 2. Check Sensors
    sensors = get_json(f"{BASE_URL}/sensors/")
    if sensors is not None:
        print(f"\n[OK] Sensor System Working: Found {len(sensors)} sensors.")
        for s in sensors[:3]:
            print(f"   - Sensor {s['id'][:8]}... (Status: {s['status']})")
    else:
        print("\n[FAIL] Sensor System Failed")

    # 3. Check Reports
    reports = get_json(f"{BASE_URL}/reports/")
    if reports is not None:
        print(f"\n[OK] Report System Working: Found {len(reports)} reports.")
        for r in reports[:3]:
            print(f"   - Report: {r['description']} (Verified: {r['verified']})")
    else:
        print("\n[FAIL] Report System Failed")

if __name__ == "__main__":
    check_system()
