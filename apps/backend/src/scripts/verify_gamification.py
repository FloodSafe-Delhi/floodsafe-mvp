from fastapi.testclient import TestClient
import sys
import os
import uuid

# Add src to path so we can import main
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.main import app

client = TestClient(app)

def test_gamification_flow():
    # 1. Create User
    username = f"gamer_{uuid.uuid4().hex[:8]}"
    print(f"Creating user: {username}...")
    response = client.post("/api/users/", json={
        "username": username,
        "email": f"{username}@example.com",
        "role": "user"
    })
    
    if response.status_code != 200:
        print(f"FAILED to create user: {response.text}")
        exit(1)
        
    user_id = response.json()["id"]
    initial_points = response.json()["points"]
    print(f"User created. Points: {initial_points}")
    
    # 2. Create Report
    print("Creating report...")
    response = client.post("/api/reports/", data={
        "user_id": user_id,
        "description": "Test flood report",
        "latitude": 28.6139,
        "longitude": 77.2090
    })
    
    if response.status_code != 200:
        print(f"FAILED to create report: {response.text}")
        exit(1)
        
    report_id = response.json()["id"]
    print(f"Report created: {report_id}")
    
    # 3. Verify Report
    print("Verifying report...")
    response = client.post(f"/api/reports/{report_id}/verify")
    
    if response.status_code != 200:
        print(f"FAILED to verify report: {response.text}")
        exit(1)
        
    verified_data = response.json()
    if not verified_data["verified"]:
        print("FAILED: Report not marked as verified")
        exit(1)
    print("Report verified.")
    
    # 4. Check User Points
    print("Checking user points...")
    response = client.get(f"/api/users/{user_id}")
    user_data = response.json()
    
    new_points = user_data["points"]
    print(f"New points: {new_points}")
    
    if new_points != initial_points + 10:
        print(f"FAILED: Points did not increase correctly. Expected {initial_points + 10}, got {new_points}")
        exit(1)
        
    # 5. Check Leaderboard
    print("Checking leaderboard...")
    response = client.get("/api/users/leaderboard/top")
    
    if response.status_code != 200:
        print(f"FAILED to get leaderboard: {response.text}")
        exit(1)
        
    leaderboard = response.json()
    found = any(u["username"] == username for u in leaderboard)
    
    if not found:
        print("FAILED: User not found in leaderboard")
        exit(1)
        
    print("✅ Gamification verification successful!")

if __name__ == "__main__":
    test_gamification_flow()
