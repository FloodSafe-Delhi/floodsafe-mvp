from fastapi.testclient import TestClient
import sys
import os
import uuid

# Add src to path so we can import main
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.main import app

client = TestClient(app)

def test_user_flow():
    # 1. Create User
    username = f"testuser_{uuid.uuid4().hex[:8]}"
    email = f"{username}@example.com"
    
    print(f"Creating user: {username}...")
    response = client.post("/api/users/", json={
        "username": username,
        "email": email,
        "role": "user"
    })
    
    if response.status_code != 200:
        print(f"FAILED to create user: {response.text}")
        exit(1)
        
    user_data = response.json()
    user_id = user_data["id"]
    print(f"User created with ID: {user_id}")
    
    # 2. Get User
    print(f"Fetching user profile...")
    response = client.get(f"/api/users/{user_id}")
    
    if response.status_code != 200:
        print(f"FAILED to get user: {response.text}")
        exit(1)
        
    fetched_data = response.json()
    assert fetched_data["username"] == username
    assert fetched_data["email"] == email
    print("✅ User verification successful!")

if __name__ == "__main__":
    test_user_flow()
