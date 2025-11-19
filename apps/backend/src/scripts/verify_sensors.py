from fastapi.testclient import TestClient
import sys
import os
import uuid
from datetime import datetime

# Add src to path so we can import main
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.main import app

client = TestClient(app)

def test_sensor_flow():
    # 1. Create Sensor
    print(f"Creating sensor...")
    response = client.post("/api/sensors/", json={
        "location_lat": 28.6139,
        "location_lng": 77.2090,
        "status": "active"
    })
    
    if response.status_code != 200:
        print(f"FAILED to create sensor: {response.text}")
        exit(1)
        
    sensor_data = response.json()
    sensor_id = sensor_data["id"]
    print(f"Sensor created with ID: {sensor_id}")
    
    # 2. List Sensors
    print(f"Listing sensors...")
    response = client.get("/api/sensors/")
    
    if response.status_code != 200:
        print(f"FAILED to list sensors: {response.text}")
        exit(1)
        
    sensors = response.json()
    found = any(s["id"] == sensor_id for s in sensors)
    if not found:
        print("FAILED: Created sensor not found in list")
        exit(1)
    print(f"Found {len(sensors)} sensors.")

    # 3. Record Reading
    print(f"Recording reading...")
    response = client.post(f"/api/sensors/{sensor_id}/readings", json={
        "sensor_id": sensor_id,
        "water_level": 5.5,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    if response.status_code != 200:
        print(f"FAILED to record reading: {response.text}")
        exit(1)
        
    print("Reading recorded.")
    
    # 4. Get Readings
    print(f"Fetching readings...")
    response = client.get(f"/api/sensors/{sensor_id}/readings")
    
    if response.status_code != 200:
        print(f"FAILED to get readings: {response.text}")
        exit(1)
        
    readings = response.json()
    if len(readings) == 0:
        print("FAILED: No readings found")
        exit(1)
        
    print(f"Found {len(readings)} readings.")
    print("✅ Sensor verification successful!")

if __name__ == "__main__":
    test_sensor_flow()
