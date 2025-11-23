import time, requests, random
URL = "http://localhost:5000/api/sensor-data"

def send_once(sensor_id, hotspot):
    payload = {
        "sensor_id": sensor_id,
        "hotspot_id": hotspot,
        "water_level": round(20 + 40*random.random(), 2),
        "soil_moisture": round(30 + 40*random.random(), 2)
    }
    try:
        r = requests.post(URL, json=payload, timeout=5)
        print(r.status_code, r.text)
    except Exception as e:
        print("Failed to send:", e)

if __name__ == "__main__":
    print("Starting simulator. Posting to", URL)
    while True:
        send_once("ESP_SIM_1", "KarolBagh")
        time.sleep(30)
