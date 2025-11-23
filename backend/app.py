import os, json
from flask import Flask, request, jsonify
from .db import SessionLocal, engine, Base
from .models import SensorReading, Prediction, RoadSegment

# Create tables automatically for MVP
Base.metadata.create_all(bind=engine)

app = Flask(__name__)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/api/sensor-data", methods=["POST"])
def ingest_sensor():
    data = request.get_json(force=True)

    sensor_id = data.get("sensor_id")
    if not sensor_id:
        return jsonify({"error": "sensor_id required"}), 400

    water_level = data.get("water_level")
    soil_moisture = data.get("soil_moisture")

    if water_level is None and soil_moisture is None:
        return jsonify({"error": "provide water_level or soil_moisture"}), 400

    hotspot_id = data.get("hotspot_id")
    ts = data.get("ts")

    db = next(get_db())
    rec = SensorReading(
        sensor_id=sensor_id,
        hotspot_id=hotspot_id,
        water_level=water_level,
        soil_moisture=soil_moisture,
        raw=json.dumps(data)
    )

    if ts:
        from dateutil import parser
        try:
            rec.ts = parser.isoparse(ts)
        except Exception:
            pass

    db.add(rec)
    db.commit()
    db.refresh(rec)

    # Simple MVP rule:
    # If water level > 40 → road becomes flooded
    try:
        if rec.water_level and rec.water_level > 40:
            seg = db.query(RoadSegment) \
                .filter(RoadSegment.name.ilike(f"%{hotspot_id}%")) \
                .first()
            if seg:
                seg.status = "flooded"
                db.commit()
    except:
        pass

    return jsonify({"ok": True, "id": rec.id})

@app.route("/api/roads", methods=["GET"])
def get_roads():
    db = next(get_db())
    from .api_helpers import roads_to_geojson
    return jsonify(roads_to_geojson(db))

@app.route("/api/hotspots", methods=["GET"])
def get_hotspots():
    db = next(get_db())
    rows = db.query(SensorReading.hotspot_id).distinct().all()
    hotspots = [r[0] for r in rows if r[0]]
    return jsonify({"hotspots": hotspots})

@app.route("/api/predictions/<hotspot_id>", methods=["GET"])
def get_predictions(hotspot_id):
    db = next(get_db())
    q = (db.query(Prediction)
         .filter(Prediction.hotspot_id == hotspot_id)
         .order_by(Prediction.ds.desc())
         .limit(48)
         .all())

    out = [{
        "ds": p.ds.isoformat(),
        "yhat": p.yhat,
        "yhat_lower": p.yhat_lower,
        "yhat_upper": p.yhat_upper
    } for p in q]

    return jsonify(out)

if __name__ == "__main__":
    port = int(os.getenv("FLASK_RUN_PORT", 5000))
    app.run(host="0.0.0.0", port=port)
