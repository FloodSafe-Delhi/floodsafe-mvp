from flask import jsonify
from sqlalchemy.orm import Session
from .models import RoadSegment
import json

def roads_to_geojson(db: Session):
    rows = db.query(RoadSegment).all()
    features = []
    for r in rows:
        try:
            geometry = None
            if r.geojson:
                geometry = json.loads(r.geojson)

            features.append({
                "type": "Feature",
                "properties": {
                    "id": r.id,
                    "name": r.name,
                    "status": r.status
                },
                "geometry": geometry
            })
        except Exception:
            continue

    return {
        "type": "FeatureCollection",
        "features": features
    }
