import sys
import os
from datetime import datetime
import random
from uuid import uuid4

# Add the parent directory to sys.path to allow importing modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.infrastructure.database import SessionLocal, engine, Base
from src.infrastructure.models import User, FloodZone, Sensor, Report

def seed_data():
    print("Seeding Database...")
    
    # Create Tables
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    
    try:
        # 1. Create Users
        if db.query(User).count() == 0:
            print("Creating Users...")
            users = [
                User(username="admin", email="admin@floodsafe.ai", role="admin", points=1000, level=10, badges='["Admin", "Guardian"]'),
                User(username="demo_user", email="demo@floodsafe.ai", role="user", points=50, level=2, badges='["First Reporter"]'),
            ]
            db.add_all(users)
            db.commit()
        
        admin_user = db.query(User).filter_by(username="admin").first()

        # 2. Create Flood Zones (Bangalore Context)
        # Using some approximate polygons for Bangalore areas
        if db.query(FloodZone).count() == 0:
            print("Creating Flood Zones...")
            zones = [
                FloodZone(
                    name="Koramangala 4th Block",
                    risk_level="High",
                    geometry="POLYGON((77.62 12.93, 77.63 12.93, 77.63 12.94, 77.62 12.94, 77.62 12.93))"
                ),
                FloodZone(
                    name="Indiranagar 100ft Road",
                    risk_level="Medium",
                    geometry="POLYGON((77.64 12.97, 77.65 12.97, 77.65 12.98, 77.64 12.98, 77.64 12.97))"
                ),
                FloodZone(
                    name="Bellandur Lake Area",
                    risk_level="Critical",
                    geometry="POLYGON((77.66 12.92, 77.68 12.92, 77.68 12.94, 77.66 12.94, 77.66 12.92))"
                )
            ]
            db.add_all(zones)
            db.commit()

        # 3. Create Sensors
        if db.query(Sensor).count() == 0:
            print("Creating Sensors...")
            sensors = [
                Sensor(location="POINT(77.625 12.935)", status="active"), # Koramangala
                Sensor(location="POINT(77.645 12.975)", status="active"), # Indiranagar
                Sensor(location="POINT(77.67 12.93)", status="warning"),  # Bellandur
            ]
            db.add_all(sensors)
            db.commit()

        # 4. Create Reports
        if db.query(Report).count() == 0:
            print("Creating Reports...")
            reports = [
                Report(
                    user_id=admin_user.id,
                    description="Water logging near Sony Signal",
                    location="POINT(77.63 12.935)",
                    media_type="text",
                    verified=True,
                    verification_score=10,
                    upvotes=5
                ),
                Report(
                    user_id=admin_user.id,
                    description="Drain overflow observed",
                    location="POINT(77.642 12.972)",
                    media_type="image",
                    verified=False,
                    verification_score=2,
                    upvotes=1
                )
            ]
            db.add_all(reports)
            db.commit()
            
        print("Database Seeded Successfully!")
        
    except Exception as e:
        print(f"Error seeding database: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_data()
