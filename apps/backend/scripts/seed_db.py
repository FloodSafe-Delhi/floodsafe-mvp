"""
Simple DB seeder for development.
Creates tables and inserts a demo user.
"""
from src.database import engine, SessionLocal
from src.models import user_report
from src.crud import create_user

def seed():
    # create tables
    user_report.Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        # create a demo user (email must be unique)
        try:
            create_user(db, name="Demo User", email="demo@floodsafe.local", password="password")
            print("Created demo user: demo@floodsafe.local / password")
        except Exception as e:
            # if already exists or fail, print and continue
            print("Could not create demo user (may already exist):", e)
    finally:
        db.close()

if __name__ == "__main__":
    seed()
