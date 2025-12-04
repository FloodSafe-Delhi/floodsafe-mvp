from src.database import engine, SessionLocal
from src.models import user_report
from src.crud import create_user

def seed():
    user_report.Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # create a demo user
        try:
            create_user(db, name="Demo User", email="demo@floodsafe.local", password="password")
            print("Created demo user: demo@floodsafe.local / password")
        except Exception as e:
            print("Could not create demo user (may already exist):", e)
    finally:
        db.close()

if __name__ == "__main__":
    seed()
