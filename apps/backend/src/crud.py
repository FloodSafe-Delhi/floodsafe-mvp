from sqlalchemy.orm import Session
from src.models.user_report import User, Report
from src.utils.security import hash_password, verify_password
from geoalchemy2.shape import from_shape
from shapely.geometry import Point
from typing import List

def create_user(db: Session, name: str, email: str, password: str) -> User:
    hashed = hash_password(password)
    user = User(name=name, email=email, hashed_password=hashed)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def get_user_by_email(db: Session, email: str) -> User | None:
    return db.query(User).filter(User.email == email).first()

def authenticate_user(db: Session, email: str, password: str) -> User | None:
    user = get_user_by_email(db, email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

def create_report(db: Session, title: str, description: str, lon: float, lat: float, image_path: str = None, user_id: int = None) -> Report:
    geom = from_shape(Point(lon, lat), srid=4326)
    report = Report(title=title, description=description, geom=geom, image_path=image_path, user_id=user_id)
    db.add(report)
    db.commit()
    db.refresh(report)
    return report

def list_reports(db: Session, limit: int = 100) -> List[Report]:
    return db.query(Report).order_by(Report.created_at.desc()).limit(limit).all()
