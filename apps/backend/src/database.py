from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from src.core.config import settings

# SQLAlchemy engine
engine = create_engine(settings.DATABASE_URL, future=True)

# session factory
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

# declarative base
Base = declarative_base()

# dependency for FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
