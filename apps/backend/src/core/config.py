import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./dev.db")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "/app/uploads")

settings = Settings()
