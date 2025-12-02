from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    PROJECT_NAME: str = "FloodSafe API"
    API_V1_STR: str = "/api"
    
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/floodsafe"
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:5173",  # Vite Frontend
        "http://localhost:5174",  # Vite Frontend (alternate)
        "http://localhost:5175",  # Vite Frontend (alternate)
        "http://localhost:5176",  # Vite Frontend (current)
        "http://localhost:3000",  # Alternative Frontend
        "http://localhost:8000",  # Swagger UI
        "http://192.168.210.102:5173",  # Network access
        "http://172.18.144.1:5173",  # Docker network access
    ]

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
