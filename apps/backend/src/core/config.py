from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "FloodSafe API"
    API_V1_STR: str = "/api"

    # Database - MUST be overridden in production via DATABASE_URL env var
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/floodsafe"

    # CORS Configuration
    # For production, set env var: BACKEND_CORS_ORIGINS=["https://your-frontend.vercel.app"]
    # Pydantic will automatically parse JSON array from env var
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:5175",  # Vite Frontend (dev only)
        "http://localhost:8000",  # Swagger UI (dev only)
    ]

    # Flag to detect if we're in production (check if DATABASE_URL changed from default)
    @property
    def is_production(self) -> bool:
        return "localhost" not in self.DATABASE_URL

    # JWT Authentication
    JWT_SECRET_KEY: str = "floodsafe-jwt-secret-change-in-production-min-32-chars"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Google OAuth
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""

    # Firebase (for phone auth)
    FIREBASE_PROJECT_ID: str = ""

    # Mapbox (for routing)
    MAPBOX_ACCESS_TOKEN: str = ""

    # ML Service Integration
    ML_SERVICE_URL: str = "http://localhost:8002"
    ML_SERVICE_ENABLED: bool = True

    # ML Routing Integration (gradual rollout)
    ML_ROUTING_ENABLED: bool = False       # Enable ML predictions in route comparison
    ML_ROUTING_WEIGHT: float = 0.3         # Weight of ML vs reports/sensors [0-1]
    ML_MIN_CONFIDENCE: float = 0.7         # Only use predictions above this confidence
    ML_CACHE_TTL_SECONDS: int = 300        # Cache ML predictions for 5 min

    # External Alerts Configuration
    RSS_FEEDS_ENABLED: bool = True         # Enable RSS news fetcher
    IMD_API_ENABLED: bool = True           # Enable IMD weather fetcher (may need IP whitelist)
    CWC_SCRAPER_ENABLED: bool = True       # Enable CWC flood forecast scraper
    TWITTER_BEARER_TOKEN: str = ""         # Twitter API v2 bearer token (optional)
    TELEGRAM_BOT_TOKEN: str = ""           # Telegram Bot API token (optional)

    # External Alerts Scheduler (minutes)
    ALERT_REFRESH_RSS_MINUTES: int = 15    # RSS feeds refresh interval
    ALERT_REFRESH_IMD_MINUTES: int = 60    # IMD refresh interval
    ALERT_REFRESH_TWITTER_MINUTES: int = 30  # Twitter refresh interval
    ALERT_REFRESH_CWC_MINUTES: int = 120   # CWC scraper refresh interval

    # SendGrid Email Service (for email verification)
    SENDGRID_API_KEY: str = ""
    SENDGRID_FROM_EMAIL: str = "noreply@floodsafe.app"
    SENDGRID_FROM_NAME: str = "FloodSafe"

    # Email Verification
    EMAIL_VERIFICATION_EXPIRE_HOURS: int = 24
    FRONTEND_URL: str = "http://localhost:5175"  # For verification redirects
    BACKEND_URL: str = "http://localhost:8000"   # For verification links in emails

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
