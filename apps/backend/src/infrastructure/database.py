from sqlalchemy import create_engine, URL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker as async_sessionmaker
from ..core.config import settings
import logging
from urllib.parse import urlparse, unquote
import os

logger = logging.getLogger(__name__)


def create_database_url() -> URL:
    """
    Create SQLAlchemy URL object, handling special characters in password.
    Supports both connection string and component-based configuration.
    """
    database_url_string = settings.DATABASE_URL

    if not database_url_string:
        raise ValueError("DATABASE_URL is empty or not set")

    try:
        # Parse the URL string
        parsed = urlparse(database_url_string)

        if not parsed.scheme:
            raise ValueError("DATABASE_URL missing scheme (should start with postgresql://)")

        # Decode URL-encoded password if present
        password = unquote(parsed.password) if parsed.password else None

        # Create URL object (handles special characters properly)
        url_obj = URL.create(
            drivername=parsed.scheme,
            username=parsed.username,
            password=password,
            host=parsed.hostname,
            port=parsed.port,
            database=parsed.path.lstrip('/') if parsed.path else None,
        )

        # Log sanitized version
        sanitized = f"{parsed.scheme}://{parsed.username}:****@{parsed.hostname}:{parsed.port}{parsed.path}"
        logger.info(f"Connecting to database: {sanitized}")

        return url_obj

    except Exception as e:
        logger.error(f"Failed to parse DATABASE_URL: {e}")
        logger.error(f"DATABASE_URL format should be: postgresql://user:password@host:port/database")
        logger.error(f"DATABASE_URL value (first 50 chars): {database_url_string[:50]}...")
        raise ValueError(f"Invalid DATABASE_URL format: {e}")


# Sync engine (for existing endpoints)
try:
    database_url = create_database_url()
    engine = create_engine(database_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info("Database sync engine initialized successfully")
except Exception as e:
    logger.error(f"CRITICAL: Failed to initialize database engine: {e}")
    logger.error(f"DATABASE_URL env var exists: {bool(os.getenv('DATABASE_URL'))}")
    raise

# Async engine (for external alerts)
try:
    # Create async version of the URL
    async_url = database_url.set(drivername="postgresql+asyncpg")
    async_engine = create_async_engine(async_url, echo=False)
    AsyncSessionLocal = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )
    logger.info("Database async engine initialized successfully")
except Exception as e:
    logger.error(f"CRITICAL: Failed to initialize async database engine: {e}")
    raise

Base = declarative_base()


def get_db():
    """Get sync database session (for existing endpoints)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db():
    """Get async database session (for external alerts)."""
    async with AsyncSessionLocal() as session:
        yield session
