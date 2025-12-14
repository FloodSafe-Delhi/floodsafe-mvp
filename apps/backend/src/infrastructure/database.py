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


# Determine if we need SSL (required for cloud databases like Supabase)
def get_connect_args(url: URL) -> dict:
    """Get connection arguments based on host (SSL for cloud, none for localhost)."""
    host = url.host or ""
    if "localhost" in host or "127.0.0.1" in host or "db" == host:
        # Local development - no SSL needed
        return {}
    else:
        # Cloud database (Supabase, etc.) - require SSL
        return {"sslmode": "require"}


# Sync engine (for existing endpoints)
try:
    database_url = create_database_url()
    connect_args = get_connect_args(database_url)
    engine = create_engine(database_url, connect_args=connect_args)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info(f"Database sync engine initialized (SSL: {'sslmode' in connect_args})")
except Exception as e:
    logger.error(f"CRITICAL: Failed to initialize database engine: {e}")
    logger.error(f"DATABASE_URL env var exists: {bool(os.getenv('DATABASE_URL'))}")
    raise

# Async engine (for external alerts)
try:
    # Create async version of the URL
    async_url = database_url.set(drivername="postgresql+asyncpg")
    # For asyncpg, SSL is passed differently
    host = database_url.host or ""
    is_cloud = "localhost" not in host and "127.0.0.1" not in host and host != "db"
    async_connect_args = {"ssl": "require"} if is_cloud else {}
    async_engine = create_async_engine(async_url, echo=False, connect_args=async_connect_args)
    AsyncSessionLocal = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )
    logger.info(f"Database async engine initialized (SSL: {is_cloud})")
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
