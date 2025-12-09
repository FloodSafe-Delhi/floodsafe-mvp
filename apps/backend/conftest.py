import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.infrastructure.database import Base, get_db
from src.main import app

# Use PostgreSQL with PostGIS for testing (required for GeoAlchemy2)
# In CI, this connects to a PostgreSQL service container
# Locally, this can use the Docker Compose database
SQLALCHEMY_TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://floodsafe:floodsafe@localhost:5432/floodsafe_test"
)


@pytest.fixture(scope="session")
def test_engine():
    """Create a test database engine with PostgreSQL/PostGIS."""
    try:
        engine = create_engine(SQLALCHEMY_TEST_DATABASE_URL)
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        yield engine
    except Exception as e:
        pytest.skip(f"PostgreSQL database not available: {e}")


@pytest.fixture(scope="function")
def test_db(test_engine):
    """Create a test database session with transaction rollback."""
    connection = test_engine.connect()
    transaction = connection.begin()

    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=connection
    )
    db = TestingSessionLocal()

    try:
        yield db
    finally:
        db.close()
        transaction.rollback()
        connection.close()


@pytest.fixture(scope="function")
def client(test_db):
    """Create a test client with database override."""
    def override_get_db():
        try:
            yield test_db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def client_no_db():
    """Create a test client without database dependency for basic endpoint tests."""
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
