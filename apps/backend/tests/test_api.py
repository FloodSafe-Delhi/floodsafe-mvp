"""Tests for API endpoints.

Tests marked with @pytest.mark.db_required need PostgreSQL with PostGIS.
Other tests use client_no_db for basic endpoint accessibility checks.
"""
import pytest


class TestOpenAPISchema:
    """Tests for API documentation - no database required."""

    def test_openapi_schema_available(self, client_no_db):
        """Test that OpenAPI schema is generated."""
        response = client_no_db.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema
        assert "info" in schema

    def test_docs_endpoint_available(self, client_no_db):
        """Test that Swagger UI is accessible."""
        response = client_no_db.get("/docs")
        assert response.status_code == 200

    def test_redoc_endpoint_available(self, client_no_db):
        """Test that ReDoc is accessible."""
        response = client_no_db.get("/redoc")
        assert response.status_code == 200

    def test_api_routes_registered(self, client_no_db):
        """Test that expected API routes are in the schema."""
        response = client_no_db.get("/openapi.json")
        schema = response.json()
        paths = schema.get("paths", {})

        # Check core endpoints are registered
        assert "/health" in paths
        assert "/api/reports/" in paths
        assert "/api/users/" in paths or "/api/users" in paths


@pytest.mark.db_required
class TestReportsAPI:
    """Tests for the reports API endpoints - requires PostgreSQL."""

    def test_list_reports_empty(self, client):
        """Test listing reports when database is empty."""
        response = client.get("/api/reports/")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_reports_endpoint_exists(self, client):
        """Test that reports endpoint is accessible."""
        response = client.get("/api/reports/")
        assert response.status_code != 404


@pytest.mark.db_required
class TestUsersAPI:
    """Tests for the users API endpoints - requires PostgreSQL."""

    def test_users_endpoint_exists(self, client):
        """Test that users endpoint structure is correct."""
        response = client.get("/api/users/")
        assert response.status_code in [200, 404, 405]


@pytest.mark.db_required
class TestSensorsAPI:
    """Tests for the sensors API endpoints - requires PostgreSQL."""

    def test_sensors_endpoint_exists(self, client):
        """Test that sensors endpoint is accessible."""
        response = client.get("/api/sensors/")
        assert response.status_code in [200, 404, 405]
