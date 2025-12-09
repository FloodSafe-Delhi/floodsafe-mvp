"""Tests for health check endpoint.

These tests use client_no_db fixture since they don't require database access.
"""
import pytest


def test_health_check(client_no_db):
    """Test that the health check endpoint returns healthy status."""
    response = client_no_db.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_health_check_method_not_allowed(client_no_db):
    """Test that POST to health endpoint is not allowed."""
    response = client_no_db.post("/health")
    assert response.status_code == 405
