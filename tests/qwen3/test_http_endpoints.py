"""HTTP integration tests for qwen3 backend."""

import pytest
import httpx


def test_health_endpoint():
    """Test /health endpoint returns correct status."""
    response = httpx.get("http://localhost:8000/health", timeout=10.0)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["backend"] == "qwen3"


def test_metrics_endpoint():
    """Test /metrics endpoint returns Prometheus metrics."""
    response = httpx.get("http://localhost:8000/metrics", timeout=10.0, follow_redirects=True)
    assert response.status_code == 200
    # Prometheus metrics should contain metric names
    assert "stt_" in response.text or "#" in response.text


def test_health_contains_backend_info():
    """Test /health endpoint returns backend information."""
    response = httpx.get("http://localhost:8000/health", timeout=10.0)
    assert response.status_code == 200
    data = response.json()

    # Should contain backend field
    assert "backend" in data
    assert data["backend"] in ["qwen3", "mock"]


def test_health_ignores_unknown_params():
    """Test /health endpoint ignores unknown query parameters."""
    # FastAPI ignores unknown query params by default
    response = httpx.get("http://localhost:8000/health?unknown=1", timeout=10.0)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_metrics_contains_backend_metrics():
    """Test /metrics contains backend-specific metrics."""
    response = httpx.get("http://localhost:8000/metrics", timeout=10.0, follow_redirects=True)
    assert response.status_code == 200
    # Should contain some stt metrics
    text = response.text
    # Prometheus format check
    assert "# HELP" in text or "# TYPE" in text or "stt_" in text
