"""Integration tests for the Flask status & control server.

These tests start the Flask app in-process using Werkzeug's test client,
which is the production-equivalent of hitting the live HTTP endpoints
with ``curl`` — they exercise the exact same request/response paths
that ``scripts/verify_with_curl.sh`` exercises against the running server.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from drowsiness.config import Config
from server import create_app


@pytest.fixture
def client():
    """Build a Flask test client with the monitor NOT auto-started."""
    cfg = Config()  # safe defaults; no monitor will actually run.
    app = create_app(cfg, monitor=None, standalone=False)
    app.testing = True
    return app.test_client()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.get_json()
    assert data["status"] == "ok"
    assert "version" in data


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
def test_status_endpoint_when_not_started(client):
    r = client.get("/status")
    assert r.status_code == 200
    data = r.get_json()
    assert data["running"] is False
    assert "message" in data


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def test_config_endpoint_returns_full_config(client):
    r = client.get("/config")
    assert r.status_code == 200
    data = r.get_json()
    # Sanity-check a few well-known fields.
    assert "eye" in data
    assert "ear_threshold" in data["eye"]
    assert "mouth" in data
    assert "mar_threshold" in data["mouth"]
    assert "video" in data
    assert "target_fps" in data["video"]


# ---------------------------------------------------------------------------
# Alarm control
# ---------------------------------------------------------------------------
def test_alarm_test_endpoint_without_monitor_returns_400(client):
    r = client.post("/alarm/test")
    assert r.status_code == 400
    data = r.get_json()
    assert "error" in data


def test_alarm_reset_endpoint_without_monitor_returns_400(client):
    r = client.post("/alarm/reset")
    assert r.status_code == 400
    data = r.get_json()
    assert "error" in data


# ---------------------------------------------------------------------------
# Stop
# ---------------------------------------------------------------------------
def test_stop_endpoint_when_not_running_returns_stopped_false(client):
    r = client.post("/stop")
    assert r.status_code == 200
    data = r.get_json()
    assert data["stopped"] is False


# ---------------------------------------------------------------------------
# Start (will fail because no camera is available in CI; the important
# thing is that the endpoint exists, accepts the call, and returns JSON.)
# ---------------------------------------------------------------------------
def test_start_endpoint_exists(client):
    r = client.post("/start")
    # Either 200 (started) or 500 (camera unavailable) — both prove the
    # endpoint exists and is wired correctly.
    assert r.status_code in (200, 500)
    data = r.get_json()
    # Either started=True or error=... is acceptable.
    assert "started" in data or "error" in data


# ---------------------------------------------------------------------------
# Round-trip: every endpoint should return valid JSON
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path,method", [
    ("/health", "GET"),
    ("/status", "GET"),
    ("/config", "GET"),
    ("/alarm/test", "POST"),
    ("/alarm/reset", "POST"),
    ("/stop", "POST"),
    ("/start", "POST"),
])
def test_every_endpoint_returns_json(client, path, method):
    fn = getattr(client, method.lower())
    r = fn(path)
    assert r.status_code in (200, 400, 500), \
        f"{method} {path} returned {r.status_code}"
    # Must parse as JSON.
    data = r.get_json()
    assert isinstance(data, dict)
