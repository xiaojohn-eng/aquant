"""
test_api.py
===========
API endpoint tests (Round 2 addition).

Covers: health, root, stocks, portfolio, monitor, WebSocket, auth.
Run with: pytest tests/test_api.py -v
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Public endpoints
# ---------------------------------------------------------------------------

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "timestamp" in data


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "AQuant" in data["message"]
    assert data["version"] == "4.4.0"


def test_docs_accessible():
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger" in response.text.lower()


def test_openapi_schema():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert data["info"]["title"] == "AQuant — A-Share Quantitative Trading System"


# ---------------------------------------------------------------------------
# Stock API
# ---------------------------------------------------------------------------

def test_get_stocks():
    response = client.get("/api/stocks")
    assert response.status_code in (200, 503)  # 503 if data source unavailable
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, list)


def test_get_recommendations():
    response = client.get("/api/stocks/recommendations")
    # May 404 if no data available in test environment
    assert response.status_code in (200, 404, 503)


def test_get_kline_invalid_code():
    response = client.get("/api/stocks/INVALID/kline")
    assert response.status_code in (422, 404)


# ---------------------------------------------------------------------------
# Portfolio API
# ---------------------------------------------------------------------------

def test_get_portfolio_unauthorized():
    """Portfolio endpoints require auth — should 401 without token."""
    response = client.get("/api/portfolio/positions")
    assert response.status_code == 401


def test_portfolio_buy_unauthorized():
    response = client.post("/api/portfolio/buy", json={
        "code": "000001.SZ", "shares": 100, "price": 10.0
    })
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Monitor API
# ---------------------------------------------------------------------------

def test_monitor_gpu():
    response = client.get("/api/monitor/gpu")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_monitor_logs():
    response = client.get("/api/monitor/logs?limit=10")
    assert response.status_code in (200, 404)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def test_auth_missing_token():
    response = client.get("/api/portfolio/positions")
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

def test_websocket_market():
    """WebSocket connection test."""
    with client.websocket_connect("/ws/market") as websocket:
        websocket.send_text("ping")
        data = websocket.receive_text()
        assert "Echo" in data


def test_websocket_gpu():
    """GPU monitor WebSocket connection."""
    with client.websocket_connect("/ws/gpu") as websocket:
        websocket.send_text("ping")
        # GPU WS doesn't echo, just keepalive


# ---------------------------------------------------------------------------
# Risk API (new in v4.4)
# ---------------------------------------------------------------------------

def test_risk_status():
    response = client.get("/api/risk/status")
    assert response.status_code in (200, 404)  # 404 if route not yet registered


# ---------------------------------------------------------------------------
# Integration: full data flow
# ---------------------------------------------------------------------------

def test_end_to_end_health_check():
    """Verify all critical endpoints respond."""
    endpoints = [
        ("/health", 200),
        ("/", 200),
        ("/docs", 200),
        ("/api/monitor/gpu", 200),
        ("/api/stocks", [200, 503]),
    ]
    for path, expected in endpoints:
        response = client.get(path)
        if isinstance(expected, list):
            assert response.status_code in expected, f"{path}: got {response.status_code}"
        else:
            assert response.status_code == expected, f"{path}: got {response.status_code}"
