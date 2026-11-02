"""Unit tests for the modeloINV Dashboard Web Backend, REST API, and WebSocket streamer.

Validates:
- Application factory creation and configuration
- Static assets mounting and serving (HTML, CSS, JS, Chart.js)
- All REST API endpoints:
  - /api/v1/portfolio
  - /api/v1/positions
  - /api/v1/trades
  - /api/v1/performance
  - /api/v1/benchmark
  - /api/v1/learning
  - /api/v1/readiness
  - /api/v1/mode (toggle between paper and live)
  - /api/v1/health
  - /api/v1/retrain
- Real-time WebSocket streaming at /ws/stream
- WebSocket broadcast events (ticks, trades, equity, mode changes)
- Input validation and HTTP error statuses (400, 404, 422)
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
from fastapi.testclient import TestClient

from modeloinv.core.models import (
    AccountInfo,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from modeloinv.dashboard.app import create_app
from modeloinv.dashboard.websockets import ConnectionManager
from modeloinv.execution.paper import SimulatedPaperBroker
from modeloinv.storage.database import DatabaseManager


@pytest.fixture
def test_broker():
    """Provides an isolated SimulatedPaperBroker initialized with €100."""
    return SimulatedPaperBroker(initial_capital_eur=100.0)


@pytest.fixture
def test_db():
    """Provides an in-memory SQLite database manager."""
    return DatabaseManager(db_path=":memory:")


@pytest.fixture
def app_and_client(test_broker, test_db):
    """Provides a configured FastAPI app and TestClient."""
    app = create_app(broker=test_broker, db_manager=test_db)
    client = TestClient(app)
    return app, client, test_broker, test_db


class TestDashboardAppFactory:
    """Tests application factory initialization and configuration."""

    def test_create_app_defaults(self):
        app = create_app()
        assert app.title == "modeloINV Trading API"
        assert hasattr(app.state, "broker")
        assert hasattr(app.state, "connection_manager")
        assert isinstance(app.state.broker, SimulatedPaperBroker)

    def test_static_files_served(self, app_and_client):
        _, client, _, _ = app_and_client

        # Root serves index.html
        res_root = client.get("/")
        assert res_root.status_code == 200
        assert "text/html" in res_root.headers.get("content-type", "")
        assert "modeloINV Dashboard" in res_root.text

        # Direct index.html
        res_idx = client.get("/index.html")
        assert res_idx.status_code == 200

        # CSS asset
        res_css = client.get("/static/css/style.css")
        assert res_css.status_code == 200
        assert "glass-panel" in res_css.text

        # JS asset
        res_js = client.get("/static/js/app.js")
        assert res_js.status_code == 200
        assert "initWebSocket" in res_js.text

        # Chart.js asset
        res_chart = client.get("/static/js/chart.min.js")
        assert res_chart.status_code == 200


class TestDashboardRestEndpoints:
    """Tests all REST API endpoints under /api/v1."""

    def test_health_endpoint(self, app_and_client):
        _, client, _, _ = app_and_client
        res = client.get("/api/v1/health")
        assert res.status_code == 200
        assert res.json() == {"status": "ok"}

    def test_portfolio_endpoint(self, app_and_client):
        _, client, broker, _ = app_and_client
        res = client.get("/api/v1/portfolio")
        assert res.status_code == 200
        data = res.json()

        assert data["currency"] == "EUR"
        assert data["initial_capital_eur"] == 100.0
        assert data["cash_eur"] == 100.0
        assert data["equity_eur"] == 100.0
        assert data["realized_pnl_eur"] == 0.0
        assert data["unrealized_pnl_eur"] == 0.0
        assert data["buying_power_eur"] == 100.0
        assert data["is_live"] is False

    def test_positions_endpoint_empty_and_populated(self, app_and_client):
        _, client, broker, _ = app_and_client

        # Initially empty
        res1 = client.get("/api/v1/positions")
        assert res1.status_code == 200
        assert res1.json() == []

        # Place order to open position
        req = OrderRequest(
            symbol="SPY",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
        )
        res_order = broker.submit_order(req, current_market_price=450.0)
        assert res_order.status == OrderStatus.FILLED

        # Verify position is reflected
        res2 = client.get("/api/v1/positions")
        assert res2.status_code == 200
        positions = res2.json()
        assert len(positions) == 1
        assert positions[0]["symbol"] == "SPY"
        assert positions[0]["quantity"] == 0.1
        assert positions[0]["entry_price"] > 0

    def test_trades_endpoint(self, app_and_client):
        _, client, broker, _ = app_and_client

        # Initially empty
        res1 = client.get("/api/v1/trades")
        assert res1.status_code == 200
        assert res1.json() == []

        # Submit trade
        req = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.5,
        )
        broker.submit_order(req, current_market_price=200.0)

        # Check trades endpoint
        res2 = client.get("/api/v1/trades")
        assert res2.status_code == 200
        trades = res2.json()
        assert len(trades) == 1
        assert trades[0]["symbol"] == "AAPL"
        assert trades[0]["side"] == "BUY"
        assert trades[0]["quantity"] == 0.5
        assert trades[0]["price"] > 0

    def test_performance_endpoint(self, app_and_client):
        _, client, _, _ = app_and_client
        res = client.get("/api/v1/performance")
        assert res.status_code == 200
        data = res.json()

        assert "cumulative_return" in data
        assert "sharpe_ratio" in data
        assert "sortino_ratio" in data
        assert "max_drawdown" in data
        assert "win_rate" in data
        assert data["cumulative_return"] == 0.0
        assert data["sharpe_ratio"] == 0.0
        assert data["sortino_ratio"] == 0.0
        assert data["max_drawdown"] == 0.0
        assert data["win_rate"] == 0.0

    def test_performance_endpoint_populated(self, app_and_client):
        _, client, _, db = app_and_client
        base_t = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=10)
        equities = [100.0, 101.5, 102.8, 104.2, 105.0]
        for i, eq in enumerate(equities):
            db.record_portfolio_snapshot({
                "timestamp": (base_t + datetime.timedelta(days=i)).isoformat(),
                "cash_eur": eq * 0.4,
                "equity_eur": eq,
                "realized_pnl_eur": eq - 100.0,
                "unrealized_pnl_eur": 0.0,
                "cumulative_return_pct": (eq - 100.0),
                "drawdown_pct": 0.0,
                "benchmark_spy_price": 450.0 + i,
                "open_positions_count": 0,
                "positions": [],
                "execution_mode": "paper",
            })
        db.record_trade({
            "trade_id": "tr_perf_1",
            "order_id": None,
            "symbol": "SPY",
            "side": "SELL",
            "quantity": 0.1,
            "price": 455.0,
            "fee_eur": 0.01,
            "slippage_eur": 0.0,
            "realized_pnl_eur": 5.0,
            "pnl_percent": 5.0,
            "holding_period_seconds": 3600.0,
            "strategy_name": "hybrid_ai",
            "model_version": "v1.1.0",
            "execution_mode": "paper",
            "timestamp": base_t.isoformat(),
        })
        res = client.get("/api/v1/performance")
        assert res.status_code == 200
        data = res.json()
        assert data["cumulative_return"] > 0.0
        assert data["sharpe_ratio"] > 1.0
        assert data["win_rate"] == 1.0

    def test_benchmark_endpoint(self, app_and_client):
        _, client, _, _ = app_and_client
        res = client.get("/api/v1/benchmark")
        assert res.status_code == 200
        data = res.json()

        assert "timestamps" in data
        assert "strategy_equity" in data
        assert "sp500_equity" in data
        assert data["timestamps"] == []
        assert data["strategy_equity"] == []
        assert data["sp500_equity"] == []

    def test_benchmark_endpoint_populated(self, app_and_client):
        _, client, _, db = app_and_client
        base_t = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=5)
        for i in range(5):
            db.record_portfolio_snapshot({
                "timestamp": (base_t + datetime.timedelta(days=i)).isoformat(),
                "cash_eur": 40.0,
                "equity_eur": 100.0 + (i * 2.0),
                "realized_pnl_eur": i * 2.0,
                "unrealized_pnl_eur": 0.0,
                "cumulative_return_pct": i * 2.0,
                "drawdown_pct": 0.0,
                "benchmark_spy_price": 450.0 + (i * 1.0),
                "benchmark_cumulative_return_pct": (i * 1.0 / 450.0) * 100.0,
                "open_positions_count": 0,
                "positions": [],
                "execution_mode": "paper",
            })
        res = client.get("/api/v1/benchmark")
        assert res.status_code == 200
        data = res.json()
        assert len(data["timestamps"]) == 5
        assert len(data["strategy_equity"]) == 5
        assert len(data["sp500_equity"]) == 5
        assert data["strategy_equity"][-1] > data["sp500_equity"][-1]

    def test_learning_endpoint(self, app_and_client):
        _, client, _, _ = app_and_client
        res = client.get("/api/v1/learning")
        assert res.status_code == 200
        data = res.json()

        assert "current_version" in data
        assert "versions" in data
        assert len(data["versions"]) >= 2
        for v in data["versions"]:
            assert "version" in v
            assert "sharpe" in v
            assert "win_rate" in v
            assert "timestamp" in v

    def test_readiness_endpoint(self, app_and_client):
        _, client, _, _ = app_and_client
        res = client.get("/api/v1/readiness")
        assert res.status_code == 200
        data = res.json()

        assert "is_live_ready" in data
        assert "gates" in data
        assert len(data["gates"]) == 8
        assert data["is_live_ready"] is False

    def test_readiness_endpoint_when_certified(self, app_and_client):
        _, client, _, db = app_and_client
        start_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=60)
        eq1 = np.linspace(100.0, 102.0, 10)
        eq2 = np.linspace(102.0, 119.0, 22)[1:]
        eq3 = np.linspace(119.0, 114.0, 10)[1:]
        eq4 = np.linspace(114.0, 122.40, 21)[1:]
        equities = np.concatenate([eq1, eq2, eq3, eq4])
        spy_prices = [450.0 * (1.0 + (i * (0.098 / 59.0))) for i in range(60)]

        for i in range(60):
            ts = start_date + datetime.timedelta(days=i)
            eq = float(equities[i])
            db.record_portfolio_snapshot({
                "timestamp": ts.isoformat(),
                "cash_eur": round(eq * 0.4, 2),
                "equity_eur": round(eq, 2),
                "realized_pnl_eur": round(eq - 100.0, 2),
                "unrealized_pnl_eur": 0.0,
                "cumulative_return_pct": round(eq - 100.0, 2),
                "drawdown_pct": 0.035,
                "benchmark_spy_price": round(spy_prices[i], 2),
                "benchmark_cumulative_return_pct": round((spy_prices[i] - 450.0) / 450.0 * 100.0, 2),
                "open_positions_count": 0,
                "positions": [],
                "execution_mode": "paper",
            })
        for i in range(64):
            is_win = (i % 8) not in (3, 7)
            db.record_trade({
                "trade_id": f"tr_cert_{i}",
                "order_id": None,
                "symbol": "SPY",
                "side": "SELL",
                "quantity": 0.05,
                "price": 450.0 + i,
                "fee_eur": 0.015,
                "slippage_eur": 0.005,
                "realized_pnl_eur": 1.25 if is_win else -0.85,
                "pnl_percent": 5.0 if is_win else -3.5,
                "holding_period_seconds": 3600.0,
                "strategy_name": "hybrid_ai",
                "model_version": "v1.1.0",
                "execution_mode": "paper",
                "timestamp": (start_date + datetime.timedelta(hours=i * 20)).isoformat(),
            })
        res = client.get("/api/v1/readiness")
        assert res.status_code == 200
        data = res.json()
        assert data["is_live_ready"] is True

    def test_mode_toggle_endpoint(self, app_and_client):
        _, client, broker, _ = app_and_client

        # Toggle to live mode
        res1 = client.post("/api/v1/mode", json={"mode": "live"})
        assert res1.status_code == 200
        d1 = res1.json()
        assert d1["status"] == "success"
        assert d1["active_mode"] == "live"
        assert d1["is_live"] is True
        assert broker.is_live is True

        # Check portfolio endpoint reflects is_live
        res_port = client.get("/api/v1/portfolio")
        assert res_port.json()["is_live"] is True

        # Toggle back to paper mode
        res2 = client.post("/api/v1/mode", json={"mode": "paper"})
        assert res2.status_code == 200
        d2 = res2.json()
        assert d2["status"] == "success"
        assert d2["active_mode"] == "paper"
        assert d2["is_live"] is False
        assert broker.is_live is False

    def test_mode_toggle_invalid_mode_raises_400(self, app_and_client):
        _, client, _, _ = app_and_client
        res = client.post("/api/v1/mode", json={"mode": "crypto_live_invalid"})
        assert res.status_code == 400
        assert "Invalid mode" in res.json()["detail"]

    def test_retrain_endpoint(self, app_and_client):
        _, client, _, _ = app_and_client
        res = client.post("/api/v1/retrain", json={"force": True})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert "retrained_version" in data


class TestDashboardWebSocketStream:
    """Tests real-time WebSocket connection and event streaming."""

    def test_websocket_connect_and_initial_tick(self, app_and_client):
        _, client, _, _ = app_and_client

        with client.websocket_connect("/ws/stream") as ws:
            msg = ws.receive_json()
            assert msg["event"] == "tick"
            assert msg["symbol"] == "SPY"
            assert msg["price"] == 450.0
            assert msg["portfolio_equity_eur"] == 100.0

    def test_connection_manager_broadcasts(self):
        import asyncio

        manager = ConnectionManager()
        assert len(manager.active_connections) == 0

        # Broadcast with no connections returns 0
        loop = asyncio.new_event_loop()
        count = loop.run_until_complete(manager.broadcast({"event": "test"}))
        assert count == 0
        loop.close()


class TestDashboardErrorHandling:
    """Tests 404, 422, and invalid payload handling."""

    def test_not_found_endpoint(self, app_and_client):
        _, client, _, _ = app_and_client
        res = client.get("/api/v1/non_existent_endpoint")
        assert res.status_code == 404

    def test_malformed_json_mode_endpoint(self, app_and_client):
        _, client, _, _ = app_and_client
        res = client.post("/api/v1/mode", json={"unexpected_key": "val"})
        assert res.status_code == 422
