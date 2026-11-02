"""Adversarial and Stress Test Suite for Milestone 5: Dashboard REST API and WebSockets.

Mandate Coverage:
1. High-frequency concurrent requests to all REST endpoints using starlette.testclient.TestClient.
2. Malformed JSON payloads, invalid types, boundary numbers, and SQL injection attempts on query parameters (e.g., /api/v1/trades?limit=-5, /api/v1/mode with unsupported modes).
3. Rapid WebSocket connect/disconnect cycles, concurrent client connections, broadcasting while clients drop unexpectedly.
4. Broker state synchronization: after POST /api/v1/mode toggles to live, ensure /api/v1/portfolio immediately reflects is_live == True, and vice versa.
5. Fallback behavior when database or registry files are empty, missing, or corrupted.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List, Optional
import unittest.mock as mock

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.testclient import TestClient
import pytest

from modeloinv.core.models import (
    AccountInfo,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Trade,
)
from modeloinv.dashboard.app import create_app
from modeloinv.dashboard.websockets import ConnectionManager
from modeloinv.execution.paper import SimulatedPaperBroker
from modeloinv.learning.readiness import LiveReadinessEvaluator
from modeloinv.learning.registry import ModelMetadata, ModelRegistry
from modeloinv.storage.database import DatabaseManager


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def clean_broker():
    """Provides a fresh SimulatedPaperBroker."""
    return SimulatedPaperBroker(initial_capital_eur=100.0)


@pytest.fixture
def clean_db():
    """Provides an in-memory SQLite database manager."""
    return DatabaseManager(db_path=":memory:")


@pytest.fixture
def app_and_client(clean_broker, clean_db):
    """Provides a configured FastAPI app and TestClient."""
    app = create_app(broker=clean_broker, db_manager=clean_db)
    client = TestClient(app)
    return app, client, clean_broker, clean_db


# =============================================================================
# 1. High-Frequency Concurrent Requests (Mandate 1)
# =============================================================================

class TestRestHighFrequencyConcurrency:
    """Stress tests REST API under concurrent multithreaded loads."""

    def test_concurrent_rest_read_burst(self, app_and_client):
        """Spins 20 concurrent threads firing 200 GET requests across all REST endpoints."""
        _, client, _, _ = app_and_client
        endpoints = [
            "/api/v1/portfolio",
            "/api/v1/positions",
            "/api/v1/trades",
            "/api/v1/performance",
            "/api/v1/benchmark",
            "/api/v1/learning",
            "/api/v1/readiness",
            "/api/v1/health",
            "/",
            "/index.html",
        ]

        def worker(idx: int) -> tuple[int, int]:
            target = endpoints[idx % len(endpoints)]
            res = client.get(target)
            return idx, res.status_code

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(worker, i) for i in range(200)]
            results = [f.result() for f in as_completed(futures)]

        assert len(results) == 200
        for idx, code in results:
            assert code == 200, f"Request {idx} failed with status code {code}"

    def test_concurrent_mixed_read_write_burst(self, app_and_client):
        """Interweaves concurrent GET and POST (/api/v1/mode, /api/v1/retrain) requests under high load."""
        _, client, broker, _ = app_and_client

        def worker(idx: int) -> tuple[int, int, str]:
            op_type = idx % 5
            if op_type == 0:
                res = client.post("/api/v1/mode", json={"mode": "live" if idx % 2 == 0 else "paper"})
                return idx, res.status_code, "mode_toggle"
            elif op_type == 1:
                res = client.post("/api/v1/retrain", json={"force": (idx % 2 == 0)})
                return idx, res.status_code, "retrain"
            elif op_type == 2:
                res = client.get("/api/v1/portfolio")
                return idx, res.status_code, "portfolio"
            elif op_type == 3:
                res = client.get("/api/v1/trades")
                return idx, res.status_code, "trades"
            else:
                res = client.get("/api/v1/performance")
                return idx, res.status_code, "performance"

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(worker, i) for i in range(250)]
            results = [f.result() for f in as_completed(futures)]

        assert len(results) == 250
        status_codes = [r[1] for r in results]
        assert all(code == 200 for code in status_codes), f"Encountered non-200 responses: {set(status_codes)}"

    def test_concurrent_trades_execution_and_rest_polling(self):
        """Concurrently generates trade executions in broker while REST queries poll trades and portfolio."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            db = DatabaseManager(db_path=db_path)
            broker = SimulatedPaperBroker(initial_capital_eur=100.0)
            app = create_app(broker=broker, db_manager=db)
            client = TestClient(app)

            def trader():
                for i in range(20):
                    req = OrderRequest(
                        symbol="SPY",
                        side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        quantity=0.05,
                    )
                    res = broker.submit_order(req, current_market_price=450.0 + i)
                    if res.status == OrderStatus.FILLED and broker.trade_history:
                        db.record_order(res)
                        db.record_trade(broker.trade_history[-1])
                    time.sleep(0.005)

            def poller(endpoint: str) -> List[int]:
                codes = []
                for _ in range(20):
                    r = client.get(endpoint)
                    codes.append(r.status_code)
                    time.sleep(0.005)
                return codes

            with ThreadPoolExecutor(max_workers=4) as executor:
                fut_trader = executor.submit(trader)
                fut_port = executor.submit(poller, "/api/v1/portfolio")
                fut_trades = executor.submit(poller, "/api/v1/trades")
                fut_pos = executor.submit(poller, "/api/v1/positions")

                fut_trader.result()
                port_codes = fut_port.result()
                trade_codes = fut_trades.result()
                pos_codes = fut_pos.result()

            assert all(c == 200 for c in port_codes)
            assert all(c == 200 for c in trade_codes)
            assert all(c == 200 for c in pos_codes)

            final_trades = client.get("/api/v1/trades").json()
            assert len(final_trades) > 0
        finally:
            if os.path.exists(db_path):
                try:
                    os.remove(db_path)
                except OSError:
                    pass


# =============================================================================
# 2. Malformed Payloads & SQL Injection Fuzzing (Mandate 2)
# =============================================================================

class TestPayloadFuzzingAndSqlInjection:
    """Stress tests parameter boundaries, malformed JSON, and SQL injection robustness."""

    @pytest.mark.parametrize(
        "query_str",
        [
            "?limit=-5",
            "?limit=0",
            "?limit=999999999999999999999",
            "?limit=abc",
            "?limit=1;%20DROP%20TABLE%20trades;--",
            "?symbol=%27%20OR%201=1;--",
            "?offset=-10",
            "?start_time=2026-99-99T99:99:99",
            "?execution_mode=paper%27%20OR%20%271%27=%271",
        ],
    )
    def test_trades_query_param_fuzzing(self, app_and_client, query_str: str):
        """Verifies trades endpoint handles negative boundaries and SQL injection strings gracefully."""
        _, client, broker, db = app_and_client

        # Record a trade to ensure table has data (quantity=0.1 at price 200 = €20, safely within €100)
        req = OrderRequest(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.1)
        res_order = broker.submit_order(req, current_market_price=200.0)
        assert res_order.status == OrderStatus.FILLED

        res = client.get(f"/api/v1/trades{query_str}")
        assert res.status_code == 200
        data = res.json()
        assert isinstance(data, list)
        assert len(data) >= 1

        # Check DB table was not destroyed
        conn = db.get_connection()
        count = conn.execute("SELECT count(*) FROM trades;").fetchone()[0]
        assert count >= 0

    def test_database_manager_trades_direct_sqli_resilience(self, clean_db):
        """Directly invokes DatabaseManager.get_trades with malicious SQL injection parameters."""
        db = clean_db
        # Record sample trade with order_id=None to adhere to foreign key constraints
        t = Trade(
            trade_id="sqli_test_1",
            order_id=None,
            symbol="SPY",
            side=OrderSide.BUY,
            quantity=1.0,
            price=450.0,
            timestamp=datetime.datetime.utcnow(),
        )
        db.record_trade(t)

        # SQL Injection attempts across all filter fields
        sqli_symbol = "SPY' OR '1'='1"
        res_symbol = db.get_trades(symbol=sqli_symbol)
        # Should return empty because literal symbol string doesn't match
        assert res_symbol == []

        sqli_mode = "paper'; DROP TABLE trades; --"
        res_mode = db.get_trades(execution_mode=sqli_mode)
        assert res_mode == []

        # Negative limit boundary
        res_neg = db.get_trades(limit=-5)
        assert isinstance(res_neg, list)

        # Verify trades table still intact and accessible
        trades_after = db.get_trades(symbol="SPY")
        assert len(trades_after) == 1
        assert trades_after[0]["symbol"] == "SPY"

    @pytest.mark.parametrize(
        "invalid_mode",
        [
            "crypto_live",
            "sandbox",
            "production",
            "test",
            "admin",
            "",
            "   ",
            "paper; DROP TABLE trades;--",
            "' OR '1'='1",
            "\x00live",
            "live\n",
            "live ",
            " paper ",
        ],
    )
    def test_mode_toggle_unsupported_modes(self, app_and_client, invalid_mode: str):
        """Verifies unsupported or injection mode strings are strictly rejected with HTTP 400."""
        _, client, broker, _ = app_and_client
        initial_is_live = broker.is_live

        # White-space padded 'live ' / ' paper ' are handled by strip(), so test if they either succeed safely or fail
        res = client.post("/api/v1/mode", json={"mode": invalid_mode})

        if invalid_mode.strip().lower() in ["paper", "live"]:
            # If strip() normalizes it safely to paper/live, status code is 200
            assert res.status_code == 200
        else:
            assert res.status_code == 400
            assert "Invalid mode" in res.json().get("detail", "")
            # Ensure broker state was unchanged
            assert broker.is_live == initial_is_live

    @pytest.mark.parametrize(
        "malformed_payload",
        [
            {"mode": 12345},
            {"mode": None},
            {"mode": True},
            {"mode": ["paper"]},
            {"mode": {"nested": "live"}},
            {"unrecognized_key": "live"},
            {},
        ],
    )
    def test_mode_toggle_malformed_json_types(self, app_and_client, malformed_payload: Any):
        """Verifies Pydantic schema validation returns 422 Unprocessable Entity for invalid types."""
        _, client, _, _ = app_and_client
        res = client.post("/api/v1/mode", json=malformed_payload)
        assert res.status_code in [400, 422]

    def test_mode_toggle_non_json_bodies(self, app_and_client):
        """Sends raw non-JSON bytes, empty body, and truncated payloads."""
        _, client, _, _ = app_and_client

        # Empty body
        res_empty = client.post("/api/v1/mode", content=b"")
        assert res_empty.status_code == 422

        # Raw non-JSON
        res_raw = client.post("/api/v1/mode", content=b"NOT_A_JSON_STRING", headers={"Content-Type": "application/json"})
        assert res_raw.status_code == 422

        # Truncated JSON
        res_trunc = client.post("/api/v1/mode", content=b'{"mode": "li', headers={"Content-Type": "application/json"})
        assert res_trunc.status_code == 422

    @pytest.mark.parametrize(
        "endpoint",
        [
            "/api/v1/portfolio",
            "/api/v1/positions",
            "/api/v1/trades",
            "/api/v1/performance",
            "/api/v1/benchmark",
            "/api/v1/learning",
            "/api/v1/readiness",
            "/api/v1/health",
        ],
    )
    def test_fuzzing_all_get_endpoints_with_adversarial_params(self, app_and_client, endpoint: str):
        """Tests every GET endpoint with XSS, SQLi, and 10,000-char buffer strings."""
        _, client, _, _ = app_and_client
        fuzz_params = [
            "?test=<script>alert('xss')</script>",
            "?sqli=1%27%20OR%20%271%27=%271",
            f"?buffer={'A' * 10000}",
        ]
        for p in fuzz_params:
            res = client.get(f"{endpoint}{p}")
            assert res.status_code == 200, f"{endpoint}{p} returned {res.status_code}"


# =============================================================================
# 3. WebSocket Stress & Lifecycle (Mandate 3)
# =============================================================================

class TestWebSocketStressAndLifecycle:
    """Stress tests WebSocket real-time streaming, concurrent connections, and disconnect tolerance."""

    def test_rapid_websocket_connect_disconnect_cycles(self, app_and_client):
        """Connects and immediately disconnects 30 WebSockets in succession."""
        app, client, broker, _ = app_and_client
        ws_mgr: ConnectionManager = app.state.connection_manager

        for i in range(30):
            with client.websocket_connect("/ws/stream") as ws:
                msg = ws.receive_json()
                assert msg["event"] == "tick"
                assert msg["symbol"] == "SPY"
                assert msg["portfolio_equity_eur"] == 100.0

        # After all context managers exit, all connections must be cleaned up
        assert len(ws_mgr.active_connections) == 0

    def test_concurrent_websocket_connections_and_broadcast(self, app_and_client):
        """Opens multiple WebSockets and broadcasts messages to all of them."""
        app, client, _, _ = app_and_client
        ws_mgr: ConnectionManager = app.state.connection_manager

        with client.websocket_connect("/ws/stream") as ws1:
            m1 = ws1.receive_json()
            assert m1["event"] == "tick"

            with client.websocket_connect("/ws/stream") as ws2:
                m2 = ws2.receive_json()
                assert m2["event"] == "tick"

                assert len(ws_mgr.active_connections) == 2

                # Trigger synchronous broadcast via event loop
                loop = asyncio.new_event_loop()
                count = loop.run_until_complete(
                    ws_mgr.broadcast_tick(symbol="AAPL", price=210.5, portfolio_equity_eur=105.0)
                )
                loop.close()

                assert count == 2
                res1 = ws1.receive_json()
                assert res1["event"] == "tick"
                assert res1["symbol"] == "AAPL"
                assert res1["price"] == 210.5

                res2 = ws2.receive_json()
                assert res2["event"] == "tick"
                assert res2["symbol"] == "AAPL"

    def test_broadcasting_with_abruptly_dropped_clients(self):
        """Stress tests ConnectionManager broadcast when clients drop abruptly with exceptions."""
        loop = asyncio.new_event_loop()
        ws_mgr = ConnectionManager()

        # Create mock websockets: some healthy, some that crash on send_json
        healthy_ws_list = []
        dead_ws_list = []

        for i in range(10):
            mock_ws = mock.AsyncMock()
            mock_ws.accept = mock.AsyncMock()
            mock_ws.send_json = mock.AsyncMock()
            healthy_ws_list.append(mock_ws)

        for i in range(10):
            broken_ws = mock.AsyncMock()
            broken_ws.accept = mock.AsyncMock()
            broken_ws.send_json = mock.AsyncMock(side_effect=ConnectionResetError("Socket reset by peer"))
            dead_ws_list.append(broken_ws)

        # Connect all 20 sockets
        for ws in healthy_ws_list + dead_ws_list:
            loop.run_until_complete(ws_mgr.connect(ws))

        assert len(ws_mgr.active_connections) == 20

        # Broadcast tick
        success_count = loop.run_until_complete(
            ws_mgr.broadcast_tick(symbol="SPY", price=450.0, portfolio_equity_eur=100.0)
        )

        # 10 healthy should succeed, 10 dead should be detected and pruned
        assert success_count == 10
        assert len(ws_mgr.active_connections) == 10
        for dead in dead_ws_list:
            assert dead not in ws_mgr.active_connections

        loop.close()

    def test_connection_manager_concurrent_disconnect_thread_safety(self):
        """Adversarial concurrency test: multiple threads simultaneously disconnecting the same/different sockets."""
        ws_mgr = ConnectionManager()
        loop = asyncio.new_event_loop()

        # Connect 50 mock sockets
        sockets = [mock.AsyncMock() for _ in range(50)]
        for s in sockets:
            loop.run_until_complete(ws_mgr.connect(s))

        assert len(ws_mgr.active_connections) == 50

        # Concurrently call disconnect from multiple worker threads
        errors = []

        def disconnect_worker(ws: Any):
            try:
                ws_mgr.disconnect(ws)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=8) as executor:
            # Each socket disconnected twice concurrently to trigger potential list.remove race condition
            futures = [executor.submit(disconnect_worker, s) for s in sockets for _ in range(2)]
            for f in futures:
                f.result()

        loop.close()

        # Ensure no ValueError exceptions escaped
        assert len(errors) == 0, f"Concurrent disconnect produced errors: {errors}"
        assert len(ws_mgr.active_connections) == 0

    def test_broadcast_event_types_payload_integrity(self):
        """Verifies schema and values of all 4 broadcast types: tick, trade, equity, mode."""
        loop = asyncio.new_event_loop()
        ws_mgr = ConnectionManager()
        received_messages = []

        mock_client = mock.AsyncMock()
        mock_client.send_json = mock.AsyncMock(side_effect=lambda msg: received_messages.append(msg))
        loop.run_until_complete(ws_mgr.connect(mock_client))

        # 1. broadcast_tick
        loop.run_until_complete(ws_mgr.broadcast_tick("SPY", 450.12345, 100.5678))
        assert received_messages[-1]["event"] == "tick"
        assert received_messages[-1]["price"] == round(450.12345, 4)
        assert received_messages[-1]["portfolio_equity_eur"] == 100.5678

        # 2. broadcast_trade
        loop.run_until_complete(ws_mgr.broadcast_trade({"symbol": "SPY", "side": "BUY", "quantity": 1.0}))
        assert received_messages[-1]["event"] == "trade_fill"
        assert received_messages[-1]["trade"]["symbol"] == "SPY"

        # 3. broadcast_equity
        loop.run_until_complete(ws_mgr.broadcast_equity(102.5, 95.0, 7.5, 0.0))
        assert received_messages[-1]["event"] == "equity_update"
        assert received_messages[-1]["equity_eur"] == 102.5
        assert received_messages[-1]["cash_eur"] == 95.0

        # 4. broadcast_mode
        loop.run_until_complete(ws_mgr.broadcast_mode("live", True))
        assert received_messages[-1]["event"] == "mode_change"
        assert received_messages[-1]["active_mode"] == "live"
        assert received_messages[-1]["is_live"] is True

        loop.close()


# =============================================================================
# 4. Broker State Synchronization (Mandate 4)
# =============================================================================

class TestBrokerStateSynchronization:
    """Verifies lockstep synchronization between REST mode toggle, broker state, and portfolio endpoint."""

    def test_mode_toggle_immediate_portfolio_synchronization(self, app_and_client):
        """Toggles mode to live and asserts immediate synchronization across broker, account, and REST portfolio."""
        _, client, broker, _ = app_and_client

        # Initial paper state
        res_init = client.get("/api/v1/portfolio").json()
        assert res_init["is_live"] is False
        assert broker.is_live is False
        assert broker.get_account().is_live is False

        # Toggle to live
        res_toggle_live = client.post("/api/v1/mode", json={"mode": "live"})
        assert res_toggle_live.status_code == 200
        d_live = res_toggle_live.json()
        assert d_live["status"] == "success"
        assert d_live["active_mode"] == "live"
        assert d_live["is_live"] is True

        # Broker state must be immediately True
        assert broker.is_live is True
        assert broker.get_account().is_live is True

        # Portfolio endpoint must immediately reflect is_live == True
        res_port_live = client.get("/api/v1/portfolio").json()
        assert res_port_live["is_live"] is True

        # Toggle back to paper
        res_toggle_paper = client.post("/api/v1/mode", json={"mode": "paper"})
        assert res_toggle_paper.status_code == 200
        d_paper = res_toggle_paper.json()
        assert d_paper["is_live"] is False

        # Broker state must be immediately False
        assert broker.is_live is False
        assert broker.get_account().is_live is False

        # Portfolio endpoint must immediately reflect is_live == False
        res_port_paper = client.get("/api/v1/portfolio").json()
        assert res_port_paper["is_live"] is False

    def test_rapid_mode_toggle_alternation_consistency(self, app_and_client):
        """Performs 50 rapid alternating mode toggles, verifying zero desynchronization at each step."""
        _, client, broker, _ = app_and_client

        for i in range(50):
            target = "live" if i % 2 == 0 else "paper"
            expected_bool = (target == "live")

            res_post = client.post("/api/v1/mode", json={"mode": target})
            assert res_post.status_code == 200
            assert res_post.json()["is_live"] == expected_bool

            # Immediate verification
            res_port = client.get("/api/v1/portfolio").json()
            assert res_port["is_live"] == expected_bool
            assert broker.is_live == expected_bool
            assert broker.get_account().is_live == expected_bool

    def test_mode_toggle_concurrent_race_synchronization(self, app_and_client):
        """Fires concurrent portfolio reads while mode toggles occur, verifying no corrupted states."""
        _, client, broker, _ = app_and_client
        errors = []

        def toggler(count: int):
            for i in range(count):
                m = "live" if i % 2 == 0 else "paper"
                r = client.post("/api/v1/mode", json={"mode": m})
                if r.status_code != 200:
                    errors.append(f"Toggler error: {r.status_code}")

        def reader(count: int):
            for _ in range(count):
                r = client.get("/api/v1/portfolio")
                if r.status_code != 200:
                    errors.append(f"Reader error: {r.status_code}")
                else:
                    data = r.json()
                    if not isinstance(data.get("is_live"), bool):
                        errors.append(f"Invalid is_live type: {data}")

        with ThreadPoolExecutor(max_workers=8) as executor:
            fut_tog = executor.submit(toggler, 40)
            fut_readers = [executor.submit(reader, 40) for _ in range(4)]

            fut_tog.result()
            for fr in fut_readers:
                fr.result()

        assert len(errors) == 0, f"Race condition errors: {errors}"

    def test_mode_toggle_websocket_notification(self, app_and_client):
        """Verifies that mode toggle sends real-time mode_change event over active WebSocket."""
        _, client, _, _ = app_and_client

        with client.websocket_connect("/ws/stream") as ws:
            # Drain initial tick
            init_msg = ws.receive_json()
            assert init_msg["event"] == "tick"

            # Toggle mode via REST
            res = client.post("/api/v1/mode", json={"mode": "live"})
            assert res.status_code == 200

            # WebSocket must receive mode_change event
            event = ws.receive_json()
            assert event["event"] == "mode_change"
            assert event["active_mode"] == "live"
            assert event["is_live"] is True


# =============================================================================
# 5. Database & Registry Corruption and Fallbacks (Mandate 5)
# =============================================================================

class TestCorruptedAndMissingStorageFallbacks:
    """Stress tests fallback resilience when databases, registries, or reports are empty, missing, or corrupted."""

    def test_empty_database_fallback(self):
        """Verifies all dashboard endpoints return valid defaults and baseline data when DB is completely empty."""
        empty_db = DatabaseManager(db_path=":memory:")
        broker = SimulatedPaperBroker(initial_capital_eur=100.0)
        app = create_app(broker=broker, db_manager=empty_db)
        client = TestClient(app)

        # 1. /portfolio
        r_port = client.get("/api/v1/portfolio")
        assert r_port.status_code == 200
        assert r_port.json()["equity_eur"] == 100.0

        # 2. /positions
        r_pos = client.get("/api/v1/positions")
        assert r_pos.status_code == 200
        assert r_pos.json() == []

        # 3. /trades
        r_trades = client.get("/api/v1/trades")
        assert r_trades.status_code == 200
        assert r_trades.json() == []

        # 4. /performance (checks no divide-by-zero or NaN)
        r_perf = client.get("/api/v1/performance")
        assert r_perf.status_code == 200
        d_perf = r_perf.json()
        assert d_perf["sharpe_ratio"] == 0.0
        assert d_perf["win_rate"] == 0.0

        # 5. /benchmark
        r_bench = client.get("/api/v1/benchmark")
        assert r_bench.status_code == 200
        d_bench = r_bench.json()
        assert d_bench["timestamps"] == []
        assert d_bench["strategy_equity"] == []
        assert d_bench["sp500_equity"] == []

        # 6. /learning
        r_learn = client.get("/api/v1/learning")
        assert r_learn.status_code == 200
        assert "current_version" in r_learn.json()

        # 7. /readiness
        r_read = client.get("/api/v1/readiness")
        assert r_read.status_code == 200
        assert len(r_read.json()["gates"]) == 8

    def test_missing_database_manager_fallback(self, clean_broker):
        """Verifies dashboard operates cleanly when db_manager is None."""
        app = create_app(broker=clean_broker, db_manager=None)
        # Explicitly ensure state db_manager is None
        app.state.db_manager = None
        client = TestClient(app)

        endpoints = [
            "/api/v1/portfolio",
            "/api/v1/positions",
            "/api/v1/trades",
            "/api/v1/performance",
            "/api/v1/benchmark",
            "/api/v1/learning",
            "/api/v1/readiness",
            "/api/v1/health",
        ]
        for ep in endpoints:
            r = client.get(ep)
            assert r.status_code == 200, f"Endpoint {ep} failed with status {r.status_code} when db_manager=None"

    def test_corrupted_database_error_handling(self, clean_broker):
        """Verifies endpoints handle database query exceptions gracefully without crashing."""
        # Create a mock database manager whose get_trades and get_portfolio_history raise sqlite3 errors
        broken_db = mock.MagicMock()
        broken_db.get_trades.side_effect = Exception("sqlite3.DatabaseError: file is encrypted or is not a database")
        broken_db.get_portfolio_history.side_effect = Exception("sqlite3.DatabaseError: database disk image is malformed")

        app = create_app(broker=clean_broker, db_manager=broken_db)
        client = TestClient(app)

        # /trades must catch exception and return []
        r_trades = client.get("/api/v1/trades")
        assert r_trades.status_code == 200
        assert r_trades.json() == []

        # /performance must catch exception and return default metrics
        r_perf = client.get("/api/v1/performance")
        assert r_perf.status_code == 200
        assert r_perf.json()["sharpe_ratio"] == 0.0
        assert r_perf.json()["win_rate"] == 0.0

        # /benchmark must catch exception and return default benchmark series
        r_bench = client.get("/api/v1/benchmark")
        assert r_bench.status_code == 200
        assert r_bench.json()["strategy_equity"] == []

    def test_missing_model_registry_fallback(self, clean_broker, clean_db):
        """Verifies /api/v1/learning falls back cleanly when model_registry is None."""
        app = create_app(broker=clean_broker, db_manager=clean_db, model_registry=None)
        app.state.model_registry = None
        client = TestClient(app)

        r_learn = client.get("/api/v1/learning")
        assert r_learn.status_code == 200
        data = r_learn.json()
        assert data["current_version"] in ["v1.1.0", "v1.2.0"]
        assert len(data["versions"]) >= 2

    def test_corrupted_learning_progress_report_fallback(self, app_and_client):
        """Verifies /api/v1/learning recovers when models/learning_progress_report.json is corrupted."""
        _, client, _, _ = app_and_client

        # Ensure models dir exists
        os.makedirs("models", exist_ok=True)
        report_path = Path("models/learning_progress_report.json")
        orig_content = None
        if report_path.exists():
            orig_content = report_path.read_text(encoding="utf-8")

        try:
            # Overwrite with truncated malformed JSON
            report_path.write_text('{"vn_version": "v1.2.0", "version_history": [{"vers', encoding="utf-8")

            res = client.get("/api/v1/learning")
            assert res.status_code == 200
            data = res.json()
            assert "current_version" in data
            assert len(data["versions"]) >= 2
        finally:
            # Restore or remove file
            if orig_content is not None:
                report_path.write_text(orig_content, encoding="utf-8")
            elif report_path.exists():
                report_path.unlink()

    def test_missing_broker_fails_with_clean_500(self, clean_db):
        """Verifies endpoints return HTTP 500 with descriptive detail when broker is not initialized."""
        app = create_app(db_manager=clean_db)
        app.state.broker = None
        client = TestClient(app)

        # /portfolio must return 500 with detail
        res_port = client.get("/api/v1/portfolio")
        assert res_port.status_code == 500
        assert "Broker not initialized" in res_port.json()["detail"]

        # /mode must return 500 with detail
        res_mode = client.post("/api/v1/mode", json={"mode": "live"})
        assert res_mode.status_code == 500
        assert "Broker not initialized" in res_mode.json()["detail"]
