"""
Empirical zero-state and dynamic probe for Challenger M5.
"""
from modeloinv.dashboard.app import create_app
from modeloinv.storage.database import DatabaseManager
from modeloinv.execution.paper import SimulatedPaperBroker
from fastapi.testclient import TestClient

def test_zero_state():
    db = DatabaseManager(":memory:")
    broker = SimulatedPaperBroker(initial_capital_eur=100.0)
    app = create_app(broker=broker, db_manager=db)
    client = TestClient(app)

    perf = client.get("/api/v1/performance").json()
    print("Empty Performance:", perf)
    assert perf["cumulative_return"] == 0.0, f"Expected 0.0, got {perf['cumulative_return']}"
    assert perf["sharpe_ratio"] == 0.0, f"Expected 0.0, got {perf['sharpe_ratio']}"
    assert perf["sortino_ratio"] == 0.0, f"Expected 0.0, got {perf['sortino_ratio']}"
    assert perf["max_drawdown"] == 0.0, f"Expected 0.0, got {perf['max_drawdown']}"
    assert perf["win_rate"] == 0.0, f"Expected 0.0, got {perf['win_rate']}"

    bench = client.get("/api/v1/benchmark").json()
    print("Empty Benchmark:", bench)
    assert bench["timestamps"] == [], "Expected empty list"
    assert bench["strategy_equity"] == [], "Expected empty list"
    assert bench["sp500_equity"] == [], "Expected empty list"

    readiness = client.get("/api/v1/readiness").json()
    print("Empty Readiness is_live_ready:", readiness["is_live_ready"])
    assert readiness["is_live_ready"] is False, "Expected is_live_ready: False on empty DB"
    assert len(readiness["gates"]) == 8, f"Expected 8 gates, got {len(readiness['gates'])}"
    assert readiness["gates"]["gate_7_min_trades"]["actual"] == 0
    assert readiness["gates"]["gate_4_sharpe_ratio"]["actual"] == 0.0

    trades = client.get("/api/v1/trades").json()
    print("Empty Trades count:", len(trades))
    assert trades == []

    positions = client.get("/api/v1/positions").json()
    print("Empty Positions count:", len(positions))
    assert positions == []

    portfolio = client.get("/api/v1/portfolio").json()
    print("Empty Portfolio equity:", portfolio["equity_eur"])
    assert portfolio["equity_eur"] == 100.0
    assert portfolio["cash_eur"] == 100.0
    assert portfolio["realized_pnl_eur"] == 0.0

    print("SUCCESS: ALL ZERO-STATE REST INVARIANTS CONFIRMED PERFECTLY!")

if __name__ == "__main__":
    test_zero_state()
