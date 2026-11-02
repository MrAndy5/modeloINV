"""Unit tests for storage subsystem: DatabaseManager (SQLite WAL) and ParquetBarCache."""

from __future__ import annotations

import concurrent.futures
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil

import pandas as pd
import pytest

from modeloinv.core.models import AssetClass, Bar, OrderRequest, OrderResult, OrderSide, OrderStatus, OrderType, Trade
from modeloinv.storage.cache import ParquetBarCache, sanitize_filename_token
from modeloinv.storage.database import DatabaseManager


# =====================================================================
# DatabaseManager (SQLite WAL) Tests
# =====================================================================

@pytest.fixture
def temp_db(tmp_path: Path):
    """Provides a fresh DatabaseManager backed by a temporary file."""
    db_file = tmp_path / "test_modeloinv.db"
    db = DatabaseManager(db_path=db_file)
    yield db
    db.close()


def test_db_initialization_and_tables(temp_db: DatabaseManager):
    """Verify tables and indexes are created and WAL mode is active."""
    conn = temp_db.get_connection()
    cursor = conn.cursor()

    # Check WAL journal mode
    journal_mode = cursor.execute("PRAGMA journal_mode;").fetchone()[0]
    assert journal_mode.lower() == "wal"

    # Check tables exist
    tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    table_names = {t[0] for t in tables}
    assert {"orders", "trades", "portfolio_snapshots", "model_versions", "system_logs"}.issubset(table_names)


def test_orders_crud_and_status_update(temp_db: DatabaseManager):
    """Verify order insertion, querying, and status updates."""
    order_req = OrderRequest(
        symbol="SPY",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=2.5,
    )
    order_id = temp_db.record_order(order_req)
    assert order_id is not None

    fetched = temp_db.get_order(order_id)
    assert fetched is not None
    assert fetched["symbol"] == "SPY"
    assert fetched["requested_quantity"] == 2.5
    assert fetched["status"] == "PENDING"

    # Update to FILLED
    updated = temp_db.update_order_status(
        order_id=order_id,
        status=OrderStatus.FILLED,
        filled_quantity=2.5,
        filled_avg_price=501.25,
        fee_eur=0.10,
        slippage_eur=0.05,
        message="Fill confirmed",
    )
    assert updated is True

    filled_order = temp_db.get_order(order_id)
    assert filled_order["status"] == "FILLED"
    assert filled_order["filled_quantity"] == 2.5
    assert filled_order["filled_avg_price"] == 501.25
    assert filled_order["fee_eur"] == 0.10
    assert filled_order["slippage_eur"] == 0.05


def test_trades_recording_and_statistics(temp_db: DatabaseManager):
    """Verify recording trades and calculating win rate and profit factor statistics."""
    now = datetime.now(timezone.utc)
    # Trade 1: Win (+20 EUR)
    temp_db.record_trade({
        "trade_id": "tr_1",
        "symbol": "SPY",
        "side": "BUY",
        "quantity": 1.0,
        "price": 500.0,
        "realized_pnl_eur": 20.0,
        "fee_eur": 0.50,
        "slippage_eur": 0.20,
        "timestamp": (now - timedelta(minutes=15)).isoformat(),
        "execution_mode": "paper",
    })
    # Trade 2: Win (+10 EUR)
    temp_db.record_trade({
        "trade_id": "tr_2",
        "symbol": "AAPL",
        "side": "BUY",
        "quantity": 2.0,
        "price": 180.0,
        "realized_pnl_eur": 10.0,
        "fee_eur": 0.50,
        "slippage_eur": 0.10,
        "timestamp": (now - timedelta(minutes=10)).isoformat(),
        "execution_mode": "paper",
    })
    # Trade 3: Loss (-15 EUR)
    temp_db.record_trade({
        "trade_id": "tr_3",
        "symbol": "BTCUSDT",
        "side": "SELL",
        "quantity": 0.05,
        "price": 65000.0,
        "realized_pnl_eur": -15.0,
        "fee_eur": 0.50,
        "slippage_eur": 0.30,
        "timestamp": now.isoformat(),
        "execution_mode": "paper",
    })

    trades = temp_db.get_trades(limit=10)
    assert len(trades) == 3
    # Ordered DESC by timestamp
    assert trades[0]["trade_id"] == "tr_3"

    stats = temp_db.get_trade_statistics(execution_mode="paper")
    assert stats["total_trades"] == 3
    assert stats["winning_trades"] == 2
    assert stats["losing_trades"] == 1
    assert stats["win_rate"] == round(2 / 3, 4)
    assert stats["total_realized_pnl"] == 15.0  # 20 + 10 - 15
    assert stats["total_fees"] == 1.50
    assert stats["total_slippage"] == 0.60
    assert stats["gross_profit"] == 30.0
    assert stats["gross_loss"] == 15.0
    assert stats["profit_factor"] == 2.0  # 30 / 15


def test_portfolio_snapshots_chronology(temp_db: DatabaseManager):
    """Verify portfolio snapshots are stored and retrieved in chronological order."""
    t1 = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 1, 1, 11, 0, tzinfo=timezone.utc)
    t3 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    temp_db.record_portfolio_snapshot({
        "timestamp": t1.isoformat(),
        "cash_eur": 100.0,
        "equity_eur": 100.0,
        "execution_mode": "paper",
    })
    temp_db.record_portfolio_snapshot({
        "timestamp": t2.isoformat(),
        "cash_eur": 80.0,
        "equity_eur": 105.0,
        "execution_mode": "paper",
    })
    temp_db.record_portfolio_snapshot({
        "timestamp": t3.isoformat(),
        "cash_eur": 80.0,
        "equity_eur": 110.0,
        "execution_mode": "paper",
    })

    history = temp_db.get_portfolio_history(execution_mode="paper")
    assert len(history) == 3
    # Ordered ASC (chronological)
    assert history[0]["equity_eur"] == 100.0
    assert history[1]["equity_eur"] == 105.0
    assert history[2]["equity_eur"] == 110.0

    latest = temp_db.get_latest_portfolio_snapshot(execution_mode="paper")
    assert latest is not None
    assert latest["equity_eur"] == 110.0


def test_model_version_registry_champion_atomicity(temp_db: DatabaseManager):
    """Verify setting a new champion model version atomically resets the old champion."""
    temp_db.record_model_version(
        version="v0.1.0",
        strategy_type="hybrid",
        metrics={"sharpe": 1.1},
        is_champion=True,
    )
    champ = temp_db.get_champion_model_version()
    assert champ is not None
    assert champ["version"] == "v0.1.0"

    # Promote candidate v0.2.0
    temp_db.record_model_version(
        version="v0.2.0",
        parent_version="v0.1.0",
        strategy_type="hybrid",
        metrics={"sharpe": 1.4},
        is_champion=True,
    )
    champ = temp_db.get_champion_model_version()
    assert champ is not None
    assert champ["version"] == "v0.2.0"

    # Verify v0.1.0 is no longer champion
    all_versions = {v["version"]: v["is_champion"] for v in temp_db.get_model_versions()}
    assert all_versions["v0.1.0"] is False
    assert all_versions["v0.2.0"] is True


def test_multithreaded_wal_concurrency(temp_db: DatabaseManager):
    """Verify concurrent reads and writes across multiple threads without locking errors."""
    def writer_task(thread_id: int):
        for i in range(20):
            temp_db.record_trade({
                "trade_id": f"t_{thread_id}_{i}",
                "symbol": "SPY",
                "side": "BUY",
                "quantity": 1.0,
                "price": 500.0 + i,
                "realized_pnl_eur": 1.0,
                "timestamp": datetime.utcnow().isoformat(),
            })

    def reader_task():
        for _ in range(20):
            temp_db.get_trades(limit=10)
            temp_db.get_trade_statistics()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for tid in range(4):
            futures.append(executor.submit(writer_task, tid))
        for _ in range(4):
            futures.append(executor.submit(reader_task))
        for f in concurrent.futures.as_completed(futures):
            f.result()

    stats = temp_db.get_trade_statistics()
    assert stats["total_trades"] == 80  # 4 writers * 20 trades


# =====================================================================
# ParquetBarCache Tests
# =====================================================================

@pytest.fixture
def temp_cache(tmp_path: Path):
    """Provides a fresh ParquetBarCache backed by a temporary directory."""
    cache_dir = tmp_path / "cache"
    return ParquetBarCache(cache_dir=cache_dir)


def test_parquet_cache_windows_filenames(temp_cache: ParquetBarCache):
    """Verify symbol token sanitization creates valid filenames on Windows."""
    assert sanitize_filename_token("EUR/USD") == "EUR_USD"
    assert sanitize_filename_token("BINANCE:BTCUSDT") == "BINANCE_BTCUSDT"
    assert sanitize_filename_token("EURUSD=X") == "EURUSD_X"

    p1 = temp_cache.get_cache_path("EUR/USD", "1d")
    assert "/" not in p1.name
    p2 = temp_cache.get_cache_path("BINANCE:BTCUSDT", "5m")
    assert ":" not in p2.name


def test_parquet_cache_roundtrip_and_deduplication(temp_cache: ParquetBarCache):
    """Verify writing, reading, and deduplication by timestamp."""
    t1 = datetime(2026, 1, 1, 9, 30, tzinfo=timezone.utc)
    t2 = datetime(2026, 1, 1, 9, 35, tzinfo=timezone.utc)
    t3 = datetime(2026, 1, 1, 9, 40, tzinfo=timezone.utc)

    bars_initial = [
        Bar(timestamp=t1, symbol="SPY", open=500.0, high=502.0, low=499.0, close=501.0, volume=1000.0),
        Bar(timestamp=t2, symbol="SPY", open=501.0, high=503.0, low=500.0, close=502.0, volume=1500.0),
    ]
    count1 = temp_cache.write_bars("SPY", "5m", bars_initial)
    assert count1 == 2
    assert temp_cache.has_cache("SPY", "5m") is True

    # Write overlapping batch: t2 (updated close) and t3
    bars_update = [
        Bar(timestamp=t2, symbol="SPY", open=501.0, high=504.0, low=500.0, close=503.5, volume=2000.0),
        Bar(timestamp=t3, symbol="SPY", open=503.5, high=505.0, low=502.0, close=504.0, volume=1200.0),
    ]
    count2 = temp_cache.write_bars("SPY", "5m", bars_update)
    assert count2 == 3  # Deduplicated from 4 to 3

    read_back = temp_cache.read_bars("SPY", "5m")
    assert len(read_back) == 3
    # Check t2 updated with new close 503.5 (keep='last')
    assert read_back[1].close == 503.5
    assert read_back[1].volume == 2000.0


def test_parquet_cache_date_slicing(temp_cache: ParquetBarCache):
    """Verify reading bars with start and end date predicate pushdown filters."""
    base = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    bars = [
        Bar(
            timestamp=base + timedelta(minutes=i * 5),
            symbol="AAPL",
            open=180.0 + i,
            high=181.0 + i,
            low=179.0 + i,
            close=180.5 + i,
            volume=500.0,
        )
        for i in range(10)
    ]
    temp_cache.write_bars("AAPL", "5m", bars)

    # Slice middle 4 bars: index 2 to 5 inclusive
    start_filter = base + timedelta(minutes=10)
    end_filter = base + timedelta(minutes=25)
    sliced = temp_cache.read_bars("AAPL", "5m", start=start_filter, end=end_filter)
    assert len(sliced) == 4
    assert sliced[0].timestamp == start_filter
    assert sliced[-1].timestamp == end_filter


def test_parquet_cache_metadata_range(temp_cache: ParquetBarCache):
    """Verify fast metadata range detection without reading entire dataset."""
    t_start = datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)
    t_end = datetime(2026, 2, 5, 0, 0, tzinfo=timezone.utc)
    bars = [
        Bar(timestamp=t_start, symbol="NVDA", open=800.0, high=810.0, low=790.0, close=805.0, volume=100.0),
        Bar(timestamp=t_end, symbol="NVDA", open=805.0, high=820.0, low=800.0, close=815.0, volume=150.0),
    ]
    temp_cache.write_bars("NVDA", "1d", bars)

    date_range = temp_cache.get_cached_range("NVDA", "1d")
    assert date_range is not None
    min_ts, max_ts = date_range
    assert min_ts == t_start
    assert max_ts == t_end


def test_parquet_cache_dataframe_read(temp_cache: ParquetBarCache):
    """Verify reading bars directly into a Pandas DataFrame."""
    now = datetime.now(timezone.utc)
    bars = [
        Bar(timestamp=now, symbol="QQQ", open=400.0, high=402.0, low=399.0, close=401.0, volume=5000.0)
    ]
    temp_cache.write_bars("QQQ", "1m", bars)

    df = temp_cache.read_dataframe("QQQ", "1m")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "close" in df.columns
    assert df.iloc[0]["close"] == 401.0


def test_parquet_cache_corrupt_file_handling(temp_cache: ParquetBarCache):
    """Verify ParquetBarCache recovers from and tolerates corrupted files without crashing."""
    target_path = temp_cache.get_cache_path("CORRUPT", "1m")
    target_path.write_bytes(b"INVALID_PARQUET_GARBAGE_PAYLOAD")

    # Metadata, read_bars, and read_dataframe must handle gracefully without raising
    assert temp_cache.get_cached_range("CORRUPT", "1m") is None
    assert temp_cache.read_bars("CORRUPT", "1m") == []
    df = temp_cache.read_dataframe("CORRUPT", "1m")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0

    # Overwriting corrupt file with valid bars recovers cache
    now = datetime.now(timezone.utc)
    recovery_bar = Bar(timestamp=now, symbol="CORRUPT", open=100.0, high=101.0, low=99.0, close=100.0, volume=100.0)
    assert temp_cache.write_bars("CORRUPT", "1m", [recovery_bar]) == 1
    assert len(temp_cache.read_bars("CORRUPT", "1m")) == 1
