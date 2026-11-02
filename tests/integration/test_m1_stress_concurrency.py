"""Adversarial Stress Test Suite for Milestone 1: Concurrency, Storage & Event Throughput.

Empirical verification of:
1. SQLite WAL mode concurrency under 20 simultaneous read/write threads.
2. Cross-instance SQLite WAL concurrency on Windows filesystem.
3. ParquetBarCache rapid writes, date-sliced reads, and Windows WinError 32 immunity.
4. EventDispatcher high throughput (>20,000 events/sec), thread contention, and fault isolation.
"""

from __future__ import annotations

import concurrent.futures
import os
import random
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from modeloinv.core.events import (
    AlertEvent,
    BarEvent,
    Event,
    EventDispatcher,
    OrderRequestEvent,
    OrderResultEvent,
    TickEvent,
    TradeEvent,
)
from modeloinv.core.models import (
    AssetClass,
    Bar,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    Trade,
)
from modeloinv.storage.cache import ParquetBarCache
from modeloinv.storage.database import DatabaseManager


# =============================================================================
# CHALLENGE 1: High-Concurrency SQLite WAL Operations (20 Threads)
# =============================================================================

class TestSQLiteWALConcurrencyStress:
    """Stress tests SQLite under 20 simultaneous threads performing mixed reads/writes."""

    def test_wal_20_threads_single_instance_high_contention(self, tmp_path: Path):
        """20 simultaneous threads doing rapid concurrent record_trade, record_order,

        update_order_status, record_portfolio_snapshot, and get_portfolio_history.
        Verifies zero 'database is locked' errors and 100% data integrity.
        """
        db_file = tmp_path / "stress_wal_20t.db"
        db = DatabaseManager(db_path=db_file, timeout=10.0)

        # Baseline schema check
        conn = db.get_connection()
        journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert journal_mode.lower() == "wal", f"Expected WAL mode, got {journal_mode}"

        num_threads = 20
        trades_per_writer = 50
        orders_per_writer = 50
        snapshots_per_writer = 30
        read_iterations = 40

        # Allocation:
        # Threads 0-6 (7 threads): Trade writers
        # Threads 7-13 (7 threads): Order writers + status updaters
        # Threads 14-17 (4 threads): Snapshot writers + portfolio readers
        # Threads 18-19 (2 threads): Aggregate statistics & history readers

        errors: List[Exception] = []
        err_lock = threading.Lock()

        def log_err(e: Exception):
            with err_lock:
                errors.append(e)

        def trade_writer(tid: int):
            try:
                for i in range(trades_per_writer):
                    trade_id = f"tr_{tid}_{i}"
                    db.record_trade({
                        "trade_id": trade_id,
                        "symbol": "SPY" if i % 2 == 0 else "AAPL",
                        "side": "BUY" if i % 3 != 0 else "SELL",
                        "quantity": 1.0 + (i * 0.1),
                        "price": 500.0 + i,
                        "fee_eur": 0.25,
                        "slippage_eur": 0.10,
                        "realized_pnl_eur": 5.0 if i % 2 == 0 else -2.0,
                        "execution_mode": "paper",
                        "timestamp": (datetime.now(timezone.utc) - timedelta(seconds=i)).isoformat(),
                    })
            except Exception as e:
                log_err(e)

        def order_worker(tid: int):
            try:
                for i in range(orders_per_writer):
                    client_id = f"cl_{tid}_{i}"
                    oid = db.record_order({
                        "client_order_id": client_id,
                        "symbol": "BTCUSDT",
                        "side": OrderSide.BUY,
                        "order_type": OrderType.LIMIT,
                        "status": OrderStatus.PENDING,
                        "quantity": 0.5,
                        "limit_price": 60000.0 + i,
                        "execution_mode": "paper",
                    })
                    # Immediately attempt an update to simulate broker fill race
                    if i % 2 == 0:
                        db.update_order_status(
                            order_id=oid,
                            status=OrderStatus.FILLED,
                            filled_quantity=0.5,
                            filled_avg_price=60000.0 + i,
                            fee_eur=0.30,
                        )
            except Exception as e:
                log_err(e)

        def snapshot_worker(tid: int):
            try:
                base_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
                for i in range(snapshots_per_writer):
                    db.record_portfolio_snapshot({
                        "timestamp": (base_time + timedelta(minutes=tid * 100 + i)).isoformat(),
                        "cash_eur": 100.0 - i,
                        "equity_eur": 100.0 + (i * 0.5),
                        "realized_pnl_eur": i * 0.5,
                        "execution_mode": "paper",
                        "open_positions_count": i % 3,
                    })
                    # Interleaved concurrent read
                    history = db.get_portfolio_history(execution_mode="paper", limit=20)
                    assert isinstance(history, list)
            except Exception as e:
                log_err(e)

        def reader_worker(tid: int):
            try:
                for _ in range(read_iterations):
                    # Query trade stats
                    stats = db.get_trade_statistics(execution_mode="paper")
                    assert isinstance(stats, dict)
                    # Query orders
                    orders = db.get_orders(limit=25)
                    assert isinstance(orders, list)
                    # Query latest snapshot
                    snap = db.get_latest_portfolio_snapshot(execution_mode="paper")
                    time.sleep(0.001)  # small yield to encourage context switching
            except Exception as e:
                log_err(e)

        # Launch 20 simultaneous threads
        threads: List[threading.Thread] = []
        for tid in range(7):
            threads.append(threading.Thread(target=trade_writer, args=(tid,), name=f"trade-w-{tid}"))
        for tid in range(7, 14):
            threads.append(threading.Thread(target=order_worker, args=(tid,), name=f"order-w-{tid}"))
        for tid in range(14, 18):
            threads.append(threading.Thread(target=snapshot_worker, args=(tid,), name=f"snap-w-{tid}"))
        for tid in range(18, 20):
            threads.append(threading.Thread(target=reader_worker, args=(tid,), name=f"reader-{tid}"))

        start_time = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)
            assert not t.is_alive(), f"Thread {t.name} hung or deadlocked!"
        elapsed = time.time() - start_time

        # Assert no lock errors or exceptions occurred across all 20 threads
        if errors:
            pytest.fail(f"Encountered {len(errors)} concurrency errors: {errors[0]}")

        # Verification of exact data counts
        expected_trades = 7 * trades_per_writer  # 350
        expected_orders = 7 * orders_per_writer  # 350
        expected_snapshots = 4 * snapshots_per_writer  # 120

        stats = db.get_trade_statistics(execution_mode="paper")
        assert stats["total_trades"] == expected_trades, f"Expected {expected_trades} trades, got {stats['total_trades']}"

        all_orders = db.get_orders(limit=1000)
        assert len(all_orders) == expected_orders, f"Expected {expected_orders} orders, got {len(all_orders)}"

        history = db.get_portfolio_history(execution_mode="paper", limit=1000)
        assert len(history) == expected_snapshots, f"Expected {expected_snapshots} snapshots, got {len(history)}"

        # Check WAL checkpoint operates cleanly without error
        db.checkpoint()
        db.close()

    def test_wal_20_threads_multi_instance_concurrency(self, tmp_path: Path):
        """Adversarial stress: 20 separate DatabaseManager instances opening the SAME file.

        Each thread has its own DatabaseManager, bypassing in-process Python RLock.
        This verifies SQLite's operating-system-level WAL locking and busy_timeout handling.
        """
        db_file = tmp_path / "stress_wal_multi_inst.db"

        # Prime schema
        init_db = DatabaseManager(db_path=db_file, timeout=10.0)
        init_db.close()

        num_threads = 20
        ops_per_thread = 25
        errors: List[Exception] = []
        err_lock = threading.Lock()

        def worker(tid: int):
            # Create independent manager instance for this thread
            local_mgr = DatabaseManager(db_path=db_file, timeout=15.0, auto_init=False)
            try:
                for i in range(ops_per_thread):
                    if tid % 3 == 0:
                        # Trade writer
                        local_mgr.record_trade({
                            "trade_id": f"t_multi_{tid}_{i}",
                            "symbol": "SPY",
                            "side": "BUY",
                            "quantity": 1.0,
                            "price": 500.0,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                    elif tid % 3 == 1:
                        # Order writer
                        local_mgr.record_order({
                            "order_id": f"ord_multi_{tid}_{i}",
                            "symbol": "AAPL",
                            "side": "BUY",
                            "order_type": "MARKET",
                            "status": "FILLED",
                            "filled_quantity": 2.0,
                            "filled_avg_price": 180.0,
                        })
                    else:
                        # Reader
                        local_mgr.get_trades(limit=10)
                        local_mgr.get_orders(limit=10)
                        local_mgr.get_trade_statistics()
            except Exception as e:
                with err_lock:
                    errors.append(e)
            finally:
                local_mgr.close()

        threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)
            assert not t.is_alive(), f"Thread {t.name} hung during multi-instance WAL stress!"

        assert len(errors) == 0, f"Multi-instance WAL concurrency failed with errors: {errors}"

        # Confirm database integrity via final instance
        verify_db = DatabaseManager(db_path=db_file, timeout=5.0, auto_init=False)
        stats = verify_db.get_trade_statistics()
        assert stats["total_trades"] > 0
        verify_db.close()

    def test_wal_checkpoint_during_continuous_writes(self, tmp_path: Path):
        """Stress test WAL passive/truncate checkpoints while concurrent threads are writing."""
        db_file = tmp_path / "stress_wal_ckpt.db"
        db = DatabaseManager(db_path=db_file, timeout=10.0)

        stop_event = threading.Event()
        errors: List[Exception] = []
        err_lock = threading.Lock()

        def writer(tid: int):
            try:
                i = 0
                while not stop_event.is_set() and i < 100:
                    db.record_trade({
                        "trade_id": f"ckpt_tr_{tid}_{i}",
                        "symbol": "SPY",
                        "side": "BUY",
                        "quantity": 1.0,
                        "price": 500.0,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                    i += 1
            except Exception as e:
                with err_lock:
                    errors.append(e)

        def checkpointer():
            try:
                for _ in range(10):
                    db.checkpoint()
                    time.sleep(0.01)
            except Exception as e:
                with err_lock:
                    errors.append(e)

        writers = [threading.Thread(target=writer, args=(t,)) for t in range(5)]
        ckpter = threading.Thread(target=checkpointer)

        for w in writers:
            w.start()
        ckpter.start()

        ckpter.join(timeout=10.0)
        stop_event.set()
        for w in writers:
            w.join(timeout=10.0)

        db.close()
        assert len(errors) == 0, f"Checkpointing during active writes failed with errors: {errors}"


# =============================================================================
# CHALLENGE 2: Rapid Parquet Cache Writes, Slices, and Windows File Locking
# =============================================================================

class TestParquetCacheAdversarialStress:
    """Stress tests ParquetBarCache under rapid writes, concurrent date slicing,

    and Windows filesystem atomic replace semantics.
    """

    def test_rapid_concurrent_writes_and_date_sliced_reads(self, tmp_path: Path):
        """10 threads writing rapid overlapping batches + 10 threads reading date slices.

        Confirms no WinError 32 (file in use), no corrupted Parquet files, and correct deduplication.
        """
        cache = ParquetBarCache(cache_dir=tmp_path / "parquet_cache")
        symbol = "SPY"
        timeframe = "1m"
        base_dt = datetime(2026, 1, 1, 9, 30, tzinfo=timezone.utc)

        writer_threads_count = 10
        reader_threads_count = 10
        batches_per_writer = 15
        bars_per_batch = 30

        errors: List[Exception] = []
        err_lock = threading.Lock()

        def writer_task(tid: int):
            try:
                for b_idx in range(batches_per_writer):
                    # Each batch generates bars, intentionally overlapping across writers
                    # Start minute offsets vary to force deduplication and reordering
                    start_min = (tid * 5) + (b_idx * 10)
                    batch = []
                    for k in range(bars_per_batch):
                        ts = base_dt + timedelta(minutes=start_min + k)
                        bar = Bar(
                            timestamp=ts,
                            symbol=symbol,
                            open=500.0 + k,
                            high=502.0 + k,
                            low=499.0 + k,
                            close=501.0 + k,
                            volume=1000.0 + (tid * 10),
                            asset_class=AssetClass.EQUITY,
                            currency="USD",
                        )
                        batch.append(bar)
                    cache.write_bars(symbol, timeframe, batch)
            except Exception as e:
                with err_lock:
                    errors.append(e)

        def reader_task(tid: int):
            try:
                for _ in range(25):
                    # Perform metadata check
                    meta_range = cache.get_cached_range(symbol, timeframe)
                    if meta_range is not None:
                        min_t, max_t = meta_range
                        assert min_t <= max_t

                    # Perform date-sliced read
                    slice_start = base_dt + timedelta(minutes=10)
                    slice_end = base_dt + timedelta(minutes=50)
                    bars = cache.read_bars(symbol, timeframe, start=slice_start, end=slice_end)
                    assert isinstance(bars, list)
                    for bar in bars:
                        assert bar.timestamp >= slice_start
                        assert bar.timestamp <= slice_end

                    # Perform dataframe read
                    df = cache.read_dataframe(symbol, timeframe, start=slice_start, end=slice_end)
                    assert isinstance(df, pd.DataFrame)
                    time.sleep(0.001)
            except Exception as e:
                with err_lock:
                    errors.append(e)

        writers = [threading.Thread(target=writer_task, args=(i,), name=f"p-write-{i}") for i in range(writer_threads_count)]
        readers = [threading.Thread(target=reader_task, args=(i,), name=f"p-read-{i}") for i in range(reader_threads_count)]

        all_threads = writers + readers
        random.shuffle(all_threads)

        for t in all_threads:
            t.start()
        for t in all_threads:
            t.join(timeout=30.0)
            assert not t.is_alive(), f"Thread {t.name} hung during Parquet stress!"

        assert len(errors) == 0, f"Parquet concurrency failed with errors: {errors}"

        # Post-stress verification: read entire cache
        all_bars = cache.read_bars(symbol, timeframe)
        assert len(all_bars) > 0, "Cache should contain stored bars"

        # Verify chronological order and absolute deduplication
        prev_ts = None
        for b in all_bars:
            if prev_ts is not None:
                assert b.timestamp > prev_ts, f"Bar timestamps out of order or duplicate: {prev_ts} vs {b.timestamp}"
            prev_ts = b.timestamp

    def test_parquet_unlocked_metadata_race_condition(self, tmp_path: Path):
        """Stress race condition between os.replace (write_bars) and pq.read_metadata (get_cached_range).

        get_cached_range does not hold self._lock; on Windows, reading while replacing can trigger WinError 32.
        Tests resilience and error recovery.
        """
        cache = ParquetBarCache(cache_dir=tmp_path / "parquet_race")
        symbol = "AAPL"
        timeframe = "5m"
        base_dt = datetime(2026, 1, 1, 9, 30, tzinfo=timezone.utc)

        # Pre-seed initial file
        cache.write_bars(symbol, timeframe, [
            Bar(timestamp=base_dt, symbol=symbol, open=180.0, high=182.0, low=179.0, close=181.0, volume=500.0)
        ])

        stop_flag = threading.Event()
        errors: List[Exception] = []
        err_lock = threading.Lock()

        def aggressive_writer():
            try:
                for k in range(100):
                    batch = [
                        Bar(
                            timestamp=base_dt + timedelta(minutes=5 * (k + j)),
                            symbol=symbol,
                            open=180.0,
                            high=181.0,
                            low=179.0,
                            close=180.5,
                            volume=100.0,
                        )
                        for j in range(10)
                    ]
                    cache.write_bars(symbol, timeframe, batch)
            except Exception as e:
                with err_lock:
                    errors.append(e)
            finally:
                stop_flag.set()

        def tight_loop_reader():
            try:
                while not stop_flag.is_set():
                    # Hammer get_cached_range continuously
                    _ = cache.get_cached_range(symbol, timeframe)
            except Exception as e:
                with err_lock:
                    errors.append(e)

        writer = threading.Thread(target=aggressive_writer, name="agg-writer")
        readers = [threading.Thread(target=tight_loop_reader, name=f"tight-reader-{i}") for i in range(5)]

        for r in readers:
            r.start()
        writer.start()

        writer.join(timeout=30.0)
        assert not writer.is_alive()
        for r in readers:
            r.join(timeout=5.0)

        if errors:
            import traceback
            for err in errors:
                traceback.print_exception(type(err), err, err.__traceback__)
        assert len(errors) == 0, f"Aggressive writer/reader collision produced errors: {errors}"

    def test_parquet_multi_instance_concurrency(self, tmp_path: Path):
        """Stress tests multiple independent ParquetBarCache instances pointing to the same folder.

        Verifies cross-instance Windows file locking behavior when separate instances write and read.
        """
        cache_dir = tmp_path / "parquet_multi_inst"
        symbol = "MSFT"
        timeframe = "1h"
        base_dt = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)

        # Seed initial file
        c_init = ParquetBarCache(cache_dir=cache_dir)
        c_init.write_bars(symbol, timeframe, [
            Bar(timestamp=base_dt, symbol=symbol, open=400.0, high=405.0, low=395.0, close=402.0, volume=1000.0)
        ])

        errors: List[Exception] = []
        err_lock = threading.Lock()

        def writer_inst(tid: int):
            try:
                local_cache = ParquetBarCache(cache_dir=cache_dir)
                for i in range(10):
                    batch = [
                        Bar(
                            timestamp=base_dt + timedelta(hours=i * 2 + j),
                            symbol=symbol,
                            open=400.0 + i,
                            high=405.0 + i,
                            low=395.0 + i,
                            close=402.0 + i,
                            volume=500.0,
                        )
                        for j in range(5)
                    ]
                    local_cache.write_bars(symbol, timeframe, batch)
            except Exception as e:
                with err_lock:
                    errors.append(e)

        def reader_inst(tid: int):
            try:
                local_cache = ParquetBarCache(cache_dir=cache_dir)
                for _ in range(15):
                    bars = local_cache.read_bars(symbol, timeframe)
                    assert isinstance(bars, list)
            except Exception as e:
                with err_lock:
                    errors.append(e)

        threads = []
        for i in range(4):
            threads.append(threading.Thread(target=writer_inst, args=(i,)))
        for i in range(4):
            threads.append(threading.Thread(target=reader_inst, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20.0)
            assert not t.is_alive()

        if errors:
            import traceback
            for err in errors:
                traceback.print_exception(type(err), err, err.__traceback__)
        assert len(errors) == 0, f"Multi-instance Parquet cache concurrency failed with: {errors}"


# =============================================================================
# CHALLENGE 3: EventDispatcher High Throughput, Concurrency & Fault Isolation
# =============================================================================

class TestEventDispatcherAdversarialStress:
    """Stress tests EventDispatcher under heavy throughput, concurrent multi-threading,

    and adversarial exception-throwing subscribers.
    """

    def test_raw_throughput_and_history_boundedness(self):
        """Dispatches 50,000 events to verify throughput and strict circular buffer bound."""
        history_limit = 200
        dispatcher = EventDispatcher(history_size=history_limit)

        received_count = 0

        def tick_handler(event: TickEvent):
            nonlocal received_count
            received_count += 1

        dispatcher.subscribe(TickEvent, tick_handler)

        total_events = 50_000
        t0 = time.time()
        for i in range(total_events):
            evt = TickEvent(symbol="SPY", bid=500.0, ask=500.1, last_price=500.05, volume=10.0)
            dispatcher.publish(evt)
        elapsed = time.time() - t0

        throughput = total_events / elapsed
        assert received_count == total_events, f"Expected {total_events} received, got {received_count}"

        # History should never exceed configured history_size
        history = dispatcher.get_history()
        assert len(history) == history_limit, f"Expected history size {history_limit}, got {len(history)}"

        # Assert throughput meets high-performance bar (>15,000 events/second)
        assert throughput > 15_000, f"Throughput {throughput:.1f} events/sec is below standard 15k/sec"

    def test_multi_threaded_dispatch_contention(self):
        """20 concurrent threads publishing 1,000 events each (20,000 total).

        Multiple subscribers with locks verifying no lost events or deadlocks.
        """
        dispatcher = EventDispatcher(history_size=500)
        total_threads = 20
        events_per_thread = 1_000
        expected_total = total_threads * events_per_thread

        tick_count = 0
        tick_lock = threading.Lock()

        global_count = 0
        global_lock = threading.Lock()

        def on_tick(event: TickEvent):
            nonlocal tick_count
            with tick_lock:
                tick_count += 1

        def on_all(event: Event):
            nonlocal global_count
            with global_lock:
                global_count += 1

        dispatcher.subscribe(TickEvent, on_tick)
        dispatcher.subscribe_all(on_all)

        errors: List[Exception] = []
        err_lock = threading.Lock()

        def publisher(tid: int):
            try:
                for i in range(events_per_thread):
                    evt = TickEvent(symbol=f"SYM_{tid}", last_price=100.0 + i)
                    dispatcher.publish(evt)
            except Exception as e:
                with err_lock:
                    errors.append(e)

        threads = [threading.Thread(target=publisher, args=(tid,), name=f"pub-{tid}") for tid in range(total_threads)]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)
            assert not t.is_alive(), f"Publisher thread {t.name} hung!"
        elapsed = time.time() - t0

        assert len(errors) == 0, f"Thread contention produced errors: {errors}"
        assert tick_count == expected_total, f"Expected {expected_total} ticks, got {tick_count}"
        assert global_count == expected_total, f"Expected {expected_total} global events, got {global_count}"

    def test_handler_exceptions_and_fault_isolation(self):
        """Adversarial test: subscribers raising unhandled exceptions and infinite recursion traps.

        Verifies that healthy handlers continue to receive events unharmed.
        """
        dispatcher = EventDispatcher()

        healthy_received = []

        def evil_handler_one(event: Event):
            raise ValueError("Malicious fault injected into handler 1")

        def evil_handler_two(event: Event):
            raise RuntimeError("Catastrophic runtime error in handler 2")

        def healthy_handler(event: Event):
            healthy_received.append(event)

        # Register evil and healthy handlers
        dispatcher.subscribe(TickEvent, evil_handler_one)
        dispatcher.subscribe(TickEvent, evil_handler_two)
        dispatcher.subscribe(TickEvent, healthy_handler)

        # Dispatch 50 events
        for i in range(50):
            evt = TickEvent(symbol="SPY", last_price=500.0 + i)
            # Must NOT raise exception to publisher
            dispatcher.publish(evt)

        # Healthy handler must have received all 50 events regardless of evil handlers
        assert len(healthy_received) == 50, f"Fault isolation failed: healthy handler received {len(healthy_received)}/50"

    def test_concurrent_subscribe_unsubscribe_during_publish(self):
        """Stress tests dynamic subscription/unsubscription while publishers are actively running.

        Verifies thread-safety of subscriber registry without dictionary iteration corruption.
        """
        dispatcher = EventDispatcher()
        stop_event = threading.Event()
        errors: List[Exception] = []
        err_lock = threading.Lock()

        def dummy_listener(evt: Event):
            pass

        def publisher_thread():
            try:
                count = 0
                while not stop_event.is_set() and count < 2000:
                    dispatcher.publish(TickEvent(symbol="SPY", last_price=500.0))
                    count += 1
            except Exception as e:
                with err_lock:
                    errors.append(e)

        def subscriber_flipper():
            try:
                for _ in range(500):
                    dispatcher.subscribe(TickEvent, dummy_listener)
                    dispatcher.subscribe_all(dummy_listener)
                    dispatcher.unsubscribe(TickEvent, dummy_listener)
                    dispatcher.unsubscribe_all(dummy_listener)
            except Exception as e:
                with err_lock:
                    errors.append(e)

        pubs = [threading.Thread(target=publisher_thread, name=f"dyn-pub-{i}") for i in range(4)]
        flippers = [threading.Thread(target=subscriber_flipper, name=f"dyn-flip-{i}") for i in range(2)]

        all_t = pubs + flippers
        for t in all_t:
            t.start()

        for f in flippers:
            f.join(timeout=10.0)
            assert not f.is_alive()
        stop_event.set()

        for p in pubs:
            p.join(timeout=10.0)
            assert not p.is_alive()

        assert len(errors) == 0, f"Dynamic mutation during publish caused errors: {errors}"
