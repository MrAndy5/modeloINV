"""Thread-safe SQLite database manager for Windows with WAL mode concurrency.

Handles persistent relational storage for trades, portfolio snapshots,
orders, model version registry, and system logs.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    """Returns current UTC timestamp in ISO8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _to_iso(dt: Union[datetime, str, None]) -> str:
    """Converts a datetime or string to ISO8601 UTC string."""
    if dt is None:
        return _utc_now_iso()
    if isinstance(dt, str):
        return dt
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat()


class DatabaseManager:
    """Thread-safe SQLite database manager for Windows with WAL mode."""

    def __init__(
        self,
        db_path: Union[str, Path] = "data/modeloinv.db",
        timeout: float = 5.0,
        auto_init: bool = True,
    ) -> None:
        """Initializes the database manager.

        Args:
            db_path: Path to SQLite database file or ':memory:'.
            timeout: Busy timeout in seconds.
            auto_init: If True, creates tables and indexes on instantiation.
        """
        self.raw_path = str(db_path)
        self.timeout = timeout
        self.is_memory = self.raw_path == ":memory:" or "mode=memory" in self.raw_path

        if not self.is_memory:
            self.db_path = str(Path(self.raw_path).resolve())
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        else:
            self.db_path = self.raw_path

        self._local = threading.local()
        self._write_lock = threading.RLock()
        self._all_connections: List[sqlite3.Connection] = []
        self._conn_lock = threading.Lock()

        # If in-memory, hold an anchor connection to prevent premature destruction
        self._anchor_conn: Optional[sqlite3.Connection] = None
        if self.is_memory:
            self._anchor_conn = self._create_connection()

        if auto_init:
            self.initialize_schema()

    def _create_connection(self) -> sqlite3.Connection:
        """Creates and configures a new sqlite3.Connection."""
        uri = self.raw_path.startswith("file:")
        conn = sqlite3.connect(
            self.db_path,
            timeout=self.timeout,
            check_same_thread=False,
            uri=uri,
        )
        conn.row_factory = sqlite3.Row

        # Execute PRAGMAs
        cursor = conn.cursor()
        if not self.is_memory:
            cursor.execute("PRAGMA journal_mode = WAL;")
            cursor.execute("PRAGMA synchronous = NORMAL;")
        cursor.execute(f"PRAGMA busy_timeout = {int(self.timeout * 1000)};")
        cursor.execute("PRAGMA foreign_keys = ON;")
        cursor.execute("PRAGMA temp_store = MEMORY;")
        cursor.execute("PRAGMA cache_size = -64000;")  # 64MB cache
        cursor.close()

        with self._conn_lock:
            self._all_connections.append(conn)

        return conn

    def get_connection(self) -> sqlite3.Connection:
        """Returns the thread-local SQLite connection, creating one if needed."""
        if self.is_memory and self._anchor_conn is not None:
            return self._anchor_conn
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._create_connection()
            self._local.conn = conn
        return conn

    def initialize_schema(self) -> None:
        """Creates database schema, tables, and indices."""
        schema_sql = """
        -- 1. Orders table
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            client_order_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            order_type TEXT NOT NULL,
            status TEXT NOT NULL,
            requested_quantity REAL,
            requested_notional REAL,
            limit_price REAL,
            stop_price REAL,
            trail_percent REAL,
            filled_quantity REAL NOT NULL DEFAULT 0.0,
            filled_avg_price REAL NOT NULL DEFAULT 0.0,
            fee_eur REAL NOT NULL DEFAULT 0.0,
            slippage_eur REAL NOT NULL DEFAULT 0.0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            message TEXT DEFAULT '',
            is_live INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
        CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
        CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at);
        CREATE INDEX IF NOT EXISTS idx_orders_client_id ON orders(client_order_id);

        -- 2. Trades table
        CREATE TABLE IF NOT EXISTS trades (
            trade_id TEXT PRIMARY KEY,
            order_id TEXT,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity REAL NOT NULL,
            price REAL NOT NULL,
            fee_eur REAL NOT NULL DEFAULT 0.0,
            slippage_eur REAL NOT NULL DEFAULT 0.0,
            realized_pnl_eur REAL NOT NULL DEFAULT 0.0,
            pnl_percent REAL NOT NULL DEFAULT 0.0,
            holding_period_seconds REAL DEFAULT 0.0,
            strategy_name TEXT NOT NULL DEFAULT 'hybrid_ai',
            model_version TEXT DEFAULT 'v0.1.0',
            execution_mode TEXT NOT NULL DEFAULT 'paper',
            timestamp TEXT NOT NULL,
            metadata_json TEXT DEFAULT '{}',
            FOREIGN KEY (order_id) REFERENCES orders (order_id) ON DELETE SET NULL
        );
        CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
        CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
        CREATE INDEX IF NOT EXISTS idx_trades_model_version ON trades(model_version);
        CREATE INDEX IF NOT EXISTS idx_trades_execution_mode ON trades(execution_mode);

        -- 3. Portfolio snapshots table
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            cash_eur REAL NOT NULL,
            equity_eur REAL NOT NULL,
            realized_pnl_eur REAL NOT NULL DEFAULT 0.0,
            unrealized_pnl_eur REAL NOT NULL DEFAULT 0.0,
            total_pnl_eur REAL NOT NULL DEFAULT 0.0,
            cumulative_return_pct REAL NOT NULL DEFAULT 0.0,
            drawdown_pct REAL NOT NULL DEFAULT 0.0,
            benchmark_spy_price REAL,
            benchmark_cumulative_return_pct REAL,
            open_positions_count INTEGER NOT NULL DEFAULT 0,
            positions_json TEXT DEFAULT '[]',
            execution_mode TEXT NOT NULL DEFAULT 'paper'
        );
        CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON portfolio_snapshots(timestamp);
        CREATE INDEX IF NOT EXISTS idx_snapshots_execution_mode ON portfolio_snapshots(execution_mode);

        -- 4. Model versions registry table
        CREATE TABLE IF NOT EXISTS model_versions (
            version TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            parent_version TEXT,
            strategy_type TEXT NOT NULL DEFAULT 'hybrid',
            parameters_json TEXT NOT NULL DEFAULT '{}',
            metrics_json TEXT NOT NULL DEFAULT '{}',
            is_champion INTEGER NOT NULL DEFAULT 0,
            is_live_ready INTEGER NOT NULL DEFAULT 0,
            artifact_path TEXT DEFAULT '',
            training_sample_count INTEGER DEFAULT 0,
            notes TEXT DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_model_versions_created ON model_versions(created_at);
        CREATE INDEX IF NOT EXISTS idx_model_versions_champion ON model_versions(is_champion);

        -- 5. System logs table
        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            level TEXT NOT NULL,
            module TEXT NOT NULL,
            message TEXT NOT NULL,
            exception_trace TEXT,
            metadata_json TEXT DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp ON system_logs(timestamp);
        CREATE INDEX IF NOT EXISTS idx_system_logs_level ON system_logs(level);
        CREATE INDEX IF NOT EXISTS idx_system_logs_module ON system_logs(module);
        """
        with self._write_lock:
            conn = self.get_connection()
            conn.executescript(schema_sql)
            conn.commit()

    def checkpoint(self) -> None:
        """Forces a WAL checkpoint to flush pending transactions to disk."""
        if not self.is_memory:
            with self._write_lock:
                conn = self.get_connection()
                try:
                    conn.execute("PRAGMA wal_checkpoint(PASSIVE);")
                except sqlite3.Error as e:
                    logger.warning("Checkpoint warning: %s", e)

    def close(self) -> None:
        """Safely closes all active connections and truncates the WAL file."""
        with self._write_lock:
            with self._conn_lock:
                if not self.is_memory and self._all_connections:
                    try:
                        self._all_connections[0].execute("PRAGMA wal_checkpoint(TRUNCATE);")
                    except sqlite3.Error:
                        pass
                for conn in self._all_connections:
                    try:
                        conn.close()
                    except sqlite3.Error:
                        pass
                self._all_connections.clear()
            self._anchor_conn = None
            if hasattr(self._local, "conn"):
                self._local.conn = None

    def clear_all_tables(self) -> None:
        """Truncates all tables for testing or resetting backtest state."""
        with self._write_lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            for tbl in ["trades", "orders", "portfolio_snapshots", "model_versions", "system_logs"]:
                cursor.execute(f"DELETE FROM {tbl};")
            conn.commit()
            cursor.close()

    def __enter__(self) -> "DatabaseManager":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # =========================================================================
    # Orders API
    # =========================================================================
    def record_order(self, order: Any) -> str:
        """Inserts or updates an order record."""
        if isinstance(order, dict):
            d = order
        else:
            d = getattr(order, "__dict__", {})

        order_id = str(d.get("order_id") or uuid.uuid4().hex)
        client_order_id = str(d.get("client_order_id") or order_id)
        symbol = str(d.get("symbol", ""))
        side = getattr(d.get("side"), "value", str(d.get("side", "BUY")))
        order_type = getattr(d.get("order_type"), "value", str(d.get("order_type", "MARKET")))
        status = getattr(d.get("status"), "value", str(d.get("status", "PENDING")))
        req_qty = d.get("quantity")
        req_notional = d.get("notional")
        limit_px = d.get("limit_price")
        stop_px = d.get("stop_price")
        trail_pct = d.get("trail_percent")
        filled_qty = float(d.get("filled_quantity", 0.0) or 0.0)
        filled_avg_px = float(d.get("filled_avg_price", 0.0) or 0.0)
        fee_eur = float(d.get("fee_eur", 0.0) or 0.0)
        slippage_eur = float(d.get("slippage_eur", 0.0) or 0.0)
        created_at = _to_iso(d.get("timestamp") or d.get("created_at"))
        updated_at = _utc_now_iso()
        message = str(d.get("message", ""))
        is_live = 1 if d.get("is_live") else 0

        sql = """
        INSERT INTO orders (
            order_id, client_order_id, symbol, side, order_type, status,
            requested_quantity, requested_notional, limit_price, stop_price, trail_percent,
            filled_quantity, filled_avg_price, fee_eur, slippage_eur,
            created_at, updated_at, message, is_live
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(order_id) DO UPDATE SET
            status = excluded.status,
            filled_quantity = excluded.filled_quantity,
            filled_avg_price = excluded.filled_avg_price,
            fee_eur = excluded.fee_eur,
            slippage_eur = excluded.slippage_eur,
            updated_at = excluded.updated_at,
            message = excluded.message;
        """
        params = (
            order_id, client_order_id, symbol, side, order_type, status,
            req_qty, req_notional, limit_px, stop_px, trail_pct,
            filled_qty, filled_avg_px, fee_eur, slippage_eur,
            created_at, updated_at, message, is_live,
        )
        with self._write_lock:
            conn = self.get_connection()
            conn.execute(sql, params)
            conn.commit()
        return order_id

    def update_order_status(
        self,
        order_id: str,
        status: Any,
        filled_quantity: Optional[float] = None,
        filled_avg_price: Optional[float] = None,
        fee_eur: Optional[float] = None,
        slippage_eur: Optional[float] = None,
        message: Optional[str] = None,
    ) -> bool:
        """Updates status and fill fields of an existing order."""
        status_str = getattr(status, "value", str(status))
        updates = ["status = ?", "updated_at = ?"]
        params: List[Any] = [status_str, _utc_now_iso()]

        if filled_quantity is not None:
            updates.append("filled_quantity = ?")
            params.append(float(filled_quantity))
        if filled_avg_price is not None:
            updates.append("filled_avg_price = ?")
            params.append(float(filled_avg_price))
        if fee_eur is not None:
            updates.append("fee_eur = ?")
            params.append(float(fee_eur))
        if slippage_eur is not None:
            updates.append("slippage_eur = ?")
            params.append(float(slippage_eur))
        if message is not None:
            updates.append("message = ?")
            params.append(str(message))

        params.append(order_id)
        sql = f"UPDATE orders SET {', '.join(updates)} WHERE order_id = ?;"

        with self._write_lock:
            conn = self.get_connection()
            cursor = conn.execute(sql, params)
            conn.commit()
            return cursor.rowcount > 0

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single order by ID."""
        conn = self.get_connection()
        row = conn.execute("SELECT * FROM orders WHERE order_id = ?;", (order_id,)).fetchone()
        return dict(row) if row else None

    def get_orders(
        self,
        symbol: Optional[str] = None,
        status: Optional[Any] = None,
        is_live: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Queries orders with optional filters, sorted by created_at DESC."""
        conditions = []
        params: List[Any] = []

        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if status:
            conditions.append("status = ?")
            params.append(getattr(status, "value", str(status)))
        if is_live is not None:
            conditions.append("is_live = ?")
            params.append(1 if is_live else 0)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM orders {where_clause} ORDER BY created_at DESC LIMIT ? OFFSET ?;"
        params.extend([limit, offset])

        conn = self.get_connection()
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # =========================================================================
    # Trades API
    # =========================================================================
    def record_trade(self, trade: Any) -> str:
        """Records an executed trade fill."""
        if isinstance(trade, dict):
            d = trade
        else:
            d = getattr(trade, "__dict__", {})

        trade_id = str(d.get("trade_id") or uuid.uuid4().hex)
        order_id = d.get("order_id")
        symbol = str(d.get("symbol", ""))
        side = getattr(d.get("side"), "value", str(d.get("side", "BUY")))
        qty = float(d.get("quantity", 0.0))
        price = float(d.get("price", 0.0))
        fee_eur = float(d.get("fee_eur", 0.0) or 0.0)
        slippage_eur = float(d.get("slippage_eur", 0.0) or 0.0)
        realized_pnl = float(d.get("realized_pnl_eur", 0.0) or 0.0)
        pnl_pct = float(d.get("pnl_percent", 0.0) or 0.0)
        holding_sec = float(d.get("holding_period_seconds", 0.0) or 0.0)
        strategy_name = str(d.get("strategy_name", "hybrid_ai"))
        model_version = str(d.get("model_version", "v0.1.0"))
        execution_mode = str(d.get("execution_mode", "paper"))
        timestamp = _to_iso(d.get("timestamp"))

        meta = d.get("metadata", d.get("signal_metadata", d.get("metadata_json", {})))
        meta_json = json.dumps(meta) if isinstance(meta, dict) else str(meta)

        sql = """
        INSERT INTO trades (
            trade_id, order_id, symbol, side, quantity, price,
            fee_eur, slippage_eur, realized_pnl_eur, pnl_percent,
            holding_period_seconds, strategy_name, model_version,
            execution_mode, timestamp, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        params = (
            trade_id, order_id, symbol, side, qty, price,
            fee_eur, slippage_eur, realized_pnl, pnl_pct,
            holding_sec, strategy_name, model_version,
            execution_mode, timestamp, meta_json,
        )
        with self._write_lock:
            conn = self.get_connection()
            conn.execute(sql, params)
            conn.commit()
        return trade_id

    def get_trades(
        self,
        symbol: Optional[str] = None,
        execution_mode: Optional[str] = None,
        model_version: Optional[str] = None,
        start_time: Optional[Union[datetime, str]] = None,
        end_time: Optional[Union[datetime, str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Queries executed trades ordered by timestamp DESC."""
        conditions = []
        params: List[Any] = []

        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if execution_mode:
            conditions.append("execution_mode = ?")
            params.append(execution_mode)
        if model_version:
            conditions.append("model_version = ?")
            params.append(model_version)
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(_to_iso(start_time))
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(_to_iso(end_time))

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM trades {where_clause} ORDER BY timestamp DESC LIMIT ? OFFSET ?;"
        params.extend([limit, offset])

        conn = self.get_connection()
        rows = conn.execute(sql, params).fetchall()
        result = []
        for r in rows:
            item = dict(r)
            try:
                item["metadata"] = json.loads(item.get("metadata_json") or "{}")
            except Exception:
                item["metadata"] = {}
            result.append(item)
        return result

    def get_trade_statistics(
        self,
        execution_mode: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculates aggregated performance statistics over closed trades."""
        conditions = []
        params: List[Any] = []
        if execution_mode:
            conditions.append("execution_mode = ?")
            params.append(execution_mode)
        if model_version:
            conditions.append("model_version = ?")
            params.append(model_version)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"""
        SELECT
            COUNT(*) AS total_trades,
            SUM(CASE WHEN realized_pnl_eur > 0 THEN 1 ELSE 0 END) AS winning_trades,
            SUM(CASE WHEN realized_pnl_eur < 0 THEN 1 ELSE 0 END) AS losing_trades,
            COALESCE(SUM(realized_pnl_eur), 0.0) AS total_realized_pnl,
            COALESCE(SUM(fee_eur), 0.0) AS total_fees,
            COALESCE(SUM(slippage_eur), 0.0) AS total_slippage,
            COALESCE(SUM(CASE WHEN realized_pnl_eur > 0 THEN realized_pnl_eur ELSE 0 END), 0.0) AS gross_profit,
            COALESCE(ABS(SUM(CASE WHEN realized_pnl_eur < 0 THEN realized_pnl_eur ELSE 0 END)), 0.0) AS gross_loss
        FROM trades {where_clause};
        """
        conn = self.get_connection()
        row = conn.execute(sql, params).fetchone()
        stats = dict(row) if row else {}

        total = stats.get("total_trades", 0)
        wins = stats.get("winning_trades", 0)
        win_rate = (wins / total) if total > 0 else 0.0
        gross_profit = stats.get("gross_profit", 0.0)
        gross_loss = stats.get("gross_loss", 0.0)
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (100.0 if gross_profit > 0 else 1.0)

        stats["win_rate"] = round(win_rate, 4)
        stats["profit_factor"] = round(profit_factor, 4)
        return stats

    # =========================================================================
    # Portfolio Snapshots API
    # =========================================================================
    def record_portfolio_snapshot(self, snapshot: Any) -> int:
        """Records an account equity snapshot."""
        if isinstance(snapshot, dict):
            d = snapshot
        else:
            d = getattr(snapshot, "__dict__", {})

        timestamp = _to_iso(d.get("timestamp"))
        cash_eur = float(d.get("cash_eur", 0.0))
        equity_eur = float(d.get("equity_eur", 0.0))
        realized_pnl = float(d.get("realized_pnl_eur", 0.0) or 0.0)
        unrealized_pnl = float(d.get("unrealized_pnl_eur", 0.0) or 0.0)
        total_pnl = float(d.get("total_pnl_eur", realized_pnl + unrealized_pnl))

        init_cap = float(d.get("initial_capital_eur", 100.0) or 100.0)
        cum_ret = float(d.get("cumulative_return_pct", ((equity_eur - init_cap) / init_cap * 100.0) if init_cap else 0.0))
        drawdown_pct = float(d.get("drawdown_pct", 0.0) or 0.0)
        spy_price = d.get("benchmark_spy_price", d.get("benchmark_sp500_price"))
        spy_cum_ret = d.get("benchmark_cumulative_return_pct", d.get("benchmark_return_pct"))

        positions = d.get("positions", [])
        if isinstance(positions, list):
            pos_json = json.dumps([p if isinstance(p, dict) else getattr(p, "__dict__", {}) for p in positions])
            pos_count = len(positions)
        else:
            pos_json = str(positions)
            pos_count = int(d.get("open_positions_count", 0))

        execution_mode = str(d.get("execution_mode", "live" if d.get("is_live") else "paper"))

        sql = """
        INSERT INTO portfolio_snapshots (
            timestamp, cash_eur, equity_eur, realized_pnl_eur, unrealized_pnl_eur,
            total_pnl_eur, cumulative_return_pct, drawdown_pct,
            benchmark_spy_price, benchmark_cumulative_return_pct,
            open_positions_count, positions_json, execution_mode
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        params = (
            timestamp, cash_eur, equity_eur, realized_pnl, unrealized_pnl,
            total_pnl, cum_ret, drawdown_pct,
            spy_price, spy_cum_ret, pos_count, pos_json, execution_mode,
        )
        with self._write_lock:
            conn = self.get_connection()
            cursor = conn.execute(sql, params)
            conn.commit()
            return cursor.lastrowid

    def get_portfolio_history(
        self,
        execution_mode: Optional[str] = None,
        start_time: Optional[Union[datetime, str]] = None,
        end_time: Optional[Union[datetime, str]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Queries portfolio snapshots in chronological order (ASC)."""
        conditions = []
        params: List[Any] = []

        if execution_mode:
            conditions.append("execution_mode = ?")
            params.append(execution_mode)
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(_to_iso(start_time))
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(_to_iso(end_time))

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""
        sql = f"SELECT * FROM portfolio_snapshots {where_clause} ORDER BY timestamp ASC {limit_clause};"

        conn = self.get_connection()
        rows = conn.execute(sql, params).fetchall()
        result = []
        for r in rows:
            item = dict(r)
            try:
                item["positions"] = json.loads(item.get("positions_json") or "[]")
            except Exception:
                item["positions"] = []
            result.append(item)
        return result

    def get_portfolio_snapshots(
        self,
        execution_mode: Optional[str] = None,
        start_time: Optional[Union[datetime, str]] = None,
        end_time: Optional[Union[datetime, str]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Alias for get_portfolio_history adhering to canonical API naming."""
        return self.get_portfolio_history(execution_mode, start_time, end_time, limit)

    def get_latest_portfolio_snapshot(
        self,
        execution_mode: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Retrieves the most recent portfolio snapshot."""
        conditions = []
        params: List[Any] = []
        if execution_mode:
            conditions.append("execution_mode = ?")
            params.append(execution_mode)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM portfolio_snapshots {where_clause} ORDER BY timestamp DESC LIMIT 1;"

        conn = self.get_connection()
        row = conn.execute(sql, params).fetchone()
        if not row:
            return None
        item = dict(row)
        try:
            item["positions"] = json.loads(item.get("positions_json") or "[]")
        except Exception:
            item["positions"] = []
        return item

    # =========================================================================
    # Model Versions API (R5)
    # =========================================================================
    def record_model_version(
        self,
        version: str,
        parent_version: Optional[str] = None,
        strategy_type: str = "hybrid",
        parameters: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        is_champion: bool = False,
        is_live_ready: bool = False,
        artifact_path: str = "",
        training_sample_count: int = 0,
        notes: str = "",
    ) -> None:
        """Records a model version in the registry."""
        params_json = json.dumps(parameters or {})
        metrics_json = json.dumps(metrics or {})
        created_at = _utc_now_iso()
        champ_int = 1 if is_champion else 0
        ready_int = 1 if is_live_ready else 0

        with self._write_lock:
            conn = self.get_connection()
            if is_champion:
                conn.execute("UPDATE model_versions SET is_champion = 0 WHERE is_champion = 1;")

            sql = """
            INSERT INTO model_versions (
                version, created_at, parent_version, strategy_type,
                parameters_json, metrics_json, is_champion, is_live_ready,
                artifact_path, training_sample_count, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(version) DO UPDATE SET
                metrics_json = excluded.metrics_json,
                is_champion = excluded.is_champion,
                is_live_ready = excluded.is_live_ready,
                artifact_path = excluded.artifact_path,
                training_sample_count = excluded.training_sample_count,
                notes = excluded.notes;
            """
            conn.execute(sql, (
                version, created_at, parent_version, strategy_type,
                params_json, metrics_json, champ_int, ready_int,
                artifact_path, training_sample_count, notes
            ))
            conn.commit()

    def get_model_versions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Lists registered model versions ordered by created_at DESC."""
        conn = self.get_connection()
        rows = conn.execute(
            "SELECT * FROM model_versions ORDER BY created_at DESC LIMIT ?;", (limit,)
        ).fetchall()
        result = []
        for r in rows:
            item = dict(r)
            item["parameters"] = json.loads(item.get("parameters_json") or "{}")
            item["metrics"] = json.loads(item.get("metrics_json") or "{}")
            item["is_champion"] = bool(item.get("is_champion"))
            item["is_live_ready"] = bool(item.get("is_live_ready"))
            result.append(item)
        return result

    def get_champion_model_version(self) -> Optional[Dict[str, Any]]:
        """Retrieves current champion model version record."""
        conn = self.get_connection()
        row = conn.execute("SELECT * FROM model_versions WHERE is_champion = 1 LIMIT 1;").fetchone()
        if not row:
            return None
        item = dict(row)
        item["parameters"] = json.loads(item.get("parameters_json") or "{}")
        item["metrics"] = json.loads(item.get("metrics_json") or "{}")
        item["is_champion"] = bool(item.get("is_champion"))
        item["is_live_ready"] = bool(item.get("is_live_ready"))
        return item

    # =========================================================================
    # System Logs API
    # =========================================================================
    def record_log(
        self,
        level: str,
        module: str,
        message: str,
        exception_trace: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Records a structured system log entry."""
        timestamp = _utc_now_iso()
        meta_json = json.dumps(metadata or {})
        sql = """
        INSERT INTO system_logs (timestamp, level, module, message, exception_trace, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?);
        """
        with self._write_lock:
            conn = self.get_connection()
            cursor = conn.execute(sql, (timestamp, level.upper(), module, message, exception_trace, meta_json))
            conn.commit()
            return cursor.lastrowid

    def get_logs(
        self,
        level: Optional[str] = None,
        module: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Queries system logs ordered by timestamp DESC."""
        conditions = []
        params: List[Any] = []
        if level:
            conditions.append("level = ?")
            params.append(level.upper())
        if module:
            conditions.append("module LIKE ?")
            params.append(f"%{module}%")

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM system_logs {where_clause} ORDER BY timestamp DESC LIMIT ?;"
        params.append(limit)

        conn = self.get_connection()
        rows = conn.execute(sql, params).fetchall()
        result = []
        for r in rows:
            item = dict(r)
            try:
                item["metadata"] = json.loads(item.get("metadata_json") or "{}")
            except Exception:
                item["metadata"] = {}
            result.append(item)
        return result
