"""Columnar Parquet historical bar cache using PyArrow.

Provides high-speed disk caching for historical OHLCV bars with timestamp
indexing, date range slicing, idempotent upserting, deduplication, and
Windows filesystem compatibility.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from modeloinv.core.models import AssetClass, Bar

logger = logging.getLogger(__name__)

BAR_ARROW_SCHEMA = pa.schema([
    ("timestamp", pa.timestamp("us", tz="UTC")),
    ("symbol", pa.string()),
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("volume", pa.float64()),
    ("asset_class", pa.string()),
    ("currency", pa.string()),
])


def normalize_datetime_utc(dt: Union[datetime, str, None]) -> Optional[datetime]:
    """Ensures datetime is a timezone-aware UTC datetime."""
    if dt is None:
        return None
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def sanitize_filename_token(token: str) -> str:
    """Sanitizes symbol tokens for Windows filesystem paths."""
    # Replace invalid Windows filename characters: \ / : * ? " < > |
    sanitized = re.sub(r'[\\/:*?"<>|=]', "_", token)
    return sanitized


class ParquetBarCache:
    """High-performance local Parquet cache for historical market bars."""

    def __init__(self, cache_dir: Union[str, Path] = "data/cache") -> None:
        """Initializes the Parquet bar cache directory.

        Args:
            cache_dir: Directory where parquet files are saved.
        """
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def get_cache_path(self, symbol: str, timeframe: str) -> Path:
        """Returns the filesystem path for a symbol and timeframe.

        Handles Windows-illegal characters in symbols like EUR/USD or BINANCE:BTCUSDT.
        """
        clean_symbol = sanitize_filename_token(symbol)
        clean_tf = sanitize_filename_token(timeframe)
        return self.cache_dir / f"{clean_symbol}_{clean_tf}.parquet"

    def has_cache(self, symbol: str, timeframe: str) -> bool:
        """Checks if a non-empty cache file exists for the symbol and timeframe."""
        path = self.get_cache_path(symbol, timeframe)
        return path.exists() and path.stat().st_size > 0

    def get_cached_range(self, symbol: str, timeframe: str) -> Optional[Tuple[datetime, datetime]]:
        """Returns (min_timestamp, max_timestamp) from file metadata in <0.1ms."""
        path = self.get_cache_path(symbol, timeframe)
        if not path.exists() or path.stat().st_size == 0:
            return None

        with self._lock:
            try:
                meta = pq.read_metadata(path)
                if meta.num_rows == 0:
                    return None

                # Attempt to read directly from row-group column statistics
                col_idx = meta.schema.names.index("timestamp")
                first_rg_stat = meta.row_group(0).column(col_idx).statistics
                last_rg_stat = meta.row_group(meta.num_row_groups - 1).column(col_idx).statistics

                if first_rg_stat and last_rg_stat and first_rg_stat.min is not None and last_rg_stat.max is not None:
                    min_ts = first_rg_stat.min
                    max_ts = last_rg_stat.max
                    if hasattr(min_ts, "to_pydatetime"):
                        min_ts = min_ts.to_pydatetime()
                    if hasattr(max_ts, "to_pydatetime"):
                        max_ts = max_ts.to_pydatetime()
                    return normalize_datetime_utc(min_ts), normalize_datetime_utc(max_ts)

                # Fallback: read only the timestamp column
                tbl = pq.read_table(path, columns=["timestamp"])
                ts_col = tbl["timestamp"].to_pylist()
                if not ts_col:
                    return None
                return normalize_datetime_utc(min(ts_col)), normalize_datetime_utc(max(ts_col))
            except Exception as e:
                logger.warning("Failed to read Parquet metadata for %s: %s", path, e)
                return None

    def write_bars(
        self,
        symbol: str,
        timeframe: str,
        bars: Sequence[Any],
    ) -> int:
        """Idempotently writes or merges bars into the Parquet cache.

        Deduplicates by timestamp, sorts chronologically, and writes atomically.

        Args:
            symbol: Ticker symbol.
            timeframe: Bar timeframe (e.g. '1m', '5m', '1d').
            bars: Sequence of Bar dataclass instances or dicts.

        Returns:
            Total count of bars stored in the cache file.
        """
        if not bars:
            return 0

        target_path = self.get_cache_path(symbol, timeframe)

        ts_list = []
        sym_list = []
        open_list = []
        high_list = []
        low_list = []
        close_list = []
        vol_list = []
        asset_list = []
        curr_list = []

        for b in bars:
            if isinstance(b, dict):
                dt = normalize_datetime_utc(b.get("timestamp"))
                sym = str(b.get("symbol", symbol))
                o = float(b.get("open", 0.0))
                h = float(b.get("high", 0.0))
                l = float(b.get("low", 0.0))
                c = float(b.get("close", 0.0))
                v = float(b.get("volume", 0.0))
                ac = str(getattr(b.get("asset_class"), "value", b.get("asset_class", "EQUITY")))
                cur = str(b.get("currency", "USD"))
            else:
                dt = normalize_datetime_utc(getattr(b, "timestamp"))
                sym = str(getattr(b, "symbol", symbol))
                o = float(getattr(b, "open", 0.0))
                h = float(getattr(b, "high", 0.0))
                l = float(getattr(b, "low", 0.0))
                c = float(getattr(b, "close", 0.0))
                v = float(getattr(b, "volume", 0.0))
                ac_raw = getattr(b, "asset_class", "EQUITY")
                ac = str(getattr(ac_raw, "value", ac_raw))
                cur = str(getattr(b, "currency", "USD"))

            ts_list.append(dt)
            sym_list.append(sym)
            open_list.append(o)
            high_list.append(h)
            low_list.append(l)
            close_list.append(c)
            vol_list.append(v)
            asset_list.append(ac)
            curr_list.append(cur)

        new_data = {
            "timestamp": ts_list,
            "symbol": sym_list,
            "open": open_list,
            "high": high_list,
            "low": low_list,
            "close": close_list,
            "volume": vol_list,
            "asset_class": asset_list,
            "currency": curr_list,
        }
        new_df = pd.DataFrame(new_data)

        with self._lock:
            # Merge with existing cache if present
            if target_path.exists() and target_path.stat().st_size > 0:
                try:
                    existing_tbl = pq.read_table(target_path)
                    existing_df = existing_tbl.to_pandas()
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                except Exception as e:
                    logger.warning("Corrupt parquet file detected, overwriting %s: %s", target_path, e)
                    combined_df = new_df
            else:
                combined_df = new_df

            # Deduplicate by timestamp and sort
            combined_df = combined_df.drop_duplicates(subset=["timestamp"], keep="last")
            combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)

            out_tbl = pa.Table.from_pandas(combined_df, schema=BAR_ARROW_SCHEMA, preserve_index=False)

            # Atomic write via temp file with retry loop for Windows file locking
            tmp_path = target_path.with_name(f"{target_path.name}.tmp.{uuid.uuid4().hex}")
            try:
                pq.write_table(out_tbl, tmp_path, compression="SNAPPY")
                max_retries = 10
                for attempt in range(max_retries):
                    try:
                        os.replace(tmp_path, target_path)
                        break
                    except (PermissionError, OSError):
                        if attempt == max_retries - 1:
                            raise
                        time.sleep(0.01 * (2 ** attempt))
            finally:
                if tmp_path.exists():
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

            return len(combined_df)

    def read_bars(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[Union[datetime, str]] = None,
        end: Optional[Union[datetime, str]] = None,
    ) -> List[Bar]:
        """Reads historical bars with predicate pushdown date range filtering."""
        target_path = self.get_cache_path(symbol, timeframe)
        if not target_path.exists() or target_path.stat().st_size == 0:
            return []

        start_dt = normalize_datetime_utc(start)
        end_dt = normalize_datetime_utc(end)

        filters = []
        if start_dt is not None:
            filters.append(("timestamp", ">=", start_dt))
        if end_dt is not None:
            filters.append(("timestamp", "<=", end_dt))

        with self._lock:
            try:
                if filters:
                    tbl = pq.read_table(target_path, filters=filters)
                else:
                    tbl = pq.read_table(target_path)
            except Exception as e:
                logger.error("Error reading Parquet cache %s: %s", target_path, e)
                return []

        if tbl.num_rows == 0:
            return []

        pydict = tbl.to_pydict()
        bars_out: List[Bar] = []

        for i in range(tbl.num_rows):
            ts = pydict["timestamp"][i]
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()
            ts = normalize_datetime_utc(ts)

            ac_val = pydict["asset_class"][i]
            try:
                ac_enum = AssetClass(ac_val)
            except Exception:
                ac_enum = AssetClass.EQUITY

            bars_out.append(
                Bar(
                    timestamp=ts,
                    symbol=pydict["symbol"][i],
                    open=pydict["open"][i],
                    high=pydict["high"][i],
                    low=pydict["low"][i],
                    close=pydict["close"][i],
                    volume=pydict["volume"][i],
                    asset_class=ac_enum,
                    currency=pydict["currency"][i],
                )
            )

        return bars_out

    def read_dataframe(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[Union[datetime, str]] = None,
        end: Optional[Union[datetime, str]] = None,
        set_index: bool = True,
    ) -> pd.DataFrame:
        """Reads historical bars directly into a Pandas DataFrame."""
        target_path = self.get_cache_path(symbol, timeframe)
        if not target_path.exists() or target_path.stat().st_size == 0:
            df = pd.DataFrame(columns=[
                "timestamp", "symbol", "open", "high", "low", "close", "volume", "asset_class", "currency"
            ])
            if set_index:
                df = df.set_index("timestamp")
            return df

        start_dt = normalize_datetime_utc(start)
        end_dt = normalize_datetime_utc(end)

        filters = []
        if start_dt is not None:
            filters.append(("timestamp", ">=", start_dt))
        if end_dt is not None:
            filters.append(("timestamp", "<=", end_dt))

        with self._lock:
            try:
                if filters:
                    tbl = pq.read_table(target_path, filters=filters)
                else:
                    tbl = pq.read_table(target_path)
            except Exception as e:
                logger.error("Error reading Parquet dataframe %s: %s", target_path, e)
                df = pd.DataFrame(columns=[
                    "timestamp", "symbol", "open", "high", "low", "close", "volume", "asset_class", "currency"
                ])
                if set_index:
                    df = df.set_index("timestamp")
                return df

        df = tbl.to_pandas()
        if set_index and "timestamp" in df.columns:
            df = df.set_index("timestamp")
        return df

    def clear_cache(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> int:
        """Cleans cached Parquet files."""
        count = 0
        with self._lock:
            if symbol and timeframe:
                path = self.get_cache_path(symbol, timeframe)
                if path.exists():
                    path.unlink()
                    count = 1
            elif symbol:
                clean_sym = sanitize_filename_token(symbol)
                for f in self.cache_dir.glob(f"{clean_sym}_*.parquet"):
                    f.unlink()
                    count += 1
            elif timeframe:
                clean_tf = sanitize_filename_token(timeframe)
                for f in self.cache_dir.glob(f"*_{clean_tf}.parquet"):
                    f.unlink()
                    count += 1
            else:
                for f in self.cache_dir.glob("*.parquet"):
                    f.unlink()
                    count += 1
        return count
