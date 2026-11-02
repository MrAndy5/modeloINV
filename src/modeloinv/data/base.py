"""Base Abstract Data Provider and Exception Hierarchy for modeloINV.

Enforces uniform contracts, mathematical boundary validation, and
asset-class inference across all market data sources.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from modeloinv.core.models import AssetClass, Bar


class DataProviderError(Exception):
    """Base exception for all data provider failures."""
    pass


class SymbolNotFoundError(DataProviderError):
    """Raised when an asset ticker cannot be found or is delisted."""
    pass


class RateLimitExceededError(DataProviderError):
    """Raised when API queries exceed rate limits (HTTP 429 / 418)."""
    pass


class ProviderConnectionError(DataProviderError):
    """Raised on network failures, timeouts, or remote HTTP 5xx errors."""
    pass


class ProviderAuthenticationError(DataProviderError):
    """Raised when API credentials are missing, rejected, or invalid."""
    pass


class DataValidationError(DataProviderError):
    """Raised when incoming market data violates price/volume invariants."""
    pass


class BaseDataProvider(ABC):
    """Abstract Base Class for all market data providers in modeloINV."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider unique identifier (e.g. 'yahoo', 'binance', 'alpaca', 'mock', 'cascade')."""
        pass

    @property
    @abstractmethod
    def supported_asset_classes(self) -> Set[AssetClass]:
        """Set of asset classes natively supported by this provider."""
        pass

    @abstractmethod
    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[Bar]:
        """Fetch historical OHLCV bars sorted chronologically (earliest to latest)."""
        pass

    @abstractmethod
    def get_latest_bar(self, symbol: str, timeframe: str = "1m") -> Bar:
        """Fetch the most recent bar for the given symbol."""
        pass

    @abstractmethod
    def get_fx_rate(self, base_currency: str = "EUR", quote_currency: str = "USD") -> float:
        """Fetch current spot FX conversion rate (e.g., EUR/USD)."""
        pass

    def validate_bar(self, bar: Bar) -> Bar:
        """Enforce mathematical and structural invariants on a Bar instance.

        Invariants:
        1. All prices must be strictly positive: open, high, low, close > 0
        2. High must bound extremes: high >= max(open, close)
        3. Low must bound extremes: low <= min(open, close)
        4. Volume must be non-negative: volume >= 0.0
        5. Timestamp must have UTC timezone information.
        """
        for field_name, val in [
            ("open", bar.open),
            ("high", bar.high),
            ("low", bar.low),
            ("close", bar.close),
            ("volume", bar.volume),
        ]:
            if val is None or math.isnan(val) or math.isinf(val):
                raise DataValidationError(f"Bar {field_name} cannot be None, NaN or Inf for {bar.symbol}: {val}")

        if bar.open <= 0 or bar.high <= 0 or bar.low <= 0 or bar.close <= 0:
            raise DataValidationError(f"Invalid non-positive price in bar: {bar}")
        if bar.high < max(bar.open, bar.close) - 1e-6:
            raise DataValidationError(f"High {bar.high} is less than max(open, close): {bar}")
        if bar.low > min(bar.open, bar.close) + 1e-6:
            raise DataValidationError(f"Low {bar.low} is greater than min(open, close): {bar}")
        if bar.volume < 0.0:
            raise DataValidationError(f"Negative volume in bar: {bar}")
        if bar.timestamp.tzinfo is None:
            bar = Bar(
                timestamp=bar.timestamp.replace(tzinfo=timezone.utc),
                symbol=bar.symbol,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                asset_class=bar.asset_class,
                currency=bar.currency,
            )
        return bar

    @staticmethod
    def infer_asset_class(symbol: str) -> AssetClass:
        """Heuristic rule engine classifying symbols into AssetClass."""
        s = symbol.upper().strip()
        if s.endswith("=X") or (len(s) == 6 and s in {"EURUSD", "GBPUSD", "USDJPY", "EURGBP"}):
            return AssetClass.FOREX
        if s.endswith(("USDT", "BUSD", "USDC", "BTC", "ETH")) or s in {"BTC", "ETH", "SOL"}:
            return AssetClass.CRYPTO
        if "/" in s or "-" in s:
            parts = s.replace("-", "/").split("/")
            if len(parts) == 2:
                if parts[1] in {"USDT", "BUSD", "USDC", "BTC", "ETH", "USD"} and parts[0] in {"BTC", "ETH", "SOL", "BNB", "XRP"}:
                    return AssetClass.CRYPTO
                if len(parts[0]) == 3 and len(parts[1]) == 3:
                    return AssetClass.FOREX
        if s in {"SPY", "QQQ", "IWM", "VTI", "VOO", "XLK", "XLF", "XLE", "XLV", "DIA", "EEM"}:
            return AssetClass.ETF
        return AssetClass.EQUITY

    @staticmethod
    def bars_to_dataframe(bars: List[Bar]) -> pd.DataFrame:
        """Convert a list of Bar objects into a pandas DataFrame indexed by UTC timestamp."""
        if not bars:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "asset_class", "currency"])
        records = [
            {
                "timestamp": b.timestamp,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
                "asset_class": b.asset_class.value if isinstance(b.asset_class, AssetClass) else b.asset_class,
                "currency": b.currency,
            }
            for b in bars
        ]
        df = pd.DataFrame.from_records(records)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        return df

    @staticmethod
    def dataframe_to_bars(
        df: pd.DataFrame,
        symbol: str,
        asset_class: Optional[AssetClass] = None,
        currency: str = "USD",
    ) -> List[Bar]:
        """Convert a pandas DataFrame back into a List[Bar]."""
        inferred_asset = asset_class or BaseDataProvider.infer_asset_class(symbol)
        bars: List[Bar] = []
        for ts, row in df.iterrows():
            timestamp = ts if isinstance(ts, datetime) else pd.to_datetime(ts).to_pydatetime()
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            bars.append(
                Bar(
                    timestamp=timestamp,
                    symbol=symbol,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume", 0.0)),
                    asset_class=inferred_asset,
                    currency=currency,
                )
            )
        return bars
