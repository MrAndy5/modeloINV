"""Public REST client for Binance market data.

Provides free cryptocurrency klines and EUR/USD ticker rates without API keys.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import requests

from modeloinv.core.models import AssetClass, Bar
from modeloinv.data.base import (
    BaseDataProvider,
    DataProviderError,
    DataValidationError,
    ProviderConnectionError,
    RateLimitExceededError,
    SymbolNotFoundError,
)

logger = logging.getLogger(__name__)


class BinanceDataProvider(BaseDataProvider):
    """Public REST client for Binance market data without API keys."""

    BASE_URLS = [
        "https://api.binance.com/api/v3",
        "https://api1.binance.com/api/v3",
        "https://api2.binance.com/api/v3",
    ]
    TIMEFRAME_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "1h": "1h",
        "1d": "1d",
        "1D": "1d",
    }

    def __init__(self, timeout: float = 10.0, max_retries: int = 3) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()

    @property
    def name(self) -> str:
        return "binance"

    @property
    def supported_asset_classes(self) -> Set[AssetClass]:
        return {AssetClass.CRYPTO}

    def _normalize_symbol(self, symbol: str) -> str:
        return symbol.upper().replace("/", "").replace("-", "").strip()

    def _request_klines(self, params: Dict[str, Any]) -> List[List[Any]]:
        """Sends GET request to Binance klines endpoint with fallback hosts."""
        last_error = None
        for attempt in range(self.max_retries):
            base_url = self.BASE_URLS[attempt % len(self.BASE_URLS)]
            url = f"{base_url}/klines"
            try:
                res = self._session.get(url, params=params, timeout=self.timeout)
                if res.status_code == 200:
                    return res.json()
                elif res.status_code == 400:
                    err = res.json()
                    if err.get("code") == -1121:
                        raise SymbolNotFoundError(f"Symbol '{params.get('symbol')}' not found on Binance.")
                    raise DataValidationError(f"Binance invalid request: {err.get('msg')}")
                elif res.status_code in (418, 429):
                    wait_sec = 2 ** attempt
                    logger.warning(f"Binance rate limit ({res.status_code}). Backing off {wait_sec}s...")
                    time.sleep(wait_sec)
                    continue
                else:
                    last_error = ProviderConnectionError(f"Binance API HTTP {res.status_code}: {res.text[:200]}")
            except (requests.ConnectionError, requests.Timeout) as e:
                last_error = ProviderConnectionError(f"Connection error to Binance: {e}")
                time.sleep(1)

        if last_error:
            raise last_error
        raise RateLimitExceededError("Binance max retries exceeded.")

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[Bar]:
        interval = self.TIMEFRAME_MAP.get(timeframe)
        if not interval:
            raise DataValidationError(f"Unsupported timeframe '{timeframe}' for Binance")
        norm_symbol = self._normalize_symbol(symbol)

        target_limit = limit or 1000
        bars: List[Bar] = []
        current_start_ms = int(start.timestamp() * 1000) if start else None
        end_ms = int(end.timestamp() * 1000) if end else None

        while True:
            chunk_limit = min(1000, target_limit - len(bars)) if limit else 1000
            params: Dict[str, Any] = {
                "symbol": norm_symbol,
                "interval": interval,
                "limit": chunk_limit,
            }
            if current_start_ms:
                params["startTime"] = current_start_ms
            if end_ms:
                params["endTime"] = end_ms

            chunk = self._request_klines(params)
            if not chunk:
                break

            for k in chunk:
                # Structure: [open_time, open, high, low, close, volume, close_time, ...]
                ts = datetime.fromtimestamp(k[0] / 1000.0, tz=timezone.utc)
                o = float(k[1])
                h = float(k[2])
                l = float(k[3])
                c = float(k[4])
                v = float(k[5])

                bar = Bar(
                    timestamp=ts,
                    symbol=norm_symbol,
                    open=o,
                    high=max(h, max(o, c)),
                    low=min(l, min(o, c)),
                    close=c,
                    volume=max(0.0, v),
                    asset_class=AssetClass.CRYPTO,
                    currency="USDT" if norm_symbol.endswith("USDT") else "USD",
                )
                bars.append(self.validate_bar(bar))

            if len(chunk) < chunk_limit or (limit and len(bars) >= target_limit):
                break
            current_start_ms = chunk[-1][6] + 1
            if end_ms and current_start_ms >= end_ms:
                break

        return bars[:limit] if limit else bars

    def get_latest_bar(self, symbol: str, timeframe: str = "1m") -> Bar:
        bars = self.get_historical_bars(symbol, timeframe=timeframe, limit=1)
        if not bars:
            raise SymbolNotFoundError(f"Symbol '{symbol}' returned no klines on Binance")
        return bars[-1]

    def get_fx_rate(self, base_currency: str = "EUR", quote_currency: str = "USD") -> float:
        base = base_currency.upper()
        quote = quote_currency.upper()
        if base == quote:
            return 1.0

        if base == "EUR" and quote in {"USD", "USDT"}:
            url = f"{self.BASE_URLS[0]}/ticker/price"
            try:
                res = self._session.get(url, params={"symbol": "EURUSDT"}, timeout=self.timeout)
                if res.status_code == 200:
                    return float(res.json()["price"])
            except Exception as e:
                logger.warning(f"Binance FX rate fetch failed: {e}")

        raise DataProviderError(f"Binance FX conversion not supported for {base}/{quote}")
