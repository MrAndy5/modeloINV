"""Direct REST client for Yahoo Finance v8 Chart API.

Fetches free equity, ETF, and Forex market data without third-party wrapper dependencies.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

import requests

from modeloinv.core.models import AssetClass, Bar
from modeloinv.data.base import (
    BaseDataProvider,
    DataValidationError,
    ProviderConnectionError,
    RateLimitExceededError,
    SymbolNotFoundError,
)

logger = logging.getLogger(__name__)


class YahooDataProvider(BaseDataProvider):
    """Direct REST client for Yahoo Finance v8 Chart API."""

    BASE_URLS = [
        "https://query1.finance.yahoo.com/v8/finance/chart",
        "https://query2.finance.yahoo.com/v8/finance/chart",
    ]
    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
    }
    TIMEFRAME_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "1h": "60m",
        "1d": "1d",
        "1D": "1d",
    }

    def __init__(self, timeout: float = 10.0, max_retries: int = 3) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update(self.DEFAULT_HEADERS)

    @property
    def name(self) -> str:
        return "yahoo"

    @property
    def supported_asset_classes(self) -> Set[AssetClass]:
        return {AssetClass.EQUITY, AssetClass.ETF, AssetClass.FOREX}

    def _execute_request(self, symbol: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Executes GET request with failover between query1 and query2 and exponential backoff."""
        clean_symbol = symbol.strip()
        last_error = None

        for attempt in range(self.max_retries):
            base_url = self.BASE_URLS[attempt % len(self.BASE_URLS)]
            url = f"{base_url}/{clean_symbol}"
            try:
                res = self._session.get(url, params=params, timeout=self.timeout)
                if res.status_code == 200:
                    data = res.json()
                    chart = data.get("chart", {})
                    if chart.get("error"):
                        err_code = chart["error"].get("code", "")
                        err_desc = chart["error"].get("description", "")
                        if "Not Found" in err_desc or err_code == "Not Found":
                            raise SymbolNotFoundError(f"Symbol '{symbol}' not found on Yahoo Finance: {err_desc}")
                        raise ProviderConnectionError(f"Yahoo chart error: {err_code} - {err_desc}")
                    return data
                elif res.status_code == 404:
                    raise SymbolNotFoundError(f"Symbol '{symbol}' not found on Yahoo Finance (HTTP 404).")
                elif res.status_code == 429:
                    wait_sec = 2 ** attempt
                    logger.warning(f"Yahoo Finance rate limit (429) for {symbol}. Backing off {wait_sec}s...")
                    time.sleep(wait_sec)
                    continue
                else:
                    last_error = ProviderConnectionError(f"Yahoo API returned HTTP {res.status_code}: {res.text[:200]}")
            except (requests.ConnectionError, requests.Timeout) as e:
                last_error = ProviderConnectionError(f"Connection failure to Yahoo for {symbol}: {e}")
                time.sleep(1)

        if last_error:
            raise last_error
        raise RateLimitExceededError(f"Exceeded max retries for {symbol} on Yahoo Finance.")

    def _parse_chart_data(self, symbol: str, data: Dict[str, Any], limit: Optional[int]) -> List[Bar]:
        """Parses Yahoo Finance v8 JSON response into a list of validated Bar objects."""
        chart_result = data.get("chart", {}).get("result")
        if not chart_result or len(chart_result) == 0:
            return []

        res = chart_result[0]
        meta = res.get("meta", {})
        currency = meta.get("currency", "USD")
        inferred_asset = self.infer_asset_class(symbol)

        timestamps = res.get("timestamp", [])
        quote = res.get("indicators", {}).get("quote", [{}])[0]
        opens = quote.get("open", [])
        highs = quote.get("high", [])
        lows = quote.get("low", [])
        closes = quote.get("close", [])
        volumes = quote.get("volume", [])

        bars: List[Bar] = []
        n = len(timestamps)

        for i in range(n):
            ts_raw = timestamps[i]
            o = opens[i] if i < len(opens) else None
            h = highs[i] if i < len(highs) else None
            l = lows[i] if i < len(lows) else None
            c = closes[i] if i < len(closes) else None
            v = volumes[i] if i < len(volumes) else None

            # Skip incomplete bars or market holiday records
            if o is None or h is None or l is None or c is None:
                continue

            vol = float(v) if v is not None else 0.0
            dt = datetime.fromtimestamp(ts_raw, tz=timezone.utc)

            # Epsilon bounding adjustment for small precision errors in vendor feeds
            o_f, h_f, l_f, c_f = float(o), float(h), float(l), float(c)
            h_bounded = max(h_f, max(o_f, c_f))
            l_bounded = min(l_f, min(o_f, c_f))

            bar = Bar(
                timestamp=dt,
                symbol=symbol.upper(),
                open=round(o_f, 4),
                high=round(h_bounded, 4),
                low=round(l_bounded, 4),
                close=round(c_f, 4),
                volume=round(max(0.0, vol), 2),
                asset_class=inferred_asset,
                currency=currency,
            )
            bars.append(self.validate_bar(bar))

        if limit and len(bars) > limit:
            bars = bars[-limit:]
        return bars

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
            raise DataValidationError(f"Unsupported timeframe '{timeframe}' for Yahoo Finance")

        params: Dict[str, Any] = {
            "interval": interval,
            "includePrePost": "false",
            "events": "div,splits",
        }
        if start:
            params["period1"] = int(start.timestamp())
        else:
            params["period1"] = int((datetime.now(timezone.utc) - timedelta(days=365)).timestamp())

        if end:
            params["period2"] = int(end.timestamp())
        else:
            params["period2"] = int(datetime.now(timezone.utc).timestamp())

        data = self._execute_request(symbol, params)
        return self._parse_chart_data(symbol, data, limit)

    def get_latest_bar(self, symbol: str, timeframe: str = "1m") -> Bar:
        bars = self.get_historical_bars(symbol, timeframe=timeframe, limit=5)
        if not bars:
            raise SymbolNotFoundError(f"No recent bars available for '{symbol}' on Yahoo Finance")
        return bars[-1]

    def get_fx_rate(self, base_currency: str = "EUR", quote_currency: str = "USD") -> float:
        base = base_currency.upper()
        quote = quote_currency.upper()
        if base == quote:
            return 1.0

        pair_symbol = f"{base}{quote}=X"
        try:
            bar = self.get_latest_bar(pair_symbol, timeframe="1d")
            return bar.close
        except Exception:
            # Try inverse pair
            inverse_symbol = f"{quote}{base}=X"
            bar = self.get_latest_bar(inverse_symbol, timeframe="1d")
            return 1.0 / bar.close
