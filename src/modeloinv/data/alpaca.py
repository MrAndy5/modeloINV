"""Alpaca Market Data v2 REST client.

Provides free-tier IEX equity and ETF bar feeds with config-driven credentials.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import requests

from modeloinv.core.models import AssetClass, Bar
from modeloinv.data.base import (
    BaseDataProvider,
    DataProviderError,
    DataValidationError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    RateLimitExceededError,
    SymbolNotFoundError,
)

logger = logging.getLogger(__name__)


class AlpacaDataProvider(BaseDataProvider):
    """Alpaca Market Data v2 REST client with free-tier IEX feed support."""

    DATA_BASE_URL = "https://data.alpaca.markets/v2"
    TIMEFRAME_MAP = {
        "1m": "1Min",
        "5m": "5Min",
        "15m": "15Min",
        "1h": "1Hour",
        "1d": "1Day",
        "1D": "1Day",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        feed: str = "iex",
        timeout: float = 10.0,
    ) -> None:
        self.api_key = (
            api_key
            or os.getenv("MODELOINV_ALPACA_API_KEY_ID")
            or os.getenv("MODELOINV_ALPACA_API_KEY")
            or os.getenv("APCA_API_KEY_ID")
            or ""
        )
        self.secret_key = (
            secret_key
            or os.getenv("MODELOINV_ALPACA_API_SECRET_KEY")
            or os.getenv("MODELOINV_ALPACA_SECRET_KEY")
            or os.getenv("APCA_API_SECRET_KEY")
            or ""
        )
        self.feed = feed
        self.timeout = timeout
        self.is_configured = bool(self.api_key and self.secret_key)

    @property
    def name(self) -> str:
        return "alpaca"

    @property
    def supported_asset_classes(self) -> Set[AssetClass]:
        return {AssetClass.EQUITY, AssetClass.ETF}

    def _get_headers(self) -> Dict[str, str]:
        if not self.is_configured:
            raise ProviderAuthenticationError(
                "Alpaca API credentials not configured (API key or secret key missing)."
            )
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Accept": "application/json",
        }

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[Bar]:
        if not self.is_configured:
            raise ProviderAuthenticationError("Alpaca credentials missing.")

        tf = self.TIMEFRAME_MAP.get(timeframe)
        if not tf:
            raise DataValidationError(f"Unsupported timeframe '{timeframe}' for Alpaca")

        url = f"{self.DATA_BASE_URL}/stocks/{symbol.upper()}/bars"
        params: Dict[str, Any] = {
            "timeframe": tf,
            "feed": self.feed,
            "limit": limit or 1000,
        }
        if start:
            params["start"] = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        if end:
            params["end"] = end.strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            res = requests.get(url, headers=self._get_headers(), params=params, timeout=self.timeout)
        except (requests.ConnectionError, requests.Timeout) as e:
            raise ProviderConnectionError(f"Connection failure to Alpaca: {e}")

        if res.status_code in (401, 403):
            raise ProviderAuthenticationError(f"Alpaca auth failed: {res.text}")
        if res.status_code == 404:
            raise SymbolNotFoundError(f"Symbol '{symbol}' not found on Alpaca")
        if res.status_code == 429:
            raise RateLimitExceededError("Alpaca rate limit exceeded (200 req/min free tier)")
        if res.status_code != 200:
            raise ProviderConnectionError(f"Alpaca API error HTTP {res.status_code}: {res.text}")

        data = res.json().get("bars") or []
        bars: List[Bar] = []
        for item in data:
            o = float(item["o"])
            h = float(item["h"])
            l = float(item["l"])
            c = float(item["c"])
            v = float(item["v"])
            ts_str = item["t"].replace("Z", "+00:00")
            ts = datetime.fromisoformat(ts_str)

            bar = Bar(
                timestamp=ts,
                symbol=symbol.upper(),
                open=o,
                high=max(h, max(o, c)),
                low=min(l, min(o, c)),
                close=c,
                volume=max(0.0, v),
                asset_class=self.infer_asset_class(symbol),
                currency="USD",
            )
            bars.append(self.validate_bar(bar))

        return bars

    def get_latest_bar(self, symbol: str, timeframe: str = "1m") -> Bar:
        bars = self.get_historical_bars(symbol, timeframe=timeframe, limit=1)
        if not bars:
            raise SymbolNotFoundError(f"No bars returned for {symbol} on Alpaca")
        return bars[-1]

    def get_fx_rate(self, base_currency: str = "EUR", quote_currency: str = "USD") -> float:
        raise DataProviderError("Alpaca does not provide spot FX market rates.")
