"""Deterministic offline market data generator using Geometric Brownian Motion.

Provides 100% offline, reproducible market bars across Equities, ETFs, Crypto, and Forex.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

import numpy as np

from modeloinv.core.models import AssetClass, Bar
from modeloinv.data.base import BaseDataProvider

logger = logging.getLogger(__name__)


class MockDataProvider(BaseDataProvider):
    """Deterministic offline market data generator using Geometric Brownian Motion."""

    PROFILES: Dict[str, Dict[str, Any]] = {
        "SPY": {"s0": 500.0, "mu": 0.10, "sigma": 0.16, "vol": 50_000_000.0, "asset": AssetClass.ETF, "curr": "USD"},
        "AAPL": {"s0": 180.0, "mu": 0.12, "sigma": 0.22, "vol": 40_000_000.0, "asset": AssetClass.EQUITY, "curr": "USD"},
        "QQQ": {"s0": 420.0, "mu": 0.14, "sigma": 0.20, "vol": 35_000_000.0, "asset": AssetClass.ETF, "curr": "USD"},
        "NVDA": {"s0": 850.0, "mu": 0.25, "sigma": 0.40, "vol": 45_000_000.0, "asset": AssetClass.EQUITY, "curr": "USD"},
        "MSFT": {"s0": 410.0, "mu": 0.11, "sigma": 0.18, "vol": 25_000_000.0, "asset": AssetClass.EQUITY, "curr": "USD"},
        "BTCUSDT": {"s0": 65000.0, "mu": 0.35, "sigma": 0.65, "vol": 25_000.0, "asset": AssetClass.CRYPTO, "curr": "USDT"},
        "ETHUSDT": {"s0": 3400.0, "mu": 0.30, "sigma": 0.70, "vol": 50_000.0, "asset": AssetClass.CRYPTO, "curr": "USDT"},
        "EURUSD=X": {"s0": 1.0850, "mu": 0.00, "sigma": 0.08, "vol": 0.0, "asset": AssetClass.FOREX, "curr": "USD"},
        "EURUSD": {"s0": 1.0850, "mu": 0.00, "sigma": 0.08, "vol": 0.0, "asset": AssetClass.FOREX, "curr": "USD"},
    }

    TIMEFRAME_DT: Dict[str, float] = {
        "1m": 1.0 / (252.0 * 390.0),
        "5m": 5.0 / (252.0 * 390.0),
        "15m": 15.0 / (252.0 * 390.0),
        "1h": 60.0 / (252.0 * 390.0),
        "1d": 1.0 / 252.0,
        "1D": 1.0 / 252.0,
    }

    def __init__(self, seed: int = 42) -> None:
        self.base_seed = seed
        self.anomalies: Dict[str, List[Dict[str, Any]]] = {}

    @property
    def name(self) -> str:
        return "mock"

    @property
    def supported_asset_classes(self) -> Set[AssetClass]:
        return {AssetClass.EQUITY, AssetClass.ETF, AssetClass.CRYPTO, AssetClass.FOREX}

    def _normalize_key(self, symbol: str) -> str:
        return symbol.upper().replace("/", "").replace("-", "").strip()

    def _get_profile(self, symbol: str) -> Dict[str, Any]:
        norm = self._normalize_key(symbol)
        if norm in self.PROFILES:
            return self.PROFILES[norm]
        if symbol.upper() in self.PROFILES:
            return self.PROFILES[symbol.upper()]

        h = abs(hash(norm))
        s0 = 50.0 + (h % 500)
        return {
            "s0": float(s0),
            "mu": 0.08,
            "sigma": 0.20,
            "vol": 1_000_000.0,
            "asset": self.infer_asset_class(symbol),
            "curr": "USD",
        }

    def inject_flash_crash(self, symbol: str, drop_pct: float = 0.20, bar_idx: Optional[int] = None) -> None:
        """Inject a flash crash drop at specific bar index (defaults to index 5)."""
        key = symbol.upper()
        self.anomalies.setdefault(key, []).append({
            "type": "flash_crash",
            "drop_pct": drop_pct,
            "bar_idx": bar_idx if bar_idx is not None else 5,
        })

    def inject_volatility_spike(self, symbol: str, multiplier: float = 5.0) -> None:
        """Inject volatility expansion for a symbol."""
        key = symbol.upper()
        self.anomalies.setdefault(key, []).append({
            "type": "vol_spike",
            "multiplier": multiplier,
        })

    def inject_gap(self, symbol: str, bars: int = 10) -> None:
        """Inject missing data gap."""
        key = symbol.upper()
        self.anomalies.setdefault(key, []).append({
            "type": "gap",
            "bars": bars,
        })

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[Bar]:
        dt = self.TIMEFRAME_DT.get(timeframe, 1.0 / 252.0)
        profile = self._get_profile(symbol)

        step_delta = timedelta(days=1) if timeframe in {"1d", "1D"} else timedelta(
            minutes=int(timeframe[:-1]) if timeframe[:-1].isdigit() else 5
        )
        now = datetime.now(timezone.utc)
        end_time = end or now
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)

        if start:
            start_time = start if start.tzinfo is not None else start.replace(tzinfo=timezone.utc)
            total_seconds = max(60, int((end_time - start_time).total_seconds()))
            bar_count = int(total_seconds / max(1, step_delta.total_seconds()))
        else:
            bar_count = limit or 100
            start_time = end_time - (step_delta * bar_count)

        if limit:
            bar_count = min(bar_count, limit)
        bar_count = max(1, bar_count)

        seed = (self.base_seed + abs(hash(symbol.upper())) + abs(hash(timeframe))) % (2**31 - 1)
        rng = np.random.RandomState(seed)

        bars: List[Bar] = []
        current_price = profile["s0"]
        current_time = start_time
        mu = profile["mu"]
        sigma = profile["sigma"]

        # Check for volatility spike anomaly
        anom_list = self.anomalies.get(symbol.upper(), [])
        for anom in anom_list:
            if anom.get("type") == "vol_spike":
                sigma *= anom.get("multiplier", 2.0)

        base_vol = profile["vol"] / (390.0 if timeframe not in {"1d", "1D"} else 1.0)

        for i in range(bar_count):
            open_p = current_price
            z = float(rng.normal(0, 1))
            ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
            close_p = open_p * np.exp(ret)

            # Check for injected flash crash anomaly
            for anom in anom_list:
                if anom.get("type") == "flash_crash" and anom.get("bar_idx") == i:
                    close_p *= (1.0 - anom.get("drop_pct", 0.20))

            intra_vol = max(1e-5, sigma * np.sqrt(dt))
            high_p = max(open_p, close_p) * (1.0 + abs(float(rng.normal(0, 1))) * intra_vol * 0.8)
            low_p = max(0.0001, min(open_p, close_p) * max(0.01, 1.0 - abs(float(rng.normal(0, 1))) * intra_vol * 0.8))
            if low_p > min(open_p, close_p):
                low_p = min(open_p, close_p)

            vol_mult = 1.0 + 2.0 * (abs(close_p - open_p) / (open_p * intra_vol + 1e-8))
            volume = (
                max(0.0, base_vol * np.exp(float(rng.normal(0, 0.3))) * vol_mult)
                if profile["asset"] != AssetClass.FOREX
                else 0.0
            )

            open_rounded = max(0.0001, round(float(open_p), 4))
            close_rounded = max(0.0001, round(float(close_p), 4))
            high_rounded = max(max(open_rounded, close_rounded), round(float(high_p), 4))
            low_rounded = max(0.0001, min(min(open_rounded, close_rounded), round(float(low_p), 4)))

            bar = Bar(
                timestamp=current_time,
                symbol=symbol.upper(),
                open=open_rounded,
                high=high_rounded,
                low=low_rounded,
                close=close_rounded,
                volume=round(float(volume), 2),
                asset_class=profile["asset"],
                currency=profile["curr"],
            )
            bars.append(self.validate_bar(bar))

            current_price = max(0.0001, close_p)
            current_time += step_delta

        return bars

    def get_latest_bar(self, symbol: str, timeframe: str = "1m") -> Bar:
        bars = self.get_historical_bars(symbol, timeframe=timeframe, limit=1)
        return bars[-1]

    def get_fx_rate(self, base_currency: str = "EUR", quote_currency: str = "USD") -> float:
        base = base_currency.upper()
        quote = quote_currency.upper()
        if base == quote:
            return 1.0
        if base == "EUR" and quote in {"USD", "USDT"}:
            return 1.0850
        if base in {"USD", "USDT"} and quote == "EUR":
            return 1.0 / 1.0850
        return 1.0

    def generate_bars(
        self,
        symbol: str,
        start_dt: datetime,
        num_bars: int = 100,
        timeframe_minutes: int = 5,
        start_price: float = 100.0,
        volatility: float = 0.002,
        drift: float = 0.0001,
        asset_class: AssetClass = AssetClass.EQUITY,
        currency: str = "USD",
    ) -> List[Bar]:
        """Convenience method matching tests/conftest.py fixture interface."""
        bars: List[Bar] = []
        current_price = start_price
        current_time = start_dt if start_dt.tzinfo is not None else start_dt.replace(tzinfo=timezone.utc)
        step_delta = timedelta(minutes=timeframe_minutes)

        seed = (self.base_seed + abs(hash(symbol.upper()))) % (2**31 - 1)
        rng = np.random.RandomState(seed)

        for _ in range(num_bars):
            ret = float(rng.normal(drift, volatility))
            open_p = current_price
            close_p = open_p * (1.0 + ret)
            high_p = max(open_p, close_p) * (1.0 + abs(float(rng.normal(0, volatility * 0.5))))
            low_p = min(open_p, close_p) * (1.0 - abs(float(rng.normal(0, volatility * 0.5))))
            vol = float(rng.randint(1000, 50000)) if asset_class != AssetClass.FOREX else 0.0

            bar = Bar(
                timestamp=current_time,
                symbol=symbol,
                open=round(open_p, 4),
                high=round(high_p, 4),
                low=round(low_p, 4),
                close=round(close_p, 4),
                volume=round(vol, 2),
                asset_class=asset_class,
                currency=currency,
            )
            bars.append(self.validate_bar(bar))
            current_price = close_p
            current_time += step_delta

        return bars
