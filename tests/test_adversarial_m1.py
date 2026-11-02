"""Adversarial stress and edge-case challenge test suite for Milestone 1.

Targets:
1. CurrencyConverter: extreme FX rates, negative rates, zero division, precision drift over 10,000 conversions.
2. ParquetBarCache: corrupted parquet files, empty files, symbols with Windows illegal characters.
3. MockDataProvider: extreme volatility parameters, long-horizon generation, NaN handling.
4. DataCascade: all network endpoints returning 500/404/429/timeout, fallback to cache and mock.
"""

from __future__ import annotations

import math
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from modeloinv.core.currency import (
    CurrencyConversionError,
    CurrencyConverter,
    FXDecompositionResult,
)
from modeloinv.core.models import AssetClass, Bar, ModelValidationError
from modeloinv.data.base import (
    BaseDataProvider,
    DataProviderError,
    DataValidationError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    RateLimitExceededError,
    SymbolNotFoundError,
)
from modeloinv.data.cascade import DataCascade
from modeloinv.data.mock import MockDataProvider
from modeloinv.storage.cache import ParquetBarCache, sanitize_filename_token


# ============================================================================
# 1. CurrencyConverter Adversarial Tests
# ============================================================================

def test_currency_negative_and_zero_rates_rejected():
    """Verify CurrencyConverter strictly rejects non-positive exchange rates."""
    conv = CurrencyConverter()

    for invalid_rate in [0.0, -0.0001, -1.0, -100.0, -1e10]:
        with pytest.raises(CurrencyConversionError, match="must be positive"):
            conv.set_rate("EURUSD", invalid_rate)

    # In decompose_pnl, negative or zero entry/exit rates must also be rejected
    with pytest.raises(CurrencyConversionError, match="FX rates must be positive"):
        CurrencyConverter.decompose_pnl(100.0, 105.0, 1.0, entry_fx_rate=0.0, exit_fx_rate=1.08)

    with pytest.raises(CurrencyConversionError, match="FX rates must be positive"):
        CurrencyConverter.decompose_pnl(100.0, 105.0, 1.0, entry_fx_rate=1.08, exit_fx_rate=-1.0)


def test_currency_extreme_fx_rates_and_boundaries():
    """Stress CurrencyConverter with sub-microscopic and astronomical FX rates."""
    conv = CurrencyConverter()

    extreme_rates = [1e-12, 1e-8, 1e-4, 1.0, 1e4, 1e8, 1e12]
    for rate in extreme_rates:
        conv.set_rate("EURUSD", rate)
        assert conv.get_rate("EUR", "USD") == rate
        # Reciprocal invariant
        expected_reciprocal = 1.0 / rate
        assert math.isclose(conv.get_rate("USD", "EUR"), expected_reciprocal, rel_tol=1e-9)

        # Round-trip conversion: EUR -> USD -> EUR
        orig_eur = 100.0
        usd = conv.convert(orig_eur, "EUR", "USD")
        back_eur = conv.convert(usd, "USD", "EUR")
        assert math.isclose(orig_eur, back_eur, rel_tol=1e-7)

    # Zero amount conversion invariant
    assert conv.convert(0.0, "EUR", "USD") == 0.0
    assert conv.convert(1e-15, "EUR", "USD") == 0.0  # isclose(0.0, abs_tol=1e-12)


def test_currency_precision_drift_10000_conversions():
    """Verify that 10,000 independent and chained conversions accumulate zero material drift."""
    conv = CurrencyConverter(fallback_rate=1.0850)
    conv.set_rate("EURUSD", 1.0850)

    # 1. 10,000 independent conversions across diverse magnitudes
    max_independent_drift = 0.0
    for i in range(1, 10001):
        amt = float(i) * 0.07391
        usd = conv.convert(amt, "EUR", "USD")
        back_eur = conv.convert(usd, "USD", "EUR")
        drift = abs(amt - back_eur)
        if drift > max_independent_drift:
            max_independent_drift = drift

    assert max_independent_drift < 1e-10, (
        f"Independent conversion drift exceeded machine precision: {max_independent_drift}"
    )

    # 2. 10,000 sequential iterative chained round-trips
    chained_val = 100.00
    for _ in range(10000):
        chained_val = conv.convert(conv.convert(chained_val, "EUR", "USD"), "USD", "EUR")

    chained_drift = abs(100.00 - chained_val)
    assert chained_drift < 1e-7, (
        f"Iterative chained drift over 10,000 round-trips exceeded tolerance: {chained_drift}"
    )


def test_currency_pnl_decomposition_algebraic_invariants():
    """Verify algebraic closure of PnL decomposition across diverse scenarios."""
    scenarios = [
        # (entry_p, exit_p, qty, entry_fx, exit_fx)
        (100.0, 110.0, 1.0, 1.08, 1.10),        # Standard asset gain, FX appreciation
        (100.0, 90.0, 2.5, 1.05, 1.02),         # Asset loss, FX depreciation
        (500.0, 500.0, 10.0, 1.08, 1.15),       # Zero asset price move, pure FX gain
        (200.0, 250.0, 5.0, 1.08, 1.08),        # Asset move, zero FX movement
        (0.0005, 0.0010, 50000.0, 1.08, 1.08),  # Micro-penny asset
        (65000.0, 68000.0, 0.015, 1.085, 1.09), # Fractional high-price crypto
        (100.0, 100.0, 0.0, 1.08, 1.08),        # Zero quantity edge case
    ]

    for ep, xp, qty, efx, xfx in scenarios:
        res = CurrencyConverter.decompose_pnl(ep, xp, qty, efx, xfx)
        # Fundamental invariant: Total PnL == Asset PnL + FX PnL
        sum_pnl = res.asset_pnl_eur + res.fx_pnl_eur
        assert math.isclose(res.total_pnl_eur, sum_pnl, abs_tol=1e-10), (
            f"PnL decomposition violated: total={res.total_pnl_eur}, sum={sum_pnl}"
        )


# ============================================================================
# 2. ParquetBarCache Adversarial Tests
# ============================================================================

def test_parquet_cache_windows_illegal_characters():
    """Verify ParquetBarCache sanitizes Windows NTFS illegal characters."""
    problematic_symbols = [
        "EUR/USD",
        "BINANCE:BTCUSDT",
        "TEST*SYMBOL?",
        'QUOTED"TICKER',
        "PIPE|SYMBOL",
        "LESS<GREATER>",
        "EQUALS=X",
        "BACK\\SLASH",
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ParquetBarCache(cache_dir=tmpdir)
        now = datetime.now(timezone.utc)
        test_bars = [
            Bar(
                timestamp=now - timedelta(minutes=i),
                symbol="TEST",
                open=100.0, high=102.0, low=98.0, close=101.0, volume=500.0
            )
            for i in range(5)
        ]

        for sym in problematic_symbols:
            count = cache.write_bars(sym, "1m", test_bars)
            assert count == 5
            assert cache.has_cache(sym, "1m")
            read_bars = cache.read_bars(sym, "1m")
            assert len(read_bars) == 5
            df = cache.read_dataframe(sym, "1m")
            assert len(df) == 5


def test_parquet_cache_empty_and_corrupt_files():
    """Test cache behavior under 0-byte and corrupted parquet files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ParquetBarCache(cache_dir=tmpdir)
        sym = "CORRUPT_TEST"
        tf = "1m"
        target_path = cache.get_cache_path(sym, tf)

        # 1. Zero-byte empty file handling
        target_path.write_bytes(b"")
        assert not cache.has_cache(sym, tf)
        assert cache.get_cached_range(sym, tf) is None
        assert cache.read_bars(sym, tf) == []
        assert len(cache.read_dataframe(sym, tf)) == 0

        # 2. Corrupt file with garbage bytes
        target_path.write_bytes(b"RANDOM_NON_PARQUET_GARBAGE_PAYLOAD")
        assert cache.has_cache(sym, tf)  # Exists and size > 0
        assert cache.get_cached_range(sym, tf) is None  # Handled gracefully
        assert cache.read_bars(sym, tf) == []  # Handled gracefully

        # read_dataframe on corrupt file gracefully returns empty DataFrame
        df_corrupt = cache.read_dataframe(sym, tf)
        assert isinstance(df_corrupt, pd.DataFrame)
        assert len(df_corrupt) == 0

        # 3. Write bars overwrites and recovers corrupt file
        now = datetime.now(timezone.utc)
        recovery_bar = Bar(timestamp=now, symbol=sym, open=100.0, high=101.0, low=99.0, close=100.0, volume=100.0)
        cnt = cache.write_bars(sym, tf, [recovery_bar])
        assert cnt == 1
        assert len(cache.read_bars(sym, tf)) == 1


# ============================================================================
# 3. MockDataProvider & Model Boundary Adversarial Tests
# ============================================================================

def test_mock_provider_deterministic_reproducibility():
    """Verify MockDataProvider with same seed generates identical series when end time is fixed."""
    fixed_end = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    m1 = MockDataProvider(seed=777)
    m2 = MockDataProvider(seed=777)

    bars1 = m1.get_historical_bars("SPY", timeframe="1d", end=fixed_end, limit=30)
    bars2 = m2.get_historical_bars("SPY", timeframe="1d", end=fixed_end, limit=30)

    assert len(bars1) == len(bars2) == 30
    for b1, b2 in zip(bars1, bars2):
        assert b1.timestamp == b2.timestamp
        assert b1.open == b2.open
        assert b1.close == b2.close
        assert b1.high == b2.high
        assert b1.low == b2.low


def test_mock_provider_extreme_volatility_spike_vulnerability():
    """Verify extreme volatility spike maintains strictly positive prices and valid bars."""
    mock = MockDataProvider(seed=42)
    # Inject massive volatility spike (50x on crypto BTCUSDT)
    mock.inject_volatility_spike("BTCUSDT", multiplier=50.0)
    bars = mock.get_historical_bars("BTCUSDT", timeframe="1d", limit=20)
    assert len(bars) == 20
    assert all(b.low > 0 for b in bars)
    assert all(b.open > 0 and b.close > 0 and b.high > 0 for b in bars)
    assert all(b.high >= max(b.open, b.close) - 1e-6 for b in bars)


def test_mock_provider_generate_bars_underflow_vulnerability():
    """Demonstrate vulnerability where arithmetic returns with high volatility drop price to <= 0."""
    mock = MockDataProvider(seed=0)
    mock.base_seed = 0
    start_dt = datetime.now(timezone.utc)
    # High volatility (1.0) causes arithmetic return (1.0 + ret) to become negative, violating invariants
    with pytest.raises((DataValidationError, ModelValidationError)):
        mock.generate_bars("AAPL", start_dt, num_bars=50, volatility=1.0)


def test_nan_validation_and_parquet_cache_poisoning_vulnerability():
    """Verify NaN prices are strictly rejected and cannot poison Parquet cache."""
    # 1. Bar rejects float('nan') in __post_init__
    with pytest.raises(ModelValidationError, match="cannot be None, NaN or Inf"):
        Bar(
            timestamp=datetime.now(timezone.utc),
            symbol="NAN_TEST",
            open=float("nan"),
            high=100.0,
            low=90.0,
            close=95.0,
            volume=1000.0,
            asset_class=AssetClass.EQUITY,
            currency="USD",
        )

    # 2. validate_bar also strictly rejects NaN
    class DummyProvider(BaseDataProvider):
        name = "dummy"
        supported_asset_classes = {AssetClass.EQUITY}
        def get_historical_bars(self, *a, **k): pass
        def get_latest_bar(self, *a, **k): pass
        def get_fx_rate(self, *a, **k): pass

    dummy = DummyProvider()
    valid_bar = Bar(
        timestamp=datetime.now(timezone.utc),
        symbol="NAN_TEST",
        open=100.0,
        high=105.0,
        low=95.0,
        close=100.0,
        volume=1000.0,
    )
    object.__setattr__(valid_bar, "open", float("nan"))
    with pytest.raises(DataValidationError, match="cannot be None, NaN or Inf"):
        dummy.validate_bar(valid_bar)


# ============================================================================
# 4. DataCascade Adversarial Tests
# ============================================================================

class MockFailingProvider(BaseDataProvider):
    def __init__(self, name: str, error_to_raise: Exception):
        self._name = name
        self.error_to_raise = error_to_raise

    @property
    def name(self) -> str:
        return self._name

    @property
    def supported_asset_classes(self):
        return {AssetClass.EQUITY, AssetClass.ETF, AssetClass.CRYPTO, AssetClass.FOREX}

    def get_historical_bars(self, symbol, timeframe="1d", start=None, end=None, limit=None):
        raise self.error_to_raise

    def get_latest_bar(self, symbol, timeframe="1m"):
        raise self.error_to_raise

    def get_fx_rate(self, base="EUR", quote="USD"):
        raise self.error_to_raise


def test_data_cascade_all_network_endpoints_fail_fallback_to_cache():
    """Verify DataCascade falls back to local cache when all upstream providers fail."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ParquetBarCache(cache_dir=tmpdir)
        now = datetime.now(timezone.utc)
        cached_bar = Bar(timestamp=now, symbol="AAPL", open=180.0, high=182.0, low=179.0, close=181.0, volume=5000.0)
        cache.write_bars("AAPL", "1d", [cached_bar])

        failing_providers = [
            MockFailingProvider("binance", ProviderConnectionError("500 Internal Server Error")),
            MockFailingProvider("yahoo", ProviderConnectionError("Connection timeout after 10s")),
            MockFailingProvider("alpaca", ProviderAuthenticationError("401 Unauthorized")),
        ]

        cascade = DataCascade(
            providers=failing_providers,
            cache=cache,
            fallback_to_cache=True,
            offline_mode=False,
        )

        bars = cascade.get_historical_bars("AAPL", timeframe="1d")
        assert len(bars) == 1
        assert bars[0].close == 181.0
        stats = cascade.get_stats()
        assert stats["cache_hits"] == 1
        assert stats["failover_events"] == 3


def test_data_cascade_all_sources_fail_mock_fallback_toggle():
    """Verify DataCascade behavior when all providers fail and cache is empty."""
    failing_providers = [
        MockFailingProvider("binance", RateLimitExceededError("429 Rate Limit")),
        MockFailingProvider("yahoo", SymbolNotFoundError("404 Not Found")),
        MockFailingProvider("alpaca", ProviderConnectionError("504 Gateway Timeout")),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ParquetBarCache(cache_dir=tmpdir)
        cascade = DataCascade(
            providers=failing_providers,
            cache=cache,
            fallback_to_cache=True,
            offline_mode=False,
        )

        # 1. When mock fallback is disabled, raises DataProviderError
        os.environ["MODELOINV_ENV"] = "production"
        os.environ["MODELOINV_ALLOW_MOCK_FALLBACK"] = "0"

        with pytest.raises(DataProviderError, match="All data sources failed"):
            cascade.get_historical_bars("UNKNOWN_SYM", timeframe="1d")

        # 2. When mock fallback is enabled, returns mock bars
        os.environ["MODELOINV_ALLOW_MOCK_FALLBACK"] = "1"
        mock_bars = cascade.get_historical_bars("UNKNOWN_SYM", timeframe="1d", limit=5)
        assert len(mock_bars) == 5

        # Reset environment
        os.environ.pop("MODELOINV_ALLOW_MOCK_FALLBACK", None)
        os.environ["MODELOINV_ENV"] = "test"


def test_data_cascade_asset_routing_priorities():
    """Verify DataCascade routes queries to asset-specific primary providers."""
    cascade = DataCascade()

    crypto_order = [p.name for p in cascade._get_routed_providers("BTCUSDT")]
    assert crypto_order[0] == "binance", f"Expected binance first for crypto, got {crypto_order}"

    fx_order = [p.name for p in cascade._get_routed_providers("EURUSD=X")]
    assert fx_order[0] == "yahoo", f"Expected yahoo first for forex, got {fx_order}"

    equity_order = [p.name for p in cascade._get_routed_providers("SPY")]
    assert equity_order[0] == "yahoo" and equity_order[1] == "alpaca", (
        f"Expected yahoo then alpaca for equities, got {equity_order}"
    )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
