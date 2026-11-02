"""Unit tests for market data providers (modeloinv.data): Base, Mock, Yahoo, Binance, Alpaca, Cascade."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from modeloinv.core.models import AssetClass, Bar
from modeloinv.data.alpaca import AlpacaDataProvider
from modeloinv.data.base import (
    BaseDataProvider,
    DataProviderError,
    DataValidationError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    RateLimitExceededError,
    SymbolNotFoundError,
)
from modeloinv.data.binance import BinanceDataProvider
from modeloinv.data.cascade import DataCascade
from modeloinv.data.mock import MockDataProvider
from modeloinv.data.yahoo import YahooDataProvider
from modeloinv.storage.cache import ParquetBarCache


# =====================================================================
# BaseDataProvider Tests
# =====================================================================

def test_infer_asset_class():
    """Verify asset class inference rules."""
    assert BaseDataProvider.infer_asset_class("SPY") == AssetClass.ETF
    assert BaseDataProvider.infer_asset_class("QQQ") == AssetClass.ETF
    assert BaseDataProvider.infer_asset_class("AAPL") == AssetClass.EQUITY
    assert BaseDataProvider.infer_asset_class("BTCUSDT") == AssetClass.CRYPTO
    assert BaseDataProvider.infer_asset_class("BTC/USDT") == AssetClass.CRYPTO
    assert BaseDataProvider.infer_asset_class("EURUSD=X") == AssetClass.FOREX
    assert BaseDataProvider.infer_asset_class("EURUSD") == AssetClass.FOREX


def test_dataframe_conversion_roundtrip():
    """Verify converting between List[Bar] and pd.DataFrame."""
    now = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    bars = [
        Bar(timestamp=now, symbol="AAPL", open=180.0, high=182.0, low=179.0, close=181.0, volume=1000.0),
        Bar(timestamp=now + timedelta(minutes=5), symbol="AAPL", open=181.0, high=183.0, low=180.0, close=182.5, volume=1200.0),
    ]
    df = BaseDataProvider.bars_to_dataframe(bars)
    assert len(df) == 2
    assert "close" in df.columns

    restored = BaseDataProvider.dataframe_to_bars(df, symbol="AAPL")
    assert len(restored) == 2
    assert restored[0].symbol == "AAPL"
    assert restored[0].close == 181.0
    assert restored[1].close == 182.5


# =====================================================================
# MockDataProvider Tests
# =====================================================================

def test_mock_provider_deterministic_reproducibility():
    """Verify identical seeds produce byte-identical bars."""
    m1 = MockDataProvider(seed=42)
    m2 = MockDataProvider(seed=42)
    dt = datetime(2026, 1, 1, 9, 30, tzinfo=timezone.utc)

    bars1 = m1.get_historical_bars("SPY", timeframe="5m", start=dt, limit=20)
    bars2 = m2.get_historical_bars("SPY", timeframe="5m", start=dt, limit=20)

    assert len(bars1) == 20
    assert len(bars2) == 20
    for b1, b2 in zip(bars1, bars2):
        assert b1.timestamp == b2.timestamp
        assert b1.close == b2.close
        assert b1.volume == b2.volume


def test_mock_provider_invariants_hold():
    """Verify all generated bars strictly respect price and volume geometry."""
    mock = MockDataProvider(seed=123)
    for sym in ["SPY", "AAPL", "BTCUSDT", "EURUSD=X", "UNKNOWN_CO"]:
        bars = mock.get_historical_bars(sym, timeframe="15m", limit=30)
        for b in bars:
            assert b.open > 0
            assert b.high >= b.low
            assert b.high >= max(b.open, b.close) - 1e-4
            assert b.low <= min(b.open, b.close) + 1e-4
            assert b.volume >= 0
            assert b.timestamp.tzinfo is not None


def test_mock_provider_anomaly_injection():
    """Verify synthetic anomaly injection (flash crash)."""
    mock = MockDataProvider(seed=42)
    mock.inject_flash_crash("SPY", drop_pct=0.30, bar_idx=3)
    bars = mock.get_historical_bars("SPY", timeframe="5m", limit=10)
    # Compare bar 3 close to bar 2 close
    assert bars[3].close < bars[2].close * 0.75


def test_mock_provider_fx_rate():
    """Verify mock provider EUR/USD spot rate tracking."""
    mock = MockDataProvider()
    assert mock.get_fx_rate("EUR", "USD") == 1.0850
    assert mock.get_fx_rate("USD", "EUR") == round(1.0 / 1.0850, 6) or abs(mock.get_fx_rate("USD", "EUR") - 1.0 / 1.0850) < 1e-6
    assert mock.get_fx_rate("EUR", "EUR") == 1.0


# =====================================================================
# YahooDataProvider Tests (Mocked HTTP)
# =====================================================================

def test_yahoo_parse_chart_data():
    """Verify YahooDataProvider correctly parses Yahoo v8 chart responses."""
    provider = YahooDataProvider()

    mock_resp_data = {
        "chart": {
            "result": [
                {
                    "meta": {"currency": "USD", "symbol": "SPY"},
                    "timestamp": [1700000000, 1700000300],
                    "indicators": {
                        "quote": [
                            {
                                "open": [500.0, 501.0],
                                "high": [502.0, 503.0],
                                "low": [499.0, 500.5],
                                "close": [501.5, 502.5],
                                "volume": [10000, 15000],
                            }
                        ]
                    },
                }
            ]
        }
    }

    with patch.object(provider, "_execute_request", return_value=mock_resp_data):
        bars = provider.get_historical_bars("SPY", timeframe="5m")
        assert len(bars) == 2
        assert bars[0].symbol == "SPY"
        assert bars[0].close == 501.5
        assert bars[1].volume == 15000.0


def test_yahoo_handles_404_symbol_not_found():
    """Verify YahooDataProvider raises SymbolNotFoundError on 404."""
    provider = YahooDataProvider()
    mock_res = MagicMock(status_code=404)
    with patch.object(provider._session, "get", return_value=mock_res):
        with pytest.raises(SymbolNotFoundError):
            provider.get_historical_bars("NON_EXISTENT_TICKER_123")


# =====================================================================
# BinanceDataProvider Tests (Mocked HTTP)
# =====================================================================

def test_binance_symbol_normalization():
    """Verify Binance symbol normalization."""
    provider = BinanceDataProvider()
    assert provider._normalize_symbol("btc/usdt") == "BTCUSDT"
    assert provider._normalize_symbol("ETH-USDT") == "ETHUSDT"


def test_binance_parse_klines():
    """Verify BinanceDataProvider correctly parses kline array."""
    provider = BinanceDataProvider()

    mock_klines = [
        [1700000000000, "65000.0", "65500.0", "64900.0", "65200.0", "10.5", 1700000299999],
        [1700000300000, "65200.0", "65800.0", "65100.0", "65600.0", "12.0", 1700000599999],
    ]

    with patch.object(provider, "_request_klines", return_value=mock_klines):
        bars = provider.get_historical_bars("BTCUSDT", timeframe="5m", limit=2)
        assert len(bars) == 2
        assert bars[0].symbol == "BTCUSDT"
        assert bars[0].close == 65200.0
        assert bars[1].volume == 12.0
        assert bars[0].asset_class == AssetClass.CRYPTO


# =====================================================================
# AlpacaDataProvider Tests
# =====================================================================

def test_alpaca_unconfigured_raises_auth_error_immediately():
    """Verify unconfigured AlpacaDataProvider raises ProviderAuthenticationError without hanging."""
    provider = AlpacaDataProvider(api_key="", secret_key="")
    assert provider.is_configured is False
    with pytest.raises(ProviderAuthenticationError, match="Alpaca credentials missing"):
        provider.get_historical_bars("AAPL")


def test_alpaca_configured_makes_request_with_headers():
    """Verify configured AlpacaDataProvider attaches API keys and parses bars."""
    provider = AlpacaDataProvider(api_key="KEY_123", secret_key="SEC_456")
    assert provider.is_configured is True

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "bars": [
            {"t": "2026-01-01T14:30:00Z", "o": 180.0, "h": 182.0, "l": 179.5, "c": 181.2, "v": 25000}
        ]
    }

    with patch("requests.get", return_value=mock_resp) as mock_get:
        bars = provider.get_historical_bars("AAPL", timeframe="1d")
        assert len(bars) == 1
        assert bars[0].close == 181.2
        # Verify request headers
        mock_get.assert_called_once()
        headers = mock_get.call_args[1]["headers"]
        assert headers["APCA-API-KEY-ID"] == "KEY_123"
        assert headers["APCA-API-SECRET-KEY"] == "SEC_456"


# =====================================================================
# DataCascade Tests
# =====================================================================

def test_cascade_asset_aware_routing():
    """Verify DataCascade routes crypto to Binance first and equities to Yahoo first."""
    cascade = DataCascade()
    crypto_providers = cascade._get_routed_providers("BTCUSDT")
    assert crypto_providers[0].name == "binance"

    equity_providers = cascade._get_routed_providers("SPY")
    assert equity_providers[0].name == "yahoo"

    forex_providers = cascade._get_routed_providers("EURUSD=X")
    assert forex_providers[0].name == "yahoo"


def test_cascade_transparent_write_through_cache(tmp_path: Path):
    """Verify DataCascade writes fetched bars through to ParquetBarCache."""
    cache = ParquetBarCache(cache_dir=tmp_path / "cache")
    mock_source = MockDataProvider(seed=42)

    cascade = DataCascade(providers=[mock_source], cache=cache, cache_on_fetch=True)
    assert cache.has_cache("SPY", "5m") is False

    bars = cascade.get_historical_bars("SPY", timeframe="5m", limit=10)
    assert len(bars) == 10
    # Parquet cache now exists and has bars
    assert cache.has_cache("SPY", "5m") is True
    cached_bars = cache.read_bars("SPY", "5m")
    assert len(cached_bars) == 10


def test_cascade_failover_to_cache_on_provider_error(tmp_path: Path):
    """Verify DataCascade falls over to local cache when all live providers fail."""
    cache = ParquetBarCache(cache_dir=tmp_path / "cache")
    # Pre-populate cache
    cached_bars = [
        Bar(timestamp=datetime.now(timezone.utc), symbol="SPY", open=500.0, high=505.0, low=495.0, close=502.0, volume=100.0)
    ]
    cache.write_bars("SPY", "1d", cached_bars)

    # Broken live provider
    broken_provider = MagicMock(spec=BaseDataProvider)
    broken_provider.name = "broken"
    broken_provider.infer_asset_class = BaseDataProvider.infer_asset_class
    broken_provider.get_historical_bars.side_effect = ProviderConnectionError("Network is down!")

    cascade = DataCascade(providers=[broken_provider], cache=cache, fallback_to_cache=True)
    bars = cascade.get_historical_bars("SPY", timeframe="1d")

    assert len(bars) == 1
    assert bars[0].close == 502.0
    assert cascade.stats["cache_hits"] == 1
    assert cascade.stats["failover_events"] >= 1


def test_cascade_offline_mode():
    """Verify DataCascade in offline_mode bypasses network and serves mock/cache."""
    mock_prov = MockDataProvider(seed=99)
    cascade = DataCascade(offline_mode=True, mock_provider=mock_prov)
    bars = cascade.get_historical_bars("SPY", timeframe="5m", limit=5)
    assert len(bars) == 5
    assert cascade.get_fx_rate("EUR", "USD") == 1.0850


def test_validate_bar_nan_and_inf_rejection():
    """Verify BaseDataProvider.validate_bar strictly rejects NaN and Inf values."""
    mock_prov = MockDataProvider()
    ts = datetime.now(timezone.utc)
    base_bar = Bar(timestamp=ts, symbol="SPY", open=100.0, high=105.0, low=95.0, close=100.0, volume=1000.0)

    for bad_val in [float("nan"), float("inf"), float("-inf")]:
        for field_name in ["open", "high", "low", "close", "volume"]:
            # Modify field via object.__setattr__ to bypass dataclass frozen check
            test_bar = Bar(timestamp=ts, symbol="SPY", open=100.0, high=105.0, low=95.0, close=100.0, volume=1000.0)
            object.__setattr__(test_bar, field_name, bad_val)
            with pytest.raises(DataValidationError, match="cannot be None, NaN or Inf"):
                mock_prov.validate_bar(test_bar)


def test_mock_provider_volatility_spike_positive_low():
    """Verify MockDataProvider maintains strictly positive prices under extreme volatility spikes."""
    mock = MockDataProvider(seed=42)
    mock.inject_volatility_spike("BTCUSDT", multiplier=50.0)
    bars = mock.get_historical_bars("BTCUSDT", timeframe="1d", limit=30)
    assert len(bars) == 30
    for b in bars:
        assert b.open >= 0.0001
        assert b.high >= 0.0001
        assert b.low >= 0.0001
        assert b.close >= 0.0001
        assert b.high >= max(b.open, b.close) - 1e-6
        assert b.low <= min(b.open, b.close) + 1e-6
