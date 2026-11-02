"""Resilient Data Provider Cascade with local Parquet caching and offline failover.

Routes asset-specific queries across Binance, Yahoo, Alpaca, and local disk cache.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from modeloinv.core.models import AssetClass, Bar
from modeloinv.data.alpaca import AlpacaDataProvider
from modeloinv.data.base import (
    BaseDataProvider,
    DataProviderError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    RateLimitExceededError,
    SymbolNotFoundError,
)
from modeloinv.data.binance import BinanceDataProvider
from modeloinv.data.mock import MockDataProvider
from modeloinv.data.yahoo import YahooDataProvider

logger = logging.getLogger(__name__)


class DataCascade(BaseDataProvider):
    """Resilient multi-provider failover pipeline with local Parquet caching."""

    def __init__(
        self,
        providers: Optional[List[BaseDataProvider]] = None,
        cache: Optional[Any] = None,  # ParquetBarCache
        fallback_to_cache: bool = True,
        cache_on_fetch: bool = True,
        offline_mode: bool = False,
        mock_provider: Optional[BaseDataProvider] = None,
    ) -> None:
        self.providers = providers if providers is not None else [
            BinanceDataProvider(),
            YahooDataProvider(),
            AlpacaDataProvider(),
        ]
        self.cache = cache
        self.fallback_to_cache = fallback_to_cache
        self.cache_on_fetch = cache_on_fetch
        self.offline_mode = offline_mode or (os.getenv("MODELOINV_ENV") == "test" and os.getenv("MODELOINV_FORCE_OFFLINE") == "1")
        self.mock_provider = mock_provider or MockDataProvider()

        self.stats: Dict[str, Any] = {
            "requests_total": 0,
            "cache_hits": 0,
            "provider_hits": {},
            "failover_events": 0,
            "errors": [],
        }

    @property
    def name(self) -> str:
        return "cascade"

    @property
    def supported_asset_classes(self) -> Set[AssetClass]:
        return {AssetClass.EQUITY, AssetClass.ETF, AssetClass.CRYPTO, AssetClass.FOREX}

    def _get_routed_providers(self, symbol: str) -> List[BaseDataProvider]:
        asset_class = self.infer_asset_class(symbol)
        if asset_class == AssetClass.CRYPTO:
            # Binance first for crypto
            return sorted(self.providers, key=lambda p: 0 if p.name == "binance" else (1 if p.name == "yahoo" else 2))
        elif asset_class == AssetClass.FOREX:
            # Yahoo first for FX
            return sorted(self.providers, key=lambda p: 0 if p.name == "yahoo" else (1 if p.name == "binance" else 2))
        else:
            # Yahoo first, Alpaca second for Equities/ETFs
            return sorted(self.providers, key=lambda p: 0 if p.name == "yahoo" else (1 if p.name == "alpaca" else 2))

    def get_stats(self) -> Dict[str, Any]:
        """Returns telemetry counters."""
        return dict(self.stats)

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[Bar]:
        self.stats["requests_total"] += 1

        # 1. Immediate offline route if configured
        if self.offline_mode:
            if self.cache and self.fallback_to_cache:
                cached = self.cache.read_bars(symbol, timeframe, start, end)
                if cached:
                    self.stats["cache_hits"] += 1
                    return cached[:limit] if limit else cached
            return self.mock_provider.get_historical_bars(symbol, timeframe, start, end, limit)

        # 2. Iterate live routed providers
        routed = self._get_routed_providers(symbol)
        errors: List[str] = []

        for provider in routed:
            try:
                bars = provider.get_historical_bars(symbol, timeframe, start, end, limit)
                if bars:
                    # Write-through to Parquet cache
                    if self.cache and self.cache_on_fetch:
                        try:
                            self.cache.write_bars(symbol, timeframe, bars)
                        except Exception as cache_err:
                            logger.warning(f"Failed to write to Parquet cache: {cache_err}")

                    p_name = provider.name
                    self.stats["provider_hits"][p_name] = self.stats["provider_hits"].get(p_name, 0) + 1
                    return bars
            except (SymbolNotFoundError, RateLimitExceededError, ProviderConnectionError, ProviderAuthenticationError, DataProviderError) as err:
                self.stats["failover_events"] += 1
                errors.append(f"{provider.name}: {str(err)}")
                logger.warning(f"Provider {provider.name} failed for {symbol}: {err}. Falling over...")
                continue

        # 3. Fallback to local Parquet cache
        if self.cache and self.fallback_to_cache:
            try:
                cached_bars = self.cache.read_bars(symbol, timeframe, start, end)
                if cached_bars:
                    self.stats["cache_hits"] += 1
                    logger.info(f"Serving {len(cached_bars)} bars for {symbol} from local Parquet cache.")
                    return cached_bars[:limit] if limit else cached_bars
            except Exception as cache_err:
                errors.append(f"cache: {str(cache_err)}")

        # 4. Fallback to mock provider if explicitly permitted or in test env
        if os.getenv("MODELOINV_ALLOW_MOCK_FALLBACK", "0") == "1" or os.getenv("MODELOINV_ENV") == "test":
            logger.info(f"Falling back to MockDataProvider for {symbol}.")
            return self.mock_provider.get_historical_bars(symbol, timeframe, start, end, limit)

        # 5. Exhausted all providers and cache
        raise DataProviderError(
            f"All data sources failed for {symbol} ({timeframe}). Errors: {'; '.join(errors)}"
        )

    def get_latest_bar(self, symbol: str, timeframe: str = "1m") -> Bar:
        if self.offline_mode:
            return self.mock_provider.get_latest_bar(symbol, timeframe)

        for provider in self._get_routed_providers(symbol):
            try:
                bar = provider.get_latest_bar(symbol, timeframe)
                if bar:
                    return bar
            except Exception:
                continue

        # Fallback to cache latest
        if self.cache:
            try:
                cached = self.cache.read_bars(symbol, timeframe)
                if cached:
                    return cached[-1]
            except Exception:
                pass

        return self.mock_provider.get_latest_bar(symbol, timeframe)

    def get_fx_rate(self, base_currency: str = "EUR", quote_currency: str = "USD") -> float:
        if base_currency.upper() == quote_currency.upper():
            return 1.0

        if self.offline_mode:
            return self.mock_provider.get_fx_rate(base_currency, quote_currency)

        for provider in self.providers:
            try:
                return provider.get_fx_rate(base_currency, quote_currency)
            except Exception:
                continue

        return 1.0850
