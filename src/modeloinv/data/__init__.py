"""Market data infrastructure package for modeloINV.

Provides normalized historical and live feeds for Equities, ETFs, Crypto, and Forex,
with resilient cascade failover, Parquet caching, and deterministic offline mocking.
"""

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

__all__ = [
    "BaseDataProvider",
    "DataProviderError",
    "SymbolNotFoundError",
    "RateLimitExceededError",
    "ProviderConnectionError",
    "ProviderAuthenticationError",
    "DataValidationError",
    "YahooDataProvider",
    "BinanceDataProvider",
    "AlpacaDataProvider",
    "DataCascade",
    "MockDataProvider",
]
