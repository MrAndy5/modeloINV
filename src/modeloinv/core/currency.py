"""Multi-Currency FX Engine for modeloINV.

Provides real-time mark-to-market EUR/USD conversion, resilient caching with
offline fallback rates, and mathematically exact P&L decomposition isolating
asset return from FX rate movement.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class CurrencyConversionError(ValueError):
    """Raised when conversion cannot be performed."""
    pass


@dataclass(frozen=True)
class FXDecompositionResult:
    """Exact mathematical breakdown of realized EUR P&L into Asset and FX return components.

    Guarantees: asset_pnl_eur + fx_pnl_eur == total_pnl_eur (within float precision).
    """
    total_pnl_eur: float
    asset_pnl_eur: float
    fx_pnl_eur: float
    asset_return_pct: float
    fx_return_pct: float
    total_return_pct: float


class CurrencyConverter:
    """Thread-safe FX conversion and mark-to-market engine with offline fallback support."""

    def __init__(
        self,
        base_currency: str = "EUR",
        fallback_rate: float = 1.0800,
        cache_ttl_seconds: int = 3600,
    ) -> None:
        self.base_currency = base_currency.upper()
        self.fallback_rate = fallback_rate
        self.cache_ttl_seconds = cache_ttl_seconds
        self._lock = threading.RLock()
        # Pair format standard: "EURUSD" = USD per 1 EUR
        self._rates: Dict[str, Tuple[float, datetime]] = {
            "EURUSD": (self.fallback_rate, datetime.utcnow()),
            "USDEUR": (1.0 / self.fallback_rate, datetime.utcnow()),
            "EURUSDT": (self.fallback_rate, datetime.utcnow()),
            "USDTEUR": (1.0 / self.fallback_rate, datetime.utcnow()),
        }

    def set_rate(
        self,
        pair: str,
        rate: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record a new exchange rate and its reciprocal pair."""
        if rate <= 0:
            raise CurrencyConversionError(f"Exchange rate must be positive, got {rate} for {pair}")

        clean_pair = pair.replace("/", "").replace("=X", "").upper()
        ts = timestamp or datetime.utcnow()

        with self._lock:
            self._rates[clean_pair] = (rate, ts)
            # Register reciprocal pair if 6-char or known currency pair
            if len(clean_pair) == 6:
                base = clean_pair[:3]
                quote = clean_pair[3:]
                inv_pair = f"{quote}{base}"
                self._rates[inv_pair] = (1.0 / rate, ts)
            elif clean_pair == "EURUSDT":
                self._rates["USDTEUR"] = (1.0 / rate, ts)
        logger.debug(f"Updated FX rate {clean_pair} = {rate:.4f} at {ts.isoformat()}")

    def get_rate(self, from_currency: str, to_currency: str) -> float:
        """Retrieve exchange rate converting from_currency to to_currency.

        Returns multiplier: to_amount = from_amount * get_rate(from_curr, to_curr).
        """
        from_curr = from_currency.upper()
        to_curr = to_currency.upper()

        if from_curr == to_curr:
            return 1.0

        pair = f"{from_curr}{to_curr}"
        with self._lock:
            if pair in self._rates:
                rate, ts = self._rates[pair]
                age = (datetime.utcnow() - ts).total_seconds()
                if age > self.cache_ttl_seconds:
                    logger.debug(
                        f"FX rate for {pair} is stale ({age:.0f}s > {self.cache_ttl_seconds}s). Using cached."
                    )
                return rate

            # Fallback for EUR/USD variations
            if pair in ("EURUSD", "EURUSDT"):
                return self.fallback_rate
            if pair in ("USDEUR", "USDTEUR"):
                return 1.0 / self.fallback_rate

        raise CurrencyConversionError(
            f"No exchange rate registered for currency conversion from {from_curr} to {to_curr}."
        )

    def convert(
        self,
        amount: float,
        from_currency: str,
        to_currency: str,
        custom_rate: Optional[float] = None,
    ) -> float:
        """Convert an amount between two currencies."""
        if math.isclose(amount, 0.0, abs_tol=1e-12):
            return 0.0

        from_curr = from_currency.upper()
        to_curr = to_currency.upper()
        if from_curr == to_curr:
            return amount

        rate = custom_rate if custom_rate is not None else self.get_rate(from_curr, to_curr)
        return amount * rate

    def to_eur(self, amount: float, from_currency: str) -> float:
        """Convenience method: convert foreign currency amount to base EUR."""
        return self.convert(amount, from_currency=from_currency, to_currency=self.base_currency)

    def from_eur(self, amount_eur: float, to_currency: str) -> float:
        """Convenience method: convert base EUR amount to target foreign currency."""
        return self.convert(amount_eur, from_currency=self.base_currency, to_currency=to_currency)

    @staticmethod
    def decompose_pnl(
        entry_price_quote: float,
        exit_price_quote: float,
        quantity: float,
        entry_fx_rate: float,
        exit_fx_rate: float,
    ) -> FXDecompositionResult:
        """Mathematically exact decomposition of EUR P&L into Asset and FX return components.

        Parameters:
            entry_price_quote: Fill price in asset currency (e.g. USD)
            exit_price_quote: Exit price in asset currency (e.g. USD)
            quantity: Filled asset quantity
            entry_fx_rate: EUR/USD rate at entry (USD per 1 EUR)
            exit_fx_rate: EUR/USD rate at exit (USD per 1 EUR)

        Returns:
            FXDecompositionResult containing exact breakdown.
        """
        if entry_fx_rate <= 0 or exit_fx_rate <= 0:
            raise CurrencyConversionError(
                f"FX rates must be positive: entry={entry_fx_rate}, exit={exit_fx_rate}"
            )

        # EUR cost basis and proceeds
        cost_eur = (quantity * entry_price_quote) / entry_fx_rate
        proceeds_eur = (quantity * exit_price_quote) / exit_fx_rate
        total_pnl_eur = proceeds_eur - cost_eur

        # Asset P&L evaluated at entry exchange rate
        asset_pnl_eur = (quantity * (exit_price_quote - entry_price_quote)) / entry_fx_rate

        # FX P&L evaluated on gross proceeds
        fx_pnl_eur = quantity * exit_price_quote * ((1.0 / exit_fx_rate) - (1.0 / entry_fx_rate))

        # Percentage returns
        total_ret_pct = (total_pnl_eur / cost_eur * 100.0) if cost_eur > 0 else 0.0
        asset_ret_pct = (
            ((exit_price_quote - entry_price_quote) / entry_price_quote * 100.0)
            if entry_price_quote > 0
            else 0.0
        )
        fx_ret_pct = (
            ((1.0 / exit_fx_rate - 1.0 / entry_fx_rate) / (1.0 / entry_fx_rate) * 100.0)
            if entry_fx_rate > 0
            else 0.0
        )

        return FXDecompositionResult(
            total_pnl_eur=total_pnl_eur,
            asset_pnl_eur=asset_pnl_eur,
            fx_pnl_eur=fx_pnl_eur,
            asset_return_pct=asset_ret_pct,
            fx_return_pct=fx_ret_pct,
            total_return_pct=total_ret_pct,
        )


# Global converter instance
_DEFAULT_CONVERTER: Optional[CurrencyConverter] = None


def get_currency_converter() -> CurrencyConverter:
    """Retrieve or create default global CurrencyConverter."""
    global _DEFAULT_CONVERTER
    if _DEFAULT_CONVERTER is None:
        _DEFAULT_CONVERTER = CurrencyConverter()
    return _DEFAULT_CONVERTER


def set_currency_converter(converter: CurrencyConverter) -> None:
    """Inject custom CurrencyConverter for testing or multi-instance operation."""
    global _DEFAULT_CONVERTER
    _DEFAULT_CONVERTER = converter
