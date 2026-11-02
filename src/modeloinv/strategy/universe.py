"""Dynamic Asset Universe Screener, Multi-Armed Bandit Allocator, and Fractional Sizer.

Features:
- DynamicUniverseScreener: Filters multi-asset candidates by liquidity, volatility (ATR/Price >= 0.5%),
  and tradability on €100 virtual capital.
- MultiArmedBanditAllocator: Rolling Sharpe softmax dynamic allocation with maximum concentration cap.
- MicroCapitalFractionalAdapter: Translates target weights to fractional order quantities
  with €1.00 minimum notional constraint and lot size precision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np


@dataclass
class CandidateAsset:
    """Represents a potential asset in the multi-asset universe."""
    symbol: str
    asset_class: str
    price: float
    volume_24h_usd: float
    atr_14: float
    sharpe_30d: float = 0.0
    adx_14: float = 20.0
    ema_50: float = 0.0
    currency: str = "USD"
    min_order_notional: float = 1.00

    @property
    def volatility_ratio(self) -> float:
        """ATR / Price ratio."""
        if self.price <= 0:
            return 0.0
        return self.atr_14 / self.price


class DynamicUniverseScreener:
    """Scans and filters multi-asset universe for optimal intraday trading candidates."""

    def __init__(
        self,
        min_volume_24h_usd: float = 5_000_000.0,
        min_volatility_ratio: float = 0.005,  # 0.5% minimum ATR/price
        max_selected_assets: int = 5,
        default_account_equity_eur: float = 100.0,
    ):
        self.min_volume_24h_usd = min_volume_24h_usd
        self.min_volatility_ratio = min_volatility_ratio
        self.max_selected_assets = max_selected_assets
        self.account_equity_eur = default_account_equity_eur

    def filter_and_rank(
        self,
        candidates: Sequence[Union[CandidateAsset, Dict[str, Any]]],
    ) -> List[CandidateAsset]:
        """Filters candidates by liquidity, volatility, and computes composite ranking score.

        Score Formula:
            Score = 0.4 * Sharpe_30d + 0.3 * |Price - EMA_50| / EMA_50 + 0.3 * (ADX / 100)
            (or volume * volatility if detailed indicators absent)
        """
        standardized: List[CandidateAsset] = []
        for c in candidates:
            if isinstance(c, CandidateAsset):
                standardized.append(c)
            elif isinstance(c, dict):
                vol_usd = float(c.get("volume_24h_usd", c.get("volume", 0.0)))
                price = float(c.get("price", 100.0))
                # Volatility can be given as ratio or atr
                if "volatility" in c:
                    vol_ratio = float(c["volatility"])
                    atr = vol_ratio * price
                else:
                    atr = float(c.get("atr_14", c.get("atr", price * 0.01)))

                standardized.append(CandidateAsset(
                    symbol=str(c.get("symbol", "UNKNOWN")),
                    asset_class=str(c.get("asset_class", "EQUITY")),
                    price=price,
                    volume_24h_usd=vol_usd,
                    atr_14=atr,
                    sharpe_30d=float(c.get("sharpe_30d", c.get("sharpe", 0.0))),
                    adx_14=float(c.get("adx_14", c.get("adx", 20.0))),
                    ema_50=float(c.get("ema_50", price)),
                    currency=str(c.get("currency", "USD")),
                    min_order_notional=float(c.get("min_order_notional", 1.0)),
                ))

        # Filter step
        eligible: List[Tuple[float, CandidateAsset]] = []
        for asset in standardized:
            # 1. Liquidity check
            if asset.volume_24h_usd < self.min_volume_24h_usd:
                continue

            # 2. Volatility threshold (ATR / Price >= 0.5%)
            if asset.volatility_ratio < self.min_volatility_ratio:
                continue

            # 3. Opportunity Score calculation
            if asset.ema_50 > 0:
                ema_dist = abs(asset.price - asset.ema_50) / asset.ema_50
            else:
                ema_dist = 0.0

            score = (
                (0.4 * max(asset.sharpe_30d, 0.0))
                + (0.3 * ema_dist)
                + (0.3 * (asset.adx_14 / 100.0))
            )

            # Boost with volume * volatility product to reward deep liquid high-beta assets
            if score == 0.0:
                score = asset.volume_24h_usd * asset.volatility_ratio

            eligible.append((score, asset))

        # Sort descending by opportunity score
        eligible.sort(key=lambda x: x[0], reverse=True)
        return [asset for _, asset in eligible[: self.max_selected_assets]]


class MultiArmedBanditAllocator:
    """Softmax rolling Sharpe allocator with maximum asset concentration constraint."""

    def __init__(self, temperature: float = 1.0, max_concentration: float = 0.50):
        self.temperature = max(temperature, 1e-3)
        self.max_concentration = max_concentration

    def allocate(self, asset_sharpes: Dict[str, float]) -> Dict[str, float]:
        """Calculates optimal portfolio weights across candidate assets.

        Formula:
            w_i = exp(Sharpe_i / temperature) / sum(exp(Sharpe_j / temperature))
        Enforces maximum single-asset concentration cap (e.g. 50%) by redistributing excess.
        """
        if not asset_sharpes:
            return {}

        symbols = list(asset_sharpes.keys())
        if len(symbols) == 1:
            sym = symbols[0]
            return {sym: min(1.0, self.max_concentration)}

        scores = np.array([asset_sharpes[sym] for sym in symbols], dtype=np.float64)

        # Numerical stability for softmax: subtract max score
        shifted = (scores - np.max(scores)) / self.temperature
        exp_vals = np.exp(shifted)
        raw_weights = exp_vals / np.sum(exp_vals)

        weights = dict(zip(symbols, raw_weights))

        # Apply maximum concentration cap iteratively
        capped_weights = self._apply_concentration_cap(weights, self.max_concentration)
        return capped_weights

    @staticmethod
    def _apply_concentration_cap(weights: Dict[str, float], cap: float) -> Dict[str, float]:
        """Recursively caps weights and normalizes the unconstrained remainder."""
        w = dict(weights)
        total = sum(w.values())
        if total <= 0:
            return {k: 1.0 / len(w) for k in w}

        # Normalize initial sum
        for k in w:
            w[k] = w[k] / total

        # If any weight exceeds cap, iteratively clip and redistribute
        for _ in range(len(w)):
            excess = 0.0
            uncapped_keys = []
            for k, val in w.items():
                if val > cap + 1e-7:
                    excess += val - cap
                    w[k] = cap
                elif val < cap:
                    uncapped_keys.append(k)

            if excess <= 1e-7 or not uncapped_keys:
                break

            uncapped_sum = sum(w[k] for k in uncapped_keys)
            if uncapped_sum > 0:
                for k in uncapped_keys:
                    w[k] += excess * (w[k] / uncapped_sum)
            else:
                for k in uncapped_keys:
                    w[k] += excess / len(uncapped_keys)

        # Final normalization check
        norm_factor = sum(w.values())
        if norm_factor > 0:
            for k in w:
                w[k] = round(float(w[k] / norm_factor), 6)

        return w


class MicroCapitalFractionalAdapter:
    """Translates target portfolio weights into executable fractional share orders on €100 capital."""

    def __init__(
        self,
        min_order_notional_eur: float = 1.00,
        default_eur_usd_rate: float = 1.08,
    ):
        self.min_order_notional_eur = min_order_notional_eur
        self.default_eur_usd_rate = default_eur_usd_rate

    def get_asset_precision(self, symbol: str, asset_class: str = "EQUITY") -> int:
        """Determines fractional lot size precision based on asset class."""
        sym = symbol.upper()
        if "BTC" in sym or "ETH" in sym or "USDT" in sym or asset_class == "CRYPTO":
            return 6
        elif "/" in sym or asset_class == "FOREX":
            return 4
        else:
            return 6  # Micro fractional shares for equities / ETFs

    def calculate_order_quantity(
        self,
        symbol: str,
        target_weight: float,
        account_equity_eur: float,
        price: float,
        asset_class: str = "EQUITY",
        currency: str = "USD",
        eur_usd_rate: Optional[float] = None,
    ) -> Optional[float]:
        """Translates portfolio allocation weight into fractional quantity for a €100 account.

        Returns:
            Fractional share/token float, or None if order value is below €1.00 minimum.
        """
        if account_equity_eur <= 0 or price <= 0:
            return None

        abs_weight = abs(target_weight)
        if abs_weight < 1e-4:
            return 0.0

        target_eur = account_equity_eur * abs_weight

        # Enforce €1.00 minimum notional threshold
        if target_eur < self.min_order_notional_eur:
            return None

        fx_rate = eur_usd_rate or self.default_eur_usd_rate

        # Normalize notional into the quote currency of the asset
        if currency.upper() == "USD":
            target_quote = target_eur * fx_rate
        else:
            target_quote = target_eur

        raw_quantity = target_quote / price
        precision = self.get_asset_precision(symbol, asset_class)
        fractional_quantity = round(raw_quantity, precision)

        return fractional_quantity if fractional_quantity > 0 else None
