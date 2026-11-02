"""Modular Slippage Models for modeloINV Execution Simulation.

Implements Fixed BPS, Bid-Ask Spread (Parkinson/Corwin-Schultz estimation),
Volume Impact (Almgren-Chriss square-root law), and Composite slippage engines.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from modeloinv.core.models import Bar, OrderRequest, OrderSide

logger = logging.getLogger(__name__)


class SlippageModel(ABC):
    """Abstract base class for all slippage calculation models."""

    @abstractmethod
    def calculate_slippage(
        self,
        order: OrderRequest,
        reference_price: float,
        bar: Optional[Bar] = None,
    ) -> float:
        """Calculate the absolute price slippage per share (non-negative).

        Parameters:
            order: Canonical order request.
            reference_price: Benchmark price (e.g. bar open or latest mid-price).
            bar: Optional OHLCV bar representing the execution interval.

        Returns:
            Absolute slippage amount per share (in quote currency, e.g. USD).
        """
        pass

    def calculate_fill_price(
        self,
        order: OrderRequest,
        reference_price: float,
        bar: Optional[Bar] = None,
    ) -> Tuple[float, float]:
        """Calculate executed fill price and slippage per share.

        Returns:
            Tuple of (executed_fill_price, slippage_per_share).
            For BUY: fill_price = reference_price + slippage.
            For SELL: fill_price = max(0.0001, reference_price - slippage).
        """
        slippage = self.calculate_slippage(order, reference_price, bar)
        slippage = max(0.0, float(slippage))

        if order.side == OrderSide.BUY:
            fill_price = reference_price + slippage
        else:
            fill_price = max(0.0001, reference_price - slippage)

        return fill_price, slippage


class FixedBpsSlippageModel(SlippageModel):
    """Fixed Basis Points (BPS) slippage model.

    Slippage is a constant percentage penalty:
    slippage = price * (bps / 10000.0)
    """

    def __init__(self, bps: float = 5.0) -> None:
        if bps < 0:
            raise ValueError(f"Slippage BPS cannot be negative, got {bps}")
        self.bps = float(bps)

    def calculate_slippage(
        self,
        order: OrderRequest,
        reference_price: float,
        bar: Optional[Bar] = None,
    ) -> float:
        if reference_price <= 0:
            return 0.0
        return reference_price * (self.bps / 10000.0)


class BidAskSpreadModel(SlippageModel):
    """Bid-Ask Spread model estimating half-spread cost from OHLC bars or quote data.

    When tick-level quotes are unavailable, estimates effective spread using
    Parkinson High-Low volatility estimator:
    sigma = sqrt( (ln(H/L))^2 / (4 * ln 2) )
    half_spread = 0.5 * sigma * price
    """

    def __init__(
        self,
        default_spread_bps: float = 5.0,
        min_spread_bps: float = 1.0,
        max_spread_bps: float = 100.0,
    ) -> None:
        self.default_spread_bps = max(0.0, float(default_spread_bps))
        self.min_spread_bps = max(0.0, float(min_spread_bps))
        self.max_spread_bps = max(self.min_spread_bps, float(max_spread_bps))

    def calculate_slippage(
        self,
        order: OrderRequest,
        reference_price: float,
        bar: Optional[Bar] = None,
    ) -> float:
        if reference_price <= 0:
            return 0.0

        if bar is not None and bar.high > bar.low and bar.low > 0:
            # Parkinson volatility estimation
            log_hl = math.log(bar.high / bar.low)
            parkinson_vol = math.sqrt((log_hl ** 2) / (4.0 * math.log(2.0)))
            spread_pct = parkinson_vol * 0.5
            # Clamp between min_spread and max_spread
            min_pct = self.min_spread_bps / 10000.0
            max_pct = self.max_spread_bps / 10000.0
            spread_pct = min(max(spread_pct, min_pct), max_pct)
            half_spread = 0.5 * spread_pct * reference_price
            return half_spread
        else:
            # Default half-spread based on configured BPS
            return reference_price * (self.default_spread_bps / (2.0 * 10000.0))


class VolumeImpactSlippageModel(SlippageModel):
    """Almgren-Chriss (2000) square-root market impact model:

    Delta P = eta * sigma_bar * sqrt(Q / V_bar) * P

    Where:
        eta: Empirical market impact constant (typically 0.10 - 0.25).
        sigma_bar: Normalized bar volatility (High - Low) / Close.
        Q: Order share quantity.
        V_bar: Total bar volume.
    """

    def __init__(
        self,
        eta: float = 0.15,
        default_volume: float = 50000.0,
        default_volatility: float = 0.01,
        max_participation_rate: float = 0.10,
    ) -> None:
        self.eta = max(0.0, float(eta))
        self.default_volume = max(1.0, float(default_volume))
        self.default_volatility = max(0.0001, float(default_volatility))
        self.max_participation_rate = float(max_participation_rate)

    def calculate_slippage(
        self,
        order: OrderRequest,
        reference_price: float,
        bar: Optional[Bar] = None,
    ) -> float:
        if reference_price <= 0:
            return 0.0

        # Estimate order share quantity
        if order.quantity is not None and order.quantity > 0:
            qty = order.quantity
        elif order.notional is not None and order.notional > 0:
            qty = order.notional / reference_price
        else:
            qty = 1.0

        # Determine bar volume and volatility
        if bar is not None and bar.volume > 0:
            bar_vol = bar.volume
            sigma = (
                (bar.high - bar.low) / bar.close
                if bar.close > 0 and bar.high >= bar.low
                else self.default_volatility
            )
        else:
            bar_vol = self.default_volume
            sigma = self.default_volatility

        sigma = max(0.0001, sigma)
        participation = qty / bar_vol
        # Clamp participation to avoid absurd blow-ups on micro-bars
        clamped_participation = min(participation, self.max_participation_rate)

        # Square-root impact
        impact_pct = self.eta * sigma * math.sqrt(clamped_participation)
        return reference_price * impact_pct


class CompositeSlippageModel(SlippageModel):
    """Composite slippage model aggregating multiple sub-models (e.g. spread + volume impact)."""

    def __init__(self, models: Optional[List[SlippageModel]] = None) -> None:
        self.models: List[SlippageModel] = list(models) if models is not None else []

    def add_model(self, model: SlippageModel) -> None:
        self.models.append(model)

    def calculate_slippage(
        self,
        order: OrderRequest,
        reference_price: float,
        bar: Optional[Bar] = None,
    ) -> float:
        if not self.models or reference_price <= 0:
            return 0.0
        return sum(m.calculate_slippage(order, reference_price, bar) for m in self.models)
