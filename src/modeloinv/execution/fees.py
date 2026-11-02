"""Transaction Cost and Fee Models for modeloINV Execution Simulation.

Supports Zero Commission (Alpaca style: SEC & FINRA TAF regulatory fees on sells),
Crypto Percentage (Binance maker/taker fees), European Retail (myInvestor flat + bps),
and Composite fee aggregation with multi-currency EUR conversion.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

from modeloinv.core.currency import CurrencyConverter, get_currency_converter
from modeloinv.core.models import OrderRequest, OrderSide, OrderType

logger = logging.getLogger(__name__)


class FeeModel(ABC):
    """Abstract base class for all fee calculation models."""

    @abstractmethod
    def calculate_fee(
        self,
        order: OrderRequest,
        fill_price: float,
        fill_qty: float,
        fx_rate: float = 1.08,
    ) -> float:
        """Calculate total transaction fee in account base currency (EUR).

        Parameters:
            order: Executed canonical order request.
            fill_price: Filled unit price (in asset quote currency, e.g. USD).
            fill_qty: Filled share/token quantity.
            fx_rate: Exchange rate (foreign currency units per 1 EUR, e.g. USD/EUR).

        Returns:
            Fee amount strictly in EUR (non-negative).
        """
        pass

    def calculate_fee_breakdown(
        self,
        order: OrderRequest,
        fill_price: float,
        fill_qty: float,
        fx_rate: float = 1.08,
    ) -> Dict[str, float]:
        """Return itemized fee components in EUR."""
        total = self.calculate_fee(order, fill_price, fill_qty, fx_rate)
        return {"total_fee_eur": total}


class ZeroCommissionFeeModel(FeeModel):
    """Alpaca US Equity zero-commission fee structure with regulatory charges.

    - Commission: $0.00
    - Buys: $0.00 regulatory fees
    - Sells:
        - SEC Section 31 Fee: notional * 0.0000278 (min $0.01 if notional > 0)
        - FINRA Trading Activity Fee (TAF): $0.000166 per share (min $0.01, max $8.30)
    All fees calculated in USD and converted to account base EUR.
    """

    SEC_RATE: float = 0.0000278
    FINRA_TAF_PER_SHARE: float = 0.000166
    FINRA_TAF_MIN: float = 0.01
    FINRA_TAF_MAX: float = 8.30

    def calculate_fee(
        self,
        order: OrderRequest,
        fill_price: float,
        fill_qty: float,
        fx_rate: float = 1.08,
    ) -> float:
        breakdown = self.calculate_fee_breakdown(order, fill_price, fill_qty, fx_rate)
        return breakdown["total_fee_eur"]

    def calculate_fee_breakdown(
        self,
        order: OrderRequest,
        fill_price: float,
        fill_qty: float,
        fx_rate: float = 1.08,
    ) -> Dict[str, float]:
        fx = max(0.0001, fx_rate)
        if order.side == OrderSide.BUY or fill_qty <= 0 or fill_price <= 0:
            return {
                "commission_eur": 0.0,
                "sec_fee_eur": 0.0,
                "finra_taf_eur": 0.0,
                "total_fee_eur": 0.0,
            }

        notional_usd = fill_price * fill_qty

        # SEC fee: rate * notional, minimum 1 cent on any sale
        sec_usd = notional_usd * self.SEC_RATE
        sec_usd = max(0.01, sec_usd) if notional_usd > 0 else 0.0

        # FINRA TAF: $0.000166/share, min $0.01, max $8.30
        taf_raw = fill_qty * self.FINRA_TAF_PER_SHARE
        taf_usd = min(max(taf_raw, self.FINRA_TAF_MIN), self.FINRA_TAF_MAX)

        total_usd = sec_usd + taf_usd
        total_eur = round(total_usd / fx, 6)

        return {
            "commission_eur": 0.0,
            "sec_fee_eur": round(sec_usd / fx, 6),
            "finra_taf_eur": round(taf_usd / fx, 6),
            "total_fee_eur": total_eur,
        }


class CryptoPercentageFeeModel(FeeModel):
    """Crypto exchange fee model (e.g. Binance standard tier).

    - Maker fee: 0.075% - 0.10% (applies to resting LIMIT orders)
    - Taker fee: 0.10% (applies to MARKET orders or immediate crossing LIMIT orders)
    """

    def __init__(
        self,
        maker_fee_pct: float = 0.00075,
        taker_fee_pct: float = 0.0010,
    ) -> None:
        self.maker_fee_pct = max(0.0, float(maker_fee_pct))
        self.taker_fee_pct = max(0.0, float(taker_fee_pct))

    def calculate_fee(
        self,
        order: OrderRequest,
        fill_price: float,
        fill_qty: float,
        fx_rate: float = 1.08,
    ) -> float:
        fx = max(0.0001, fx_rate)
        if fill_qty <= 0 or fill_price <= 0:
            return 0.0

        fee_rate = (
            self.maker_fee_pct
            if order.order_type == OrderType.LIMIT
            else self.taker_fee_pct
        )
        notional_quote = fill_price * fill_qty
        fee_quote = notional_quote * fee_rate

        # If quoted in USD/USDT, convert to EUR
        fee_eur = fee_quote / fx
        return round(fee_eur, 6)


class EuropeanRetailFeeModel(FeeModel):
    """European retail broker fee structure (e.g. myInvestor / DEGIRO).

    - Equity orders: Flat fee €1.50 + 0.12% of trade notional (min €2.00, max €10.00).
    - FX conversion markup: 0.25% for foreign currency (USD) trades from EUR base account.
    """

    def __init__(
        self,
        flat_fee_eur: float = 1.50,
        variable_pct: float = 0.0012,
        min_fee_eur: float = 2.00,
        max_fee_eur: float = 10.00,
        fx_markup_pct: float = 0.0025,
        is_foreign_asset: bool = True,
    ) -> None:
        self.flat_fee_eur = max(0.0, float(flat_fee_eur))
        self.variable_pct = max(0.0, float(variable_pct))
        self.min_fee_eur = max(0.0, float(min_fee_eur))
        self.max_fee_eur = max(self.min_fee_eur, float(max_fee_eur))
        self.fx_markup_pct = max(0.0, float(fx_markup_pct))
        self.is_foreign_asset = is_foreign_asset

    def calculate_fee(
        self,
        order: OrderRequest,
        fill_price: float,
        fill_qty: float,
        fx_rate: float = 1.08,
    ) -> float:
        breakdown = self.calculate_fee_breakdown(order, fill_price, fill_qty, fx_rate)
        return breakdown["total_fee_eur"]

    def calculate_fee_breakdown(
        self,
        order: OrderRequest,
        fill_price: float,
        fill_qty: float,
        fx_rate: float = 1.08,
    ) -> Dict[str, float]:
        if fill_qty <= 0 or fill_price <= 0:
            return {
                "commission_eur": 0.0,
                "fx_markup_eur": 0.0,
                "total_fee_eur": 0.0,
            }

        fx = max(0.0001, fx_rate)
        notional_eur = (fill_price * fill_qty) / fx if self.is_foreign_asset else (fill_price * fill_qty)

        # Base commission: flat + variable, clamped between min and max
        raw_comm = self.flat_fee_eur + (notional_eur * self.variable_pct)
        comm_eur = min(max(raw_comm, self.min_fee_eur), self.max_fee_eur)

        # FX markup fee
        fx_fee_eur = (notional_eur * self.fx_markup_pct) if self.is_foreign_asset else 0.0

        total_eur = round(comm_eur + fx_fee_eur, 6)
        return {
            "commission_eur": round(comm_eur, 6),
            "fx_markup_eur": round(fx_fee_eur, 6),
            "total_fee_eur": total_eur,
        }


class CompositeFeeModel(FeeModel):
    """Composite fee model aggregating multiple fee structures."""

    def __init__(self, models: Optional[List[FeeModel]] = None) -> None:
        self.models: List[FeeModel] = list(models) if models is not None else []

    def add_model(self, model: FeeModel) -> None:
        self.models.append(model)

    def calculate_fee(
        self,
        order: OrderRequest,
        fill_price: float,
        fill_qty: float,
        fx_rate: float = 1.08,
    ) -> float:
        return sum(m.calculate_fee(order, fill_price, fill_qty, fx_rate) for m in self.models)
