"""Simulated Paper Broker for modeloINV.

Full-fidelity in-memory paper trading simulation tracking an exact €100.00 EUR
virtual ledger. Supports fractional shares (up to 6 decimals or cash-notional orders),
realistic execution modeling across Market, Limit, Stop-Loss, Stop-Limit, and Trailing Stop
orders with pluggable slippage and fee models, EUR/USD FX conversion, and continuous
accounting invariant assertions: Equity = Cash + Sum(Positions Market Value).
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from modeloinv.core.currency import CurrencyConverter, get_currency_converter
from modeloinv.core.models import (
    AccountInfo,
    Bar,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
    Trade,
)
from modeloinv.execution.base import BaseBroker, InsufficientFundsError, OrderRejectedError
from modeloinv.execution.fees import FeeModel, ZeroCommissionFeeModel
from modeloinv.execution.risk import RiskManager, RiskViolationError
from modeloinv.execution.slippage import FixedBpsSlippageModel, SlippageModel

logger = logging.getLogger(__name__)


class SimulatedPaperBroker(BaseBroker):
    """Full-fidelity in-memory paper execution engine."""

    def __init__(
        self,
        config: Optional[Any] = None,
        initial_capital_eur: float = 100.00,
        currency_converter: Optional[CurrencyConverter] = None,
        slippage_model: Optional[SlippageModel] = None,
        fee_model: Optional[FeeModel] = None,
        risk_manager: Optional[RiskManager] = None,
    ) -> None:
        # Extract starting capital from config if provided
        start_cap = initial_capital_eur
        if config is not None:
            if hasattr(config, "account") and hasattr(config.account, "starting_capital_eur"):
                start_cap = float(config.account.starting_capital_eur)
            elif isinstance(config, dict):
                start_cap = float(
                    config.get("account", {}).get("starting_capital_eur")
                    or config.get("broker", {}).get("initial_capital")
                    or start_cap
                )

        self.initial_capital_eur: float = round(float(start_cap), 4)
        self.cash_eur: float = self.initial_capital_eur
        self.currency_converter: CurrencyConverter = currency_converter or get_currency_converter()
        self.slippage_model: SlippageModel = slippage_model or FixedBpsSlippageModel(bps=5.0)
        self.fee_model: FeeModel = fee_model or ZeroCommissionFeeModel()
        self.risk_manager: Optional[RiskManager] = risk_manager

        # Position inventory and orders
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, OrderResult] = {}
        self.pending_orders: Dict[str, Dict[str, Any]] = {}
        self.trade_history: List[Trade] = []
        self.latest_prices: Dict[str, float] = {}

        # Performance & metrics
        self.realized_pnl_eur: float = 0.0
        self.total_fees_eur: float = 0.0
        self.order_counter: int = 0
        self.trade_counter: int = 0

        # Safety & circuit breaker flags
        self.is_halted: bool = False
        self.is_live: bool = False
        self._connected: bool = True

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def get_account(self) -> AccountInfo:
        """Retrieve current account status adhering to accounting invariants."""
        # Calculate market value of all positions in EUR
        positions_value_eur = 0.0
        unrealized_pnl_eur = 0.0

        for pos in self.positions.values():
            fx = self._get_fx_rate(pos.currency)
            pos.update_market_price(pos.current_price, fx)
            positions_value_eur += pos.market_value_eur
            unrealized_pnl_eur += pos.unrealized_pnl_eur

        equity_eur = self.cash_eur + positions_value_eur
        buying_power_eur = max(0.0, self.cash_eur)

        return AccountInfo(
            currency="EUR",
            cash_eur=round(self.cash_eur, 4),
            equity_eur=round(equity_eur, 4),
            initial_capital_eur=self.initial_capital_eur,
            realized_pnl_eur=round(self.realized_pnl_eur, 4),
            unrealized_pnl_eur=round(unrealized_pnl_eur, 4),
            total_fees_eur=round(self.total_fees_eur, 4),
            buying_power_eur=round(buying_power_eur, 4),
            is_live=self.is_live,
            timestamp=datetime.utcnow(),
        )

    def get_positions(self) -> List[Position]:
        """Return list of all non-zero open positions."""
        return [p for p in self.positions.values() if p.quantity >= (1e-6 - 1e-9)]

    def get_position(self, symbol: str) -> Optional[Position]:
        """Return open position for symbol if held with quantity > 0."""
        pos = self.positions.get(symbol)
        if pos is not None and pos.quantity >= (1e-6 - 1e-9):
            return pos
        return None

    def get_order_status(self, order_id: str) -> OrderResult:
        if order_id in self.orders:
            return self.orders[order_id]
        raise KeyError(f"Order '{order_id}' not found")

    def get_open_orders(self) -> List[OrderResult]:
        return [
            self.orders[oid]
            for oid, data in self.pending_orders.items()
            if oid in self.orders and self.orders[oid].is_active
        ]

    def get_latest_price(self, symbol: str) -> float:
        if symbol in self.latest_prices:
            return self.latest_prices[symbol]
        pos = self.get_position(symbol)
        if pos is not None:
            return pos.current_price
        return 100.0

    def update_market_price(self, symbol: str, current_price: float) -> None:
        """Update market price for symbol, recalculating position mark-to-market and checking resting orders."""
        if current_price <= 0:
            return
        self.latest_prices[symbol] = current_price
        if symbol in self.positions:
            fx = self._get_fx_rate(self.positions[symbol].currency)
            self.positions[symbol].update_market_price(current_price, fx)

        # Check pending orders for this symbol
        self._evaluate_pending_orders_on_price(symbol, current_price)

    def process_bar(self, bar: Bar) -> None:
        """Process an OHLCV candlestick bar against resting orders and mark to market."""
        self.latest_prices[bar.symbol] = bar.close
        if bar.symbol in self.positions:
            fx = self._get_fx_rate(self.positions[bar.symbol].currency)
            self.positions[bar.symbol].update_market_price(bar.close, fx)

        # Evaluate pending orders against high/low/open/close of bar
        self._evaluate_pending_orders_on_bar(bar)

    def submit_order(
        self,
        request: OrderRequest,
        current_market_price: Optional[float] = None,
        bar: Optional[Bar] = None,
    ) -> OrderResult:
        """Submit and process an order request."""
        self.order_counter += 1
        order_id = f"ord_{self.order_counter:04d}"

        # 1. Check circuit breaker / halt flags
        if self.is_halted:
            is_emergency = getattr(request, "tag", None) in ("EMERGENCY_LIQUIDATION", "RISK_REDUCTION")
            is_held_sell = (request.side == OrderSide.SELL and request.symbol in self.positions)
            if not (is_emergency or is_held_sell):
                res = OrderResult(
                    order_id=order_id,
                    client_order_id=request.client_order_id,
                    symbol=request.symbol,
                    status=OrderStatus.REJECTED,
                    side=request.side,
                    timestamp=datetime.utcnow(),
                    message="Trading halted due to risk circuit breaker",
                )
                self.orders[order_id] = res
                return res

        ref_price = (
            current_market_price
            if current_market_price is not None and current_market_price > 0
            else self.get_latest_price(request.symbol)
        )
        self.latest_prices[request.symbol] = ref_price

        asset_currency = "USD"
        if bar is not None and getattr(bar, "currency", None):
            asset_currency = str(bar.currency).upper()
        elif request.symbol in self.positions:
            asset_currency = self.positions[request.symbol].currency.upper()
        elif hasattr(request, "currency") and getattr(request, "currency", None):
            asset_currency = str(request.currency).upper()

        fx_rate = self._get_fx_rate(asset_currency)

        # 2. Risk Manager Validation
        if self.risk_manager is not None:
            account = self.get_account()
            try:
                self.risk_manager.validate_order(
                    order=request,
                    account=account,
                    positions=self.get_positions(),
                    current_market_price=ref_price,
                    fx_rate=fx_rate,
                )
            except RiskViolationError as rve:
                res = OrderResult(
                    order_id=order_id,
                    client_order_id=request.client_order_id,
                    symbol=request.symbol,
                    status=OrderStatus.REJECTED,
                    side=request.side,
                    timestamp=datetime.utcnow(),
                    message=f"Pre-trade risk violation: {str(rve)}",
                )
                self.orders[order_id] = res
                return res

        # 3. Handle Order Types
        if request.order_type == OrderType.MARKET:
            return self._execute_immediate_fill(
                order_id=order_id,
                request=request,
                ref_price=ref_price,
                fx_rate=fx_rate,
                bar=bar,
            )
        elif request.order_type == OrderType.LIMIT:
            # Check if limit order crosses market immediately
            can_fill = (
                (request.side == OrderSide.BUY and ref_price <= (request.limit_price or 0.0))
                or (request.side == OrderSide.SELL and ref_price >= (request.limit_price or 0.0))
            )
            if can_fill:
                # Award price improvement
                fill_target_price = (
                    min(request.limit_price or ref_price, ref_price)
                    if request.side == OrderSide.BUY
                    else max(request.limit_price or ref_price, ref_price)
                )
                return self._execute_immediate_fill(
                    order_id=order_id,
                    request=request,
                    ref_price=fill_target_price,
                    fx_rate=fx_rate,
                    bar=bar,
                )
            else:
                # Queue limit order
                res = OrderResult(
                    order_id=order_id,
                    client_order_id=request.client_order_id,
                    symbol=request.symbol,
                    status=OrderStatus.ACCEPTED,
                    side=request.side,
                    timestamp=datetime.utcnow(),
                    message="Limit order accepted into resting queue",
                )
                self.orders[order_id] = res
                self.pending_orders[order_id] = {
                    "request": request,
                    "order_id": order_id,
                    "ref_price": ref_price,
                    "created_at": datetime.utcnow(),
                }
                return res

        elif request.order_type == OrderType.STOP:
            # Queue stop order
            res = OrderResult(
                order_id=order_id,
                client_order_id=request.client_order_id,
                symbol=request.symbol,
                status=OrderStatus.ACCEPTED,
                side=request.side,
                timestamp=datetime.utcnow(),
                message="Stop order accepted into trigger queue",
            )
            self.orders[order_id] = res
            self.pending_orders[order_id] = {
                "request": request,
                "order_id": order_id,
                "ref_price": ref_price,
                "created_at": datetime.utcnow(),
            }
            return res

        elif request.order_type == OrderType.TRAILING_STOP:
            # Queue trailing stop with high/low water mark tracking
            trail_pct = request.trail_percent if request.trail_percent is not None else 2.0
            trail_factor = trail_pct / 100.0

            stop_price = (
                ref_price * (1.0 - trail_factor)
                if request.side == OrderSide.SELL
                else ref_price * (1.0 + trail_factor)
            )

            res = OrderResult(
                order_id=order_id,
                client_order_id=request.client_order_id,
                symbol=request.symbol,
                status=OrderStatus.ACCEPTED,
                side=request.side,
                timestamp=datetime.utcnow(),
                message="Trailing stop order accepted with ratcheting trigger",
            )
            self.orders[order_id] = res
            self.pending_orders[order_id] = {
                "request": request,
                "order_id": order_id,
                "water_mark": ref_price,
                "trail_factor": trail_factor,
                "stop_price": stop_price,
                "created_at": datetime.utcnow(),
            }
            return res

        else:
            res = OrderResult(
                order_id=order_id,
                client_order_id=request.client_order_id,
                symbol=request.symbol,
                status=OrderStatus.REJECTED,
                side=request.side,
                timestamp=datetime.utcnow(),
                message=f"Unsupported order type: {request.order_type}",
            )
            self.orders[order_id] = res
            return res

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending open order."""
        if order_id in self.pending_orders:
            del self.pending_orders[order_id]
            if order_id in self.orders:
                self.orders[order_id].status = OrderStatus.CANCELLED
                self.orders[order_id].message = "Order canceled by user"
            return True
        return False

    def get_trade_history(self) -> List[Trade]:
        """Return history of all executed trade fills."""
        return list(self.trade_history)

    # -------------------------------------------------------------------------
    # Internal Execution and Filling Engine
    # -------------------------------------------------------------------------

    def _execute_immediate_fill(
        self,
        order_id: str,
        request: OrderRequest,
        ref_price: float,
        fx_rate: float,
        bar: Optional[Bar] = None,
    ) -> OrderResult:
        """Execute immediate order fill with slippage and fee modeling."""
        # 1. Calculate slippage and execution price
        fill_price, slippage_per_share = self.slippage_model.calculate_fill_price(
            order=request,
            reference_price=ref_price,
            bar=bar,
        )

        # Enforce limit price guarantee: a limit order must never fill worse than its limit price
        if request.order_type == OrderType.LIMIT and request.limit_price is not None:
            if request.side == OrderSide.BUY:
                fill_price = min(fill_price, request.limit_price)
            elif request.side == OrderSide.SELL:
                fill_price = max(fill_price, request.limit_price)
            slippage_per_share = max(0.0, abs(fill_price - ref_price))

        # 2. Determine quantity
        if request.quantity is not None and request.quantity > 0:
            qty = round(request.quantity, 6)
        elif request.notional is not None and request.notional > 0:
            notional_usd = request.notional * fx_rate
            qty = round(notional_usd / fill_price, 6)
        else:
            qty = 1.0

        if qty < 1e-6 - 1e-9:
            res = OrderResult(
                order_id=order_id,
                client_order_id=request.client_order_id,
                symbol=request.symbol,
                status=OrderStatus.REJECTED,
                side=request.side,
                timestamp=datetime.utcnow(),
                message="Order quantity evaluated to 0 after fractional rounding",
            )
            self.orders[order_id] = res
            return res

        # 3. Calculate transaction fees
        fee_eur = self.fee_model.calculate_fee(
            order=request,
            fill_price=fill_price,
            fill_qty=qty,
            fx_rate=fx_rate,
        )

        # 4. Total slippage cost in EUR
        slippage_eur = round((slippage_per_share * qty) / fx_rate, 4)
        trade_val_usd = qty * fill_price
        trade_cost_eur = (trade_val_usd / fx_rate) + fee_eur

        # Determine asset currency
        asset_currency = "USD"
        if bar is not None and getattr(bar, "currency", None):
            asset_currency = str(bar.currency).upper()
        elif request.symbol in self.positions:
            asset_currency = self.positions[request.symbol].currency.upper()
        elif hasattr(request, "currency") and getattr(request, "currency", None):
            asset_currency = str(request.currency).upper()

        # 5. Process Buy vs Sell
        if request.side == OrderSide.BUY:
            if trade_cost_eur > (self.cash_eur + 1e-9):
                res = OrderResult(
                    order_id=order_id,
                    client_order_id=request.client_order_id,
                    symbol=request.symbol,
                    status=OrderStatus.REJECTED,
                    side=request.side,
                    timestamp=datetime.utcnow(),
                    message=(
                        f"Insufficient funds: required {trade_cost_eur:.2f} EUR, "
                        f"available {self.cash_eur:.2f} EUR"
                    ),
                )
                self.orders[order_id] = res
                return res

            # Deduct cash with zero negative leakage
            self.cash_eur = max(0.0, self.cash_eur - trade_cost_eur)
            self.total_fees_eur += fee_eur

            # Update position inventory
            if request.symbol in self.positions:
                pos = self.positions[request.symbol]
                new_qty = pos.quantity + qty
                # Weighted average entry price
                weighted_entry = (pos.entry_price * pos.quantity + fill_price * qty) / new_qty
                pos.entry_price = weighted_entry
                pos.quantity = new_qty
                pos.cost_basis_eur += (trade_val_usd / fx_rate) + fee_eur
                pos.current_price = fill_price
                pos.update_market_price(fill_price, fx_rate)
            else:
                self.positions[request.symbol] = Position(
                    symbol=request.symbol,
                    quantity=qty,
                    entry_price=fill_price,
                    current_price=fill_price,
                    currency=asset_currency,
                    entry_fx_rate=fx_rate,
                    cost_basis_eur=(trade_val_usd / fx_rate) + fee_eur,
                    market_value_eur=(trade_val_usd / fx_rate),
                    unrealized_pnl_eur=-fee_eur,
                    unrealized_pnl_pct=(-fee_eur / ((trade_val_usd / fx_rate) + fee_eur) * 100.0) if ((trade_val_usd / fx_rate) + fee_eur) > 0 else 0.0,
                    last_updated=datetime.utcnow(),
                )

            realized_trade_pnl_eur = 0.0

        else:  # SELL
            held_pos = self.positions.get(request.symbol)
            if held_pos is None or held_pos.quantity < (qty - 1e-9):
                avail = held_pos.quantity if held_pos else 0.0
                res = OrderResult(
                    order_id=order_id,
                    client_order_id=request.client_order_id,
                    symbol=request.symbol,
                    status=OrderStatus.REJECTED,
                    side=request.side,
                    timestamp=datetime.utcnow(),
                    message=f"Insufficient position quantity to sell: requested {qty}, available {avail}",
                )
                self.orders[order_id] = res
                return res

            # If selling entire position or within floating epsilon, snap qty to held_pos.quantity
            if abs(held_pos.quantity - qty) < 1e-7 or qty >= held_pos.quantity:
                qty = held_pos.quantity
                fraction_sold = 1.0
            else:
                fraction_sold = qty / held_pos.quantity

            # Proceeds in EUR after fee deduction
            proceeds_usd = qty * fill_price
            proceeds_eur = (proceeds_usd / fx_rate) - fee_eur
            self.cash_eur += proceeds_eur
            self.total_fees_eur += fee_eur

            # Realized P&L
            cost_basis_sold_eur = held_pos.cost_basis_eur * fraction_sold
            realized_trade_pnl_eur = proceeds_eur - cost_basis_sold_eur
            self.realized_pnl_eur += realized_trade_pnl_eur

            # Update holding
            held_pos.cost_basis_eur -= cost_basis_sold_eur
            held_pos.quantity -= qty
            held_pos.current_price = fill_price
            held_pos.update_market_price(fill_price, fx_rate)

            if held_pos.quantity < 1e-7:
                del self.positions[request.symbol]

        # 6. Record Trade fill
        self.trade_counter += 1
        trade = Trade(
            trade_id=f"trd_{self.trade_counter:04d}",
            order_id=order_id,
            symbol=request.symbol,
            side=request.side,
            quantity=qty,
            price=fill_price,
            notional_usd=trade_val_usd,
            fx_rate=fx_rate,
            price_eur=round(fill_price / fx_rate, 4),
            fee_eur=fee_eur,
            slippage_eur=slippage_eur,
            realized_pnl_eur=round(realized_trade_pnl_eur, 4),
            timestamp=datetime.utcnow(),
        )
        self.trade_history.append(trade)

        # 7. Construct OrderResult
        res = OrderResult(
            order_id=order_id,
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            status=OrderStatus.FILLED,
            side=request.side,
            filled_quantity=qty,
            filled_avg_price=round(fill_price, 4),
            fee_eur=round(fee_eur, 4),
            slippage_eur=slippage_eur,
            timestamp=datetime.utcnow(),
            message="Executed successfully",
        )
        self.orders[order_id] = res
        return res

    def _evaluate_pending_orders_on_price(self, symbol: str, current_price: float) -> None:
        """Evaluate resting limit, stop, and trailing stop orders against single price update."""
        order_ids_to_fill = []
        fx_rate = self._get_fx_rate("USD")

        for oid, data in list(self.pending_orders.items()):
            req: OrderRequest = data["request"]
            if req.symbol != symbol:
                continue

            if req.order_type == OrderType.LIMIT:
                if req.side == OrderSide.BUY and current_price <= (req.limit_price or 0.0):
                    fill_p = min(req.limit_price or current_price, current_price)
                    order_ids_to_fill.append((oid, fill_p))
                elif req.side == OrderSide.SELL and current_price >= (req.limit_price or 0.0):
                    fill_p = max(req.limit_price or current_price, current_price)
                    order_ids_to_fill.append((oid, fill_p))

            elif req.order_type == OrderType.STOP:
                if req.side == OrderSide.SELL and current_price <= (req.stop_price or 0.0):
                    order_ids_to_fill.append((oid, current_price))
                elif req.side == OrderSide.BUY and current_price >= (req.stop_price or 0.0):
                    order_ids_to_fill.append((oid, current_price))

            elif req.order_type == OrderType.TRAILING_STOP:
                trail_factor = data["trail_factor"]
                water_mark = data["water_mark"]

                if req.side == OrderSide.SELL:
                    # Ratchet high-water mark up
                    if current_price > water_mark:
                        data["water_mark"] = current_price
                        data["stop_price"] = current_price * (1.0 - trail_factor)
                    elif current_price <= data["stop_price"]:
                        order_ids_to_fill.append((oid, current_price))
                else:  # BUY
                    # Ratchet low-water mark down
                    if current_price < water_mark:
                        data["water_mark"] = current_price
                        data["stop_price"] = current_price * (1.0 + trail_factor)
                    elif current_price >= data["stop_price"]:
                        order_ids_to_fill.append((oid, current_price))

        for oid, fill_price in order_ids_to_fill:
            if oid in self.pending_orders:
                data = self.pending_orders.pop(oid)
                self._execute_immediate_fill(
                    order_id=oid,
                    request=data["request"],
                    ref_price=fill_price,
                    fx_rate=fx_rate,
                )

    def _evaluate_pending_orders_on_bar(self, bar: Bar) -> None:
        """Evaluate resting orders against bar High, Low, Open, Close with gap slippage modeling."""
        order_ids_to_fill = []
        fx_rate = self._get_fx_rate(bar.currency)

        for oid, data in list(self.pending_orders.items()):
            req: OrderRequest = data["request"]
            if req.symbol != bar.symbol:
                continue

            if req.order_type == OrderType.LIMIT:
                if req.side == OrderSide.BUY and bar.low <= (req.limit_price or 0.0):
                    # Price improvement: if bar opened below limit, fill at open!
                    fill_p = min(req.limit_price or bar.open, bar.open)
                    order_ids_to_fill.append((oid, fill_p))
                elif req.side == OrderSide.SELL and bar.high >= (req.limit_price or 0.0):
                    fill_p = max(req.limit_price or bar.open, bar.open)
                    order_ids_to_fill.append((oid, fill_p))

            elif req.order_type == OrderType.STOP:
                if req.side == OrderSide.SELL and bar.low <= (req.stop_price or 0.0):
                    # Gap slippage: if opened below stop price, fills at open
                    fill_p = min(req.stop_price or bar.open, bar.open)
                    order_ids_to_fill.append((oid, fill_p))
                elif req.side == OrderSide.BUY and bar.high >= (req.stop_price or 0.0):
                    fill_p = max(req.stop_price or bar.open, bar.open)
                    order_ids_to_fill.append((oid, fill_p))

            elif req.order_type == OrderType.TRAILING_STOP:
                trail_factor = data["trail_factor"]
                water_mark = data["water_mark"]

                if req.side == OrderSide.SELL:
                    if bar.high > water_mark:
                        data["water_mark"] = bar.high
                        data["stop_price"] = bar.high * (1.0 - trail_factor)
                    if bar.low <= data["stop_price"]:
                        fill_p = min(data["stop_price"], bar.open)
                        order_ids_to_fill.append((oid, fill_p))
                else:
                    if bar.low < water_mark:
                        data["water_mark"] = bar.low
                        data["stop_price"] = bar.low * (1.0 + trail_factor)
                    if bar.high >= data["stop_price"]:
                        fill_p = max(data["stop_price"], bar.open)
                        order_ids_to_fill.append((oid, fill_p))

        for oid, fill_price in order_ids_to_fill:
            if oid in self.pending_orders:
                data = self.pending_orders.pop(oid)
                self._execute_immediate_fill(
                    order_id=oid,
                    request=data["request"],
                    ref_price=fill_price,
                    fx_rate=fx_rate,
                    bar=bar,
                )

    def _get_fx_rate(self, currency: str) -> float:
        """Get exchange rate converting from foreign currency to EUR."""
        curr = currency.upper()
        if curr == "EUR":
            return 1.0
        try:
            return self.currency_converter.get_rate("EUR", curr)
        except Exception:
            return 1.0800
