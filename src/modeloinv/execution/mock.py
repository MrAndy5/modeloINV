"""Deterministic Mock Broker for Unit and Automated Integration Testing.

Provides scriptable fills, predictable price feeds, and injectable failure modes
(network timeouts, exchange rejections, partial fills, margin violations) with
zero network or database dependencies.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from modeloinv.core.models import (
    AccountInfo,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from modeloinv.execution.base import (
    BaseBroker,
    BrokerConnectionError,
    InsufficientFundsError,
    OrderNotFoundError,
    OrderRejectedError,
)

logger = logging.getLogger(__name__)


class MockBroker(BaseBroker):
    """Deterministic scriptable broker for offline testing and error injection."""

    def __init__(
        self,
        config: Optional[Any] = None,
        initial_capital_eur: float = 100.00,
        eur_usd_rate: float = 1.08,
    ) -> None:
        self.initial_capital_eur = float(initial_capital_eur)
        self.cash_eur = self.initial_capital_eur
        self.eur_usd_rate = float(eur_usd_rate)

        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, OrderResult] = {}
        self.custom_prices: Dict[str, float] = {}
        self.order_counter: int = 0

        # Injected failure modes
        self._injected_rejection_reason: Optional[str] = None
        self._inject_timeout: bool = False
        self._partial_fill_pct: Optional[float] = None
        self._inject_insufficient_funds: bool = False
        self._market_open: bool = True
        self._connected: bool = True

    def connect(self) -> bool:
        if self._inject_timeout:
            raise BrokerConnectionError("Mock network timeout during connect")
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    # -------------------------------------------------------------------------
    # Fault Injection API
    # -------------------------------------------------------------------------

    def inject_rejection(self, reason: str = "Injected mock rejection") -> None:
        """Force subsequent submit_order calls to return REJECTED status."""
        self._injected_rejection_reason = reason

    def inject_network_timeout(self, enable: bool = True) -> None:
        """Force subsequent broker calls to raise BrokerConnectionError."""
        self._inject_timeout = enable

    def inject_partial_fill(self, fill_pct: float = 0.50) -> None:
        """Force orders to fill partially by specified fraction (0.0 to 1.0)."""
        self._partial_fill_pct = max(0.01, min(0.99, float(fill_pct)))

    def inject_insufficient_funds(self, enable: bool = True) -> None:
        """Force orders to fail with insufficient funds."""
        self._inject_insufficient_funds = enable

    def set_market_open(self, is_open: bool) -> None:
        """Control market open/closed status."""
        self._market_open = is_open

    def set_custom_price(self, symbol: str, price: float) -> None:
        """Set mock market price for symbol."""
        self.custom_prices[symbol] = max(0.0001, float(price))
        if symbol in self.positions:
            self.positions[symbol].update_market_price(price, self.eur_usd_rate)

    def clear_injections(self) -> None:
        """Clear all injected faults and error behaviors."""
        self._injected_rejection_reason = None
        self._inject_timeout = False
        self._partial_fill_pct = None
        self._inject_insufficient_funds = False
        self._market_open = True

    # -------------------------------------------------------------------------
    # BaseBroker Methods
    # -------------------------------------------------------------------------

    def get_account(self) -> AccountInfo:
        if self._inject_timeout:
            raise BrokerConnectionError("Mock network timeout")

        positions_val_eur = sum(
            (p.quantity * p.current_price) / self.eur_usd_rate for p in self.positions.values()
        )
        equity_eur = self.cash_eur + positions_val_eur
        return AccountInfo(
            currency="EUR",
            cash_eur=round(self.cash_eur, 4),
            equity_eur=round(equity_eur, 4),
            initial_capital_eur=self.initial_capital_eur,
            realized_pnl_eur=round(equity_eur - self.initial_capital_eur, 4),
            unrealized_pnl_eur=round(positions_val_eur, 4),
            buying_power_eur=round(max(0.0, self.cash_eur), 4),
            is_live=False,
            timestamp=datetime.utcnow(),
        )

    def get_positions(self) -> List[Position]:
        if self._inject_timeout:
            raise BrokerConnectionError("Mock network timeout")
        return [p for p in self.positions.values() if p.quantity > 1e-6]

    def get_position(self, symbol: str) -> Optional[Position]:
        if self._inject_timeout:
            raise BrokerConnectionError("Mock network timeout")
        pos = self.positions.get(symbol)
        if pos is not None and pos.quantity > 1e-6:
            return pos
        return None

    def get_latest_price(self, symbol: str) -> float:
        if symbol in self.custom_prices:
            return self.custom_prices[symbol]
        pos = self.get_position(symbol)
        if pos is not None:
            return pos.current_price
        return 100.0

    def is_market_open(self, symbol: str) -> bool:
        return self._market_open

    def submit_order(
        self,
        request: OrderRequest,
        current_market_price: Optional[float] = None,
    ) -> OrderResult:
        if self._inject_timeout:
            raise BrokerConnectionError("Mock network timeout on order submission")

        self.order_counter += 1
        order_id = f"mock_{self.order_counter:04d}"

        # 1. Injected Rejection
        if self._injected_rejection_reason is not None:
            res = OrderResult(
                order_id=order_id,
                client_order_id=request.client_order_id,
                symbol=request.symbol,
                status=OrderStatus.REJECTED,
                side=request.side,
                timestamp=datetime.utcnow(),
                message=self._injected_rejection_reason,
            )
            self.orders[order_id] = res
            return res

        # 2. Market Closed check
        if not self._market_open:
            res = OrderResult(
                order_id=order_id,
                client_order_id=request.client_order_id,
                symbol=request.symbol,
                status=OrderStatus.REJECTED,
                side=request.side,
                timestamp=datetime.utcnow(),
                message="Market is closed",
            )
            self.orders[order_id] = res
            return res

        ref_price = (
            current_market_price
            if current_market_price is not None and current_market_price > 0
            else self.get_latest_price(request.symbol)
        )

        # Calculate requested quantity
        if request.quantity is not None and request.quantity > 0:
            qty = round(request.quantity, 6)
        elif request.notional is not None and request.notional > 0:
            notional_usd = request.notional * self.eur_usd_rate
            qty = round(notional_usd / ref_price, 6)
        else:
            qty = 1.0

        # Slippage: 2 bps
        slippage_rate = 0.0002
        exec_price = (
            ref_price * (1.0 + slippage_rate)
            if request.side == OrderSide.BUY
            else ref_price * (1.0 - slippage_rate)
        )

        trade_cost_eur = (qty * exec_price) / self.eur_usd_rate
        fee_eur = round(0.01 / self.eur_usd_rate, 4)

        # 3. Injected or Organic Insufficient Funds check
        if self._inject_insufficient_funds or (
            request.side == OrderSide.BUY and (trade_cost_eur + fee_eur) > self.cash_eur
        ):
            res = OrderResult(
                order_id=order_id,
                client_order_id=request.client_order_id,
                symbol=request.symbol,
                status=OrderStatus.REJECTED,
                side=request.side,
                timestamp=datetime.utcnow(),
                message=f"Insufficient funds: required {trade_cost_eur:.2f} EUR, available {self.cash_eur:.2f} EUR",
            )
            self.orders[order_id] = res
            return res

        # 4. Partial Fill injection
        fill_qty = qty
        status = OrderStatus.FILLED
        if self._partial_fill_pct is not None:
            fill_qty = round(qty * self._partial_fill_pct, 6)
            status = OrderStatus.PARTIALLY_FILLED

        # 5. Position and Cash ledger update
        if request.side == OrderSide.BUY:
            actual_cost_eur = (fill_qty * exec_price) / self.eur_usd_rate + fee_eur
            self.cash_eur -= actual_cost_eur
            if request.symbol in self.positions:
                pos = self.positions[request.symbol]
                new_q = pos.quantity + fill_qty
                pos.entry_price = (pos.entry_price * pos.quantity + exec_price * fill_qty) / new_q
                pos.quantity = new_q
                pos.current_price = exec_price
            else:
                self.positions[request.symbol] = Position(
                    symbol=request.symbol,
                    quantity=fill_qty,
                    entry_price=exec_price,
                    current_price=exec_price,
                    currency="USD",
                    entry_fx_rate=self.eur_usd_rate,
                    cost_basis_eur=actual_cost_eur,
                    market_value_eur=actual_cost_eur,
                )
        else:
            held = self.positions.get(request.symbol)
            if held is None or held.quantity < fill_qty:
                res = OrderResult(
                    order_id=order_id,
                    client_order_id=request.client_order_id,
                    symbol=request.symbol,
                    status=OrderStatus.REJECTED,
                    side=request.side,
                    timestamp=datetime.utcnow(),
                    message="Insufficient position quantity to sell",
                )
                self.orders[order_id] = res
                return res

            proceeds_eur = (fill_qty * exec_price) / self.eur_usd_rate - fee_eur
            self.cash_eur += proceeds_eur
            held.quantity -= fill_qty
            if held.quantity <= 1e-6:
                del self.positions[request.symbol]

        res = OrderResult(
            order_id=order_id,
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            status=status,
            side=request.side,
            filled_quantity=fill_qty,
            filled_avg_price=round(exec_price, 4),
            fee_eur=fee_eur,
            slippage_eur=round(abs(exec_price - ref_price) * fill_qty / self.eur_usd_rate, 4),
            timestamp=datetime.utcnow(),
            message="Executed by MockBroker",
        )
        self.orders[order_id] = res
        return res

    def cancel_order(self, order_id: str) -> bool:
        if self._inject_timeout:
            raise BrokerConnectionError("Mock network timeout on cancel_order")
        if order_id in self.orders and self.orders[order_id].is_active:
            self.orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False

    def get_order_status(self, order_id: str) -> OrderResult:
        if self._inject_timeout:
            raise BrokerConnectionError("Mock network timeout on get_order_status")
        if order_id in self.orders:
            return self.orders[order_id]
        raise OrderNotFoundError(f"Mock order {order_id} not found")

    def get_open_orders(self) -> List[OrderResult]:
        return [o for o in self.orders.values() if o.is_active]
