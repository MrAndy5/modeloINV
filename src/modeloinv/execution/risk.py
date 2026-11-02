"""Pre-Trade Risk Guardrails and Circuit Breakers for modeloINV.

Enforces institutional risk controls:
1. Hard Kill Switch with automated emergency liquidation.
2. Daily Drawdown Stop (Circuit Breaker) & Portfolio High-Water Mark Drawdown Stop.
3. Maximum Position Sizing (% of equity) and Maximum Open Positions.
4. Micro-Capital Minimum Lot (€1.00) and Cash Invariant Verification.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from modeloinv.core.models import (
    AccountInfo,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderType,
    Position,
)
from modeloinv.execution.base import BaseBroker, BrokerError

logger = logging.getLogger(__name__)


class RiskViolationError(BrokerError):
    """Raised when an order violates pre-trade risk constraints."""
    pass


class RiskManager:
    """Institutional pre-trade risk engine ensuring capital preservation on €100 capital."""

    def __init__(
        self,
        max_position_weight: float = 0.35,
        max_open_positions: int = 5,
        daily_drawdown_limit: float = 0.10,
        max_drawdown_limit: float = 0.15,
        kill_switch_drawdown: float = 0.25,
        min_order_notional_eur: float = 1.00,
        initial_capital_eur: float = 100.00,
    ) -> None:
        self.max_position_weight = float(max_position_weight)
        self.max_open_positions = int(max_open_positions)
        self.daily_drawdown_limit = float(daily_drawdown_limit)
        self.max_drawdown_limit = float(max_drawdown_limit)
        self.kill_switch_drawdown = float(kill_switch_drawdown)
        self.min_order_notional_eur = float(min_order_notional_eur)
        self.initial_capital_eur = float(initial_capital_eur)

        # State tracking
        self.kill_switch_active: bool = False
        self.kill_switch_reason: str = ""
        self.circuit_breaker_tripped: bool = False
        self.circuit_breaker_reason: str = ""

        self.high_water_mark_eur: float = initial_capital_eur
        self.daily_start_equity_eur: float = initial_capital_eur
        self.current_day: Optional[date] = None
        self.current_equity_eur: float = initial_capital_eur

    def update_equity(
        self,
        equity_eur: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Update current portfolio equity, evaluate drawdowns, and trigger circuit breakers if breached."""
        ts = timestamp or datetime.utcnow()
        day = ts.date()

        if self.current_day is None:
            self.current_day = day
        elif day != self.current_day:
            self.current_day = day
            self.daily_start_equity_eur = equity_eur
            # Reset daily circuit breaker on new trading day if kill switch is not active
            if self.circuit_breaker_tripped and "daily" in self.circuit_breaker_reason.lower():
                logger.info(f"New trading day {day}: resetting daily circuit breaker.")
                self.circuit_breaker_tripped = False
                self.circuit_breaker_reason = ""

        self.current_equity_eur = equity_eur

        # Update high-water mark
        if equity_eur > self.high_water_mark_eur:
            self.high_water_mark_eur = equity_eur

        # Check total drawdown from high-water mark
        if self.high_water_mark_eur > 0:
            total_dd = (self.high_water_mark_eur - equity_eur) / self.high_water_mark_eur
            if total_dd >= self.kill_switch_drawdown:
                self.activate_kill_switch(
                    f"Kill switch triggered: drawdown {total_dd*100:.2f}% >= {self.kill_switch_drawdown*100:.2f}%"
                )
            elif total_dd >= self.max_drawdown_limit:
                self.trip_circuit_breaker(
                    f"Circuit breaker tripped: max drawdown {total_dd*100:.2f}% >= {self.max_drawdown_limit*100:.2f}%"
                )

        # Check daily drawdown
        if self.daily_start_equity_eur > 0:
            daily_dd = (self.daily_start_equity_eur - equity_eur) / self.daily_start_equity_eur
            if daily_dd >= self.daily_drawdown_limit:
                self.trip_circuit_breaker(
                    f"Circuit breaker tripped: daily drawdown {daily_dd*100:.2f}% >= {self.daily_drawdown_limit*100:.2f}%"
                )

    def validate_order(
        self,
        order: OrderRequest,
        account: AccountInfo,
        positions: List[Position],
        current_market_price: float,
        fx_rate: float = 1.08,
    ) -> bool:
        """Execute all pre-trade risk validations.

        Raises:
            RiskViolationError if any guardrail is breached.
        Returns:
            True if all checks pass.
        """
        # 0. Emergency liquidation or risk-reduction orders unconditionally bypass all halts
        if getattr(order, "tag", None) in ("EMERGENCY_LIQUIDATION", "RISK_REDUCTION"):
            return True

        # Check if order reduces an existing held position
        has_held_position = any(
            p.symbol == order.symbol and p.quantity >= (1e-6 - 1e-9) for p in positions
        )
        is_risk_reducing = (order.side == OrderSide.SELL and has_held_position)

        # 1. Kill Switch check
        if self.kill_switch_active and not is_risk_reducing:
            raise RiskViolationError(f"Kill switch active: {self.kill_switch_reason}")

        # 2. Circuit Breaker check (halts new risk entries / BUY orders)
        if self.circuit_breaker_tripped and not is_risk_reducing:
            raise RiskViolationError(f"Risk circuit breaker tripped: {self.circuit_breaker_reason}")

        # Update equity state from account info
        self.update_equity(account.equity_eur)
        if self.kill_switch_active and not is_risk_reducing:
            raise RiskViolationError(f"Kill switch active: {self.kill_switch_reason}")
        if self.circuit_breaker_tripped and not is_risk_reducing:
            raise RiskViolationError(f"Risk circuit breaker tripped: {self.circuit_breaker_reason}")

        fx = max(0.0001, fx_rate)
        ref_price = current_market_price if current_market_price > 0 else 100.0

        # Calculate requested order notional in EUR
        if order.notional is not None and order.notional > 0:
            order_notional_eur = order.notional
            qty = (order.notional * fx) / ref_price
        elif order.quantity is not None and order.quantity > 0:
            qty = order.quantity
            order_notional_usd = qty * ref_price
            order_notional_eur = order_notional_usd / fx
        else:
            raise RiskViolationError("Order must specify positive quantity or notional.")

        # Sell orders reducing existing positions bypass position limits and buying power checks
        if order.side == OrderSide.SELL:
            if has_held_position:
                return True
            raise RiskViolationError(f"Cannot sell unowned symbol {order.symbol}: no open position held.")

        # 3. Minimum Order Notional check
        if order_notional_eur < (self.min_order_notional_eur - 1e-6):
            raise RiskViolationError(
                f"Order notional €{order_notional_eur:.2f} is below minimum lot €{self.min_order_notional_eur:.2f}"
            )

        # 4. Maximum Open Positions limit
        open_symbols = {p.symbol for p in positions if abs(p.quantity) > 1e-6}
        if order.symbol not in open_symbols and len(open_symbols) >= self.max_open_positions:
            raise RiskViolationError(
                f"Cannot open new position in {order.symbol}: "
                f"already at maximum {self.max_open_positions} open positions."
            )

        # 5. Maximum Position Size limit
        existing_val_eur = 0.0
        for p in positions:
            if p.symbol == order.symbol:
                existing_val_eur = (p.quantity * ref_price) / fx
                break

        total_post_notional_eur = existing_val_eur + order_notional_eur
        max_allowed_eur = account.equity_eur * self.max_position_weight

        if total_post_notional_eur > (max_allowed_eur + 1e-4):
            raise RiskViolationError(
                f"Position size in {order.symbol} would reach €{total_post_notional_eur:.2f}, "
                f"exceeding max position size of €{max_allowed_eur:.2f} ({self.max_position_weight*100:.1f}% of equity €{account.equity_eur:.2f})"
            )

        # 6. Purchasing power / Cash check
        if order_notional_eur > (account.buying_power_eur + 1e-4):
            raise RiskViolationError(
                f"Insufficient buying power: required €{order_notional_eur:.2f}, available €{account.buying_power_eur:.2f}"
            )

        return True

    def trip_circuit_breaker(self, reason: str) -> None:
        """Halt new order entries due to intraday loss or risk breach."""
        self.circuit_breaker_tripped = True
        self.circuit_breaker_reason = reason
        logger.warning(f"CIRCUIT BREAKER TRIPPED: {reason}")

    def reset_circuit_breaker(self) -> None:
        """Manually or scheduled reset of circuit breaker."""
        self.circuit_breaker_tripped = False
        self.circuit_breaker_reason = ""
        logger.info("Circuit breaker has been reset.")

    def activate_kill_switch(self, reason: str = "Manual activation") -> None:
        """Activate hard kill switch."""
        self.kill_switch_active = True
        self.kill_switch_reason = reason
        self.circuit_breaker_tripped = True
        self.circuit_breaker_reason = reason
        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")

    def deactivate_kill_switch(self) -> None:
        """Deactivate kill switch after manual verification."""
        self.kill_switch_active = False
        self.kill_switch_reason = ""
        self.circuit_breaker_tripped = False
        self.circuit_breaker_reason = ""
        logger.info("Kill switch deactivated.")

    def emergency_liquidate(self, broker: BaseBroker) -> List[OrderResult]:
        """Liquidate all open positions to 100% cash and cancel all pending orders."""
        logger.critical("EMERGENCY LIQUIDATION INITIATED")
        results: List[OrderResult] = []

        # 1. Cancel all open orders
        for order in broker.get_open_orders():
            try:
                broker.cancel_order(order.order_id)
            except Exception as e:
                logger.error(f"Failed to cancel order {order.order_id} during liquidation: {e}")

        # 2. Liquidate all positions
        for pos in broker.get_positions():
            if pos.quantity >= (1e-6 - 1e-9):
                sell_req = OrderRequest(
                    symbol=pos.symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=pos.quantity,
                    tag="EMERGENCY_LIQUIDATION",
                )
                try:
                    res = broker.submit_order(sell_req)
                    results.append(res)
                except Exception as e:
                    logger.error(f"Failed to liquidate position {pos.symbol}: {e}")

        # 3. Trip kill switch
        self.activate_kill_switch("Activated following emergency liquidation")
        return results
