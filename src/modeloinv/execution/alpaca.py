"""Alpaca REST API v2 Paper and Live Broker Adapter for modeloINV.

Integrates with Alpaca REST API v2 Paper Trading Sandbox (https://paper-api.alpaca.markets)
and Live Trading gateway. Enforces a strict Virtual Sub-Account Partition (€100.00 virtual
capital ceiling) over Alpaca's default $100k sandbox account, ensuring execution fidelity
and preventing over-leveraging on micro-capital strategies.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import requests

from modeloinv.core.currency import CurrencyConverter, get_currency_converter
from modeloinv.core.models import (
    AccountInfo,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
)
from modeloinv.execution.base import (
    BaseBroker,
    BrokerConnectionError,
    BrokerError,
    InsufficientFundsError,
    InvalidOrderError,
    OrderNotFoundError,
    OrderRejectedError,
)

logger = logging.getLogger(__name__)


class AlpacaPaperBroker(BaseBroker):
    """Alpaca REST API v2 adapter with strict €100 virtual capital ceiling."""

    DEFAULT_PAPER_URL = "https://paper-api.alpaca.markets"
    DEFAULT_LIVE_URL = "https://api.alpaca.markets"

    def __init__(
        self,
        config: Optional[Any] = None,
        live: bool = False,
        api_key_id: Optional[str] = None,
        api_secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        virtual_capital_eur: float = 100.00,
        currency_converter: Optional[CurrencyConverter] = None,
        http_session: Optional[Any] = None,
    ) -> None:
        self.live = live
        self.virtual_capital_eur = float(virtual_capital_eur)
        self.virtual_cash_eur = self.virtual_capital_eur
        self.currency_converter = currency_converter or get_currency_converter()
        self.http_session = http_session or requests.Session()

        self.api_key_id: str = api_key_id or ""
        self.api_secret_key: str = api_secret_key or ""
        self.base_url: str = base_url or (self.DEFAULT_LIVE_URL if self.live else self.DEFAULT_PAPER_URL)

        # Extract settings from config if available
        if config is not None:
            if hasattr(config, "is_live"):
                self.live = config.is_live()
            if hasattr(config, "alpaca"):
                self.api_key_id = config.alpaca.api_key_id
                self.api_secret_key = config.alpaca.api_secret_key
                self.base_url = (
                    config.alpaca.live_base_url if self.live else config.alpaca.base_url
                )
                if hasattr(config.alpaca, "virtual_capital_eur"):
                    self.virtual_capital_eur = float(config.alpaca.virtual_capital_eur)
                    self.virtual_cash_eur = self.virtual_capital_eur
            elif isinstance(config, dict):
                alpaca_sec = config.get("alpaca", {})
                self.api_key_id = (
                    alpaca_sec.get("api_key_id")
                    or config.get("api_key_id")
                    or self.api_key_id
                )
                self.api_secret_key = (
                    alpaca_sec.get("api_secret_key")
                    or config.get("api_secret_key")
                    or self.api_secret_key
                )
                default_url = self.DEFAULT_LIVE_URL if self.live else self.DEFAULT_PAPER_URL
                self.base_url = alpaca_sec.get("base_url") or default_url
                if "virtual_capital_eur" in alpaca_sec:
                    self.virtual_capital_eur = float(alpaca_sec["virtual_capital_eur"])
                    self.virtual_cash_eur = self.virtual_capital_eur

        # Fallback to environment variables
        if not self.api_key_id:
            self.api_key_id = (
                os.getenv("MODELOINV_ALPACA_API_KEY_ID")
                or os.getenv("MODELOINV_ALPACA_API_KEY")
                or os.getenv("APCA_API_KEY_ID")
                or ""
            )
        if not self.api_secret_key:
            self.api_secret_key = (
                os.getenv("MODELOINV_ALPACA_API_SECRET_KEY")
                or os.getenv("MODELOINV_ALPACA_SECRET_KEY")
                or os.getenv("APCA_API_SECRET_KEY")
                or ""
            )

        # Local virtual positions and order book tracking
        self.virtual_positions: Dict[str, Position] = {}
        self.virtual_orders: Dict[str, OrderResult] = {}
        self._connected = False

    @property
    def is_live(self) -> bool:
        """Returns True if broker is in live execution mode."""
        return self.live

    @is_live.setter
    def is_live(self, value: bool) -> None:
        """Updates live mode and synchronizes base URL."""
        self.live = bool(value)
        self.base_url = self.DEFAULT_LIVE_URL if self.live else self.DEFAULT_PAPER_URL

    def _get_headers(self) -> Dict[str, str]:
        if not self.api_key_id or not self.api_secret_key:
            raise BrokerConnectionError(
                "Alpaca API credentials missing. Provide APCA_API_KEY_ID and APCA_API_SECRET_KEY."
            )
        return {
            "APCA-API-KEY-ID": self.api_key_id,
            "APCA-API-SECRET-KEY": self.api_secret_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def connect(self) -> bool:
        """Validate API credentials and connectivity."""
        if not self.api_key_id or not self.api_secret_key:
            if self.live:
                raise BrokerConnectionError("Live trading requires configured Alpaca API credentials.")
            logger.warning("AlpacaPaperBroker running in offline/unauthenticated simulation mode.")
            self._connected = True
            return True

        url = f"{self.base_url}/v2/account"
        try:
            res = self.http_session.get(url, headers=self._get_headers(), timeout=10.0)
            if res.status_code == 200:
                self._connected = True
                logger.info(f"Connected to Alpaca API at {self.base_url}")
                return True
            else:
                raise BrokerConnectionError(
                    f"Alpaca connection failed HTTP {res.status_code}: {res.text}"
                )
        except requests.RequestException as e:
            if not self.live:
                logger.warning(f"Could not connect to Alpaca network ({e}), running in mock mode.")
                self._connected = True
                return True
            raise BrokerConnectionError(f"Network error connecting to Alpaca: {e}")

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    # -------------------------------------------------------------------------
    # Virtual Sub-Account Partition Enforcement
    # -------------------------------------------------------------------------

    def get_account(self) -> AccountInfo:
        """Retrieve account status enforcing strict €100 virtual partition."""
        fx = self._get_fx_rate()

        # If live credentials are valid, query live endpoint to sync positions/status
        if self.api_key_id and self.api_secret_key:
            try:
                res = self.http_session.get(
                    f"{self.base_url}/v2/account",
                    headers=self._get_headers(),
                    timeout=5.0,
                )
                if res.status_code == 200:
                    raw = res.json()
                    # Alpaca raw balances (typically $100,000 in sandbox)
                    raw_cash_usd = float(raw.get("cash", 0.0))
                    raw_equity_usd = float(raw.get("equity", 0.0))
                    raw_bp_usd = float(raw.get("buying_power", 0.0))

                    # Enforce €100 virtual capital ceiling
                    virtual_cap_usd = self.virtual_capital_eur * fx
                    effective_equity_usd = min(raw_equity_usd, virtual_cap_usd)
                    effective_cash_usd = min(raw_cash_usd, effective_equity_usd)
                    effective_bp_usd = min(raw_bp_usd, effective_cash_usd)

                    return AccountInfo(
                        currency="EUR",
                        cash_eur=round(effective_cash_usd / fx, 4),
                        equity_eur=round(effective_equity_usd / fx, 4),
                        initial_capital_eur=self.virtual_capital_eur,
                        realized_pnl_eur=0.0,
                        unrealized_pnl_eur=round((effective_equity_usd - effective_cash_usd) / fx, 4),
                        buying_power_eur=round(effective_bp_usd / fx, 4),
                        is_live=self.live,
                        timestamp=datetime.utcnow(),
                    )
            except Exception as e:
                logger.warning(f"Failed to query Alpaca live account, using virtual ledger: {e}")

        # Local virtual partition fallback
        positions_val_eur = sum(
            (p.quantity * p.current_price) / fx for p in self.virtual_positions.values()
        )
        unrealized_pnl_eur = sum(
            ((p.current_price - p.entry_price) * p.quantity) / fx
            for p in self.virtual_positions.values()
        )
        equity_eur = self.virtual_cash_eur + positions_val_eur
        return AccountInfo(
            currency="EUR",
            cash_eur=round(self.virtual_cash_eur, 4),
            equity_eur=round(equity_eur, 4),
            initial_capital_eur=self.virtual_capital_eur,
            realized_pnl_eur=0.0,
            unrealized_pnl_eur=round(unrealized_pnl_eur, 4),
            buying_power_eur=round(max(0.0, self.virtual_cash_eur), 4),
            is_live=self.live,
            timestamp=datetime.utcnow(),
        )

    def get_positions(self) -> List[Position]:
        """Retrieve open positions with EUR currency conversions."""
        if self.api_key_id and self.api_secret_key:
            try:
                res = self.http_session.get(
                    f"{self.base_url}/v2/positions",
                    headers=self._get_headers(),
                    timeout=5.0,
                )
                if res.status_code == 200:
                    positions = []
                    fx = self._get_fx_rate()
                    for item in res.json():
                        p = self._parse_alpaca_position(item, fx)
                        positions.append(p)
                    return positions
            except Exception as e:
                logger.warning(f"Failed to fetch Alpaca positions: {e}")

        return list(self.virtual_positions.values())

    def submit_order(
        self,
        request: OrderRequest,
        current_market_price: Optional[float] = None,
    ) -> OrderResult:
        """Submit order to Alpaca REST API after validating virtual €100 partition."""
        fx = self._get_fx_rate()
        ref_price = current_market_price or self.get_latest_price(request.symbol) or 100.0

        # Calculate requested notional in EUR
        if request.notional is not None and request.notional > 0:
            order_notional_eur = request.notional
            qty = (request.notional * fx) / ref_price
        elif request.quantity is not None and request.quantity > 0:
            qty = request.quantity
            order_notional_usd = qty * ref_price
            order_notional_eur = order_notional_usd / fx
        else:
            order_notional_eur = 1.0
            qty = (order_notional_eur * fx) / ref_price

        # Enforce virtual €100 ceiling
        if request.side == OrderSide.BUY:
            current_positions_eur = sum(
                (p.quantity * p.current_price) / fx for p in self.get_positions()
            )
            total_projected_eur = current_positions_eur + order_notional_eur

            if total_projected_eur > (self.virtual_capital_eur + 1e-4):
                res = OrderResult(
                    order_id=f"ord_alp_{len(self.virtual_orders)+1:04d}",
                    client_order_id=request.client_order_id,
                    symbol=request.symbol,
                    status=OrderStatus.REJECTED,
                    side=request.side,
                    timestamp=datetime.utcnow(),
                    message=(
                        f"Order rejected by Virtual Sub-Account Partition: "
                        f"projected €{total_projected_eur:.2f} exceeds virtual ceiling €{self.virtual_capital_eur:.2f}"
                    ),
                )
                self.virtual_orders[res.order_id] = res
                return res

        elif request.side == OrderSide.SELL:
            held_pos = self.virtual_positions.get(request.symbol)
            if held_pos is None or held_pos.quantity < (qty - 1e-9):
                avail = held_pos.quantity if held_pos else 0.0
                res = OrderResult(
                    order_id=f"ord_alp_{len(self.virtual_orders)+1:04d}",
                    client_order_id=request.client_order_id,
                    symbol=request.symbol,
                    status=OrderStatus.REJECTED,
                    side=request.side,
                    timestamp=datetime.utcnow(),
                    message=f"Insufficient position quantity to sell: requested {qty}, available {avail}",
                )
                self.virtual_orders[res.order_id] = res
                return res

        # Attempt REST API submission if credentials configured
        if self.api_key_id and self.api_secret_key:
            payload = self._to_alpaca_payload(request, qty, order_notional_eur, fx)
            try:
                res = self.http_session.post(
                    f"{self.base_url}/v2/orders",
                    headers=self._get_headers(),
                    json=payload,
                    timeout=10.0,
                )
                if res.status_code in (200, 201):
                    order_data = res.json()
                    order_result = self._parse_alpaca_order(order_data, request, fx)
                    self.virtual_orders[order_result.order_id] = order_result
                    return order_result
                elif res.status_code in (403, 422):
                    res_json = res.json() if res.headers.get("content-type") == "application/json" else {}
                    msg = res_json.get("message", res.text)
                    order_result = OrderResult(
                        order_id=f"ord_alp_{len(self.virtual_orders)+1:04d}",
                        client_order_id=request.client_order_id,
                        symbol=request.symbol,
                        status=OrderStatus.REJECTED,
                        side=request.side,
                        timestamp=datetime.utcnow(),
                        message=f"Alpaca rejected order (HTTP {res.status_code}): {msg}",
                    )
                    self.virtual_orders[order_result.order_id] = order_result
                    return order_result
                else:
                    raise BrokerError(f"Alpaca order submission failed HTTP {res.status_code}: {res.text}")
            except requests.RequestException as e:
                if self.live:
                    raise BrokerConnectionError(f"Failed to submit live order to Alpaca: {e}")
                logger.warning(f"Alpaca API call failed ({e}), falling back to local virtual fill.")

        # Local deterministic virtual execution fallback
        order_id = f"ord_alp_{len(self.virtual_orders)+1:04d}"
        exec_price = ref_price
        fee_eur = 0.0
        if request.side == OrderSide.SELL:
            # Regulatory fees
            fee_eur = round((0.0000278 * (qty * exec_price) + max(0.01, 0.000166 * qty)) / fx, 6)

        trade_cost_eur = (qty * exec_price) / fx
        if request.side == OrderSide.BUY:
            self.virtual_cash_eur -= (trade_cost_eur + fee_eur)
            if request.symbol in self.virtual_positions:
                curr = self.virtual_positions[request.symbol]
                new_q = curr.quantity + qty
                curr.entry_price = (curr.entry_price * curr.quantity + exec_price * qty) / new_q
                curr.quantity = new_q
                curr.current_price = exec_price
            else:
                self.virtual_positions[request.symbol] = Position(
                    symbol=request.symbol,
                    quantity=qty,
                    entry_price=exec_price,
                    current_price=exec_price,
                    currency="USD",
                    entry_fx_rate=fx,
                    cost_basis_eur=trade_cost_eur,
                    market_value_eur=trade_cost_eur,
                )
        else:
            held_pos = self.virtual_positions.get(request.symbol)
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
                self.virtual_orders[order_id] = res
                return res

            if abs(held_pos.quantity - qty) < 1e-7 or qty >= held_pos.quantity:
                qty = held_pos.quantity

            trade_cost_eur = (qty * exec_price) / fx
            fee_eur = round((0.0000278 * (qty * exec_price) + max(0.01, 0.000166 * qty)) / fx, 6)

            self.virtual_cash_eur += (trade_cost_eur - fee_eur)
            held_pos.quantity -= qty
            if held_pos.quantity < 1e-7:
                del self.virtual_positions[request.symbol]

        res = OrderResult(
            order_id=order_id,
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            status=OrderStatus.FILLED,
            side=request.side,
            filled_quantity=qty,
            filled_avg_price=exec_price,
            fee_eur=fee_eur,
            slippage_eur=0.0,
            timestamp=datetime.utcnow(),
            message="Executed in Alpaca virtual partition",
        )
        self.virtual_orders[order_id] = res
        return res

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order on Alpaca."""
        if self.api_key_id and self.api_secret_key:
            try:
                res = self.http_session.delete(
                    f"{self.base_url}/v2/orders/{order_id}",
                    headers=self._get_headers(),
                    timeout=5.0,
                )
                if res.status_code in (200, 204):
                    if order_id in self.virtual_orders:
                        self.virtual_orders[order_id].status = OrderStatus.CANCELLED
                    return True
            except Exception as e:
                logger.warning(f"Failed to cancel Alpaca order {order_id}: {e}")

        if order_id in self.virtual_orders:
            self.virtual_orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False

    def get_order_status(self, order_id: str) -> OrderResult:
        if order_id in self.virtual_orders:
            return self.virtual_orders[order_id]
        raise OrderNotFoundError(f"Order '{order_id}' not found on Alpaca adapter")

    # -------------------------------------------------------------------------
    # Alpaca Serialization / Deserialization Helpers
    # -------------------------------------------------------------------------

    def _to_alpaca_payload(
        self,
        request: OrderRequest,
        qty: float,
        notional_eur: float,
        fx_rate: float,
    ) -> Dict[str, Any]:
        """Convert canonical OrderRequest into Alpaca v2 REST JSON payload."""
        payload: Dict[str, Any] = {
            "symbol": request.symbol.upper(),
            "side": request.side.value.lower(),
            "type": request.order_type.value.lower(),
            "time_in_force": request.time_in_force.value.lower(),
            "client_order_id": request.client_order_id,
        }

        if request.quantity is not None and request.quantity > 0:
            payload["qty"] = str(round(request.quantity, 6))
        elif request.notional is not None and request.notional > 0:
            # Alpaca notional in USD
            notional_usd = round(request.notional * fx_rate, 2)
            payload["notional"] = str(notional_usd)
        else:
            payload["qty"] = str(round(qty, 6))

        if request.order_type == OrderType.LIMIT and request.limit_price is not None:
            payload["limit_price"] = str(round(request.limit_price, 2))
        elif request.order_type == OrderType.STOP and request.stop_price is not None:
            payload["stop_price"] = str(round(request.stop_price, 2))
        elif request.order_type == OrderType.TRAILING_STOP and request.trail_percent is not None:
            payload["trail_percent"] = str(round(request.trail_percent, 2))

        return payload

    def _parse_alpaca_order(
        self,
        data: Dict[str, Any],
        original_request: OrderRequest,
        fx_rate: float,
    ) -> OrderResult:
        """Parse Alpaca JSON order response into canonical OrderResult."""
        alpaca_status = data.get("status", "pending_new").lower()
        status_map = {
            "new": OrderStatus.ACCEPTED,
            "accepted": OrderStatus.ACCEPTED,
            "pending_new": OrderStatus.PENDING,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
            "expired": OrderStatus.EXPIRED,
        }
        status = status_map.get(alpaca_status, OrderStatus.SUBMITTED)

        filled_qty = float(data.get("filled_qty", 0.0) or 0.0)
        filled_price = float(data.get("filled_avg_price", 0.0) or 0.0)

        fee_eur = 0.0
        if original_request.side == OrderSide.SELL and filled_qty > 0 and filled_price > 0:
            fee_eur = round((0.0000278 * (filled_qty * filled_price) + max(0.01, 0.000166 * filled_qty)) / fx_rate, 6)

        return OrderResult(
            order_id=str(data.get("id", "")),
            client_order_id=str(data.get("client_order_id", original_request.client_order_id)),
            symbol=str(data.get("symbol", original_request.symbol)),
            status=status,
            side=original_request.side,
            filled_quantity=filled_qty,
            filled_avg_price=filled_price,
            fee_eur=fee_eur,
            slippage_eur=0.0,
            timestamp=datetime.utcnow(),
            message=f"Alpaca order status: {alpaca_status}",
        )

    def _parse_alpaca_position(self, data: Dict[str, Any], fx_rate: float) -> Position:
        """Parse Alpaca JSON position item into canonical Position with EUR metrics."""
        qty = float(data.get("qty", 0.0))
        entry_price = float(data.get("avg_entry_price", 0.0))
        current_price = float(data.get("current_price", entry_price))
        market_val_usd = float(data.get("market_value", qty * current_price))
        cost_basis_usd = float(data.get("cost_basis", qty * entry_price))

        pos = Position(
            symbol=str(data.get("symbol", "")),
            quantity=qty,
            entry_price=entry_price,
            current_price=current_price,
            currency="USD",
            entry_fx_rate=fx_rate,
            cost_basis_eur=cost_basis_usd / fx_rate,
            market_value_eur=market_val_usd / fx_rate,
            unrealized_pnl_eur=(market_val_usd - cost_basis_usd) / fx_rate,
            unrealized_pnl_pct=float(data.get("unrealized_plpc", 0.0)) * 100.0,
        )
        return pos

    def _get_fx_rate(self) -> float:
        try:
            return self.currency_converter.get_rate("EUR", "USD")
        except Exception:
            return 1.0800
