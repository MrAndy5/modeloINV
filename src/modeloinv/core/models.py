"""Canonical Domain Models and Enums for modeloINV.

Provides strongly typed dataclasses for Market Data, Orders, Trades, Positions,
Account Status, and Strategy Signals with rigorous boundary validation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    TRAILING_STOP = "TRAILING_STOP"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class AssetClass(str, Enum):
    EQUITY = "EQUITY"
    ETF = "ETF"
    CRYPTO = "CRYPTO"
    FOREX = "FOREX"


class TimeInForce(str, Enum):
    DAY = "DAY"
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"


class SignalAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class ModelValidationError(ValueError):
    """Raised when domain model invariant validation fails."""
    pass


@dataclass(frozen=True)
class Bar:
    """Immutable representation of an OHLCV candlestick bar.

    Frozen and hashable for caching and thread-safe data pipelines.
    """
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    asset_class: AssetClass = AssetClass.EQUITY
    currency: str = "USD"

    def __post_init__(self) -> None:
        for field_name, val in [
            ("open", self.open),
            ("high", self.high),
            ("low", self.low),
            ("close", self.close),
            ("volume", self.volume),
        ]:
            if val is None or math.isnan(val) or math.isinf(val):
                raise ModelValidationError(f"Bar {field_name} cannot be None, NaN or Inf for {self.symbol}")

        eps = 1e-7
        if self.high < self.low - eps:
            raise ModelValidationError(
                f"Bar high ({self.high}) cannot be less than low ({self.low}) for {self.symbol}"
            )
        if self.high < max(self.open, self.close) - eps:
            raise ModelValidationError(
                f"Bar high ({self.high}) cannot be less than open/close ({self.open}, {self.close})"
            )
        if self.low > min(self.open, self.close) + eps:
            raise ModelValidationError(
                f"Bar low ({self.low}) cannot be greater than open/close ({self.open}, {self.close})"
            )
        if self.volume < 0:
            raise ModelValidationError(
                f"Bar volume ({self.volume}) cannot be negative for {self.symbol}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "asset_class": str(self.asset_class.value if isinstance(self.asset_class, AssetClass) else self.asset_class),
            "currency": self.currency,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Bar:
        ts = data["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        asset_class = data.get("asset_class", AssetClass.EQUITY)
        if isinstance(asset_class, str):
            asset_class = AssetClass(asset_class)
        return cls(
            timestamp=ts,
            symbol=str(data["symbol"]),
            open=float(data["open"]),
            high=float(data["high"]),
            low=float(data["low"]),
            close=float(data["close"]),
            volume=float(data["volume"]),
            asset_class=asset_class,
            currency=str(data.get("currency", "USD")),
        )


@dataclass
class OrderRequest:
    """Order submission specification supporting both share quantities and cash notional.

    Tailored for fractional share trading on micro-capital (€100) accounts.
    """
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Optional[float] = None
    notional: Optional[float] = None
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trail_percent: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    client_order_id: str = field(
        default_factory=lambda: datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    )
    tag: str = ""

    def validate(self) -> None:
        """Validate order request parameters."""
        if not self.symbol:
            raise ModelValidationError("Order symbol cannot be empty.")
        if self.quantity is None and self.notional is None:
            raise ModelValidationError("Order must specify either 'quantity' or 'notional'.")
        if self.quantity is not None and self.quantity <= 0:
            raise ModelValidationError(f"Order quantity must be positive, got {self.quantity}.")
        if self.notional is not None and self.notional <= 0:
            raise ModelValidationError(f"Order notional must be positive, got {self.notional}.")
        if self.order_type == OrderType.LIMIT and (self.limit_price is None or self.limit_price <= 0):
            raise ModelValidationError("Limit order requires positive 'limit_price'.")
        if self.order_type == OrderType.STOP and (self.stop_price is None or self.stop_price <= 0):
            raise ModelValidationError("Stop order requires positive 'stop_price'.")
        if self.order_type == OrderType.TRAILING_STOP and (
            self.trail_percent is None or self.trail_percent <= 0
        ):
            raise ModelValidationError("Trailing stop order requires positive 'trail_percent'.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": str(self.side.value if isinstance(self.side, OrderSide) else self.side),
            "order_type": str(self.order_type.value if isinstance(self.order_type, OrderType) else self.order_type),
            "quantity": self.quantity,
            "notional": self.notional,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "trail_percent": self.trail_percent,
            "time_in_force": str(self.time_in_force.value if isinstance(self.time_in_force, TimeInForce) else self.time_in_force),
            "tag": self.tag,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OrderRequest:
        side = data["side"]
        if isinstance(side, str):
            side = OrderSide(side)
        order_type = data["order_type"]
        if isinstance(order_type, str):
            order_type = OrderType(order_type)
        tif = data.get("time_in_force", TimeInForce.DAY)
        if isinstance(tif, str):
            tif = TimeInForce(tif)
        req = cls(
            symbol=str(data["symbol"]),
            side=side,
            order_type=order_type,
            quantity=float(data["quantity"]) if data.get("quantity") is not None else None,
            notional=float(data["notional"]) if data.get("notional") is not None else None,
            limit_price=float(data["limit_price"]) if data.get("limit_price") is not None else None,
            stop_price=float(data["stop_price"]) if data.get("stop_price") is not None else None,
            trail_percent=float(data["trail_percent"]) if data.get("trail_percent") is not None else None,
            time_in_force=tif,
            client_order_id=str(data.get("client_order_id", "")),
            tag=str(data.get("tag", "")),
        )
        return req


@dataclass
class OrderResult:
    """Execution outcome and lifecycle status returned by broker for an order request."""
    order_id: str
    client_order_id: str
    symbol: str
    status: OrderStatus
    side: OrderSide
    filled_quantity: float = 0.0
    filled_avg_price: float = 0.0
    fee_eur: float = 0.0
    slippage_eur: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    message: str = ""

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    @property
    def is_active(self) -> bool:
        return self.status in {
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIALLY_FILLED,
        }

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "status": str(self.status.value if isinstance(self.status, OrderStatus) else self.status),
            "side": str(self.side.value if isinstance(self.side, OrderSide) else self.side),
            "filled_quantity": self.filled_quantity,
            "filled_avg_price": self.filled_avg_price,
            "fee_eur": self.fee_eur,
            "slippage_eur": self.slippage_eur,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OrderResult:
        status = data["status"]
        if isinstance(status, str):
            status = OrderStatus(status)
        side = data["side"]
        if isinstance(side, str):
            side = OrderSide(side)
        ts = data.get("timestamp", datetime.utcnow())
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return cls(
            order_id=str(data["order_id"]),
            client_order_id=str(data.get("client_order_id", "")),
            symbol=str(data["symbol"]),
            status=status,
            side=side,
            filled_quantity=float(data.get("filled_quantity", 0.0)),
            filled_avg_price=float(data.get("filled_avg_price", 0.0)),
            fee_eur=float(data.get("fee_eur", 0.0)),
            slippage_eur=float(data.get("slippage_eur", 0.0)),
            timestamp=ts,
            message=str(data.get("message", "")),
        )


@dataclass
class Trade:
    """Completed fill event for database persistence, P&L reporting, and continuous learning feedback."""
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    notional_usd: float = 0.0
    fx_rate: float = 1.0
    price_eur: float = 0.0
    fee_eur: float = 0.0
    slippage_eur: float = 0.0
    realized_pnl_eur: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    strategy_id: str = ""
    signal_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": str(self.side.value if isinstance(self.side, OrderSide) else self.side),
            "quantity": self.quantity,
            "price": self.price,
            "notional_usd": self.notional_usd,
            "fx_rate": self.fx_rate,
            "price_eur": self.price_eur,
            "fee_eur": self.fee_eur,
            "slippage_eur": self.slippage_eur,
            "realized_pnl_eur": self.realized_pnl_eur,
            "timestamp": self.timestamp.isoformat(),
            "strategy_id": self.strategy_id,
            "signal_metadata": self.signal_metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Trade:
        side = data["side"]
        if isinstance(side, str):
            side = OrderSide(side)
        ts = data.get("timestamp", datetime.utcnow())
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return cls(
            trade_id=str(data["trade_id"]),
            order_id=str(data["order_id"]),
            symbol=str(data["symbol"]),
            side=side,
            quantity=float(data["quantity"]),
            price=float(data["price"]),
            notional_usd=float(data.get("notional_usd", 0.0)),
            fx_rate=float(data.get("fx_rate", 1.0)),
            price_eur=float(data.get("price_eur", 0.0)),
            fee_eur=float(data.get("fee_eur", 0.0)),
            slippage_eur=float(data.get("slippage_eur", 0.0)),
            realized_pnl_eur=float(data.get("realized_pnl_eur", 0.0)),
            timestamp=ts,
            strategy_id=str(data.get("strategy_id", "")),
            signal_metadata=dict(data.get("signal_metadata", {})),
        )


@dataclass
class Position:
    """Open holding tracking quantity, entry cost, and real-time EUR mark-to-market."""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    currency: str = "USD"
    entry_fx_rate: float = 1.0
    cost_basis_eur: float = 0.0
    market_value_eur: float = 0.0
    unrealized_pnl_eur: float = 0.0
    unrealized_pnl_pct: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def update_market_price(self, current_price: float, fx_rate: float) -> None:
        """Update current market price and FX rate, recalculating market value and unrealized P&L in EUR."""
        self.current_price = current_price
        if self.currency.upper() == "EUR" or fx_rate <= 0:
            self.market_value_eur = self.quantity * self.current_price
        else:
            self.market_value_eur = (self.quantity * self.current_price) / fx_rate

        self.unrealized_pnl_eur = self.market_value_eur - self.cost_basis_eur
        if self.cost_basis_eur > 0:
            self.unrealized_pnl_pct = (self.unrealized_pnl_eur / self.cost_basis_eur) * 100.0
        else:
            self.unrealized_pnl_pct = 0.0
        self.last_updated = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "currency": self.currency,
            "entry_fx_rate": self.entry_fx_rate,
            "cost_basis_eur": self.cost_basis_eur,
            "market_value_eur": self.market_value_eur,
            "unrealized_pnl_eur": self.unrealized_pnl_eur,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Position:
        ts = data.get("last_updated", datetime.utcnow())
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return cls(
            symbol=str(data["symbol"]),
            quantity=float(data["quantity"]),
            entry_price=float(data["entry_price"]),
            current_price=float(data.get("current_price", data["entry_price"])),
            currency=str(data.get("currency", "USD")),
            entry_fx_rate=float(data.get("entry_fx_rate", 1.0)),
            cost_basis_eur=float(data.get("cost_basis_eur", 0.0)),
            market_value_eur=float(data.get("market_value_eur", 0.0)),
            unrealized_pnl_eur=float(data.get("unrealized_pnl_eur", 0.0)),
            unrealized_pnl_pct=float(data.get("unrealized_pnl_pct", 0.0)),
            last_updated=ts,
        )


@dataclass
class AccountInfo:
    """Account balance, purchasing power, and cumulative performance state."""
    currency: str = "EUR"
    cash_eur: float = 100.00
    equity_eur: float = 100.00
    initial_capital_eur: float = 100.00
    realized_pnl_eur: float = 0.0
    unrealized_pnl_eur: float = 0.0
    total_fees_eur: float = 0.0
    buying_power_eur: float = 100.00
    is_live: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def total_return_pct(self) -> float:
        if self.initial_capital_eur <= 0:
            return 0.0
        return ((self.equity_eur - self.initial_capital_eur) / self.initial_capital_eur) * 100.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "currency": self.currency,
            "cash_eur": self.cash_eur,
            "equity_eur": self.equity_eur,
            "initial_capital_eur": self.initial_capital_eur,
            "realized_pnl_eur": self.realized_pnl_eur,
            "unrealized_pnl_eur": self.unrealized_pnl_eur,
            "total_fees_eur": self.total_fees_eur,
            "buying_power_eur": self.buying_power_eur,
            "is_live": self.is_live,
            "total_return_pct": self.total_return_pct,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AccountInfo:
        ts = data.get("timestamp", datetime.utcnow())
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return cls(
            currency=str(data.get("currency", "EUR")),
            cash_eur=float(data.get("cash_eur", 100.0)),
            equity_eur=float(data.get("equity_eur", 100.0)),
            initial_capital_eur=float(data.get("initial_capital_eur", 100.0)),
            realized_pnl_eur=float(data.get("realized_pnl_eur", 0.0)),
            unrealized_pnl_eur=float(data.get("unrealized_pnl_eur", 0.0)),
            total_fees_eur=float(data.get("total_fees_eur", 0.0)),
            buying_power_eur=float(data.get("buying_power_eur", 100.0)),
            is_live=bool(data.get("is_live", False)),
            timestamp=ts,
        )


@dataclass
class TradingSignal:
    """Actionable output produced by the hybrid strategy engine (Quant + ML + RL)."""
    timestamp: datetime
    symbol: str
    target_weight: float  # -1.0 to 1.0
    action: str           # "BUY", "SELL", "HOLD"
    confidence: float = 0.0     # 0.0 to 1.0
    quant_score: float = 0.0    # -1.0 to 1.0
    ml_score: float = 0.0       # -1.0 to 1.0
    rl_action: int = 0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "target_weight": self.target_weight,
            "action": self.action,
            "confidence": self.confidence,
            "quant_score": self.quant_score,
            "ml_score": self.ml_score,
            "rl_action": self.rl_action,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TradingSignal:
        ts = data["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return cls(
            timestamp=ts,
            symbol=str(data["symbol"]),
            target_weight=float(data["target_weight"]),
            action=str(data["action"]),
            confidence=float(data.get("confidence", 0.0)),
            quant_score=float(data.get("quant_score", 0.0)),
            ml_score=float(data.get("ml_score", 0.0)),
            rl_action=int(data.get("rl_action", 0)),
            stop_loss=float(data["stop_loss"]) if data.get("stop_loss") is not None else None,
            take_profit=float(data["take_profit"]) if data.get("take_profit") is not None else None,
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio telemetry for historical equity curves and WebSocket broadcast."""
    timestamp: datetime
    cash_eur: float
    equity_eur: float
    positions_value_eur: float
    realized_pnl_eur: float
    unrealized_pnl_eur: float
    total_fees_eur: float
    benchmark_sp500_price: Optional[float] = None
    benchmark_return_pct: float = 0.0
    portfolio_return_pct: float = 0.0
    alpha_pct: float = 0.0
    drawdown_pct: float = 0.0
    high_water_mark_eur: float = 100.00

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "cash_eur": self.cash_eur,
            "equity_eur": self.equity_eur,
            "positions_value_eur": self.positions_value_eur,
            "realized_pnl_eur": self.realized_pnl_eur,
            "unrealized_pnl_eur": self.unrealized_pnl_eur,
            "total_fees_eur": self.total_fees_eur,
            "benchmark_sp500_price": self.benchmark_sp500_price,
            "benchmark_return_pct": self.benchmark_return_pct,
            "portfolio_return_pct": self.portfolio_return_pct,
            "alpha_pct": self.alpha_pct,
            "drawdown_pct": self.drawdown_pct,
            "high_water_mark_eur": self.high_water_mark_eur,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PortfolioSnapshot:
        ts = data["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return cls(
            timestamp=ts,
            cash_eur=float(data["cash_eur"]),
            equity_eur=float(data["equity_eur"]),
            positions_value_eur=float(data["positions_value_eur"]),
            realized_pnl_eur=float(data["realized_pnl_eur"]),
            unrealized_pnl_eur=float(data["unrealized_pnl_eur"]),
            total_fees_eur=float(data["total_fees_eur"]),
            benchmark_sp500_price=float(data["benchmark_sp500_price"]) if data.get("benchmark_sp500_price") is not None else None,
            benchmark_return_pct=float(data.get("benchmark_return_pct", 0.0)),
            portfolio_return_pct=float(data.get("portfolio_return_pct", 0.0)),
            alpha_pct=float(data.get("alpha_pct", 0.0)),
            drawdown_pct=float(data.get("drawdown_pct", 0.0)),
            high_water_mark_eur=float(data.get("high_water_mark_eur", 100.0)),
        )
