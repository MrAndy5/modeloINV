"""modeloINV Core Subsystem Package.

Exposes canonical domain models, currency converter, and event dispatcher.
"""

from modeloinv.core.currency import (
    CurrencyConversionError,
    CurrencyConverter,
    FXDecompositionResult,
    get_currency_converter,
    set_currency_converter,
)
from modeloinv.core.events import (
    AccountEvent,
    AlertEvent,
    BarEvent,
    Event,
    EventDispatcher,
    ModeChangeEvent,
    OrderRequestEvent,
    OrderResultEvent,
    SignalEvent,
    SnapshotEvent,
    TickEvent,
    TradeEvent,
    get_event_bus,
    set_event_bus,
)
from modeloinv.core.models import (
    AccountInfo,
    AssetClass,
    Bar,
    ModelValidationError,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    PortfolioSnapshot,
    Position,
    SignalAction,
    TimeInForce,
    Trade,
    TradingSignal,
)

__all__ = [
    # Enums
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "AssetClass",
    "TimeInForce",
    "SignalAction",
    # Models
    "Bar",
    "OrderRequest",
    "OrderResult",
    "Trade",
    "Position",
    "AccountInfo",
    "TradingSignal",
    "PortfolioSnapshot",
    "ModelValidationError",
    # Currency
    "CurrencyConverter",
    "FXDecompositionResult",
    "CurrencyConversionError",
    "get_currency_converter",
    "set_currency_converter",
    # Events
    "Event",
    "TickEvent",
    "BarEvent",
    "SignalEvent",
    "OrderRequestEvent",
    "OrderResultEvent",
    "TradeEvent",
    "AccountEvent",
    "SnapshotEvent",
    "ModeChangeEvent",
    "AlertEvent",
    "EventDispatcher",
    "get_event_bus",
    "set_event_bus",
]
