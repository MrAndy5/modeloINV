"""Lightweight Event Dispatcher for modeloINV.

Supports hybrid synchronous and asynchronous event dispatching, typed subscriptions,
circular buffer event history, and fault-tolerant handler isolation.
"""

from __future__ import annotations

import asyncio
import collections
import inspect
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional, Type, TypeVar, Union

from modeloinv.core.models import (
    AccountInfo,
    Bar,
    OrderRequest,
    OrderResult,
    PortfolioSnapshot,
    Trade,
    TradingSignal,
)

logger = logging.getLogger(__name__)

E = TypeVar("E", bound="Event")
HandlerSync = Callable[[Any], None]
HandlerAsync = Callable[[Any], Awaitable[None]]
Handler = Union[HandlerSync, HandlerAsync]


class EventDispatchError(Exception):
    """Raised when an unrecoverable event dispatch error occurs."""
    pass


@dataclass
class Event:
    """Base event class for all system telemetry and lifecycle triggers."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def event_name(self) -> str:
        return self.__class__.__name__


@dataclass
class TickEvent(Event):
    symbol: str = ""
    bid: float = 0.0
    ask: float = 0.0
    last_price: float = 0.0
    volume: float = 0.0


@dataclass
class BarEvent(Event):
    bar: Optional[Bar] = None


@dataclass
class SignalEvent(Event):
    signal: Optional[TradingSignal] = None


@dataclass
class OrderRequestEvent(Event):
    request: Optional[OrderRequest] = None


@dataclass
class OrderResultEvent(Event):
    result: Optional[OrderResult] = None


@dataclass
class TradeEvent(Event):
    trade: Optional[Trade] = None


@dataclass
class AccountEvent(Event):
    account: Optional[AccountInfo] = None


@dataclass
class SnapshotEvent(Event):
    snapshot: Optional[PortfolioSnapshot] = None


@dataclass
class ModeChangeEvent(Event):
    old_mode: str = ""
    new_mode: str = ""


@dataclass
class AlertEvent(Event):
    level: str = "INFO"  # "INFO", "WARNING", "ERROR", "CRITICAL"
    source: str = ""
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class EventDispatcher:
    """Thread-safe synchronous and asynchronous event bus."""

    def __init__(self, history_size: int = 100) -> None:
        self._history_size = history_size
        self._lock = threading.RLock()
        self._subscribers: Dict[Type[Event], List[Handler]] = collections.defaultdict(list)
        self._global_subscribers: List[Handler] = []
        self._history: Deque[Event] = collections.deque(maxlen=history_size)
        self._is_emitting_alert = False

    def subscribe(self, event_cls: Type[E], handler: Handler) -> None:
        """Subscribe a callable or coroutine to a specific Event class."""
        with self._lock:
            if handler not in self._subscribers[event_cls]:
                self._subscribers[event_cls].append(handler)
        logger.debug(f"Subscribed {handler} to {event_cls.__name__}")

    def subscribe_all(self, handler: Handler) -> None:
        """Subscribe a callable to all dispatched events (wildcard)."""
        with self._lock:
            if handler not in self._global_subscribers:
                self._global_subscribers.append(handler)

    def unsubscribe(self, event_cls: Type[E], handler: Handler) -> None:
        """Unsubscribe a handler from a specific Event class."""
        with self._lock:
            if handler in self._subscribers[event_cls]:
                self._subscribers[event_cls].remove(handler)

    def unsubscribe_all(self, handler: Handler) -> None:
        """Unsubscribe a handler from all registrations."""
        with self._lock:
            if handler in self._global_subscribers:
                self._global_subscribers.remove(handler)
            for handlers in self._subscribers.values():
                if handler in handlers:
                    handlers.remove(handler)

    def clear(self) -> None:
        """Remove all subscriptions and reset history."""
        with self._lock:
            self._subscribers.clear()
            self._global_subscribers.clear()
            self._history.clear()

    def get_history(self) -> List[Event]:
        """Return a copy of recent dispatched events."""
        with self._lock:
            return list(self._history)

    def _execute_handler_sync(self, handler: Handler, event: Event) -> None:
        """Safely execute a synchronous handler or schedule an async coroutine."""
        try:
            if inspect.iscoroutinefunction(handler):
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(handler(event))
                except RuntimeError:
                    # No active event loop in thread; run to completion
                    asyncio.run(handler(event))
            else:
                handler(event)
        except Exception as e:
            logger.exception(f"Error executing handler {handler} for event {event.event_name}: {e}")
            self._emit_alert_safely(
                level="ERROR",
                source=handler.__name__ if hasattr(handler, "__name__") else "handler",
                message=str(e),
                details={"event": event.event_name, "event_id": event.event_id},
            )

    def _emit_alert_safely(self, level: str, source: str, message: str, details: Dict[str, Any]) -> None:
        """Emit an AlertEvent with recursion protection."""
        if self._is_emitting_alert:
            return
        self._is_emitting_alert = True
        try:
            alert = AlertEvent(level=level, source=source, message=message, details=details)
            self.publish(alert)
        finally:
            self._is_emitting_alert = False

    def publish(self, event: Event) -> None:
        """Synchronously publish an event to all matching subscribers."""
        with self._lock:
            self._history.append(event)
            handlers_to_call: List[Handler] = []
            for cls in type(event).__mro__:
                if issubclass(cls, Event) and cls in self._subscribers:
                    handlers_to_call.extend(self._subscribers[cls])
            handlers_to_call.extend(self._global_subscribers)

        for handler in handlers_to_call:
            self._execute_handler_sync(handler, event)

    async def publish_async(self, event: Event) -> None:
        """Asynchronously publish an event, awaiting coroutines and calling synchronous handlers."""
        with self._lock:
            self._history.append(event)
            handlers_to_call: List[Handler] = []
            for cls in type(event).__mro__:
                if issubclass(cls, Event) and cls in self._subscribers:
                    handlers_to_call.extend(self._subscribers[cls])
            handlers_to_call.extend(self._global_subscribers)

        coros = []
        for handler in handlers_to_call:
            try:
                if inspect.iscoroutinefunction(handler):
                    coros.append(handler(event))
                else:
                    handler(event)
            except Exception as e:
                logger.exception(f"Error in handler {handler} during publish_async: {e}")

        if coros:
            await asyncio.gather(*coros, return_exceptions=True)


# Global event bus singleton
_DEFAULT_BUS: Optional[EventDispatcher] = None


def get_event_bus() -> EventDispatcher:
    """Retrieve global default EventDispatcher singleton."""
    global _DEFAULT_BUS
    if _DEFAULT_BUS is None:
        _DEFAULT_BUS = EventDispatcher()
    return _DEFAULT_BUS


def set_event_bus(bus: EventDispatcher) -> None:
    """Inject custom EventDispatcher for testing or isolation."""
    global _DEFAULT_BUS
    _DEFAULT_BUS = bus
