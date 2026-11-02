"""Unit tests for core subsystem: models, currency engine, and event dispatcher."""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import math

import pytest

from modeloinv.core import (
    AccountEvent,
    AccountInfo,
    AlertEvent,
    AssetClass,
    Bar,
    BarEvent,
    CurrencyConversionError,
    CurrencyConverter,
    Event,
    EventDispatcher,
    FXDecompositionResult,
    ModelValidationError,
    OrderRequest,
    OrderRequestEvent,
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
    get_currency_converter,
    get_event_bus,
    set_currency_converter,
    set_event_bus,
)


# =====================================================================
# Canonical Domain Models Tests
# =====================================================================

def test_bar_immutability():
    """Verify Bar is frozen and immutable."""
    bar = Bar(
        timestamp=datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
        symbol="SPY",
        open=450.0,
        high=455.0,
        low=449.0,
        close=452.0,
        volume=10000.0,
        asset_class=AssetClass.EQUITY,
        currency="USD",
    )
    with pytest.raises(FrozenInstanceError):
        bar.close = 460.0  # type: ignore


def test_bar_boundary_validations():
    """Verify Bar invariant checks enforce valid OHLCV geometry."""
    ts = datetime.utcnow()
    # High < Low
    with pytest.raises(ModelValidationError, match="cannot be less than low"):
        Bar(timestamp=ts, symbol="TEST", open=100.0, high=90.0, low=95.0, close=92.0, volume=10.0)

    # High < Open
    with pytest.raises(ModelValidationError, match="cannot be less than open/close"):
        Bar(timestamp=ts, symbol="TEST", open=100.0, high=99.0, low=90.0, close=95.0, volume=10.0)

    # Low > Close
    with pytest.raises(ModelValidationError, match="cannot be greater than open/close"):
        Bar(timestamp=ts, symbol="TEST", open=100.0, high=105.0, low=98.0, close=95.0, volume=10.0)

    # Negative Volume
    with pytest.raises(ModelValidationError, match="cannot be negative"):
        Bar(timestamp=ts, symbol="TEST", open=100.0, high=105.0, low=95.0, close=100.0, volume=-5.0)


def test_bar_serialization_roundtrip():
    """Verify Bar to_dict and from_dict preserve all fields."""
    dt = datetime(2026, 3, 10, 14, 30, tzinfo=timezone.utc)
    bar = Bar(
        timestamp=dt,
        symbol="BTCUSDT",
        open=65000.0,
        high=66000.0,
        low=64500.0,
        close=65800.0,
        volume=125.5,
        asset_class=AssetClass.CRYPTO,
        currency="USDT",
    )
    d = bar.to_dict()
    restored = Bar.from_dict(d)
    assert restored.symbol == bar.symbol
    assert restored.open == bar.open
    assert restored.high == bar.high
    assert restored.low == bar.low
    assert restored.close == bar.close
    assert restored.volume == bar.volume
    assert restored.asset_class == bar.asset_class
    assert restored.currency == bar.currency


def test_order_request_validation():
    """Verify OrderRequest validates inputs strictly."""
    # Empty symbol
    with pytest.raises(ModelValidationError, match="Order symbol cannot be empty"):
        req = OrderRequest(symbol="", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1.0)
        req.validate()

    # Missing both quantity and notional
    with pytest.raises(ModelValidationError, match="either 'quantity' or 'notional'"):
        req = OrderRequest(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET)
        req.validate()

    # Non-positive quantity
    with pytest.raises(ModelValidationError, match="quantity must be positive"):
        req = OrderRequest(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.0)
        req.validate()

    # Limit order missing limit_price
    with pytest.raises(ModelValidationError, match="requires positive 'limit_price'"):
        req = OrderRequest(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=1.0)
        req.validate()

    # Stop order missing stop_price
    with pytest.raises(ModelValidationError, match="requires positive 'stop_price'"):
        req = OrderRequest(symbol="AAPL", side=OrderSide.SELL, order_type=OrderType.STOP, quantity=1.0)
        req.validate()

    # Trailing stop missing trail_percent
    with pytest.raises(ModelValidationError, match="requires positive 'trail_percent'"):
        req = OrderRequest(symbol="AAPL", side=OrderSide.SELL, order_type=OrderType.TRAILING_STOP, quantity=1.0)
        req.validate()


def test_order_result_lifecycle_properties():
    """Verify OrderResult state properties."""
    pending = OrderResult(order_id="1", client_order_id="c1", symbol="SPY", status=OrderStatus.PENDING, side=OrderSide.BUY)
    assert pending.is_active is True
    assert pending.is_filled is False
    assert pending.is_terminal is False

    filled = OrderResult(order_id="2", client_order_id="c2", symbol="SPY", status=OrderStatus.FILLED, side=OrderSide.BUY, filled_quantity=1.0)
    assert filled.is_filled is True
    assert filled.is_terminal is True
    assert filled.is_active is False

    cancelled = OrderResult(order_id="3", client_order_id="c3", symbol="SPY", status=OrderStatus.CANCELLED, side=OrderSide.BUY)
    assert cancelled.is_terminal is True
    assert cancelled.is_active is False


def test_position_mark_to_market():
    """Verify Position recalculates market value and unrealized P&L in EUR."""
    # 2 shares of SPY bought at 500 USD when EUR/USD = 1.00 -> cost basis = 1000 EUR
    pos = Position(
        symbol="SPY",
        quantity=2.0,
        entry_price=500.0,
        current_price=500.0,
        currency="USD",
        entry_fx_rate=1.00,
        cost_basis_eur=1000.0,
    )
    # Price rises to 550 USD, EUR/USD moves to 1.10
    # Market value in USD = 2 * 550 = 1100 USD
    # Market value in EUR = 1100 / 1.10 = 1000 EUR
    # Unrealized P&L = 1000 - 1000 = 0.0 EUR
    pos.update_market_price(current_price=550.0, fx_rate=1.10)
    assert round(pos.market_value_eur, 2) == 1000.0
    assert round(pos.unrealized_pnl_eur, 2) == 0.0

    # Price rises to 660 USD, EUR/USD moves to 1.10
    # Market value in USD = 2 * 660 = 1320 USD
    # Market value in EUR = 1320 / 1.10 = 1200 EUR
    # Unrealized P&L = 1200 - 1000 = +200 EUR (+20%)
    pos.update_market_price(current_price=660.0, fx_rate=1.10)
    assert round(pos.market_value_eur, 2) == 1200.0
    assert round(pos.unrealized_pnl_eur, 2) == 200.0
    assert round(pos.unrealized_pnl_pct, 2) == 20.0


def test_account_info_total_return():
    """Verify AccountInfo computes total return percentage from initial capital."""
    acc = AccountInfo(
        cash_eur=50.0,
        equity_eur=125.0,
        initial_capital_eur=100.0,
    )
    assert acc.total_return_pct == 25.0

    # Zero initial capital guard
    acc_zero = AccountInfo(initial_capital_eur=0.0, equity_eur=0.0)
    assert acc_zero.total_return_pct == 0.0


# =====================================================================
# Multi-Currency FX Engine Tests
# =====================================================================

def test_currency_converter_rates_and_reciprocal():
    """Verify CurrencyConverter rate setting and reciprocal calculation."""
    conv = CurrencyConverter(fallback_rate=1.08)
    assert conv.get_rate("EUR", "USD") == 1.08
    assert math.isclose(conv.get_rate("USD", "EUR"), 1.0 / 1.08, rel_tol=1e-7)

    # Update EURUSD rate to 1.10
    conv.set_rate("EURUSD", 1.10)
    assert conv.get_rate("EUR", "USD") == 1.10
    assert math.isclose(conv.get_rate("USD", "EUR"), 1.0 / 1.10, rel_tol=1e-7)

    # Conversion amounts
    assert round(conv.to_eur(110.0, "USD"), 2) == 100.0
    assert round(conv.from_eur(100.0, "USD"), 2) == 110.0


def test_currency_converter_invalid_rate():
    """Verify CurrencyConverter rejects negative or zero rates."""
    conv = CurrencyConverter()
    with pytest.raises(CurrencyConversionError, match="must be positive"):
        conv.set_rate("EURUSD", -1.0)
    with pytest.raises(CurrencyConversionError, match="must be positive"):
        conv.set_rate("EURUSD", 0.0)


def test_currency_exact_pnl_decomposition():
    """Verify mathematically exact P&L decomposition into asset and FX return components."""
    # Scenario: Buy 10 shares of AAPL at 150 USD with EUR/USD = 1.00
    # Cost basis = 10 * 150 / 1.00 = 1500 EUR
    # Sell at 180 USD with EUR/USD = 1.20
    # Gross proceeds = 10 * 180 / 1.20 = 1500 EUR
    # Total EUR P&L = 1500 - 1500 = 0.0 EUR
    res = CurrencyConverter.decompose_pnl(
        entry_price_quote=150.0,
        exit_price_quote=180.0,
        quantity=10.0,
        entry_fx_rate=1.00,
        exit_fx_rate=1.20,
    )
    # Asset gained 30 USD/share * 10 / 1.00 = +300 EUR
    # FX lost: 1800 * (1/1.20 - 1/1.00) = 1800 * (0.8333 - 1.0) = -300 EUR
    assert math.isclose(res.total_pnl_eur, 0.0, abs_tol=1e-9)
    assert math.isclose(res.asset_pnl_eur, 300.0, abs_tol=1e-9)
    assert math.isclose(res.fx_pnl_eur, -300.0, abs_tol=1e-9)
    # Proof of exact identity: asset_pnl + fx_pnl == total_pnl
    assert math.isclose(res.asset_pnl_eur + res.fx_pnl_eur, res.total_pnl_eur, abs_tol=1e-12)


def test_currency_decomposition_bull_scenario():
    """Verify decomposition when both asset and FX move favorably."""
    res = CurrencyConverter.decompose_pnl(
        entry_price_quote=100.0,
        exit_price_quote=120.0,
        quantity=5.0,
        entry_fx_rate=1.10,
        exit_fx_rate=1.05,  # EUR weakened against USD -> USD assets worth more in EUR
    )
    assert res.asset_pnl_eur > 0
    assert res.fx_pnl_eur > 0
    assert math.isclose(res.asset_pnl_eur + res.fx_pnl_eur, res.total_pnl_eur, abs_tol=1e-12)


# =====================================================================
# Lightweight Event Dispatcher Tests
# =====================================================================

def test_event_dispatcher_sync_subscription():
    """Verify synchronous event publication and subscription."""
    bus = EventDispatcher(history_size=50)
    received = []

    def on_tick(event: Event):
        received.append(event)

    bus.subscribe(BarEvent, on_tick)

    bar = Bar(
        timestamp=datetime.utcnow(),
        symbol="SPY",
        open=500.0,
        high=505.0,
        low=495.0,
        close=502.0,
        volume=100.0,
    )
    event = BarEvent(bar=bar)
    bus.publish(event)

    assert len(received) == 1
    assert received[0] is event
    assert len(bus.get_history()) == 1


def test_event_dispatcher_async_subscription():
    """Verify asynchronous event publication and subscription."""
    bus = EventDispatcher()
    received = []

    async def async_handler(event: Event):
        received.append(event)

    bus.subscribe(AccountEvent, async_handler)

    acc = AccountInfo(cash_eur=100.0, equity_eur=100.0)
    event = AccountEvent(account=acc)

    async def run():
        await bus.publish_async(event)

    asyncio.run(run())
    assert len(received) == 1
    assert received[0] is event


def test_event_dispatcher_fault_isolation():
    """Verify broken subscriber does not crash publisher or block other subscribers."""
    bus = EventDispatcher()
    order_results = []
    alerts = []

    def failing_handler(event: Event):
        raise RuntimeError("Subscriber explosion!")

    def good_handler(event: Event):
        order_results.append(event)

    def alert_listener(alert: AlertEvent):
        alerts.append(alert)

    bus.subscribe(OrderRequestEvent, failing_handler)
    bus.subscribe(OrderRequestEvent, good_handler)
    bus.subscribe(AlertEvent, alert_listener)

    req = OrderRequest(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1.0)
    event = OrderRequestEvent(request=req)

    # Publishing should not raise, good_handler should receive event, and AlertEvent emitted
    bus.publish(event)
    assert len(order_results) == 1
    assert len(alerts) == 1
    assert alerts[0].level == "ERROR"
    assert "Subscriber explosion!" in alerts[0].message


def test_event_dispatcher_history_limit():
    """Verify circular history buffer respects maxlen limit."""
    bus = EventDispatcher(history_size=5)
    for i in range(10):
        bus.publish(Event())
    history = bus.get_history()
    assert len(history) == 5


def test_bar_nan_and_inf_validation():
    """Verify Bar.__post_init__ strictly rejects NaN and Inf values."""
    ts = datetime.now(timezone.utc)
    for bad_val in [float("nan"), float("inf"), float("-inf")]:
        with pytest.raises(ModelValidationError, match="cannot be None, NaN or Inf"):
            Bar(timestamp=ts, symbol="TEST", open=bad_val, high=105.0, low=95.0, close=100.0, volume=1000.0)

        with pytest.raises(ModelValidationError, match="cannot be None, NaN or Inf"):
            Bar(timestamp=ts, symbol="TEST", open=100.0, high=bad_val, low=95.0, close=100.0, volume=1000.0)

        with pytest.raises(ModelValidationError, match="cannot be None, NaN or Inf"):
            Bar(timestamp=ts, symbol="TEST", open=100.0, high=105.0, low=bad_val, close=100.0, volume=1000.0)

        with pytest.raises(ModelValidationError, match="cannot be None, NaN or Inf"):
            Bar(timestamp=ts, symbol="TEST", open=100.0, high=105.0, low=95.0, close=bad_val, volume=1000.0)

        with pytest.raises(ModelValidationError, match="cannot be None, NaN or Inf"):
            Bar(timestamp=ts, symbol="TEST", open=100.0, high=105.0, low=95.0, close=100.0, volume=bad_val)
