"""Comprehensive Unit Test Suite for modeloINV Execution Subsystem (Milestone 2).

Tests:
1. BaseBroker & BrokerFactory dynamic backend instantiations.
2. Slippage Models (FixedBps, BidAskSpread Parkinson, VolumeImpact Almgren-Chriss, Composite).
3. Fee Models (ZeroCommission SEC/FINRA, CryptoPercentage, EuropeanRetail, Composite).
4. Pre-trade Risk Guardrails (RiskManager, circuit breakers, drawdown stops, kill switch, emergency liquidation).
5. SimulatedPaperBroker (€100 virtual ledger, fractional precision, Market/Limit/Stop/Trailing Stop, invariants).
6. AlpacaPaperBroker (Virtual Sub-Account Partition ceiling, payload serialization, error handling).
7. MockBroker (deterministic execution and error injection).
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from modeloinv.config import AppConfig, ExecutionConfig
from modeloinv.core.currency import CurrencyConverter
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
)
from modeloinv.execution.alpaca import AlpacaPaperBroker
from modeloinv.execution.base import (
    BaseBroker,
    BrokerConnectionError,
    BrokerError,
    BrokerFactory,
    InsufficientFundsError,
    OrderNotFoundError,
    OrderRejectedError,
)
from modeloinv.execution.fees import (
    CompositeFeeModel,
    CryptoPercentageFeeModel,
    EuropeanRetailFeeModel,
    FeeModel,
    ZeroCommissionFeeModel,
)
from modeloinv.execution.mock import MockBroker
from modeloinv.execution.paper import SimulatedPaperBroker
from modeloinv.execution.risk import RiskManager, RiskViolationError
from modeloinv.execution.slippage import (
    BidAskSpreadModel,
    CompositeSlippageModel,
    FixedBpsSlippageModel,
    SlippageModel,
    VolumeImpactSlippageModel,
)


# =====================================================================
# 1. Base Broker & Factory Tests
# =====================================================================

class TestBrokerFactoryAndBase:
    """Tests BrokerFactory resolution and BaseBroker interface."""

    def test_factory_creates_simulated_paper_broker(self):
        broker = BrokerFactory.create_broker("simulated")
        assert isinstance(broker, SimulatedPaperBroker)
        assert broker.cash_eur == 100.00

    def test_factory_creates_mock_broker(self):
        broker = BrokerFactory.create_broker("mock")
        assert isinstance(broker, MockBroker)
        assert broker.cash_eur == 100.00

    def test_factory_creates_alpaca_paper_broker(self):
        broker = BrokerFactory.create_broker("alpaca")
        assert isinstance(broker, AlpacaPaperBroker)
        assert not broker.live

    def test_factory_creates_alpaca_live_broker(self):
        broker = BrokerFactory.create_broker("alpaca_live")
        assert isinstance(broker, AlpacaPaperBroker)
        assert broker.live

    def test_factory_with_config_dict(self):
        cfg = {"broker": {"backend": "mock", "initial_capital": 250.0}}
        broker = BrokerFactory.create_broker(cfg)
        assert isinstance(broker, MockBroker)

    def test_factory_with_app_config(self):
        config = AppConfig()
        config.execution.broker = "paper"
        broker = BrokerFactory.create(config)
        assert isinstance(broker, SimulatedPaperBroker)

    def test_factory_unsupported_backend_raises_error(self):
        with pytest.raises(ValueError, match="Unsupported broker backend"):
            BrokerFactory.create_broker("nonexistent_broker_backend")

    def test_base_broker_defaults(self):
        broker = MockBroker()
        assert broker.connect() is True
        assert broker.is_connected() is True
        assert broker.is_market_open("AAPL") is True
        broker.disconnect()
        assert broker.is_connected() is False


# =====================================================================
# 2. Slippage Model Tests
# =====================================================================

class TestSlippageModels:
    """Validates Fixed BPS, Bid-Ask Spread, Volume Impact, and Composite slippage."""

    def test_fixed_bps_slippage(self):
        model = FixedBpsSlippageModel(bps=10.0)  # 10 bps = 0.1%
        req_buy = OrderRequest(symbol="SPY", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1.0)
        req_sell = OrderRequest(symbol="SPY", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=1.0)

        # 10 bps on $100 = $0.10
        slip = model.calculate_slippage(req_buy, reference_price=100.0)
        assert pytest.approx(slip, rel=1e-5) == 0.10

        fill_buy, slip_buy = model.calculate_fill_price(req_buy, reference_price=100.0)
        assert pytest.approx(fill_buy, rel=1e-5) == 100.10
        assert pytest.approx(slip_buy, rel=1e-5) == 0.10

        fill_sell, slip_sell = model.calculate_fill_price(req_sell, reference_price=100.0)
        assert pytest.approx(fill_sell, rel=1e-5) == 99.90
        assert pytest.approx(slip_sell, rel=1e-5) == 0.10

    def test_fixed_bps_negative_rejected(self):
        with pytest.raises(ValueError, match="cannot be negative"):
            FixedBpsSlippageModel(bps=-5.0)

    def test_bid_ask_spread_model_fallback(self):
        model = BidAskSpreadModel(default_spread_bps=10.0)
        req = OrderRequest(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1.0)

        # Default half-spread = 0.5 * 10 bps = 5 bps on $200 = $0.10
        slip = model.calculate_slippage(req, reference_price=200.0)
        assert pytest.approx(slip, rel=1e-4) == 0.10

    def test_bid_ask_spread_model_with_bar(self):
        model = BidAskSpreadModel()
        bar = Bar(
            timestamp=datetime.datetime.utcnow(),
            symbol="SPY",
            open=100.0,
            high=102.0,
            low=98.0,
            close=101.0,
            volume=10000.0,
        )
        req = OrderRequest(symbol="SPY", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1.0)
        slip = model.calculate_slippage(req, reference_price=100.0, bar=bar)
        assert slip > 0.0

    def test_volume_impact_almgren_chriss_model(self):
        model = VolumeImpactSlippageModel(eta=0.15)
        bar = Bar(
            timestamp=datetime.datetime.utcnow(),
            symbol="NVDA",
            open=100.0,
            high=105.0,
            low=95.0,
            close=100.0,
            volume=10000.0,
        )
        # Small order: 100 shares out of 10000 volume -> participation = 0.01 -> sqrt(0.01) = 0.10
        req_small = OrderRequest(symbol="NVDA", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0)
        slip_small = model.calculate_slippage(req_small, reference_price=100.0, bar=bar)

        # Larger order: 400 shares -> participation = 0.04 -> sqrt(0.04) = 0.20 -> 2x impact of small order
        req_large = OrderRequest(symbol="NVDA", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=400.0)
        slip_large = model.calculate_slippage(req_large, reference_price=100.0, bar=bar)

        assert slip_large > slip_small
        assert pytest.approx(slip_large / slip_small, rel=1e-3) == 2.0

    def test_composite_slippage_model(self):
        m1 = FixedBpsSlippageModel(bps=5.0)  # 0.05 on $100
        m2 = FixedBpsSlippageModel(bps=10.0) # 0.10 on $100
        composite = CompositeSlippageModel([m1, m2])

        req = OrderRequest(symbol="TEST", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1.0)
        slip = composite.calculate_slippage(req, reference_price=100.0)
        assert pytest.approx(slip, rel=1e-5) == 0.15


# =====================================================================
# 3. Transaction Cost & Fee Model Tests
# =====================================================================

class TestFeeModels:
    """Validates Zero-Commission regulatory fees, Crypto fees, and European Retail fees."""

    def test_zero_commission_buy_is_free(self):
        model = ZeroCommissionFeeModel()
        req_buy = OrderRequest(symbol="SPY", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.1)
        fee_eur = model.calculate_fee(req_buy, fill_price=500.0, fill_qty=0.1, fx_rate=1.08)
        assert fee_eur == 0.0

    def test_zero_commission_sell_applies_sec_and_finra_taf(self):
        model = ZeroCommissionFeeModel()
        req_sell = OrderRequest(symbol="SPY", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=10.0)
        # Sell 10 shares at $500 = $5,000 notional
        # SEC: $5000 * 0.0000278 = $0.139
        # FINRA TAF: 10 * 0.000166 = $0.00166 -> min $0.01 applies
        # Total USD = $0.139 + $0.01 = $0.149
        breakdown = model.calculate_fee_breakdown(req_sell, fill_price=500.0, fill_qty=10.0, fx_rate=1.08)
        assert breakdown["sec_fee_eur"] > 0.0
        assert breakdown["finra_taf_eur"] > 0.0
        assert breakdown["total_fee_eur"] > 0.0
        assert pytest.approx(breakdown["sec_fee_eur"] + breakdown["finra_taf_eur"], rel=1e-4) == breakdown["total_fee_eur"]

    def test_crypto_percentage_fee_model(self):
        model = CryptoPercentageFeeModel(maker_fee_pct=0.00075, taker_fee_pct=0.0010)
        # Market order: taker fee 0.10% on $1000 notional = $1.00 USD -> in EUR at 1.08 = 0.9259 EUR
        req_market = OrderRequest(symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.01)
        fee_taker = model.calculate_fee(req_market, fill_price=100000.0, fill_qty=0.01, fx_rate=1.08)
        assert pytest.approx(fee_taker, rel=1e-3) == 1.0 / 1.08

        # Limit order: maker fee 0.075% on $1000 notional = $0.75 USD
        req_limit = OrderRequest(symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=0.01, limit_price=99000.0)
        fee_maker = model.calculate_fee(req_limit, fill_price=100000.0, fill_qty=0.01, fx_rate=1.08)
        assert fee_maker < fee_taker
        assert pytest.approx(fee_maker, rel=1e-3) == 0.75 / 1.08

    def test_european_retail_fee_model(self):
        # myInvestor style: €1.50 flat + 0.12% variable, min €2.00, max €10.00 + 0.25% FX markup
        model = EuropeanRetailFeeModel(
            flat_fee_eur=1.50,
            variable_pct=0.0012,
            min_fee_eur=2.00,
            max_fee_eur=10.00,
            fx_markup_pct=0.0025,
            is_foreign_asset=True,
        )
        # Trade $50 notional = €46.30
        # raw comm = 1.50 + 46.30 * 0.0012 = 1.555 -> clamped to min €2.00
        # FX markup = 46.30 * 0.0025 = €0.11575
        # Total = €2.1158
        req = OrderRequest(symbol="VUSA", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1.0)
        breakdown = model.calculate_fee_breakdown(req, fill_price=50.0, fill_qty=1.0, fx_rate=1.08)
        assert breakdown["commission_eur"] == 2.00
        assert breakdown["fx_markup_eur"] > 0.0
        assert breakdown["total_fee_eur"] > 2.00

    def test_composite_fee_model(self):
        m1 = CryptoPercentageFeeModel(taker_fee_pct=0.001)
        m2 = CryptoPercentageFeeModel(taker_fee_pct=0.002)
        composite = CompositeFeeModel([m1, m2])
        req = OrderRequest(symbol="ETHUSDT", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1.0)
        fee = composite.calculate_fee(req, fill_price=100.0, fill_qty=1.0, fx_rate=1.0)
        assert pytest.approx(fee, rel=1e-5) == 0.30


# =====================================================================
# 4. Pre-Trade Risk Guardrail Tests
# =====================================================================

class TestRiskManager:
    """Validates Kill Switch, Drawdown Stops, Position Limits, and Emergency Liquidation."""

    @pytest.fixture
    def risk_manager(self) -> RiskManager:
        return RiskManager(
            max_position_weight=0.35,  # 35% max per position
            max_open_positions=3,     # max 3 positions
            daily_drawdown_limit=0.10,# 10% daily drawdown stop
            max_drawdown_limit=0.15,  # 15% overall drawdown stop
            kill_switch_drawdown=0.25,# 25% hard kill switch
            min_order_notional_eur=1.00,
            initial_capital_eur=100.00,
        )

    def test_kill_switch_blocks_orders(self, risk_manager):
        account = AccountInfo(cash_eur=100.0, equity_eur=100.0, initial_capital_eur=100.0, buying_power_eur=100.0)
        req = OrderRequest(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.1)

        # Before activation: passes
        assert risk_manager.validate_order(req, account, [], current_market_price=150.0) is True

        # Activate kill switch
        risk_manager.activate_kill_switch("Manual test halt")
        assert risk_manager.kill_switch_active is True

        with pytest.raises(RiskViolationError, match="Kill switch active"):
            risk_manager.validate_order(req, account, [], current_market_price=150.0)

        # Deactivate
        risk_manager.deactivate_kill_switch()
        assert risk_manager.validate_order(req, account, [], current_market_price=150.0) is True

    def test_daily_drawdown_circuit_breaker(self, risk_manager):
        account = AccountInfo(cash_eur=100.0, equity_eur=100.0, initial_capital_eur=100.0, buying_power_eur=100.0)
        req = OrderRequest(symbol="SPY", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.05)

        # Simulate equity dropping from €100 to €89 (11% drop > 10% daily limit)
        risk_manager.update_equity(89.0)
        assert risk_manager.circuit_breaker_tripped is True

        account.equity_eur = 89.0
        # BUY orders are rejected
        with pytest.raises(RiskViolationError, match="circuit breaker"):
            risk_manager.validate_order(req, account, [], current_market_price=500.0)

        # Risk-reducing SELL orders on held positions are accepted even when circuit breaker is tripped
        held_pos = [Position(symbol="SPY", quantity=0.05, entry_price=500.0, current_price=450.0)]
        sell_req = OrderRequest(symbol="SPY", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=0.05)
        assert risk_manager.validate_order(sell_req, account, held_pos, current_market_price=450.0) is True

    def test_max_position_size_limit(self, risk_manager):
        account = AccountInfo(cash_eur=100.0, equity_eur=100.0, initial_capital_eur=100.0, buying_power_eur=100.0)
        # Attempt to buy €45 notional on €100 account (45% > 35% limit)
        req_too_large = OrderRequest(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, notional=45.0)

        with pytest.raises(RiskViolationError, match="exceeding max position size"):
            risk_manager.validate_order(req_too_large, account, [], current_market_price=150.0)

        # Order below 35% succeeds
        req_ok = OrderRequest(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, notional=25.0)
        assert risk_manager.validate_order(req_ok, account, [], current_market_price=150.0) is True

    def test_max_open_positions_limit(self, risk_manager):
        account = AccountInfo(cash_eur=100.0, equity_eur=100.0, initial_capital_eur=100.0, buying_power_eur=100.0)
        positions = [
            Position(symbol="SPY", quantity=0.05, entry_price=500.0, current_price=500.0),
            Position(symbol="QQQ", quantity=0.05, entry_price=400.0, current_price=400.0),
            Position(symbol="NVDA", quantity=0.2, entry_price=100.0, current_price=100.0),
        ]
        # Attempt to open 4th position when limit is 3
        req_4th = OrderRequest(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, notional=10.0)

        with pytest.raises(RiskViolationError, match="maximum 3 open positions"):
            risk_manager.validate_order(req_4th, account, positions, current_market_price=150.0)

    def test_minimum_order_notional_rejection(self, risk_manager):
        account = AccountInfo(cash_eur=100.0, equity_eur=100.0, initial_capital_eur=100.0, buying_power_eur=100.0)
        req_micro = OrderRequest(symbol="SPY", side=OrderSide.BUY, order_type=OrderType.MARKET, notional=0.50)

        with pytest.raises(RiskViolationError, match="below minimum lot"):
            risk_manager.validate_order(req_micro, account, [], current_market_price=500.0)

    def test_emergency_liquidation(self, risk_manager):
        broker = SimulatedPaperBroker(initial_capital_eur=100.0, risk_manager=risk_manager)
        # Open 2 positions
        broker.submit_order(OrderRequest(symbol="SPY", side=OrderSide.BUY, order_type=OrderType.MARKET, notional=20.0), current_market_price=500.0)
        broker.submit_order(OrderRequest(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, notional=20.0), current_market_price=150.0)
        assert len(broker.get_positions()) == 2

        # Simulate circuit breaker tripped
        risk_manager.update_equity(85.0)
        assert risk_manager.circuit_breaker_tripped is True

        # Trigger emergency liquidation - must succeed even under active circuit breaker
        results = risk_manager.emergency_liquidate(broker)
        assert len(results) == 2
        assert all(r.status == OrderStatus.FILLED for r in results)
        assert len(broker.get_positions()) == 0
        assert risk_manager.kill_switch_active is True

    def test_emergency_liquidation_with_active_kill_switch(self, risk_manager):
        broker = SimulatedPaperBroker(initial_capital_eur=100.0, risk_manager=risk_manager)
        broker.submit_order(OrderRequest(symbol="SPY", side=OrderSide.BUY, order_type=OrderType.MARKET, notional=20.0), current_market_price=500.0)
        assert len(broker.get_positions()) == 1

        risk_manager.activate_kill_switch("Risk officer override")
        assert risk_manager.kill_switch_active is True

        results = risk_manager.emergency_liquidate(broker)
        assert len(results) == 1
        assert results[0].status == OrderStatus.FILLED
        assert len(broker.get_positions()) == 0


# =====================================================================
# 5. Simulated Paper Broker Tests (€100 Virtual Ledger & Order Mechanics)
# =====================================================================

class TestSimulatedPaperBroker:
    """Tests the full-fidelity paper simulation engine and accounting invariants."""

    def test_initial_ledger_starts_at_exact_100_eur(self):
        broker = SimulatedPaperBroker(initial_capital_eur=100.0)
        acc = broker.get_account()
        assert acc.currency == "EUR"
        assert acc.cash_eur == 100.00
        assert acc.equity_eur == 100.00
        assert acc.initial_capital_eur == 100.00
        assert acc.buying_power_eur == 100.00
        assert acc.realized_pnl_eur == 0.0
        assert acc.unrealized_pnl_eur == 0.0
        assert len(broker.get_positions()) == 0

    def test_fractional_share_purchase_and_invariant(self):
        broker = SimulatedPaperBroker(initial_capital_eur=100.0)
        # SPY is $500, with EURUSD=1.08, 0.05 shares = $25 USD = €23.15 EUR
        req = OrderRequest(symbol="SPY", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.052145)
        res = broker.submit_order(req, current_market_price=500.0)

        assert res.status == OrderStatus.FILLED
        assert res.filled_quantity == 0.052145
        assert len(broker.get_positions()) == 1

        pos = broker.get_position("SPY")
        assert pos is not None
        assert pos.quantity == 0.052145

        # Check invariant: Equity = Cash + Position Value
        acc = broker.get_account()
        assert pytest.approx(acc.cash_eur + pos.market_value_eur, abs=0.01) == acc.equity_eur

    def test_notional_cash_order_execution(self):
        broker = SimulatedPaperBroker(initial_capital_eur=100.0)
        # Spend exactly €25.00 EUR on NVDA at $100
        req = OrderRequest(symbol="NVDA", side=OrderSide.BUY, order_type=OrderType.MARKET, notional=25.0)
        res = broker.submit_order(req, current_market_price=100.0)

        assert res.status == OrderStatus.FILLED
        assert res.filled_quantity > 0.20
        assert broker.cash_eur < 76.0  # Cash decremented by ~€25
        assert broker.get_account().equity_eur == pytest.approx(100.0, abs=0.5)

    def test_roundtrip_trade_realizes_pnl(self):
        broker = SimulatedPaperBroker(initial_capital_eur=100.0)
        # Buy 1 share at $100
        buy_req = OrderRequest(symbol="TEST", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.5)
        res_buy = broker.submit_order(buy_req, current_market_price=100.0)
        assert res_buy.status == OrderStatus.FILLED

        # Sell 1 share at $120 (gain of $10 USD = €9.26 EUR)
        sell_req = OrderRequest(symbol="TEST", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=0.5)
        res_sell = broker.submit_order(sell_req, current_market_price=120.0)
        assert res_sell.status == OrderStatus.FILLED

        # Position should now be closed
        assert len(broker.get_positions()) == 0
        acc = broker.get_account()
        assert acc.cash_eur > 100.00
        assert acc.equity_eur > 100.00
        assert acc.realized_pnl_eur > 0.0

    def test_insufficient_funds_rejection(self):
        broker = SimulatedPaperBroker(initial_capital_eur=100.0)
        # Attempt to buy €150.00 with €100 cash
        req = OrderRequest(symbol="BRK.A", side=OrderSide.BUY, order_type=OrderType.MARKET, notional=150.0)
        res = broker.submit_order(req, current_market_price=600000.0)

        assert res.status == OrderStatus.REJECTED
        assert "Insufficient funds" in res.message
        assert broker.cash_eur == 100.00

    def test_insufficient_position_quantity_sell_rejection(self):
        broker = SimulatedPaperBroker(initial_capital_eur=100.0)
        # Attempt to sell without holding
        sell_req = OrderRequest(symbol="AAPL", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=1.0)
        res = broker.submit_order(sell_req, current_market_price=150.0)

        assert res.status == OrderStatus.REJECTED
        assert "Insufficient position" in res.message

    def test_limit_order_resting_and_fill_with_price_improvement(self):
        broker = SimulatedPaperBroker(initial_capital_eur=100.0)
        # Market is $100, Limit Buy is $95 -> order should rest in queue
        limit_buy = OrderRequest(
            symbol="SPY",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            limit_price=95.0,
        )
        res_accept = broker.submit_order(limit_buy, current_market_price=100.0)
        assert res_accept.status == OrderStatus.ACCEPTED
        assert len(broker.get_open_orders()) == 1

        # Feed bar where Low hits $94 and Open is $94.5 (price improvement)
        bar = Bar(
            timestamp=datetime.datetime.utcnow(),
            symbol="SPY",
            open=94.5,
            high=96.0,
            low=94.0,
            close=95.5,
            volume=50000.0,
        )
        broker.process_bar(bar)

        # Order should now be filled
        assert len(broker.get_open_orders()) == 0
        assert len(broker.get_positions()) == 1
        pos = broker.get_position("SPY")
        assert pos is not None
        # Should fill at open ($94.5) + slippage, better than limit $95.0
        assert pos.entry_price < 95.0

    def test_stop_loss_order_gap_slippage(self):
        broker = SimulatedPaperBroker(initial_capital_eur=100.0)
        # Buy 0.5 shares at $100
        broker.submit_order(OrderRequest(symbol="SPY", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.5), current_market_price=100.0)

        # Place Sell Stop at $90
        stop_req = OrderRequest(
            symbol="SPY",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            quantity=0.5,
            stop_price=90.0,
        )
        res_stop = broker.submit_order(stop_req, current_market_price=100.0)
        assert res_stop.status == OrderStatus.ACCEPTED

        # Overnight gap down: Bar opens at $85 (breaching stop $90)
        bar_gap = Bar(
            timestamp=datetime.datetime.utcnow(),
            symbol="SPY",
            open=85.0,
            high=86.0,
            low=84.0,
            close=85.5,
            volume=20000.0,
        )
        broker.process_bar(bar_gap)

        # Order should be filled at gap open ($85) minus slippage
        assert len(broker.get_positions()) == 0
        trades = broker.get_trade_history()
        assert len(trades) == 2
        last_trade = trades[-1]
        assert last_trade.price <= 85.0  # Captured gap slippage!

    def test_trailing_stop_ratchets_and_triggers(self):
        broker = SimulatedPaperBroker(initial_capital_eur=100.0)
        broker.submit_order(OrderRequest(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.2), current_market_price=100.0)

        # Trailing stop with 5% trail (trail_percent=5.0)
        trail_req = OrderRequest(
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.TRAILING_STOP,
            quantity=0.2,
            trail_percent=5.0,
        )
        res_trail = broker.submit_order(trail_req, current_market_price=100.0)
        assert res_trail.status == OrderStatus.ACCEPTED

        # Price rises to $120 -> stop ratchets to $120 * 0.95 = $114
        broker.update_market_price("AAPL", 120.0)
        assert len(broker.get_positions()) == 1  # Not triggered

        # Price pulls back to $118 -> above $114, does not trigger
        broker.update_market_price("AAPL", 118.0)
        assert len(broker.get_positions()) == 1

        # Price drops to $113 -> breaches $114, triggers market sell!
        broker.update_market_price("AAPL", 113.0)
        assert len(broker.get_positions()) == 0
        assert len(broker.get_open_orders()) == 0

    def test_limit_order_fill_price_guarantee_no_adverse_slippage(self):
        """Limit orders must never fill worse than limit price."""
        broker = SimulatedPaperBroker(initial_capital_eur=100.0, slippage_model=FixedBpsSlippageModel(bps=20.0))
        # Buy limit at $100.00
        buy_lim = OrderRequest(symbol="SPY", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=0.1, limit_price=100.0)
        res_buy = broker.submit_order(buy_lim, current_market_price=100.0)
        assert res_buy.status == OrderStatus.FILLED
        assert res_buy.filled_avg_price <= 100.0

        # Sell limit at $100.00
        sell_lim = OrderRequest(symbol="SPY", side=OrderSide.SELL, order_type=OrderType.LIMIT, quantity=0.1, limit_price=100.0)
        res_sell = broker.submit_order(sell_lim, current_market_price=100.0)
        assert res_sell.status == OrderStatus.FILLED
        assert res_sell.filled_avg_price >= 100.0

    def test_tight_trailing_stop_percent(self):
        """Tight trailing stops (0.5% and 1.0%) calculate correct stop price threshold."""
        broker = SimulatedPaperBroker(initial_capital_eur=100.0)
        broker.submit_order(OrderRequest(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.5), current_market_price=200.0)

        # 0.5% trailing stop on $200 -> stop_price = $200 * (1 - 0.005) = $199.00
        trail_req = OrderRequest(symbol="AAPL", side=OrderSide.SELL, order_type=OrderType.TRAILING_STOP, quantity=0.5, trail_percent=0.5)
        res = broker.submit_order(trail_req, current_market_price=200.0)
        assert res.status == OrderStatus.ACCEPTED
        pending = broker.pending_orders[res.order_id]
        assert pending["trail_factor"] == 0.005
        assert pytest.approx(pending["stop_price"], rel=1e-4) == 199.00

    def test_microtrade_1e6_shares_execution(self):
        """Broker cleanly handles fractional microtrade down to 1e-6 (0.000001) shares."""
        broker = SimulatedPaperBroker(initial_capital_eur=100.0)
        req = OrderRequest(symbol="SPY", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.000001)
        res = broker.submit_order(req, current_market_price=500.0)
        assert res.status == OrderStatus.FILLED
        assert res.filled_quantity == 0.000001
        pos = broker.get_position("SPY")
        assert pos is not None
        assert pos.quantity == 0.000001

        # Sell the 1e-6 position
        sell_req = OrderRequest(symbol="SPY", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=0.000001)
        sell_res = broker.submit_order(sell_req, current_market_price=500.0)
        assert sell_res.status == OrderStatus.FILLED
        assert len(broker.get_positions()) == 0

    def test_buy_fee_included_in_cost_basis_and_realized_pnl(self):
        """Cost basis includes buy fees, making realized P&L symmetric with net cash loss."""
        fee_model = EuropeanRetailFeeModel(flat_fee_eur=2.0, variable_pct=0.0, min_fee_eur=2.0, max_fee_eur=2.0)
        broker = SimulatedPaperBroker(initial_capital_eur=100.0, fee_model=fee_model)
        broker.submit_order(OrderRequest(symbol="ABC", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1.0), current_market_price=10.0)
        broker.submit_order(OrderRequest(symbol="ABC", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=1.0), current_market_price=10.0)
        acc = broker.get_account()
        cash_loss = 100.0 - acc.cash_eur
        assert pytest.approx(cash_loss, rel=1e-3) == -acc.realized_pnl_eur


# =====================================================================
# 6. Alpaca Paper Broker Tests (Virtual Sub-Account Partition)
# =====================================================================

class TestAlpacaPaperBroker:
    """Tests AlpacaPaperBroker virtual partition ceiling and JSON payload translation."""

    def test_virtual_capital_partition_rejects_overbudget_order(self):
        broker = AlpacaPaperBroker(virtual_capital_eur=100.0)
        # Attempt to buy €250 on Alpaca sandbox (even though sandbox has $100k)
        req_overbudget = OrderRequest(
            symbol="SPY",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            notional=250.0,
        )
        res = broker.submit_order(req_overbudget, current_market_price=500.0)
        assert res.status == OrderStatus.REJECTED
        assert "Virtual Sub-Account Partition" in res.message

    def test_virtual_partition_accepts_underbudget_order(self):
        broker = AlpacaPaperBroker(virtual_capital_eur=100.0)
        req_ok = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            notional=30.0,
        )
        res = broker.submit_order(req_ok, current_market_price=150.0)
        assert res.status == OrderStatus.FILLED
        assert len(broker.get_positions()) == 1
        assert broker.get_account().equity_eur <= 101.0

    def test_alpaca_payload_serialization(self):
        broker = AlpacaPaperBroker()
        req = OrderRequest(
            symbol="MSFT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.123456,
            limit_price=420.50,
            time_in_force=TimeInForce.GTC,
            client_order_id="test_client_001",
        )
        payload = broker._to_alpaca_payload(req, qty=0.123456, notional_eur=50.0, fx_rate=1.08)
        assert payload["symbol"] == "MSFT"
        assert payload["side"] == "buy"
        assert payload["type"] == "limit"
        assert payload["time_in_force"] == "gtc"
        assert payload["qty"] == "0.123456"
        assert payload["limit_price"] == "420.5"
        assert payload["client_order_id"] == "test_client_001"

    def test_alpaca_live_mode_without_credentials_raises_error(self):
        broker = AlpacaPaperBroker(live=True, api_key_id="", api_secret_key="")
        with pytest.raises(BrokerConnectionError, match="credentials"):
            broker.connect()

    def test_alpaca_fallback_rejects_unowned_sell_and_reports_unrealized_pnl(self):
        broker = AlpacaPaperBroker(virtual_capital_eur=100.0)
        # Attempt to sell unowned stock
        res = broker.submit_order(OrderRequest(symbol="TSLA", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=5.0), current_market_price=200.0)
        assert res.status == OrderStatus.REJECTED
        assert "Insufficient position" in res.message
        assert broker.get_account().cash_eur == 100.0

        # Buy €50 of AAPL at $100 -> unrealized P&L is 0.0 before price movement
        broker.submit_order(OrderRequest(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, notional=50.0), current_market_price=100.0)
        acc = broker.get_account()
        assert pytest.approx(acc.unrealized_pnl_eur, abs=1e-4) == 0.0


# =====================================================================
# 7. Mock Broker Tests (Error Injection & Determinism)
# =====================================================================

class TestMockBroker:
    """Tests deterministic execution and error injection capabilities."""

    def test_mock_broker_injected_rejection(self):
        broker = MockBroker()
        broker.inject_rejection("Simulated exchange halt")
        req = OrderRequest(symbol="SPY", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1.0)
        res = broker.submit_order(req)
        assert res.status == OrderStatus.REJECTED
        assert res.message == "Simulated exchange halt"

        # Clear injection
        broker.clear_injections()
        res_ok = broker.submit_order(req)
        assert res_ok.status == OrderStatus.FILLED

    def test_mock_broker_injected_timeout(self):
        broker = MockBroker()
        broker.inject_network_timeout(True)
        with pytest.raises(BrokerConnectionError, match="timeout"):
            broker.get_account()

        broker.clear_injections()
        acc = broker.get_account()
        assert acc.cash_eur == 100.00

    def test_mock_broker_injected_partial_fill(self):
        broker = MockBroker()
        broker.inject_partial_fill(0.40)  # 40% partial fill
        req = OrderRequest(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1.0)
        res = broker.submit_order(req, current_market_price=100.0)
        assert res.status == OrderStatus.PARTIALLY_FILLED
        assert pytest.approx(res.filled_quantity, rel=1e-5) == 0.40

    def test_mock_broker_order_cancellation_and_lookup(self):
        broker = MockBroker()
        # Submit order that is rejected or queued
        req = OrderRequest(symbol="TEST", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1.0)
        res = broker.submit_order(req)
        assert res.status == OrderStatus.FILLED
        lookup = broker.get_order_status(res.order_id)
        assert lookup.order_id == res.order_id

        # Nonexistent order raises OrderNotFoundError
        with pytest.raises(OrderNotFoundError):
            broker.get_order_status("nonexistent_id")

    def test_simulated_paper_broker_order_cancellation(self):
        broker = SimulatedPaperBroker(initial_capital_eur=100.0)
        req = OrderRequest(symbol="SPY", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=0.1, limit_price=50.0)
        res = broker.submit_order(req, current_market_price=100.0)
        assert res.status == OrderStatus.ACCEPTED
        assert len(broker.get_open_orders()) == 1

        # Cancel open order
        assert broker.cancel_order(res.order_id) is True
        assert len(broker.get_open_orders()) == 0
        assert broker.get_order_status(res.order_id).status == OrderStatus.CANCELLED

    def test_simulated_paper_broker_fx_drift_mark_to_market(self):
        converter = CurrencyConverter(fallback_rate=1.00)
        broker = SimulatedPaperBroker(initial_capital_eur=100.0, currency_converter=converter)

        # Buy 0.5 shares of SPY at $100 when EURUSD=1.00 -> Cost = €50.00 EUR
        broker.submit_order(OrderRequest(symbol="SPY", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.5), current_market_price=100.0)
        acc_before = broker.get_account()
        assert pytest.approx(acc_before.cash_eur, abs=0.1) == 50.00
        assert pytest.approx(acc_before.equity_eur, abs=0.1) == 100.00

        # EUR strengthens to 1.25 USD per 1 EUR (asset USD value is still $50, but in EUR it is only €40)
        converter.set_rate("EURUSD", 1.25)
        acc_after = broker.get_account()
        # Equity should drop to ~€50 cash + €40 position = €90
        assert pytest.approx(acc_after.equity_eur, abs=0.1) == 90.00
        assert acc_after.unrealized_pnl_eur < 0.0
