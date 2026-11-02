"""Adversarial Challenge Test Suite for Milestone 2 (Execution, Slippage, Risk & Partitioning).

Written by Challenger 2 (`challenger_m2_2`) to empirically stress-test:
1. Volume Impact Slippage Model under extreme zero / spike volume and Q >= V.
2. Pre-Trade Risk Circuit Breakers under artificial drawdowns.
3. Emergency Kill Switch and Liquidation execution.
4. Alpaca Sub-Account Partition (€100 virtual capital ceiling and short-selling edge cases).
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
import pytest

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
)
from modeloinv.execution.alpaca import AlpacaPaperBroker
from modeloinv.execution.fees import ZeroCommissionFeeModel
from modeloinv.execution.paper import SimulatedPaperBroker
from modeloinv.execution.risk import RiskManager, RiskViolationError
from modeloinv.execution.slippage import (
    BidAskSpreadModel,
    CompositeSlippageModel,
    FixedBpsSlippageModel,
    VolumeImpactSlippageModel,
)


# ============================================================================
# 1. Volume Impact Slippage Model Adversarial Stress Tests
# ============================================================================

class TestVolumeImpactSlippageAdversarial:
    """Adversarial stress-testing of VolumeImpactSlippageModel under extreme conditions."""

    def test_volume_zero_fallback(self):
        """Confirm non-infinite, finite, strictly positive execution prices when bar volume is zero."""
        model = VolumeImpactSlippageModel(eta=0.15, default_volume=50000.0)
        bar_zero_vol = Bar(
            symbol="SPY",
            timestamp=datetime.utcnow(),
            open=500.0,
            high=505.0,
            low=495.0,
            close=500.0,
            volume=0.0,
        )
        req_buy = OrderRequest(symbol="SPY", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=10.0)
        req_sell = OrderRequest(symbol="SPY", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=10.0)

        # Slippage calculation
        slip = model.calculate_slippage(req_buy, 500.0, bar_zero_vol)
        assert not math.isinf(slip), "Slippage must not be infinite when V=0"
        assert not math.isnan(slip), "Slippage must not be NaN when V=0"
        assert slip >= 0.0, "Slippage must be non-negative"

        # Fill price calculation
        buy_fill, buy_slip = model.calculate_fill_price(req_buy, 500.0, bar_zero_vol)
        sell_fill, sell_slip = model.calculate_fill_price(req_sell, 500.0, bar_zero_vol)

        assert buy_fill > 500.0, "Buy fill price must reflect positive slippage penalty"
        assert 0.0 < sell_fill < 500.0, "Sell fill price must be strictly positive and discounted"
        assert not math.isinf(buy_fill) and not math.isinf(sell_fill)

    def test_extreme_order_quantity_exceeding_volume_clipping(self):
        """Test orders where Q >> V (e.g. Q=1,000,000 vs V=100); verify proper participation clipping."""
        max_rate = 0.10
        model = VolumeImpactSlippageModel(eta=0.20, max_participation_rate=max_rate)
        bar_low_vol = Bar(
            symbol="ILLIQ",
            timestamp=datetime.utcnow(),
            open=50.0,
            high=52.0,
            low=48.0,
            close=50.0,
            volume=100.0,
        )

        req_1k = OrderRequest(symbol="ILLIQ", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1000.0)
        req_1m = OrderRequest(symbol="ILLIQ", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1000000.0)

        slip_1k = model.calculate_slippage(req_1k, 50.0, bar_low_vol)
        slip_1m = model.calculate_slippage(req_1m, 50.0, bar_low_vol)

        # Because Q / V for both (10.0 and 10,000.0) exceed max_participation_rate (0.10),
        # both must be clipped to sqrt(0.10) exactly, resulting in identical bounded slippage
        expected_sigma = (52.0 - 48.0) / 50.0
        expected_impact = 0.20 * expected_sigma * math.sqrt(max_rate) * 50.0
        assert math.isclose(slip_1k, expected_impact, rel_tol=1e-5), "Slippage must be clipped at max_participation_rate"
        assert math.isclose(slip_1m, expected_impact, rel_tol=1e-5), "Massive orders must be strictly bounded"

        # Check execution price non-infinite and strictly positive
        fill_price, _ = model.calculate_fill_price(req_1m, 50.0, bar_low_vol)
        assert 0.0 < fill_price < 1000.0, "Fill price must remain realistic and strictly positive"

    def test_spike_volume_convergence(self):
        """Test orders during massive volume spikes (V = 10^12); verify slippage approaches zero without underflow."""
        model = VolumeImpactSlippageModel(eta=0.15)
        bar_spike = Bar(
            symbol="SPY",
            timestamp=datetime.utcnow(),
            open=500.0,
            high=501.0,
            low=499.0,
            close=500.0,
            volume=1e12,
        )
        req = OrderRequest(symbol="SPY", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0)

        slip = model.calculate_slippage(req, 500.0, bar_spike)
        assert 0.0 <= slip < 0.001, "Slippage during spike volume must be near-zero"
        fill_price, _ = model.calculate_fill_price(req, 500.0, bar_spike)
        assert math.isclose(fill_price, 500.0, abs_tol=1e-3)

    def test_sell_fill_price_never_zero_or_negative_under_extreme_impact(self):
        """Verify that sell fill prices are strictly positive (> 0) even under absurd market impact."""
        # Configured with massive impact factor
        hyper_impact_model = VolumeImpactSlippageModel(eta=500.0, max_participation_rate=1.0)
        bar = Bar(
            symbol="CRASH",
            timestamp=datetime.utcnow(),
            open=10.0,
            high=20.0,
            low=1.0,
            close=10.0,
            volume=10.0,
        )
        req_sell = OrderRequest(symbol="CRASH", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=100.0)

        fill_price, slip = hyper_impact_model.calculate_fill_price(req_sell, 10.0, bar)
        assert fill_price > 0.0, f"Sell fill price {fill_price} must be strictly positive"
        assert fill_price >= 0.0001, "Sell fill price must respect the minimum execution floor"

    def test_corrupted_bar_inputs_resilience(self):
        """Verify Bar validation prevents negative volume and VolumeImpactSlippage handles edge cases."""
        model = VolumeImpactSlippageModel()
        req = OrderRequest(symbol="TEST", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=5.0)

        # Bar constructor rejects negative volume
        with pytest.raises(Exception):
            Bar(symbol="T", timestamp=datetime.utcnow(), open=100.0, high=105.0, low=95.0, close=100.0, volume=-500.0)

        # Bar with valid fields but zero volume
        bar_valid = Bar(symbol="T", timestamp=datetime.utcnow(), open=100.0, high=105.0, low=95.0, close=100.0, volume=0.0)
        slip = model.calculate_slippage(req, 100.0, bar_valid)
        assert slip > 0.0 and not math.isinf(slip)


# ============================================================================
# 2. Risk Circuit Breaker Adversarial Tests
# ============================================================================

class TestRiskCircuitBreakerAdversarial:
    """Adversarial stress-testing of RiskManager circuit breakers under artificial drawdowns."""

    def test_artificial_daily_drawdown_trips_circuit_breaker(self):
        """Inject artificial 6% drawdown on 5% daily limit; confirm circuit breaker trips and blocks new orders."""
        rm = RiskManager(daily_drawdown_limit=0.05, initial_capital_eur=100.0)
        assert not rm.circuit_breaker_tripped

        # Equity drops from €100.0 to €94.0 (6% loss)
        rm.update_equity(94.0)
        assert rm.circuit_breaker_tripped is True
        assert "daily drawdown 6.00%" in rm.circuit_breaker_reason

        # Any new buy order must be rejected
        account = AccountInfo(cash_eur=94.0, equity_eur=94.0, initial_capital_eur=100.0, buying_power_eur=94.0)
        order_buy = OrderRequest(symbol="SPY", side=OrderSide.BUY, order_type=OrderType.MARKET, notional=10.0)

        with pytest.raises(RiskViolationError, match="Risk circuit breaker tripped"):
            rm.validate_order(order_buy, account, [], current_market_price=500.0)

    def test_paper_broker_enforces_circuit_breaker_halt(self):
        """Simulate SimulatedPaperBroker rejecting incoming orders when circuit breaker trips."""
        rm = RiskManager(daily_drawdown_limit=0.10, initial_capital_eur=100.0)
        broker = SimulatedPaperBroker(initial_capital_eur=100.0, risk_manager=rm)

        # Trip breaker via drawdown
        rm.update_equity(88.0)
        assert rm.circuit_breaker_tripped is True

        # Submit buy order through broker
        req = OrderRequest(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, notional=15.0)
        res = broker.submit_order(req, current_market_price=150.0)

        assert res.status == OrderStatus.REJECTED
        assert "circuit breaker" in res.message.lower()

    def test_daily_circuit_breaker_resets_on_new_day(self):
        """Verify that daily circuit breaker resets on new day when kill switch is NOT active."""
        rm = RiskManager(daily_drawdown_limit=0.05, initial_capital_eur=100.0)
        t0 = datetime(2026, 10, 26, 14, 0, 0)
        rm.update_equity(94.0, timestamp=t0)
        assert rm.circuit_breaker_tripped is True

        # Advance to next trading day
        t1 = datetime(2026, 10, 27, 9, 30, 0)
        rm.update_equity(94.0, timestamp=t1)
        assert rm.circuit_breaker_tripped is False


# ============================================================================
# 3. Emergency Kill Switch and Liquidation Vulnerability Tests
# ============================================================================

class TestEmergencyKillSwitchVulnerability:
    """Empirical investigation into emergency liquidation failure modes."""

    def test_kill_switch_blocks_all_subsequent_orders(self):
        """Verify that activating kill switch halts further trading."""
        rm = RiskManager(initial_capital_eur=100.0)
        rm.activate_kill_switch("Manual test shutdown")
        assert rm.kill_switch_active is True

        account = AccountInfo(cash_eur=100.0, equity_eur=100.0, initial_capital_eur=100.0, buying_power_eur=100.0)
        req = OrderRequest(symbol="SPY", side=OrderSide.BUY, order_type=OrderType.MARKET, notional=10.0)

        with pytest.raises(RiskViolationError, match="Kill switch active"):
            rm.validate_order(req, account, [], current_market_price=500.0)

    def test_empirical_vulnerability_circuit_breaker_blocks_emergency_liquidation(self):
        """Verify that emergency liquidation succeeds even when circuit breaker is tripped by drawdown."""
        rm = RiskManager(daily_drawdown_limit=0.10, initial_capital_eur=100.0)
        broker = SimulatedPaperBroker(initial_capital_eur=100.0, risk_manager=rm)

        # 1. Open a position in SPY
        buy_res = broker.submit_order(
            OrderRequest(symbol="SPY", side=OrderSide.BUY, order_type=OrderType.MARKET, notional=20.0),
            current_market_price=500.0,
        )
        assert buy_res.status == OrderStatus.FILLED
        assert len(broker.get_positions()) == 1

        # 2. Market crash trips the daily circuit breaker (15% loss)
        rm.update_equity(85.0)
        assert rm.circuit_breaker_tripped is True

        # 3. Execute emergency liquidation
        liq_results = rm.emergency_liquidate(broker)
        assert len(liq_results) == 1

        liq_order = liq_results[0]
        # In remediated codebase, emergency liquidation succeeds and liquidates 100% of positions
        assert liq_order.status == OrderStatus.FILLED
        assert len(broker.get_positions()) == 0

    def test_empirical_vulnerability_active_kill_switch_blocks_emergency_liquidation(self):
        """Verify that emergency liquidation succeeds even when kill switch is already active."""
        rm = RiskManager(initial_capital_eur=100.0)
        broker = SimulatedPaperBroker(initial_capital_eur=100.0, risk_manager=rm)

        # 1. Open position
        broker.submit_order(
            OrderRequest(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, notional=20.0),
            current_market_price=150.0,
        )
        assert len(broker.get_positions()) == 1

        # 2. Kill switch triggered
        rm.activate_kill_switch("Hard stop triggered by risk officer")
        assert rm.kill_switch_active is True

        # 3. Trigger emergency liquidation
        liq_results = rm.emergency_liquidate(broker)
        assert len(liq_results) == 1
        # Emergency liquidation orders bypass pre-trade kill switch block
        assert liq_results[0].status == OrderStatus.FILLED
        assert len(broker.get_positions()) == 0


# ============================================================================
# 4. Alpaca Virtual Sub-Account Partition Adversarial Tests
# ============================================================================

class TestAlpacaVirtualPartitionAdversarial:
    """Adversarial stress-testing of AlpacaPaperBroker virtual €100 ceiling enforcement."""

    def test_alpaca_rejects_single_order_exceeding_100_eur(self):
        """Attempting to buy €200 of assets on virtual €100 partition must be strictly rejected."""
        broker = AlpacaPaperBroker(virtual_capital_eur=100.0)

        # Request €200 of SPY
        req_200 = OrderRequest(symbol="SPY", side=OrderSide.BUY, order_type=OrderType.MARKET, notional=200.0)
        res = broker.submit_order(req_200, current_market_price=500.0)

        assert res.status == OrderStatus.REJECTED
        assert "exceeds virtual ceiling" in res.message

    def test_alpaca_rejects_cumulative_orders_exceeding_100_eur(self):
        """First order of €80 succeeds; second order of €30 exceeds €100 ceiling and is rejected."""
        broker = AlpacaPaperBroker(virtual_capital_eur=100.0)

        # First order: €80
        r1 = broker.submit_order(
            OrderRequest(symbol="SPY", side=OrderSide.BUY, order_type=OrderType.MARKET, notional=80.0),
            current_market_price=500.0,
        )
        assert r1.status == OrderStatus.FILLED

        # Second order: €30 (cumulative €110 > €100)
        r2 = broker.submit_order(
            OrderRequest(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, notional=30.0),
            current_market_price=150.0,
        )
        assert r2.status == OrderStatus.REJECTED
        assert "exceeds virtual ceiling" in r2.message

    def test_empirical_vulnerability_alpaca_unowned_short_sell_cash_inflation(self):
        """Verify that Alpaca fallback rejects selling unowned shares, preventing phantom cash inflation."""
        broker = AlpacaPaperBroker(virtual_capital_eur=100.0)
        initial_cash = broker.get_account().cash_eur
        assert initial_cash == 100.0

        # Attempt to sell 100 shares of TSLA without owning any
        phantom_sell = OrderRequest(
            symbol="TSLA",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=100.0,
        )
        res = broker.submit_order(phantom_sell, current_market_price=200.0)

        # In remediated implementation, unowned short sell is rejected and cash remains €100.0
        assert res.status == OrderStatus.REJECTED
        assert broker.get_account().cash_eur == 100.0
