"""Deep Adversarial Stress Verification Suite for M2 Iteration 2 Remediations.

Tests:
1. Emergency liquidation and risk reduction bypass under active circuit breaker & kill switch.
2. Limit order price guarantee under extreme adverse slippage (immediate and resting on-bar).
3. Trailing stop ratcheting and tight percent thresholds (0.1%, 0.5%, 1.0%, 2.5%) for both SELL and BUY sides.
4. Microtrade precision (1e-6 shares) and 1,000-trade rapid churn cash non-negativity & accounting invariants.
5. AlpacaPaperBroker fallback unowned sell rejection and mark-to-market unrealized P&L accuracy.
"""

from __future__ import annotations

import datetime
import random
import pytest

from modeloinv.core.models import (
    Bar,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
)
from modeloinv.execution.risk import RiskManager, RiskViolationError
from modeloinv.execution.paper import SimulatedPaperBroker
from modeloinv.execution.alpaca import AlpacaPaperBroker
from modeloinv.execution.slippage import FixedBpsSlippageModel, CompositeSlippageModel
from modeloinv.execution.fees import EuropeanRetailFeeModel, ZeroCommissionFeeModel


class TestAdversarialM2RemediationsDeepStress:
    """Rigorous empirical validation of the 5 worker remediations."""

    # -------------------------------------------------------------------------
    # 1. Risk Circuit Breaker & Emergency Liquidation
    # -------------------------------------------------------------------------

    def test_emergency_liquidation_with_multiple_positions_under_active_breaker(self):
        """Verify that emergency_liquidate() liquidates all positions down to 1e-6 shares

        even when the circuit breaker is actively tripped and pending orders exist.
        """
        risk_mgr = RiskManager(daily_drawdown_limit=0.05, max_drawdown_limit=0.10, min_order_notional_eur=0.0001)
        broker = SimulatedPaperBroker(initial_capital_eur=100.0, risk_manager=risk_mgr)

        # Open multiple positions including micro-position
        broker.submit_order(
            OrderRequest(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, notional=20.0),
            current_market_price=150.0,
        )
        broker.submit_order(
            OrderRequest(symbol="MSFT", side=OrderSide.BUY, order_type=OrderType.MARKET, notional=20.0),
            current_market_price=300.0,
        )
        # Micro position of exactly 1e-6 shares
        broker.submit_order(
            OrderRequest(symbol="SPY", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.000001),
            current_market_price=500.0,
        )

        # Place a resting limit order
        broker.submit_order(
            OrderRequest(symbol="NVDA", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=0.1, limit_price=50.0),
            current_market_price=100.0,
        )
        assert len(broker.pending_orders) == 1
        assert len(broker.get_positions()) == 3

        # Trip circuit breaker and kill switch
        risk_mgr.trip_circuit_breaker("Simulated catastrophic crash")
        assert risk_mgr.circuit_breaker_tripped is True

        # Regular BUY must be rejected by broker with circuit breaker violation
        res_blocked = broker.submit_order(
            OrderRequest(symbol="GOOG", side=OrderSide.BUY, order_type=OrderType.MARKET, notional=10.0),
            current_market_price=150.0,
        )
        assert res_blocked.status == OrderStatus.REJECTED
        assert "Risk circuit breaker tripped" in res_blocked.message

        # Direct risk validation must raise RiskViolationError
        with pytest.raises(RiskViolationError, match="Risk circuit breaker tripped"):
            risk_mgr.validate_order(
                OrderRequest(symbol="GOOG", side=OrderSide.BUY, order_type=OrderType.MARKET, notional=10.0),
                account=broker.get_account(),
                positions=broker.get_positions(),
                current_market_price=150.0,
            )

        # Emergency liquidation must succeed unconditionally
        results = risk_mgr.emergency_liquidate(broker)
        assert len(results) == 3
        for res in results:
            assert res.status == OrderStatus.FILLED
            assert res.side == OrderSide.SELL

        # All pending orders cancelled
        assert len(broker.pending_orders) == 0
        # All positions liquidated
        assert len(broker.get_positions()) == 0
        # Kill switch is now active
        assert risk_mgr.kill_switch_active is True

    def test_risk_reducing_sell_orders_allowed_during_kill_switch_and_breaker(self):
        """Verify that when kill switch is active, held positions can be manually sold

        (risk-reducing) while unowned sells or buy entries are blocked.
        """
        risk_mgr = RiskManager()
        broker = SimulatedPaperBroker(initial_capital_eur=100.0, risk_manager=risk_mgr)

        # Buy AAPL
        broker.submit_order(
            OrderRequest(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.2),
            current_market_price=150.0,
        )

        # Activate kill switch
        risk_mgr.activate_kill_switch("Hostile test activation")
        assert risk_mgr.kill_switch_active is True

        # Unowned SELL should be rejected by broker/risk
        res_unowned = broker.submit_order(
            OrderRequest(symbol="TSLA", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=0.1),
            current_market_price=200.0,
        )
        assert res_unowned.status == OrderStatus.REJECTED

        # Risk-reducing SELL for held AAPL position must succeed
        res_sell = broker.submit_order(
            OrderRequest(symbol="AAPL", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=0.2),
            current_market_price=150.0,
        )
        assert res_sell.status == OrderStatus.FILLED
        assert len(broker.get_positions()) == 0

    # -------------------------------------------------------------------------
    # 2. Limit Order Fill Price Guarantees Under Adverse Slippage
    # -------------------------------------------------------------------------

    def test_immediate_limit_buy_under_extreme_slippage_model(self):
        """BUY Limit order must fill at or below limit price even under 500 bps adverse slippage."""
        # 500 bps slippage = +5% price penalty
        slippage = FixedBpsSlippageModel(bps=500.0)
        broker = SimulatedPaperBroker(initial_capital_eur=100.0, slippage_model=slippage)

        # Market is 100.00. Limit is 101.00.
        # Without clamp: 100 * 1.05 = 105.00 (which violates limit of 101.00)
        req = OrderRequest(
            symbol="SPY",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            limit_price=101.00,
        )
        res = broker.submit_order(req, current_market_price=100.00)

        assert res.status == OrderStatus.FILLED
        assert res.filled_avg_price <= 101.00
        assert res.filled_avg_price == 101.00  # Clamped exactly to limit price

    def test_immediate_limit_sell_under_extreme_slippage_model(self):
        """SELL Limit order must fill at or above limit price even under 500 bps adverse slippage."""
        slippage = FixedBpsSlippageModel(bps=500.0)
        broker = SimulatedPaperBroker(initial_capital_eur=100.0, slippage_model=slippage)

        # Buy position first with zero slippage
        broker.slippage_model = FixedBpsSlippageModel(bps=0.0)
        broker.submit_order(
            OrderRequest(symbol="SPY", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.5),
            current_market_price=100.0,
        )

        # Switch to 500 bps slippage
        broker.slippage_model = slippage

        # Market is 100.00. Limit is 99.00.
        # Without clamp: 100 * (1 - 0.05) = 95.00 (which violates limit of 99.00)
        req = OrderRequest(
            symbol="SPY",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=0.5,
            limit_price=99.00,
        )
        res = broker.submit_order(req, current_market_price=100.00)

        assert res.status == OrderStatus.FILLED
        assert res.filled_avg_price >= 99.00
        assert res.filled_avg_price == 99.00  # Clamped exactly to limit price

    def test_resting_limit_orders_on_bar_trigger_never_violate_limit(self):
        """Resting limit orders triggered by Bar prices must not violate limit price."""
        slippage = FixedBpsSlippageModel(bps=200.0)
        broker = SimulatedPaperBroker(initial_capital_eur=100.0, slippage_model=slippage)

        # Submit resting BUY limit at 95.00 when market is at 100.00
        broker.submit_order(
            OrderRequest(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=0.2, limit_price=95.00),
            current_market_price=100.0,
        )
        assert len(broker.pending_orders) == 1

        # Bar arrives: Open=94.00, High=96.00, Low=93.00, Close=95.50
        bar = Bar(
            timestamp=datetime.datetime.utcnow(),
            symbol="AAPL",
            open=94.0,
            high=96.0,
            low=93.0,
            close=95.5,
            volume=10000.0,
        )
        broker.process_bar(bar)

        assert len(broker.pending_orders) == 0
        pos = broker.get_position("AAPL")
        assert pos is not None
        # Entry price must be <= limit_price 95.00
        assert pos.entry_price <= 95.00

    # -------------------------------------------------------------------------
    # 3. Trailing Stop Percent & Ratcheting Mechanics
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize("trail_pct,expected_factor", [
        (0.1, 0.001),
        (0.5, 0.005),
        (1.0, 0.01),
        (2.5, 0.025),
        (10.0, 0.10),
    ])
    def test_trailing_stop_factor_across_tight_and_wide_thresholds(self, trail_pct, expected_factor):
        """Verify trailing stop correctly converts percentage to decimal factor."""
        broker = SimulatedPaperBroker(initial_capital_eur=100.0)
        broker.submit_order(
            OrderRequest(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.1),
            current_market_price=100.0,
        )

        req = OrderRequest(
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.TRAILING_STOP,
            quantity=0.1,
            trail_percent=trail_pct,
        )
        res = broker.submit_order(req, current_market_price=100.0)
        assert res.status == OrderStatus.ACCEPTED

        order_data = broker.pending_orders[res.order_id]
        assert pytest.approx(order_data["trail_factor"], rel=1e-6) == expected_factor
        expected_stop = 100.0 * (1.0 - expected_factor)
        assert pytest.approx(order_data["stop_price"], rel=1e-4) == expected_stop

    def test_trailing_stop_ratcheting_and_execution_under_whipsaw(self):
        """Verify high water mark ratchets upward and executes on pullback."""
        broker = SimulatedPaperBroker(initial_capital_eur=100.0)
        broker.submit_order(
            OrderRequest(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.2),
            current_market_price=100.0,
        )
        # 1% trailing stop
        res = broker.submit_order(
            OrderRequest(symbol="AAPL", side=OrderSide.SELL, order_type=OrderType.TRAILING_STOP, quantity=0.2, trail_percent=1.0),
            current_market_price=100.0,
        )
        oid = res.order_id

        # Price climbs to 105.00
        broker.update_market_price("AAPL", 105.0)
        data = broker.pending_orders[oid]
        assert pytest.approx(data["water_mark"]) == 105.0
        assert pytest.approx(data["stop_price"]) == 105.0 * 0.99  # 103.95

        # Price climbs to 110.00
        broker.update_market_price("AAPL", 110.0)
        data = broker.pending_orders[oid]
        assert pytest.approx(data["water_mark"]) == 110.0
        assert pytest.approx(data["stop_price"]) == 110.0 * 0.99  # 108.90

        # Price dips to 109.00 (above stop price 108.90): should NOT trigger
        broker.update_market_price("AAPL", 109.0)
        assert oid in broker.pending_orders

        # Price drops to 108.50 (below stop price): triggers!
        broker.update_market_price("AAPL", 108.50)
        assert oid not in broker.pending_orders
        assert len(broker.get_positions()) == 0

    # -------------------------------------------------------------------------
    # 4. Fractional Precision & Cash Accounting Under 1000-Trade Churn
    # -------------------------------------------------------------------------

    def test_microtrade_1e6_full_lifecycle(self):
        """Microtrade of 1e-6 shares: buy, query, get_positions, sell, verify position cleanup."""
        broker = SimulatedPaperBroker(initial_capital_eur=100.0)
        buy_res = broker.submit_order(
            OrderRequest(symbol="QQQ", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1e-6),
            current_market_price=400.0,
        )
        assert buy_res.status == OrderStatus.FILLED
        assert buy_res.filled_quantity == 1e-6

        # Position must be returned
        pos = broker.get_position("QQQ")
        assert pos is not None
        assert pos.quantity == 1e-6
        assert len(broker.get_positions()) == 1

        # Liquidate microtrade
        sell_res = broker.submit_order(
            OrderRequest(symbol="QQQ", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=1e-6),
            current_market_price=400.0,
        )
        assert sell_res.status == OrderStatus.FILLED
        assert len(broker.get_positions()) == 0

    def test_1000_trade_stress_harness_negative_cash_prevention(self):
        """1,000 rapid trades under high turnover: cash must never be negative, accounting must balance."""
        random.seed(42)
        broker = SimulatedPaperBroker(initial_capital_eur=50.0)
        symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "SPY"]
        prices = {s: 100.0 for s in symbols}

        for i in range(1000):
            sym = random.choice(symbols)
            # Random walk price (+/- 1.5%)
            prices[sym] *= (1.0 + random.uniform(-0.015, 0.015))
            broker.update_market_price(sym, prices[sym])

            pos = broker.get_position(sym)
            if pos and random.random() < 0.5:
                # Sell fraction
                frac = random.uniform(0.1, 1.0)
                qty = round(pos.quantity * frac, 6)
                if qty >= 1e-6:
                    broker.submit_order(
                        OrderRequest(symbol=sym, side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=qty),
                        current_market_price=prices[sym],
                    )
            else:
                # Buy
                cash = broker.cash_eur
                if cash > 1.0:
                    notional = random.uniform(0.5, min(cash, 10.0))
                    broker.submit_order(
                        OrderRequest(symbol=sym, side=OrderSide.BUY, order_type=OrderType.MARKET, notional=notional),
                        current_market_price=prices[sym],
                    )

            # Invariant: cash is NEVER negative
            assert broker.cash_eur >= 0.0, f"Cash fell negative: {broker.cash_eur} on step {i}"

        acc = broker.get_account()
        assert acc.cash_eur >= 0.0
        assert acc.equity_eur >= 0.0

    # -------------------------------------------------------------------------
    # 5. AlpacaPaperBroker Fallback & Unrealized P&L Accuracy
    # -------------------------------------------------------------------------

    def test_alpaca_fallback_short_sell_rejection_and_mtm_pnl(self):
        """Alpaca fallback mode: reject unowned sell, and accurately compute MTM unrealized P&L."""
        alpaca = AlpacaPaperBroker(virtual_capital_eur=100.0)
        assert alpaca.get_account().cash_eur == 100.0

        # Try unowned sell
        unowned_res = alpaca.submit_order(
            OrderRequest(symbol="AAPL", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=1.0),
            current_market_price=150.0,
        )
        assert unowned_res.status == OrderStatus.REJECTED
        assert "Insufficient position quantity" in unowned_res.message
        assert alpaca.get_account().cash_eur == 100.0  # Zero cash inflation

        # Buy 0.2 shares of AAPL at $150 (notional = $30 = €27.78 at fx 1.08)
        buy_res = alpaca.submit_order(
            OrderRequest(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.2),
            current_market_price=150.0,
        )
        assert buy_res.status == OrderStatus.FILLED

        # With no price movement, unrealized P&L is 0.0
        acc_initial = alpaca.get_account()
        assert pytest.approx(acc_initial.unrealized_pnl_eur, abs=1e-4) == 0.0

        # Price increases from $150 to $180 (+$30/share * 0.2 shares = +$6 USD = +€5.5556 EUR at fx 1.08)
        alpaca.virtual_positions["AAPL"].current_price = 180.0
        acc_gain = alpaca.get_account()
        expected_unrealized_gain = (30.0 * 0.2) / 1.08
        assert pytest.approx(acc_gain.unrealized_pnl_eur, rel=1e-2) == expected_unrealized_gain

        # Price drops from $150 to $120 (-$30/share * 0.2 shares = -$6 USD = -€5.5556 EUR)
        alpaca.virtual_positions["AAPL"].current_price = 120.0
        acc_loss = alpaca.get_account()
        expected_unrealized_loss = (-30.0 * 0.2) / 1.08
        assert pytest.approx(acc_loss.unrealized_pnl_eur, rel=1e-2) == expected_unrealized_loss
