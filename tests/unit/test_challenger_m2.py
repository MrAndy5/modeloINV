"""Empirical Adversarial Stress Test Suite for Milestone 2 (Challenger M2).

Stress tests:
1. 500+ rapid random trades & rebalancing: accounting invariant residual verification.
2. Fractional share precision down to 0.000001 (1e-6) shares: rejection and truncation test.
3. Limit order price guarantee: verification that limit orders never fill worse than limit price.
4. Trailing stop ratcheting under tight thresholds (<= 1.0% trail) and whipsaws.
5. Insufficient funds boundary: negative cash balance prevention.
6. Transaction fee accounting: symmetry between buy/sell fees and realized P&L.
7. AlpacaPaperBroker virtual partition accounting: unrealized P&L vs market value.
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
from modeloinv.execution.paper import SimulatedPaperBroker
from modeloinv.execution.alpaca import AlpacaPaperBroker
from modeloinv.execution.fees import EuropeanRetailFeeModel, ZeroCommissionFeeModel
from modeloinv.execution.slippage import FixedBpsSlippageModel


class TestChallengerM2StressSuite:
    """Empirical stress tests designed to challenge M2 execution assumptions."""

    def test_500_rapid_trades_accounting_invariant(self):
        """Execute 500 random trades on €100 virtual ledger and verify Equity == Cash + sum(Positions)."""
        random.seed(1337)
        broker = SimulatedPaperBroker(initial_capital_eur=100.0)
        symbols = ["AAPL", "MSFT", "SPY", "NVDA", "AMZN"]
        prices = {"AAPL": 150.0, "MSFT": 300.0, "SPY": 500.0, "NVDA": 120.0, "AMZN": 180.0}

        max_residual = 0.0

        for _ in range(500):
            sym = random.choice(symbols)
            # Price random walk (+/- 2%)
            prices[sym] *= (1.0 + random.uniform(-0.02, 0.02))
            broker.update_market_price(sym, prices[sym])

            pos = broker.get_position(sym)
            if pos and random.random() < 0.45:
                # Sell partial or entire position
                sell_fraction = random.uniform(0.2, 1.0)
                sell_qty = round(pos.quantity * sell_fraction, 6)
                if sell_qty > 1e-6:
                    req = OrderRequest(symbol=sym, side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=sell_qty)
                    broker.submit_order(req, current_market_price=prices[sym])
            else:
                # Buy fractional shares with small notional
                available_cash = max(0.0, broker.cash_eur)
                if available_cash >= 1.5:
                    spend = random.uniform(1.0, min(8.0, available_cash * 0.4))
                    req = OrderRequest(symbol=sym, side=OrderSide.BUY, order_type=OrderType.MARKET, notional=spend)
                    broker.submit_order(req, current_market_price=prices[sym])

            # Invariant check: raw internal equity vs Cash + sum(MarketValue)
            fx = 1.08
            unrounded_pos_value = sum(p.quantity * p.current_price / fx for p in broker.positions.values())
            unrounded_equity = broker.cash_eur + unrounded_pos_value

            pos_val_method = sum(p.market_value_eur for p in broker.positions.values())
            computed_equity = broker.cash_eur + pos_val_method

            residual = abs(unrounded_equity - computed_equity)
            if residual > max_residual:
                max_residual = residual

        assert len(broker.trade_history) >= 400
        assert max_residual < 1e-10

    def test_fractional_precision_microtrade_1e6_rejection(self):
        """Test micro-trades down to 0.000001 shares (DISPATCH.md requirement).

        A compliant broker supporting 6-decimal micro-trades must fill 0.000001 shares.
        """
        broker = SimulatedPaperBroker(initial_capital_eur=100.0)
        req = OrderRequest(
            symbol="SPY",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.000001,  # Exactly 1e-6 shares
        )
        res = broker.submit_order(req, current_market_price=500.0)

        assert res.status == OrderStatus.FILLED
        assert res.filled_quantity == 0.000001
        pos = broker.get_position("SPY")
        assert pos is not None
        assert pos.quantity == 0.000001

    def test_limit_order_adverse_slippage_violates_limit_price(self):
        """Test whether limit orders suffer adverse slippage worse than limit price."""
        broker = SimulatedPaperBroker(initial_capital_eur=100.0, slippage_model=FixedBpsSlippageModel(bps=10.0))

        # Limit Buy at $100.00 when market is at $100.00
        limit_buy = OrderRequest(
            symbol="SPY",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            limit_price=100.00,
        )
        res_buy = broker.submit_order(limit_buy, current_market_price=100.00)

        # Expected limit order invariant: filled_avg_price <= limit_price
        assert res_buy.status == OrderStatus.FILLED
        assert res_buy.filled_avg_price <= limit_buy.limit_price

    def test_trailing_stop_tight_percentage_inversion_bug(self):
        """Test trailing stop percentage interpretation for tight intraday stops (<= 1.0%)."""
        broker = SimulatedPaperBroker(initial_capital_eur=100.0)
        # Buy 1 share at $100
        broker.submit_order(
            OrderRequest(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1.0),
            current_market_price=100.0,
        )

        # Submit 1.0% trailing stop
        req_1pct = OrderRequest(
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.TRAILING_STOP,
            quantity=1.0,
            trail_percent=1.0,  # 1% trailing stop
        )
        res_1pct = broker.submit_order(req_1pct, current_market_price=100.0)
        assert res_1pct.status == OrderStatus.ACCEPTED

        pending_order = list(broker.pending_orders.values())[0]
        # A 1% trailing stop on $100 should have stop_price = $99.00
        assert pending_order["trail_factor"] == 0.01
        assert pytest.approx(pending_order["stop_price"], rel=1e-4) == 99.00

    def test_insufficient_funds_negative_cash_leakage(self):
        """Test that order costing more than available cash is rejected and cash remains non-negative."""
        broker = SimulatedPaperBroker(initial_capital_eur=10.0, slippage_model=FixedBpsSlippageModel(bps=0.0))
        broker.cash_eur = 10.0000

        # At FX 1.08, 0.108001 shares at $100 = $10.8001 / 1.08 = 10.00009259 EUR
        # This exceeds €10.0000 by 0.00009259 EUR (< 1e-4 tolerance)
        req = OrderRequest(symbol="SPY", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.108001)
        res = broker.submit_order(req, current_market_price=100.0)

        # Order must be rejected and cash must not leak negative
        assert res.status == OrderStatus.REJECTED
        assert broker.cash_eur >= 0.0

    def test_buy_fee_omission_from_realized_pnl(self):
        """Test that buy transaction fees are included in cost basis and realized P&L reporting."""
        fee_model = EuropeanRetailFeeModel(
            flat_fee_eur=2.0,
            variable_pct=0.0,
            min_fee_eur=2.0,
            max_fee_eur=2.0,
            fx_markup_pct=0.0,
            is_foreign_asset=False,
        )
        broker = SimulatedPaperBroker(initial_capital_eur=100.0, fee_model=fee_model)

        # Buy 1 share at $10
        broker.submit_order(
            OrderRequest(symbol="XYZ", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1.0),
            current_market_price=10.0,
        )
        # Sell 1 share at same price $10
        broker.submit_order(
            OrderRequest(symbol="XYZ", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=1.0),
            current_market_price=10.0,
        )

        acc = broker.get_account()
        cash_loss = 100.0 - acc.cash_eur
        reported_loss = -acc.realized_pnl_eur

        assert pytest.approx(cash_loss, rel=1e-2) == 4.01
        assert pytest.approx(reported_loss, rel=1e-2) == 4.01

    def test_alpaca_virtual_partition_unrealized_pnl_inflation(self):
        """Test AlpacaPaperBroker get_account() unrealized P&L reporting in virtual partition."""
        broker = AlpacaPaperBroker(virtual_capital_eur=100.0)
        broker.submit_order(
            OrderRequest(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, notional=50.0),
            current_market_price=100.0,
        )

        acc = broker.get_account()
        # Since stock price has not moved, unrealized P&L should be 0.0 EUR
        assert pytest.approx(acc.unrealized_pnl_eur, abs=1e-4) == 0.0
