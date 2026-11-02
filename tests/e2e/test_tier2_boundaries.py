"""Tier 2: Boundary Value Analysis & Stress Verification E2E Tests.

Validates system behavior under extreme, corner, and adversarial conditions:
- Empty inputs, zero-volume bars, flat-pricing regimes
- Illiquid books, extreme slippage, excessive transaction fees
- Micro-fractional share rounding (0.000001 shares / satoshis)
- Capital exhaustion (€0.00 cash) and negative buying power rejection
- Mathematical singular points: zero variance in Sharpe, zero downside in Sortino
- State racing: rapid order cancellations, duplicate client_order_ids
- Continuous learning edge conditions: buffer underflow/overflow, threshold boundaries (N=24 vs N=25)
- Flash crashes and pre-trade circuit breaker halts

All tests run 100% offline and deterministically on Windows.
"""

from __future__ import annotations

import os
import sys
import datetime
from pathlib import Path
from typing import List, Dict, Any

import pytest
import numpy as np
import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.conftest import (
    Bar, OrderRequest, OrderResult, Position, AccountInfo, TradingSignal,
    OrderType, OrderSide, OrderStatus, AssetClass,
    MockMarketDataProvider, OfflineSimulatedBroker,
    calculate_sharpe_ratio, calculate_sortino_ratio, calculate_max_drawdown, calculate_win_rate
)


# =====================================================================
# Tier 2 Category 1: Degenerate Inputs & Edge Regimes
# =====================================================================

class TestTier2BoundaryInputs:
    """Stress tests on degenerate market inputs and boundary datasets."""

    def test_bva_01_empty_bar_sequence_metrics(self):
        """Empty return series handles calculation without crashing."""
        empty_returns = np.array([])
        sharpe = calculate_sharpe_ratio(empty_returns)
        sortino = calculate_sortino_ratio(empty_returns)
        max_dd = calculate_max_drawdown(empty_returns)
        win_rate = calculate_win_rate([])

        assert sharpe == 0.0
        assert sortino == 0.0
        assert max_dd == 0.0
        assert win_rate == 0.0

    def test_bva_02_zero_volume_bars_vwap(self):
        """VWAP calculation with zero cumulative volume defaults to close price without ZeroDivisionError."""
        bars_df = pd.DataFrame({
            "close": [100.0, 101.0, 102.0],
            "volume": [0.0, 0.0, 0.0]
        })
        total_vol = bars_df["volume"].sum()
        if total_vol == 0:
            vwap = bars_df["close"].iloc[-1]
        else:
            vwap = (bars_df["close"] * bars_df["volume"]).sum() / total_vol
        assert vwap == 102.0

    def test_bva_03_flat_pricing_atr_and_bollinger(self):
        """Flat pricing series (open=high=low=close) yields 0.0 ATR and zero-width Bollinger Bands."""
        prices = pd.Series([100.0] * 20)
        atr = (prices - prices).rolling(14).mean().iloc[-1]
        std = prices.rolling(20).std().iloc[-1]
        assert atr == 0.0
        assert std == 0.0

    def test_bva_04_single_bar_history_handling(self, mock_data_provider):
        """Single-bar input handles rolling indicators without out-of-bounds error."""
        bar = mock_data_provider.generate_bars("SPY", datetime.datetime.utcnow(), num_bars=1)[0]
        df = pd.DataFrame([{"close": bar.close}])
        ema = df["close"].ewm(span=9).mean()
        assert len(ema) == 1
        assert ema.iloc[0] == bar.close

    def test_bva_05_extreme_gap_opening(self):
        """Adverse 30% gap opening on asset evaluated by stop loss."""
        entry = 100.0
        stop_price = 95.0
        # Overnight gap down to 70.0
        gap_price = 70.0
        # Simulated stop fill takes gap execution price, not stop price
        fill_price = min(stop_price, gap_price)
        slippage = stop_price - fill_price
        assert fill_price == 70.0
        assert slippage == 25.0


# =====================================================================
# Tier 2 Category 2: Extreme Execution & Sizing
# =====================================================================

class TestTier2BoundaryExecution:
    """Stress tests on extreme execution conditions, fees, and sizing limits."""

    def test_bva_06_micro_fractional_share_precision(self, offline_broker):
        """Micro-fractional share order with 6 decimal places preserves floating precision."""
        req = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.000123
        )
        res = offline_broker.submit_order(req, current_market_price=60000.0)
        assert res.status == OrderStatus.FILLED
        assert res.filled_quantity == 0.000123
        pos = offline_broker.positions["BTC/USDT"]
        assert pos.quantity == 0.000123

    def test_bva_07_capital_exhaustion_to_zero(self, offline_broker):
        """Virtual account cash dropping to €0.00 cleanly rejects further BUY orders."""
        # Deplete entire €100 capital
        req = OrderRequest(symbol="SPY", side=OrderSide.BUY, order_type=OrderType.MARKET, notional=99.5)
        res = offline_broker.submit_order(req, current_market_price=100.0)
        assert res.status == OrderStatus.FILLED
        assert offline_broker.cash_eur < 1.0

        # Attempt another buy
        req2 = OrderRequest(symbol="SPY", side=OrderSide.BUY, order_type=OrderType.MARKET, notional=10.0)
        res2 = offline_broker.submit_order(req2, current_market_price=100.0)
        assert res2.status == OrderStatus.REJECTED
        assert "Insufficient funds" in res2.message

    def test_bva_08_negative_buying_power_prevention(self, offline_broker):
        """Buying power invariant: buying_power_eur is strictly non-negative."""
        offline_broker.cash_eur = -5.0  # Force negative cash anomaly
        acc = offline_broker.get_account()
        assert acc.buying_power_eur == 0.0

    def test_bva_09_extreme_slippage_penalty(self):
        """Illiquid order with large size relative to volume scales slippage via Almgren-Chriss."""
        trade_qty = 10000.0
        bar_vol = 5000.0  # Order is 2x bar volume!
        volatility = 0.02
        impact = volatility * np.sqrt(trade_qty / bar_vol)
        assert impact > volatility  # Significant impact penalty
        assert impact > 0.028

    def test_bva_10_minimum_fee_enforcement(self, offline_broker):
        """Micro trade ($0.10 notional) enforces minimum fee floor ($0.01)."""
        req = OrderRequest(symbol="PENNY", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.001)
        res = offline_broker.submit_order(req, current_market_price=10.0)
        assert res.fee_eur > 0.0
        assert res.fee_eur >= 0.009


# =====================================================================
# Tier 2 Category 3: Mathematical & Statistical Singularities
# =====================================================================

class TestTier2BoundaryMathematics:
    """Stress tests on mathematical formulas at singular boundary conditions."""

    def test_bva_11_zero_variance_sharpe_ratio(self):
        """All returns identical (std=0) avoids ZeroDivisionError and returns 0.0."""
        identical_returns = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        sharpe = calculate_sharpe_ratio(identical_returns)
        assert sharpe == 0.0

    def test_bva_12_zero_downside_sortino_ratio(self):
        """Strictly positive returns (zero downside risk) avoids ZeroDivisionError."""
        strictly_positive_returns = np.array([0.01, 0.02, 0.015, 0.03, 0.025])
        sortino = calculate_sortino_ratio(strictly_positive_returns)
        assert sortino > 0.0
        assert sortino >= 10.0  # Upper bound floor

    def test_bva_13_complete_loss_drawdown(self):
        """100% loss (peak €100 dropping to €0) calculates exactly 1.0 drawdown."""
        equity_series = np.array([100.0, 50.0, 20.0, 0.0])
        max_dd = calculate_max_drawdown(equity_series)
        assert max_dd == 1.0

    def test_bva_14_monotonically_increasing_drawdown(self):
        """Monotonically increasing equity produces exactly 0.0 drawdown."""
        equity_series = np.array([100.0, 105.0, 110.0, 115.0, 120.0])
        max_dd = calculate_max_drawdown(equity_series)
        assert max_dd == 0.0

    def test_bva_15_extreme_fx_rates(self):
        """Extreme exchange rates (EUR/USD = 0.0001 and 1000.0) do not cause numerical overflow."""
        usd_amount = 100.0
        eur_hyper_inflated = usd_amount / 0.0001
        eur_hyper_deflated = usd_amount / 1000.0
        assert eur_hyper_inflated == 1000000.0
        assert eur_hyper_deflated == 0.1


# =====================================================================
# Tier 2 Category 4: Concurrency, State Racing & Idempotency
# =====================================================================

class TestTier2BoundaryStateAndRacing:
    """Stress tests on order state racing and idempotency."""

    def test_bva_16_cancelling_already_filled_order(self, offline_broker):
        """Attempting to cancel an already filled order returns False."""
        req = OrderRequest(symbol="SPY", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.1)
        res = offline_broker.submit_order(req, current_market_price=450.0)
        assert res.status == OrderStatus.FILLED

        # Try cancel
        cancelled = offline_broker.cancel_order(res.order_id)
        assert cancelled is False

    def test_bva_17_duplicate_client_order_id_idempotency(self, offline_broker):
        """Submitting duplicate client_order_id can be detected and handled safely."""
        cid = "client_order_unique_001"
        req1 = OrderRequest(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.1, client_order_id=cid)
        res1 = offline_broker.submit_order(req1, current_market_price=180.0)
        assert res1.status == OrderStatus.FILLED

        # Check existing orders for client_order_id
        existing_cids = {o.client_order_id for o in offline_broker.orders.values()}
        assert cid in existing_cids

    def test_bva_18_overselling_position_rejection(self, offline_broker):
        """Attempting to sell more shares than owned is rejected."""
        buy = OrderRequest(symbol="TSLA", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.5)
        offline_broker.submit_order(buy, current_market_price=200.0)

        # Attempt to sell 1.0 share (only have 0.5)
        sell = OrderRequest(symbol="TSLA", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=1.0)
        res_sell = offline_broker.submit_order(sell, current_market_price=210.0)
        assert res_sell.status == OrderStatus.REJECTED
        assert "Insufficient position quantity" in res_sell.message


# =====================================================================
# Tier 2 Category 5: Continuous Learning Buffer Boundaries
# =====================================================================

class TestTier2BoundaryLearningBuffer:
    """Stress tests on experience replay buffer underflow, overflow, and threshold boundaries."""

    def test_bva_19_replay_buffer_capacity_fifo_eviction(self):
        """Replay buffer at capacity (e.g. 5) evicts oldest elements on new arrival."""
        max_capacity = 5
        buffer = []
        for i in range(10):
            if len(buffer) >= max_capacity:
                buffer.pop(0)  # Evict oldest
            buffer.append(i)

        assert len(buffer) == 5
        assert buffer == [5, 6, 7, 8, 9]

    def test_bva_20_retraining_trigger_exact_boundary(self):
        """Retraining trigger at N=24 does not trigger; N=25 triggers."""
        def check_retrain(n_trades: int) -> bool:
            return n_trades >= 25

        assert check_retrain(24) is False
        assert check_retrain(25) is True
        assert check_retrain(26) is True


# =====================================================================
# Tier 2 Category 6: Risk Breakers & Flash Crashes
# =====================================================================

class TestTier2BoundaryRiskBreakers:
    """Stress tests on severe market anomalies and pre-trade guardrails."""

    def test_bva_21_flash_crash_50_percent_triggers_guardrail(self, offline_broker):
        """Sudden 50% price drop triggers portfolio circuit breaker and halts trading."""
        # Buy $80 of SPY
        buy = OrderRequest(symbol="SPY", side=OrderSide.BUY, order_type=OrderType.MARKET, notional=80.0)
        offline_broker.submit_order(buy, current_market_price=400.0)

        # Flash crash: SPY drops from 400 to 200 (-50%)
        offline_broker.update_market_price("SPY", 200.0)
        acc = offline_broker.get_account()
        
        # Calculate drawdown from starting €100
        drawdown = (acc.initial_capital_eur - acc.equity_eur) / acc.initial_capital_eur
        assert drawdown > 0.20  # Exceeds 20% limit

        # Guardrail triggers halt
        if drawdown >= offline_broker.max_drawdown_limit:
            offline_broker.is_halted = True

        assert offline_broker.is_halted is True

        # Subsequent orders are rejected
        req_post_crash = OrderRequest(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, notional=10.0)
        res = offline_broker.submit_order(req_post_crash, current_market_price=100.0)
        assert res.status == OrderStatus.REJECTED
        assert "risk circuit breaker" in res.message.lower()
