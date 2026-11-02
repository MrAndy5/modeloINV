"""Adversarial Stress Test Suite for Milestone 3 (Sizing, Micro-Capital & Hybrid Strategy Engine).

Written by Challenger M3.2 (`challenger_m3_2`) to empirically stress-test:
1. Micro-Capital Fractional Adapter on €100 virtual capital:
   - Sub-€1.00 allocation rejection (< €1.00 minimum notional threshold).
   - Extreme asset prices ($50,000 / $100,000 BTC vs $0.00005 penny tokens / $650,000 BRK.A).
   - Multi-currency FX conversions (EUR base vs USD quote vs native EUR assets).
   - Boundary equities (blown €0 / negative accounts, zero/negative prices, zero weight).
2. RL State Bounds & Numerical Stability (`RLStateBuilder`):
   - 9-dimensional state vector invariance across extreme drawdowns (99% / 200%).
   - Extreme volatility spikes (ATR / price explosion, zero price division protection).
   - Robustness against NaN, +Inf, -Inf inputs with `np.nan_to_num`.
3. Cost-Penalized Reward Bounds (`CostPenalizedReward`):
   - Strict [-10.0, 10.0] clipping preventing runaway gradients under massive P&L or catastrophic losses.
   - Quadratic drawdown penalty threshold and escalation under severe drawdowns.
   - Exact fee and slippage subtraction.
4. Multi-Timeframe Bar Synchronizer Lookahead Checks (`MultiTimeframeBarSynchronizer`):
   - Zero lookahead bias during incomplete 5m, 15m, 1h bar accumulation.
   - Strict emission only after timeframe interval closure.
   - Exact OHLCV aggregation and volume conservation across 1m -> 5m -> 15m -> 1h.
   - Multi-symbol independence and bar gap resilience.
5. Dynamic Universe Screener & Multi-Armed Bandit Allocator:
   - Hard 50% single-asset concentration cap under extreme Sharpe disparity.
   - Liquidity and volatility filters ($5M volume, 0.5% ATR ratio).
6. Hybrid Strategy Engine Signal Geometry:
   - Long ($SL < Price < TP$) and Short ($TP < Price < SL$) geometry verification.
   - Fractional sizing integration with €100 capital in signal metadata.
"""

from __future__ import annotations

import datetime
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest
import torch

from modeloinv.core.models import AssetClass, Bar, TradingSignal
from modeloinv.strategy.engine import (
    HybridStrategyEngine,
    MultiTimeframeBarSynchronizer,
)
from modeloinv.strategy.indicators import (
    TechnicalIndicatorSuite,
    calculate_atr,
    calculate_vwap,
)
from modeloinv.strategy.regime import MarketRegimeClassifier, RegimeState, compute_composite_quant_score
from modeloinv.strategy.rl_agent import (
    ACTION_WEIGHT_MAP,
    CostPenalizedReward,
    PolicyAgent,
    PolicyNetwork,
    RLStateBuilder,
)
from modeloinv.strategy.universe import (
    CandidateAsset,
    DynamicUniverseScreener,
    MicroCapitalFractionalAdapter,
    MultiArmedBanditAllocator,
)


# ============================================================================
# 1. Micro-Capital Fractional Adapter (€100 Virtual Capital) Stress Tests
# ============================================================================

class TestMicroCapitalFractionalAdapterAdversarial:
    """Adversarial testing of MicroCapitalFractionalAdapter on €100 starting capital."""

    @pytest.fixture
    def adapter(self) -> MicroCapitalFractionalAdapter:
        return MicroCapitalFractionalAdapter(min_order_notional_eur=1.00, default_eur_usd_rate=1.08)

    def test_sub_one_euro_notional_rejection(self, adapter: MicroCapitalFractionalAdapter):
        """Enforces that target allocations below €1.00 on €100 capital are strictly rejected (return None)."""
        # €100 equity, 0.5% weight = €0.50 -> must reject
        qty_50c = adapter.calculate_order_quantity(
            symbol="AAPL", target_weight=0.005, account_equity_eur=100.0, price=150.0
        )
        assert qty_50c is None, "Order of €0.50 notional must be rejected by €1.00 minimum threshold"

        # €100 equity, 0.99% weight = €0.99 -> must reject
        qty_99c = adapter.calculate_order_quantity(
            symbol="AAPL", target_weight=0.0099, account_equity_eur=100.0, price=150.0
        )
        assert qty_99c is None, "Order of €0.99 notional must be rejected by €1.00 minimum threshold"

        # €100 equity, 1.00% weight = €1.00 -> must pass
        qty_1eur = adapter.calculate_order_quantity(
            symbol="AAPL", target_weight=0.01, account_equity_eur=100.0, price=150.0
        )
        assert qty_1eur is not None and qty_1eur > 0, "Order of €1.00 notional must be accepted"

    def test_severely_degraded_account_equity_drawdown(self, adapter: MicroCapitalFractionalAdapter):
        """Verifies order threshold adapts correctly when account is drawn down from €100 to €10 or €1."""
        # Account suffered 90% drawdown, equity is now €10.00
        # 5% weight on €10.00 is €0.50 -> reject
        qty_drawn_down_small = adapter.calculate_order_quantity(
            symbol="MSFT", target_weight=0.05, account_equity_eur=10.0, price=300.0
        )
        assert qty_drawn_down_small is None

        # 10% weight on €10.00 is €1.00 -> pass
        qty_drawn_down_valid = adapter.calculate_order_quantity(
            symbol="MSFT", target_weight=0.10, account_equity_eur=10.0, price=300.0
        )
        assert qty_drawn_down_valid is not None and qty_drawn_down_valid > 0

        # Account at €1.00 (99% drawdown): only 100% full allocation meets €1.00 threshold
        qty_last_euro = adapter.calculate_order_quantity(
            symbol="MSFT", target_weight=1.00, account_equity_eur=1.00, price=300.0
        )
        assert qty_last_euro is not None and qty_last_euro > 0

    def test_blown_or_insolvent_account(self, adapter: MicroCapitalFractionalAdapter):
        """Zero or negative account balance must always return None."""
        assert adapter.calculate_order_quantity("AAPL", 0.5, 0.0, 150.0) is None
        assert adapter.calculate_order_quantity("AAPL", 0.5, -25.0, 150.0) is None

    def test_extreme_high_asset_prices(self, adapter: MicroCapitalFractionalAdapter):
        """Stress-tests micro-allocations on very expensive assets: BTC ($50k, $100k) and Berkshire Hathaway ($650k)."""
        # BTC at $50,000: €10 allocation on €100 account (10% weight)
        # €10 * 1.08 = $10.80 -> 10.80 / 50000 = 0.000216 BTC (crypto precision 6)
        qty_btc = adapter.calculate_order_quantity(
            symbol="BTC/USD", target_weight=0.10, account_equity_eur=100.0, price=50000.0, asset_class="CRYPTO"
        )
        assert qty_btc == pytest.approx(0.000216, abs=1e-6)

        # BTC at $100,000: €5 allocation (5% weight)
        # €5 * 1.08 = $5.40 -> 5.40 / 100000 = 0.000054 BTC
        qty_btc_100k = adapter.calculate_order_quantity(
            symbol="BTC/USD", target_weight=0.05, account_equity_eur=100.0, price=100000.0, asset_class="CRYPTO"
        )
        assert qty_btc_100k == pytest.approx(0.000054, abs=1e-6)

        # BRK.A at $650,000: €1.00 allocation (1% weight)
        # €1.00 * 1.08 = $1.08 -> 1.08 / 650000 = 0.0000016615 -> rounds to 0.000002 (equity precision 6)
        qty_brk = adapter.calculate_order_quantity(
            symbol="BRK.A", target_weight=0.01, account_equity_eur=100.0, price=650000.0, asset_class="EQUITY"
        )
        assert qty_brk == pytest.approx(0.000002, abs=1e-6)

        # Asset priced at $10,000,000 per share with €1.00 allocation:
        # $1.08 / 10,000,000 = 0.000000108 -> rounds to 0.000000 at 6 decimals
        # Adapter MUST return None rather than a 0.0 ghost order
        qty_super_expensive = adapter.calculate_order_quantity(
            symbol="TITAN", target_weight=0.01, account_equity_eur=100.0, price=10000000.0, asset_class="EQUITY"
        )
        assert qty_super_expensive is None, "Sub-precision quantity rounded to 0.0 must return None"

    def test_extreme_low_penny_asset_prices(self, adapter: MicroCapitalFractionalAdapter):
        """Stress-tests micro-allocations on penny stocks and micro-cap crypto ($0.00005)."""
        # Memecoin at $0.00005: €10 allocation (10% weight)
        # €10 * 1.08 = $10.80 -> 10.80 / 0.00005 = 216,000 tokens
        qty_penny_crypto = adapter.calculate_order_quantity(
            symbol="DOGE2", target_weight=0.10, account_equity_eur=100.0, price=0.00005, asset_class="CRYPTO"
        )
        assert qty_penny_crypto == pytest.approx(216000.0, abs=1e-3)

        # Penny equity at $0.15: €50 allocation (50% weight)
        # €50 * 1.08 = $54.00 -> 54.00 / 0.15 = 360.0 shares
        qty_penny_stock = adapter.calculate_order_quantity(
            symbol="PENN", target_weight=0.50, account_equity_eur=100.0, price=0.15, asset_class="EQUITY"
        )
        assert qty_penny_stock == pytest.approx(360.0, abs=1e-3)

    def test_multi_currency_fx_conversions(self, adapter: MicroCapitalFractionalAdapter):
        """Verifies currency conversion: USD assets use FX rate, EUR assets do not double-convert."""
        # USD Asset: €50 allocation with EUR/USD = 1.08 -> $54.00 / $100.0 = 0.54 shares
        qty_usd = adapter.calculate_order_quantity(
            symbol="SPY", target_weight=0.50, account_equity_eur=100.0, price=100.0, currency="USD"
        )
        assert qty_usd == pytest.approx(0.54, abs=1e-5)

        # Native EUR Asset: €50 allocation with Price = €100.0 -> exactly 0.50 shares
        qty_eur = adapter.calculate_order_quantity(
            symbol="SAP.DE", target_weight=0.50, account_equity_eur=100.0, price=100.0, currency="EUR"
        )
        assert qty_eur == pytest.approx(0.50, abs=1e-5)

        # EUR/USD Rate fluctuation override (e.g. rate drops to 0.95 or jumps to 1.25)
        qty_weak_eur = adapter.calculate_order_quantity(
            symbol="SPY", target_weight=0.50, account_equity_eur=100.0, price=100.0, currency="USD", eur_usd_rate=0.95
        )
        assert qty_weak_eur == pytest.approx(0.475, abs=1e-5)

        qty_strong_eur = adapter.calculate_order_quantity(
            symbol="SPY", target_weight=0.50, account_equity_eur=100.0, price=100.0, currency="USD", eur_usd_rate=1.25
        )
        assert qty_strong_eur == pytest.approx(0.625, abs=1e-5)

    def test_zero_and_negative_target_weights(self, adapter: MicroCapitalFractionalAdapter):
        """Verifies zero weight returns 0.0 (exit) and negative weight produces positive quantity for shorting."""
        # Zero weight -> explicit 0.0
        qty_flat = adapter.calculate_order_quantity(
            symbol="AAPL", target_weight=0.0, account_equity_eur=100.0, price=150.0
        )
        assert qty_flat == 0.0

        # Micro-dust weight below threshold 1e-4 -> 0.0
        qty_dust = adapter.calculate_order_quantity(
            symbol="AAPL", target_weight=1e-5, account_equity_eur=100.0, price=150.0
        )
        assert qty_dust == 0.0

        # Negative weight -0.25 (short): must return positive quantity magnitude
        qty_short = adapter.calculate_order_quantity(
            symbol="AAPL", target_weight=-0.25, account_equity_eur=100.0, price=150.0
        )
        qty_long = adapter.calculate_order_quantity(
            symbol="AAPL", target_weight=0.25, account_equity_eur=100.0, price=150.0
        )
        assert qty_short is not None and qty_short > 0
        assert qty_short == qty_long

    def test_invalid_price_and_equity_boundary(self, adapter: MicroCapitalFractionalAdapter):
        """Ensures non-positive prices return None."""
        assert adapter.calculate_order_quantity("AAPL", 0.5, 100.0, 0.0) is None
        assert adapter.calculate_order_quantity("AAPL", 0.5, 100.0, -10.0) is None


# ============================================================================
# 2. RL State Bounds & Numerical Stability Stress Tests
# ============================================================================

class TestRLStateBuilderAdversarial:
    """Adversarial stress-testing of RLStateBuilder under market stress and invalid data."""

    def test_state_shape_and_dtype(self):
        """State must strictly produce 9-dimensional float32 vector."""
        state = RLStateBuilder.build_state(
            quant_score=0.5, ml_score=0.2, ml_confidence=0.8,
            adx=25.0, atr=1.5, price=100.0, current_weight=0.25,
            unrealized_pnl_pct=2.5, drawdown_pct=1.0,
        )
        assert isinstance(state, np.ndarray)
        assert state.shape == (9,)
        assert state.dtype == np.float32

    def test_catastrophic_drawdown_bounds(self):
        """Drawdown from 0% to 200% must map strictly to [0.0, 1.0]."""
        # Standard percentage: 95% drawdown -> normalized to 0.95
        state_95 = RLStateBuilder.build_state(0, 0, 0, 0, 1, 100, 0, 0, drawdown_pct=95.0)
        assert 0.0 <= state_95[7] <= 1.0
        assert state_95[7] == pytest.approx(0.95, abs=1e-4)

        # Ratio representation: 0.95 -> 0.95
        state_ratio = RLStateBuilder.build_state(0, 0, 0, 0, 1, 100, 0, 0, drawdown_pct=0.95)
        assert state_ratio[7] == pytest.approx(0.95, abs=1e-4)

        # Leveraged catastrophic loss: 250% drawdown -> clipped to 1.0
        state_250 = RLStateBuilder.build_state(0, 0, 0, 0, 1, 100, 0, 0, drawdown_pct=250.0)
        assert state_250[7] == 1.0

        # Impossible negative drawdown: -15% -> clipped to 0.0
        state_neg = RLStateBuilder.build_state(0, 0, 0, 0, 1, 100, 0, 0, drawdown_pct=-15.0)
        assert state_neg[7] == 0.0

    def test_extreme_pnl_percentage_bounds(self):
        """Unrealized P&L from -500% to +1000% must map strictly to [-1.0, 1.0]."""
        # Massive gain +800%
        state_gain = RLStateBuilder.build_state(0, 0, 0, 0, 1, 100, 0, unrealized_pnl_pct=800.0, drawdown_pct=0)
        assert state_gain[6] == 1.0

        # Total loss -100%
        state_loss = RLStateBuilder.build_state(0, 0, 0, 0, 1, 100, 0, unrealized_pnl_pct=-100.0, drawdown_pct=0)
        assert state_loss[6] == -1.0

        # Margin call -400%
        state_margin = RLStateBuilder.build_state(0, 0, 0, 0, 1, 100, 0, unrealized_pnl_pct=-400.0, drawdown_pct=0)
        assert state_margin[6] == -1.0

    def test_volatility_ratio_explosion_protection(self):
        """Flash crash or extreme ATR (ATR >> Price) must clamp volatility ratio to 0.1 without ZeroDivision."""
        # ATR = 500, Price = 100 -> ratio 5.0 -> clipped to 0.1
        state_high_vol = RLStateBuilder.build_state(0, 0, 0, 0, atr=500.0, price=100.0, current_weight=0, unrealized_pnl_pct=0, drawdown_pct=0)
        assert state_high_vol[4] == 0.1

        # Price = 0.0 (safe_price fallback ensures no ZeroDivision)
        state_zero_p = RLStateBuilder.build_state(0, 0, 0, 0, atr=5.0, price=0.0, current_weight=0, unrealized_pnl_pct=0, drawdown_pct=0)
        assert np.isfinite(state_zero_p[4])
        assert state_zero_p[4] == 0.1

        # Negative price
        state_neg_p = RLStateBuilder.build_state(0, 0, 0, 0, atr=5.0, price=-100.0, current_weight=0, unrealized_pnl_pct=0, drawdown_pct=0)
        assert np.isfinite(state_neg_p[4])

    def test_nan_and_inf_sanitization(self):
        """Ensures that inputs with NaN, +Inf, -Inf are sanitized to 0.0 with zero unhandled exceptions."""
        state_nan = RLStateBuilder.build_state(
            quant_score=float("nan"),
            ml_score=float("inf"),
            ml_confidence=float("-inf"),
            adx=float("nan"),
            atr=float("nan"),
            price=float("nan"),
            current_weight=float("nan"),
            unrealized_pnl_pct=float("nan"),
            drawdown_pct=float("nan"),
        )
        assert not np.isnan(state_nan).any(), "State vector must never contain NaN"
        assert not np.isinf(state_nan).any(), "State vector must never contain Inf"
        assert np.all(state_nan >= -1.0) and np.all(state_nan <= 1.0)


# ============================================================================
# 3. Cost-Penalized Reward Bounds Stress Tests
# ============================================================================

class TestCostPenalizedRewardAdversarial:
    """Adversarial stress-testing of CostPenalizedReward clipping and penalties."""

    @pytest.fixture
    def reward_fn(self) -> CostPenalizedReward:
        return CostPenalizedReward(lambda_fee=1.0, lambda_slippage=1.0, lambda_drawdown=10.0, drawdown_threshold=0.05)

    def test_runaway_reward_clipping_bounds(self, reward_fn: CostPenalizedReward):
        """Reward must be strictly clamped to [-10.0, 10.0] regardless of input magnitude."""
        # Massive gain
        r_huge_win = reward_fn.calculate_reward(delta_pnl=1_000_000.0)
        assert r_huge_win == 10.0

        # Massive loss
        r_huge_loss = reward_fn.calculate_reward(delta_pnl=-1_000_000.0)
        assert r_huge_loss == -10.0

        # Extreme fees and slippage
        r_heavy_costs = reward_fn.calculate_reward(delta_pnl=5.0, fee=500.0, slippage=500.0)
        assert r_heavy_costs == -10.0

    def test_drawdown_penalty_activation_threshold(self, reward_fn: CostPenalizedReward):
        """Verifies drawdown penalty is zero below 5% threshold, and increases quadratically above it."""
        # Drawdown <= 5% -> penalty = 0.0
        r_under = reward_fn.calculate_reward(delta_pnl=1.0, drawdown=0.03)
        assert r_under == 1.0

        r_at_threshold = reward_fn.calculate_reward(delta_pnl=1.0, drawdown=0.05)
        assert r_at_threshold == 1.0

        # Drawdown 10%: excess = 0.05 -> penalty = 10.0 * (0.05^2) = 0.025
        r_10pct = reward_fn.calculate_reward(delta_pnl=1.0, drawdown=0.10)
        assert r_10pct == pytest.approx(1.0 - 0.025, abs=1e-5)

        # Drawdown 25%: excess = 0.20 -> penalty = 10.0 * (0.20^2) = 0.40
        r_25pct = reward_fn.calculate_reward(delta_pnl=1.0, drawdown=0.25)
        assert r_25pct == pytest.approx(1.0 - 0.40, abs=1e-5)

        # Drawdown 75%: excess = 0.70 -> penalty = 10.0 * (0.70^2) = 4.90
        r_75pct = reward_fn.calculate_reward(delta_pnl=1.0, drawdown=0.75)
        assert r_75pct == pytest.approx(1.0 - 4.90, abs=1e-5)

    def test_fee_and_slippage_exact_penalization(self, reward_fn: CostPenalizedReward):
        """Verifies exact linear deduction of transaction fees and slippage."""
        r = reward_fn.calculate_reward(delta_pnl=2.50, fee=0.35, slippage=0.15, drawdown=0.02)
        expected = 2.50 - 0.35 - 0.15
        assert r == pytest.approx(expected, abs=1e-5)


# ============================================================================
# 4. Multi-Timeframe Bar Synchronizer Lookahead Checks
# ============================================================================

class TestMultiTimeframeSynchronizerLookahead:
    """Adversarial stress-testing of MultiTimeframeBarSynchronizer for lookahead bias and temporal leakage."""

    @pytest.fixture
    def synchronizer(self) -> MultiTimeframeBarSynchronizer:
        return MultiTimeframeBarSynchronizer(supported_timeframes=("5m", "15m", "1h"))

    def test_zero_lookahead_during_incomplete_5m_bar(self, synchronizer: MultiTimeframeBarSynchronizer):
        """Ingesting bars 09:30, 09:31, 09:32, 09:33, 09:34 must emit NO 5m bar until 09:35 is ingested."""
        start = datetime.datetime(2026, 1, 15, 9, 30)

        for minute in range(5):
            bar = Bar(
                timestamp=start + datetime.timedelta(minutes=minute),
                symbol="AAPL",
                open=100.0 + minute,
                high=102.0 + minute,
                low=99.0 + minute,
                close=101.0 + minute,
                volume=1000.0,
            )
            emitted = synchronizer.ingest_bar(bar)
            assert emitted["5m"] is None, f"Lookahead violation: 5m bar emitted prematurely at minute {minute}"
            assert len(synchronizer.get_bars("AAPL", "5m")) == 0

    def test_emission_at_window_boundary_with_exact_aggregation(self, synchronizer: MultiTimeframeBarSynchronizer):
        """At 09:35 (the first bar of the NEXT window), exactly one 5m bar must be emitted covering 09:30-09:34."""
        start = datetime.datetime(2026, 1, 15, 9, 30)

        # Ingest 09:30 to 09:34
        for minute in range(5):
            bar = Bar(
                timestamp=start + datetime.timedelta(minutes=minute),
                symbol="AAPL",
                open=100.0 + minute,
                high=105.0 + minute,
                low=95.0 - minute,
                close=102.0 + minute,
                volume=500.0,
            )
            synchronizer.ingest_bar(bar)

        # Now ingest 09:35 (first bar of next window) with distinct anomalous prices
        bar_35 = Bar(
            timestamp=start + datetime.timedelta(minutes=5),
            symbol="AAPL",
            open=999.0,
            high=999.0,
            low=999.0,
            close=999.0,
            volume=99999.0,
        )
        emitted = synchronizer.ingest_bar(bar_35)

        # Must emit completed 5m bar
        b5 = emitted["5m"]
        assert b5 is not None, "5m bar should be emitted at the boundary 09:35"
        assert b5.timestamp == start + datetime.timedelta(minutes=5)

        # Check OHLCV aggregation strictly reflects 09:30..09:34
        assert b5.open == 100.0, "5m Open must be 09:30 bar Open"
        assert b5.close == 102.0 + 4, "5m Close must be 09:34 bar Close"
        assert b5.high == 105.0 + 4, "5m High must be max across 09:30..09:34"
        assert b5.low == 95.0 - 4, "5m Low must be min across 09:30..09:34"
        assert b5.volume == 5 * 500.0, "5m Volume must be sum across 09:30..09:34"

        # LOOKAHEAD INTEGRITY CHECK: anomalous 09:35 values must NOT leak into the emitted 5m bar!
        assert b5.open != 999.0
        assert b5.close != 999.0
        assert b5.volume != 99999.0

    def test_simultaneous_multi_timeframe_alignment_and_volume_conservation(
        self, synchronizer: MultiTimeframeBarSynchronizer
    ):
        """Feeds 65 consecutive 1m bars from 09:00 to 10:04 and tests 5m, 15m, and 1h emission timing."""
        start = datetime.datetime(2026, 1, 15, 9, 0)
        total_vol_60m = 0.0

        for minute in range(65):
            vol = 100.0 + minute
            if minute < 60:
                total_vol_60m += vol

            bar = Bar(
                timestamp=start + datetime.timedelta(minutes=minute),
                symbol="MSFT",
                open=200.0 + (minute * 0.1),
                high=201.0 + (minute * 0.1),
                low=199.0 + (minute * 0.1),
                close=200.5 + (minute * 0.1),
                volume=vol,
            )
            emitted = synchronizer.ingest_bar(bar)

            # At minute 60 (10:00:00), 1h bar should be emitted for 09:00-09:59
            if minute == 60:
                assert emitted["1h"] is not None, "1h bar must be emitted when bar 10:00 is ingested"
                assert emitted["1h"].timestamp == datetime.datetime(2026, 1, 15, 10, 0)
            elif minute < 60:
                assert emitted["1h"] is None, f"Lookahead violation: 1h bar emitted prematurely at minute {minute}"

        # Inspect completed historical bars
        bars_5m = synchronizer.get_bars("MSFT", "5m")
        bars_15m = synchronizer.get_bars("MSFT", "15m")
        bars_1h = synchronizer.get_bars("MSFT", "1h")

        # 65 minutes ingested -> 12 full 5m bars, 4 full 15m bars, 1 full 1h bar
        assert len(bars_5m) == 12, f"Expected 12 completed 5m bars, got {len(bars_5m)}"
        assert len(bars_15m) == 4, f"Expected 4 completed 15m bars, got {len(bars_15m)}"
        assert len(bars_1h) == 1, f"Expected 1 completed 1h bar, got {len(bars_1h)}"

        # Volume conservation: 1h bar volume must match sum of 1m bars 0..59
        assert bars_1h[0].volume == pytest.approx(total_vol_60m, abs=1e-4)

        # Sum of four 15m bars must equal 1h bar volume
        vol_15m_sum = sum(b.volume for b in bars_15m)
        assert vol_15m_sum == pytest.approx(total_vol_60m, abs=1e-4)

        # Sum of first twelve 5m bars must equal 1h bar volume
        vol_5m_sum = sum(b.volume for b in bars_5m[:12])
        assert vol_5m_sum == pytest.approx(total_vol_60m, abs=1e-4)

    def test_multi_symbol_independence(self, synchronizer: MultiTimeframeBarSynchronizer):
        """Ensures that alternating bars across symbols (AAPL and TSLA) do not mix state or cross-contaminate."""
        start = datetime.datetime(2026, 1, 15, 9, 30)

        for minute in range(6):
            ts = start + datetime.timedelta(minutes=minute)
            # Ingest AAPL bar ($150)
            synchronizer.ingest_bar(Bar(ts, "AAPL", 150.0, 151.0, 149.0, 150.5, 1000.0))
            # Ingest TSLA bar ($250)
            synchronizer.ingest_bar(Bar(ts, "TSLA", 250.0, 252.0, 248.0, 251.0, 2000.0))

        aapl_5m = synchronizer.get_bars("AAPL", "5m")
        tsla_5m = synchronizer.get_bars("TSLA", "5m")

        assert len(aapl_5m) == 1
        assert len(tsla_5m) == 1
        assert aapl_5m[0].open == 150.0
        assert tsla_5m[0].open == 250.0
        assert aapl_5m[0].volume == 5000.0
        assert tsla_5m[0].volume == 10000.0


# ============================================================================
# 5. Bandit Allocator & Universe Screener Adversarial Tests
# ============================================================================

class TestBanditAllocatorAndUniverseScreenerAdversarial:
    """Adversarial stress-testing of DynamicUniverseScreener and MultiArmedBanditAllocator."""

    def test_bandit_allocator_strict_50_pct_concentration_cap(self):
        """Even if one asset has an astronomical Sharpe ratio, its allocation must be capped at 50%."""
        allocator = MultiArmedBanditAllocator(temperature=1.0, max_concentration=0.50)
        # Extreme disparity: Asset A Sharpe = 100.0 vs Asset B Sharpe = -50.0
        sharpes = {"SUPER_ASSET": 100.0, "DOG_ASSET": -50.0}
        weights = allocator.allocate(sharpes)

        assert "SUPER_ASSET" in weights and "DOG_ASSET" in weights
        # Concentration cap enforcement
        assert weights["SUPER_ASSET"] <= 0.500001, f"Asset exceeded 50% concentration cap: {weights['SUPER_ASSET']}"
        assert weights["DOG_ASSET"] >= 0.499999, f"Remainder was not redistributed to second asset: {weights['DOG_ASSET']}"
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-4)

    def test_bandit_allocator_multi_asset_distribution(self):
        """Tests allocation with 5 assets, ensuring cap and total weight sum to 1.0."""
        allocator = MultiArmedBanditAllocator(temperature=1.0, max_concentration=0.50)
        sharpes = {"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0, "E": 1.0}
        weights = allocator.allocate(sharpes)

        for sym, w in weights.items():
            assert 0.0 <= w <= 0.500001, f"Weight {sym}={w} violated concentration cap"
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-4)

    def test_bandit_allocator_single_asset_safety(self):
        """Single asset allocation must be capped at max_concentration (50%), holding rest in cash."""
        allocator = MultiArmedBanditAllocator(max_concentration=0.50)
        weights = allocator.allocate({"ONLY_ONE": 2.5})
        assert weights["ONLY_ONE"] == 0.50

    def test_dynamic_universe_screener_rejections(self):
        """Verifies screener rejects illiquid (< $5M) and low-volatility (< 0.5% ATR) assets."""
        screener = DynamicUniverseScreener(min_volume_24h_usd=5_000_000.0, min_volatility_ratio=0.005)

        candidates = [
            # 1. Illiquid: $4.9M volume -> reject
            CandidateAsset(symbol="ILLIQ", asset_class="EQUITY", price=100.0, volume_24h_usd=4_999_999.0, atr_14=1.0),
            # 2. Low volatility: ATR/price = 0.2% -> reject
            CandidateAsset(symbol="LOWVOL", asset_class="EQUITY", price=100.0, volume_24h_usd=10_000_000.0, atr_14=0.2),
            # 3. Invalid zero price -> reject
            CandidateAsset(symbol="ZEROP", asset_class="EQUITY", price=0.0, volume_24h_usd=10_000_000.0, atr_14=1.0),
            # 4. Valid prime candidate: $20M volume, 1.5% ATR -> accept
            CandidateAsset(symbol="PRIME", asset_class="EQUITY", price=100.0, volume_24h_usd=20_000_000.0, atr_14=1.5, sharpe_30d=2.0),
        ]

        selected = screener.filter_and_rank(candidates)
        assert len(selected) == 1
        assert selected[0].symbol == "PRIME"


# ============================================================================
# 6. Hybrid Strategy Engine Signal Geometry & Execution Stress Tests
# ============================================================================

class TestHybridStrategyEngineAdversarial:
    """Adversarial stress-testing of HybridStrategyEngine signal synthesis and stop geometry."""

    @pytest.fixture
    def bars_series(self) -> List[Bar]:
        """Creates 50 deterministic bars with positive trend."""
        start = datetime.datetime(2026, 1, 15, 9, 30)
        bars = []
        p = 100.0
        for i in range(50):
            p += 0.5
            bars.append(Bar(
                timestamp=start + datetime.timedelta(minutes=i),
                symbol="AAPL",
                open=p - 0.2, high=p + 0.8, low=p - 0.7, close=p + 0.3,
                volume=1000.0 + (i * 10.0),
                asset_class=AssetClass.EQUITY,
                currency="USD",
            ))
        return bars

    def test_signal_stop_loss_and_take_profit_geometry(self, bars_series: List[Bar]):
        """Verifies geometric invariants: Long: SL < Price < TP; Short: TP < Price < SL; Flat: SL/TP is None."""
        engine = HybridStrategyEngine(k_take_profit=3.0, k_stop_loss=2.0, default_account_equity_eur=100.0)

        sig = engine.generate_signal("AAPL", bars=bars_series, account_equity_eur=100.0)
        curr_p = bars_series[-1].close

        assert isinstance(sig, TradingSignal)
        assert -1.0 <= sig.target_weight <= 1.0
        assert 0.0 <= sig.confidence <= 1.0

        if sig.target_weight > 0:
            assert sig.stop_loss is not None and sig.take_profit is not None
            assert sig.stop_loss < curr_p < sig.take_profit, (
                f"Long geometry violated: SL ({sig.stop_loss}) < P ({curr_p}) < TP ({sig.take_profit})"
            )
        elif sig.target_weight < 0:
            assert sig.stop_loss is not None and sig.take_profit is not None
            assert sig.take_profit < curr_p < sig.stop_loss, (
                f"Short geometry violated: TP ({sig.take_profit}) < P ({curr_p}) < SL ({sig.stop_loss})"
            )
        else:
            assert sig.stop_loss is None
            assert sig.take_profit is None

    def test_signal_metadata_micro_capital_sizing(self, bars_series: List[Bar]):
        """Verifies signal metadata includes fractional order quantity calculated on €100 capital."""
        engine = HybridStrategyEngine(default_account_equity_eur=100.0)
        sig = engine.generate_signal("AAPL", bars=bars_series, account_equity_eur=100.0)

        assert "fractional_order_quantity" in sig.metadata
        assert "account_equity_eur" in sig.metadata
        assert sig.metadata["account_equity_eur"] == 100.0

        qty = sig.metadata["fractional_order_quantity"]
        if abs(sig.target_weight) > 0.05:
            assert qty is not None and qty > 0, "Non-zero allocation on €100 capital should have fractional quantity"

    def test_engine_empty_bars_safety(self):
        """Engine must handle empty bar sequence gracefully without raising exceptions."""
        engine = HybridStrategyEngine()
        sig = engine.generate_signal("EMPTY", bars=[])
        assert sig.action == "HOLD"
        assert sig.target_weight == 0.0
        assert sig.stop_loss is None
        assert sig.take_profit is None
        assert sig.metadata.get("reason") == "no_bars_available"
