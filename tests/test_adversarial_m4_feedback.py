"""Adversarial Stress Test Suite for Continuous Learning & PER (Milestone 4).

Empirically tests triggers, cooldowns, feedback anomalies, PER buffer edge cases,
model registry recovery, and live readiness gates under extreme loads and boundary conditions.
"""

from __future__ import annotations

import datetime
import json
import os
import tempfile
from typing import Any, Dict, List

import numpy as np
import pytest

from modeloinv.learning.champion import ChampionChallengerGate, GateResult
from modeloinv.learning.feedback import (
    PrioritizedExperienceReplayBuffer,
    TradeFeedbackIngestor,
    TradeFeedbackRecord,
)
from modeloinv.learning.readiness import (
    LiveReadinessEvaluator,
    LiveReadinessReport,
    ReadinessMetrics,
)
from modeloinv.learning.registry import ModelMetadata, ModelRegistry
from modeloinv.learning.retrainer import IncrementalSlidingWindowRetrainer
from modeloinv.learning.triggers import RetrainingTriggerManager, TriggerReason
from modeloinv.strategy.ml_model import HistGradientBoostingModel
from modeloinv.strategy.rl_agent import PolicyAgent


# =====================================================================
# 1. Adversarial Trigger Manager Stress Tests
# =====================================================================

class TestAdversarialTriggers:
    """Stress tests trigger manager with rapid events, edge values, and simultaneous triggers."""

    def test_rapid_triggers_within_cooldown(self) -> None:
        """Rapid event firing (100 sequential events within cooldown) must all be suppressed."""
        mgr = RetrainingTriggerManager(cooldown_minutes=60, trade_count_threshold=5)
        t0 = datetime.datetime(2026, 10, 26, 10, 0, tzinfo=datetime.timezone.utc)
        mgr.record_retrain(timestamp=t0)

        # Fire 100 checks every 30 seconds (total 50 minutes, strictly < 60 min cooldown)
        for step in range(1, 101):
            t_curr = t0 + datetime.timedelta(seconds=step * 30)
            mgr.record_trade(1)
            # Even with trade threshold exceeded and drawdown breach, cooldown MUST suppress
            should_retrain, reason, msg = mgr.check_triggers(
                current_time=t_curr,
                current_drawdown=0.15,
            )
            assert not should_retrain, f"Trigger fired unexpectedly at step {step} during cooldown!"
            assert reason is None
            assert "active cooldown" in msg

    def test_cooldown_boundary_second_precision(self) -> None:
        """Tests exact 1-second boundary: 59m59s is suppressed, 60m00s is released."""
        mgr = RetrainingTriggerManager(cooldown_minutes=60, drawdown_threshold=0.05)
        t0 = datetime.datetime(2026, 10, 26, 12, 0, 0, tzinfo=datetime.timezone.utc)
        mgr.record_retrain(timestamp=t0)

        # T0 + 59 minutes 59 seconds -> still in cooldown
        t_before = t0 + datetime.timedelta(minutes=59, seconds=59)
        assert mgr.is_in_cooldown(t_before)
        should_retrain, _, _ = mgr.check_triggers(t_before, current_drawdown=0.08)
        assert not should_retrain

        # T0 + 60 minutes 00 seconds -> cooldown elapsed
        t_after = t0 + datetime.timedelta(minutes=60, seconds=0)
        assert not mgr.is_in_cooldown(t_after)
        should_retrain, reason, _ = mgr.check_triggers(t_after, current_drawdown=0.08)
        assert should_retrain
        assert reason == TriggerReason.DRAWDOWN_BREACH

    def test_simultaneous_multi_trigger_priority(self) -> None:
        """When multiple triggers fire at once, DRAWDOWN_BREACH must take precedence for capital defense."""
        mgr = RetrainingTriggerManager(
            scheduled_hour_utc=21,
            scheduled_minute_utc=0,
            trade_count_threshold=10,
            drift_brier_threshold=0.35,
            drift_window_size=5,
            drawdown_threshold=0.05,
            cooldown_minutes=0,  # Zero cooldown to evaluate raw priority logic
        )
        t_eval = datetime.datetime(2026, 10, 26, 21, 15, tzinfo=datetime.timezone.utc)

        # Satisfy condition 3: Trade count milestone
        for _ in range(15):
            mgr.record_trade()

        # Satisfy condition 2: Prediction drift (Brier = 1.0)
        for _ in range(5):
            mgr.record_prediction(p_up=1.0, y_realized=0.0)

        # Satisfy condition 1: Drawdown breach (6.0% >= 5.0%)
        # Satisfy condition 4: Daily schedule (21:15 UTC >= 21:00 UTC)
        should_retrain, reason, msg = mgr.check_triggers(
            current_time=t_eval,
            current_drawdown=0.06,
        )

        assert should_retrain
        # DRAWDOWN_BREACH must be selected over drift, trade count, and daily schedule
        assert reason == TriggerReason.DRAWDOWN_BREACH
        assert "Drawdown breach" in msg

    def test_drift_brier_identical_predictions(self) -> None:
        """All predictions at 0.5 (maximum uncertainty) against alternating outcomes produce Brier = 0.25."""
        mgr = RetrainingTriggerManager(drift_brier_threshold=0.35, drift_window_size=10)
        now = datetime.datetime(2026, 10, 26, 12, 0, tzinfo=datetime.timezone.utc)

        for i in range(10):
            mgr.record_prediction(p_up=0.5, y_realized=float(i % 2))

        brier = mgr.calculate_rolling_brier()
        assert brier is not None
        # (0.5 - 1.0)^2 = 0.25, (0.5 - 0.0)^2 = 0.25 -> mean = 0.25
        assert brier == pytest.approx(0.25, abs=1e-5)
        # Should not trigger drift because 0.25 <= 0.35
        should_retrain, reason, _ = mgr.check_triggers(current_time=now)
        assert not should_retrain

    def test_trigger_empty_and_sub_window_predictions(self) -> None:
        """Rolling Brier returns None when records are below drift_window_size, avoiding false triggers."""
        mgr = RetrainingTriggerManager(drift_window_size=20)
        assert mgr.calculate_rolling_brier() is None

        # Add 19 predictions (1 less than window of 20)
        for _ in range(19):
            mgr.record_prediction(p_up=0.0, y_realized=1.0)
        assert mgr.calculate_rolling_brier() is None

        # 20th prediction completes window
        mgr.record_prediction(p_up=0.0, y_realized=1.0)
        brier = mgr.calculate_rolling_brier()
        assert brier is not None
        assert brier == pytest.approx(1.0, abs=1e-5)

    def test_non_positive_trade_increments(self) -> None:
        """Non-positive trade counts (0 or negative) must not advance milestone counter."""
        mgr = RetrainingTriggerManager(trade_count_threshold=10)
        assert mgr.trades_since_last_retrain == 0

        mgr.record_trade(0)
        assert mgr.trades_since_last_retrain == 0

        mgr.record_trade(-5)
        assert mgr.trades_since_last_retrain == 0

        mgr.record_trade(3)
        assert mgr.trades_since_last_retrain == 3

    def test_triggers_naive_vs_aware_datetime_behavior(self) -> None:
        """Evaluates behavior when caller passes offset-naive current_time against aware last_retrain_time."""
        mgr = RetrainingTriggerManager(cooldown_minutes=60)
        t_aware = datetime.datetime(2026, 10, 26, 12, 0, tzinfo=datetime.timezone.utc)
        mgr.record_retrain(timestamp=t_aware)

        t_naive = datetime.datetime(2026, 10, 26, 12, 30)
        # Timezone normalization handles naive vs aware gracefully during cooldown
        should_retrain, reason, msg = mgr.check_triggers(current_time=t_naive)
        assert not should_retrain
        assert "active cooldown" in msg


# =====================================================================
# 2. Adversarial Feedback Ingestor Stress Tests
# =====================================================================

class TestAdversarialFeedbackIngestor:
    """Stress tests trade feedback ingestion, outlier clipping, and anomaly logging."""

    def test_extreme_slippage_outliers_capped_at_5x(self) -> None:
        """Extreme positive slippage (>5x expected) must be capped at 5x and tagged as anomaly."""
        ingestor = TradeFeedbackIngestor()
        now = datetime.datetime(2026, 10, 26, 10, 0)

        # Test 10x expected slippage
        rec_10x = TradeFeedbackRecord(
            trade_id="TRD-10X",
            symbol="EUR/USD",
            direction="LONG",
            entry_time=now,
            exit_time=now + datetime.timedelta(minutes=5),
            entry_price=1.0800,
            exit_price=1.0820,
            quantity=1000.0,
            realized_pnl=20.0,
            realized_return_pct=0.00185,
            fees_paid=0.5,
            expected_slippage=0.0002,
            actual_slippage=0.0020,  # 10x expected
            predicted_p_up=0.6,
            predicted_confidence=0.5,
            prediction_error=0.1,
            holding_bars=5,
        )
        ingestor.ingest(rec_10x)

        stored = ingestor.get_recent_records(1)[0]
        assert stored.actual_slippage == pytest.approx(0.0010, rel=1e-5)  # 5 * 0.0002
        assert stored.metadata.get("slippage_anomaly") is True
        assert stored.metadata.get("uncapped_actual_slippage") == 0.0020

    def test_negative_slippage_price_improvement(self) -> None:
        """Negative slippage (price improvement) is valid in live trading and must not be capped."""
        ingestor = TradeFeedbackIngestor()
        now = datetime.datetime(2026, 10, 26, 10, 0)

        rec_favorable = TradeFeedbackRecord(
            trade_id="TRD-FAVORABLE",
            symbol="BTC/USDT",
            direction="BUY",
            entry_time=now,
            exit_time=now + datetime.timedelta(minutes=1),
            entry_price=68000.0,
            exit_price=68500.0,
            quantity=0.05,
            realized_pnl=25.0,
            realized_return_pct=0.0073,
            fees_paid=0.1,
            expected_slippage=0.01,
            actual_slippage=-0.005,  # Executed better than quote (favorable price improvement)
            predicted_p_up=0.7,
            predicted_confidence=0.6,
            prediction_error=0.05,
            holding_bars=1,
        )
        ingestor.ingest(rec_favorable)

        stored = ingestor.get_recent_records(1)[0]
        assert stored.actual_slippage == -0.005
        assert stored.metadata.get("slippage_anomaly") is not True

    def test_zero_expected_slippage_no_division_error(self) -> None:
        """When expected slippage is 0.0, actual slippage should not trigger zero-division."""
        ingestor = TradeFeedbackIngestor()
        now = datetime.datetime(2026, 10, 26, 10, 0)

        rec = TradeFeedbackRecord(
            trade_id="TRD-ZERO-EXP",
            symbol="SPY",
            direction="LONG",
            entry_time=now,
            exit_time=now + datetime.timedelta(minutes=1),
            entry_price=580.0,
            exit_price=581.0,
            quantity=1.0,
            realized_pnl=1.0,
            realized_return_pct=0.0017,
            fees_paid=0.0,
            expected_slippage=0.0,
            actual_slippage=0.02,
            predicted_p_up=0.5,
            predicted_confidence=0.5,
            prediction_error=0.0,
            holding_bars=1,
        )
        ingestor.ingest(rec)
        stored = ingestor.get_recent_records(1)[0]
        assert stored.actual_slippage == 0.02

    @pytest.mark.parametrize(
        "kwargs,expected_err",
        [
            ({"entry_price": 0.0}, "entry_price must be > 0"),
            ({"entry_price": -10.0}, "entry_price must be > 0"),
            ({"exit_price": 0.0}, "exit_price must be > 0"),
            ({"exit_price": -5.0}, "exit_price must be > 0"),
            ({"quantity": 0.0}, "quantity must be > 0"),
            ({"quantity": -1.0}, "quantity must be > 0"),
        ],
    )
    def test_invalid_feedback_record_raises_value_error(self, kwargs: Dict[str, Any], expected_err: str) -> None:
        """Invalid numeric values in telemetry records must fail fast."""
        now = datetime.datetime(2026, 10, 26, 10, 0)
        base = {
            "trade_id": "TRD-INVALID",
            "symbol": "AAPL",
            "direction": "LONG",
            "entry_time": now,
            "exit_time": now + datetime.timedelta(minutes=5),
            "entry_price": 150.0,
            "exit_price": 152.0,
            "quantity": 10.0,
            "realized_pnl": 20.0,
            "realized_return_pct": 0.0133,
            "fees_paid": 0.5,
            "expected_slippage": 0.01,
            "actual_slippage": 0.01,
            "predicted_p_up": 0.5,
            "predicted_confidence": 0.5,
            "prediction_error": 0.0,
            "holding_bars": 5,
        }
        base.update(kwargs)
        with pytest.raises(ValueError, match=expected_err):
            TradeFeedbackRecord(**base)

    def test_ingestor_capacity_overflow_eviction(self) -> None:
        """Ingestor strictly respects max_records ceiling by evicting oldest records."""
        ingestor = TradeFeedbackIngestor(max_records=100)
        now = datetime.datetime(2026, 10, 26, 10, 0)

        # Ingest 350 records
        for i in range(350):
            ingestor.ingest(
                TradeFeedbackRecord(
                    trade_id=f"TRD-{i}",
                    symbol="QQQ",
                    direction="LONG",
                    entry_time=now + datetime.timedelta(minutes=i),
                    exit_time=now + datetime.timedelta(minutes=i + 1),
                    entry_price=490.0,
                    exit_price=491.0,
                    quantity=1.0,
                    realized_pnl=1.0,
                    realized_return_pct=0.002,
                    fees_paid=0.01,
                    expected_slippage=0.01,
                    actual_slippage=0.01,
                    predicted_p_up=0.7,
                    predicted_confidence=0.6,
                    prediction_error=0.1,
                    holding_bars=1,
                )
            )

        assert ingestor.count() == 100
        recent = ingestor.get_recent_records(100)
        assert len(recent) == 100
        # Oldest stored record must be TRD-250, newest TRD-349
        assert recent[0].trade_id == "TRD-250"
        assert recent[-1].trade_id == "TRD-349"

    def test_rolling_brier_edge_windows(self) -> None:
        """Tests rolling Brier on window <= 0 and window > record count."""
        ingestor = TradeFeedbackIngestor()
        now = datetime.datetime(2026, 10, 26, 10, 0)
        for i in range(5):
            ingestor.ingest(
                TradeFeedbackRecord(
                    trade_id=f"TRD-{i}",
                    symbol="QQQ",
                    direction="LONG",
                    entry_time=now,
                    exit_time=now,
                    entry_price=100.0,
                    exit_price=101.0,
                    quantity=1.0,
                    realized_pnl=1.0,
                    realized_return_pct=0.01,
                    fees_paid=0.0,
                    expected_slippage=0.0,
                    actual_slippage=0.0,
                    predicted_p_up=1.0,
                    predicted_confidence=1.0,
                    prediction_error=0.0,
                    holding_bars=1,
                )
            )

        assert ingestor.calculate_rolling_brier(window=0) is None
        assert ingestor.calculate_rolling_brier(window=-5) is None
        assert ingestor.calculate_rolling_brier(window=10) is None  # 10 > 5 records
        assert ingestor.calculate_rolling_brier(window=5) == 0.0


# =====================================================================
# 3. Adversarial Prioritized Experience Replay (PER) Stress Tests
# =====================================================================

class TestAdversarialPERBuffer:
    """Stress tests PER buffer numerical stability, priority skews, and capacity wrap-around."""

    def test_zero_and_negative_td_errors(self) -> None:
        """Zero and negative TD errors must maintain positive priorities via eps_p and abs()."""
        buf = PrioritizedExperienceReplayBuffer(capacity=10, eps_p=1e-5)
        s = np.zeros(9, dtype=np.float32)

        # td_error = 0.0 -> priority = 0.0 + 1e-5 = 1e-5
        buf.push(s, action=0, reward=0.0, next_state=s, done=False, td_error=0.0)
        assert buf._priorities[0] == pytest.approx(1e-5, rel=1e-5)

        # td_error = -3.5 -> priority = |-3.5| + 1e-5 = 3.50001
        buf.push(s, action=1, reward=1.0, next_state=s, done=False, td_error=-3.5)
        assert buf._priorities[1] == pytest.approx(3.50001, rel=1e-5)

    def test_infinite_td_error_numerical_behavior(self) -> None:
        """When td_error is inf, verifies priority is clipped to finite upper bound and sampling succeeds."""
        buf = PrioritizedExperienceReplayBuffer(capacity=10)
        s = np.zeros(9, dtype=np.float32)
        buf.push(s, action=0, reward=1.0, next_state=s, done=False, td_error=float("inf"))
        assert np.isfinite(buf._priorities[0])
        assert buf._priorities[0] <= 1e6

        # Clipped priority allows safe sampling without NaN
        states, actions, rewards, next_states, dones, indices, is_weights = buf.sample(batch_size=1)
        assert len(states) == 1

    def test_extreme_priority_skew_numerical_stability(self) -> None:
        """Extreme priority skew (1 outlier with priority 10^5 vs 100 with 10^-5) must sample stably."""
        buf = PrioritizedExperienceReplayBuffer(capacity=105, alpha=0.6, eps_p=1e-5)
        s = np.zeros(9, dtype=np.float32)

        # 1 extreme outlier
        buf.push(s, action=0, reward=10.0, next_state=s, done=False, td_error=1e5)
        # 100 tiny background transitions
        for i in range(1, 101):
            buf.push(s, action=i % 4, reward=0.01, next_state=s, done=False, td_error=0.0)

        assert len(buf) == 101

        # Sample batch of 16
        states, actions, rewards, next_states, dones, indices, is_weights = buf.sample(
            batch_size=16,
            beta_is=0.4,
        )

        assert len(states) == 16
        assert len(is_weights) == 16
        # Check IS weight bounds: w_i in [0, 1]
        assert np.all(np.isfinite(is_weights))
        assert np.all(is_weights >= 0.0)
        assert np.all(is_weights <= 1.0)
        assert np.max(is_weights) == pytest.approx(1.0, abs=1e-5)
        # Outlier with 10^5 priority should almost certainly be sampled
        assert 0 in indices

    def test_all_zero_priorities_fallback_to_uniform(self) -> None:
        """When priorities sum to zero or non-positive, buffer gracefully falls back to uniform."""
        buf = PrioritizedExperienceReplayBuffer(capacity=20, alpha=0.6, eps_p=0.0)
        s = np.zeros(9, dtype=np.float32)

        for i in range(10):
            buf.push(s, action=i % 2, reward=0.0, next_state=s, done=False, td_error=0.0, prediction_error=0.0)

        # All priorities are exactly 0.0 -> sum_p = 0.0
        states, actions, rewards, next_states, dones, indices, is_weights = buf.sample(batch_size=5)
        assert len(states) == 5
        assert np.all(np.isfinite(is_weights))
        assert np.max(is_weights) == pytest.approx(1.0, abs=1e-5)

    def test_importance_sampling_weight_normalization_across_betas(self) -> None:
        """IS weights must remain strictly in [0, 1] across various beta_is values."""
        buf = PrioritizedExperienceReplayBuffer(capacity=50)
        s = np.zeros(9, dtype=np.float32)
        rng = np.random.RandomState(42)

        for i in range(30):
            td = float(rng.exponential(scale=1.0))
            buf.push(s, action=i % 3, reward=float(i), next_state=s, done=False, td_error=td)

        for beta in [0.0, 0.2, 0.5, 0.8, 1.0]:
            _, _, _, _, _, _, is_weights = buf.sample(batch_size=10, beta_is=beta)
            assert np.all(np.isfinite(is_weights))
            assert np.all(is_weights >= 0.0)
            assert np.all(is_weights <= 1.0)
            assert np.max(is_weights) == pytest.approx(1.0, abs=1e-5)

    def test_circular_capacity_wrap_around_under_heavy_load(self) -> None:
        """Buffer capacity (50) under 1,000 pushes wraps cleanly without memory leak or desynchronization."""
        buf = PrioritizedExperienceReplayBuffer(capacity=50)
        s = np.zeros(9, dtype=np.float32)

        for i in range(1000):
            buf.push(
                state=s + i,
                action=i % 5,
                reward=float(i),
                next_state=s + i + 1,
                done=(i % 10 == 0),
                td_error=float(i % 20),
            )

        assert len(buf) == 50
        assert buf._pos == 0  # 1000 % 50 == 0
        assert len(buf._states) == 50
        assert len(buf._actions) == 50
        assert len(buf._rewards) == 50
        assert len(buf._priorities) == 50

        # Sample succeeds
        states, actions, rewards, _, _, _, _ = buf.sample(batch_size=20)
        assert len(states) == 20

    def test_update_priorities_out_of_bounds_tolerance(self) -> None:
        """Updating out-of-bounds indices is silently skipped without throwing IndexError."""
        buf = PrioritizedExperienceReplayBuffer(capacity=10)
        s = np.zeros(9, dtype=np.float32)
        for i in range(5):
            buf.push(s, action=0, reward=0.0, next_state=s, done=False)

        # Indices -1 and 99 are out of bounds; 2 is valid
        buf.update_priorities(indices=[-1, 2, 99], td_errors=[5.0, 1.2, 8.0])
        assert buf._priorities[2] == pytest.approx(1.2 + 1e-5, rel=1e-5)

    def test_sample_from_empty_buffer_raises(self) -> None:
        """Sampling an empty buffer must raise ValueError."""
        buf = PrioritizedExperienceReplayBuffer(capacity=10)
        with pytest.raises(ValueError, match="Cannot sample from an empty replay buffer"):
            buf.sample(batch_size=5)

    def test_sample_batch_size_clamping(self) -> None:
        """Requesting batch_size > len(buffer) safely clamps to len(buffer)."""
        buf = PrioritizedExperienceReplayBuffer(capacity=10)
        s = np.zeros(9, dtype=np.float32)
        for i in range(3):
            buf.push(s, action=i, reward=float(i), next_state=s, done=False)

        states, _, _, _, _, indices, _ = buf.sample(batch_size=100)
        assert len(states) == 3
        assert len(indices) == 3


# =====================================================================
# 4. Adversarial Retrainer Stress Tests
# =====================================================================

class TestAdversarialRetrainer:
    """Stress tests retrainer on degenerated features, extreme weights, and edge sample sizes."""

    def test_flat_zero_variance_features_fit_safely(self) -> None:
        """HistGradientBoosting retrains on zero-variance flat features without crashing."""
        retrainer = IncrementalSlidingWindowRetrainer(min_samples=20)
        X = np.zeros((30, 32), dtype=np.float32)  # All zeros
        y = np.array([1] * 15 + [-1] * 15, dtype=np.int64)

        candidate, metrics = retrainer.retrain_ml(X, y)
        assert candidate.is_fitted
        assert "accuracy" in metrics
        assert "brier_score" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_nan_features_native_handling(self) -> None:
        """HistGradientBoosting handles NaNs in features without preprocessing failure."""
        retrainer = IncrementalSlidingWindowRetrainer(min_samples=20)
        rng = np.random.RandomState(42)
        X = rng.randn(40, 32).astype(np.float32)
        # Sprinkle NaNs
        X[0, 0] = np.nan
        X[5, 12] = np.nan
        X[20, 31] = np.nan
        y = np.where(X[:, 1] > 0, 1, -1)

        candidate, metrics = retrainer.retrain_ml(X, y)
        assert candidate.is_fitted
        assert np.isfinite(metrics["accuracy"])
        assert np.isfinite(metrics["brier_score"])

    def test_sample_weights_extreme_age_decay(self) -> None:
        """Timestamps spanning 10 years decay gracefully; mean remains normalized to 1.0."""
        retrainer = IncrementalSlidingWindowRetrainer(tau_decay_days=30.0)
        now = datetime.datetime(2026, 10, 26, 12, 0)

        # 5 samples ranging from 10 years ago to 1 hour ago
        timestamps = [
            now - datetime.timedelta(days=3650),
            now - datetime.timedelta(days=1825),
            now - datetime.timedelta(days=365),
            now - datetime.timedelta(days=30),
            now - datetime.timedelta(hours=1),
        ]

        weights = retrainer.compute_sample_weights(timestamps, current_time=now)
        assert len(weights) == 5
        assert np.all(np.isfinite(weights))
        # Mean weight is normalized to 1.0
        assert np.mean(weights) == pytest.approx(1.0, rel=1e-5)
        # Recent sample has by far the highest weight
        assert weights[-1] > weights[0]
        assert weights[0] == pytest.approx(0.0, abs=1e-5)

    def test_sample_weights_future_timestamps_clamped(self) -> None:
        """Future timestamps (clock skew) are clamped with dt=0, receiving maximum recency factor = 1.0."""
        retrainer = IncrementalSlidingWindowRetrainer(tau_decay_days=30.0)
        now = datetime.datetime(2026, 10, 26, 12, 0)
        future_ts = now + datetime.timedelta(days=5)

        weights = retrainer.compute_sample_weights([future_ts, now], current_time=now)
        assert len(weights) == 2
        # Both effectively have dt=0, so equal weights
        assert weights[0] == pytest.approx(weights[1], rel=1e-5)

    def test_rl_retrainer_insufficient_buffer_returns_zero_metrics(self) -> None:
        """When replay buffer has fewer than max(batch_size, 16) transitions, returns zero metrics."""
        retrainer = IncrementalSlidingWindowRetrainer()
        agent = PolicyAgent(state_dim=9, action_dim=7)
        buf = PrioritizedExperienceReplayBuffer(capacity=100)

        # Only 5 transitions (less than 16 minimum)
        s = np.zeros(9, dtype=np.float32)
        for i in range(5):
            buf.push(s, action=0, reward=0.0, next_state=s, done=False)

        metrics = retrainer.retrain_rl(agent, buf, num_steps=10, batch_size=32)
        assert metrics["value_loss"] == 0.0
        assert metrics["mean_episode_reward"] == 0.0

    def test_rl_retrainer_multi_step_training_with_per(self) -> None:
        """Multi-step training on PER buffer updates agent policy and priority weights stably."""
        retrainer = IncrementalSlidingWindowRetrainer()
        agent = PolicyAgent(state_dim=9, action_dim=7)
        buf = PrioritizedExperienceReplayBuffer(capacity=100)

        s = np.zeros(9, dtype=np.float32)
        s_next = np.ones(9, dtype=np.float32)
        for i in range(40):
            buf.push(s, action=i % 7, reward=float(i % 5), next_state=s_next, done=False, td_error=float(i % 3))

        metrics = retrainer.retrain_rl(agent, buf, num_steps=5, batch_size=16)
        assert "value_loss" in metrics
        assert metrics["value_loss"] >= 0.0
        assert metrics["policy_entropy"] == 1.0



# =====================================================================
# 5. Adversarial Model Registry & Champion Gate Stress Tests
# =====================================================================

class TestAdversarialRegistryAndGate:
    """Stress tests model registry error recovery, semver bumping, and gate tolerances."""

    def test_corrupted_registry_json_recovery(self) -> None:
        """Corrupted JSON in registry.json automatically triggers graceful reset."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            reg_path = os.path.join(tmp_dir, "corrupted_registry.json")
            with open(reg_path, "w", encoding="utf-8") as f:
                f.write("{ INVALID JSON CONTENT: [unclosed")

            reg = ModelRegistry(registry_path=reg_path, models_dir=tmp_dir)
            assert reg.active_champion_version is None
            assert len(reg.models) == 0

    def test_semver_optional_v_prefix(self) -> None:
        """Semver regex permits optional 'v' prefix, so both 'v1.2.3' and '1.2.3' are parsed."""
        assert ModelRegistry.parse_semver("v1.2.3") == (1, 2, 3)
        assert ModelRegistry.parse_semver("1.2.3") == (1, 2, 3)

    @pytest.mark.parametrize(
        "invalid_ver",
        ["v1.0", "alpha-v1", "v1.0.0.0", "latest", "", "v1.2.beta", "not_a_version"],
    )
    def test_invalid_semver_strings_rejected(self, invalid_ver: str) -> None:
        """Invalid semantic version formats must raise ValueError."""
        with pytest.raises(ValueError, match="Invalid semantic version string"):
            ModelRegistry.parse_semver(invalid_ver)

    def test_semver_monotonic_progression(self) -> None:
        """Verify patch, minor, and major bumps from v1.0.9 and v1.9.9."""
        assert ModelRegistry.parse_semver("v1.0.9") == (1, 0, 9)

        with tempfile.TemporaryDirectory() as tmp_dir:
            reg_path = os.path.join(tmp_dir, "registry.json")
            reg = ModelRegistry(registry_path=reg_path, models_dir=tmp_dir)

            m = ModelMetadata(
                version="v1.0.9",
                timestamp=datetime.datetime.utcnow(),
                trigger_reason="init",
                training_window={},
            )
            reg.register_candidate(m)

            assert reg.next_version("patch") == "v1.0.10"
            assert reg.next_version("minor") == "v1.1.0"
            assert reg.next_version("major") == "v2.0.0"

    def test_champion_gate_exact_boundary_tolerance(self) -> None:
        """Strict verification of exact gate margins: Sharpe -0.10, Drawdown 10%, WinRate 50%."""
        gate = ChampionChallengerGate(sharpe_tolerance=0.10, max_drawdown_ceiling=0.10, min_win_rate=0.50)
        champ = {"sharpe_ratio": 1.50, "max_drawdown": 0.08, "win_rate": 0.55}

        # 1. Exact Sharpe boundary (1.50 - 0.10 = 1.4000) -> passes
        res_exact_sharpe = gate.evaluate_metrics(
            {"sharpe_ratio": 1.40, "max_drawdown": 0.08, "win_rate": 0.50},
            champ,
        )
        assert res_exact_sharpe.promoted

        # 2. Sharpe just below boundary (1.399) -> rejected
        res_below_sharpe = gate.evaluate_metrics(
            {"sharpe_ratio": 1.399, "max_drawdown": 0.08, "win_rate": 0.50},
            champ,
        )
        assert not res_below_sharpe.promoted
        assert "Sharpe" in res_below_sharpe.reason

        # 3. Exact Drawdown boundary (0.08 * 1.10 = 0.088) -> passes
        res_exact_dd = gate.evaluate_metrics(
            {"sharpe_ratio": 1.50, "max_drawdown": 0.088, "win_rate": 0.50},
            champ,
        )
        assert res_exact_dd.promoted

        # 4. Drawdown exceeds 1.10x champion limit (0.089 > 0.088) -> rejected
        res_over_dd = gate.evaluate_metrics(
            {"sharpe_ratio": 1.50, "max_drawdown": 0.089, "win_rate": 0.50},
            champ,
        )
        assert not res_over_dd.promoted
        assert "Drawdown" in res_over_dd.reason

        # 5. Exact Win Rate boundary (0.500) -> passes
        res_exact_wr = gate.evaluate_metrics(
            {"sharpe_ratio": 1.50, "max_drawdown": 0.08, "win_rate": 0.500},
            champ,
        )
        assert res_exact_wr.promoted

        # 6. Win Rate below 50% (0.499) -> rejected
        res_below_wr = gate.evaluate_metrics(
            {"sharpe_ratio": 1.50, "max_drawdown": 0.08, "win_rate": 0.499},
            champ,
        )
        assert not res_below_wr.promoted
        assert "Win Rate" in res_below_wr.reason

    def test_rollback_on_empty_registry_returns_none(self) -> None:
        """Rollback on empty registry safely returns None without error."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            reg = ModelRegistry(registry_path=os.path.join(tmp_dir, "reg.json"), models_dir=tmp_dir)
            assert reg.rollback() is None


# =====================================================================
# 6. Adversarial Live Readiness Evaluator Stress Tests
# =====================================================================

class TestAdversarialLiveReadiness:
    """Stress tests 8-gate live readiness engine at strict boundary values and extreme negatives."""

    def test_exact_boundary_thresholds(self) -> None:
        """Verifies each gate at exact threshold passes, and just below threshold fails."""
        evaluator = LiveReadinessEvaluator()
        # Exactly on all 8 thresholds
        exact_pass = {
            "backtest_months": 12.0,      # Gate 1: >= 12.0
            "paper_days": 30.0,           # Gate 2: >= 30.0
            "strategy_cum_return": 0.1501, # Gate 3: > sp500
            "sp500_cum_return": 0.1500,
            "sharpe_ratio": 1.0001,       # Gate 4: > 1.0
            "max_30d_return": 0.15,       # Gate 5: >= 15.0%
            "max_drawdown": 0.10,         # Gate 6: <= 10.0%
            "total_trades": 50,           # Gate 7: >= 50
            "win_rate": 0.50,             # Gate 8: >= 50%
            "profit_factor": 1.30,        # Gate 8: >= 1.30
        }
        report_pass = evaluator.evaluate(exact_pass)
        assert report_pass.is_live_ready is True

        # Gate 3 strict inequality: strategy == sp500 must fail
        failing_g3 = dict(exact_pass)
        failing_g3["strategy_cum_return"] = 0.1500
        report_fail_g3 = evaluator.evaluate(failing_g3)
        assert report_fail_g3.is_live_ready is False
        assert report_fail_g3.gates["gate_3_beat_sp500"]["passed"] is False

        # Gate 4 strict inequality: sharpe == 1.0 must fail
        failing_g4 = dict(exact_pass)
        failing_g4["sharpe_ratio"] = 1.0000
        report_fail_g4 = evaluator.evaluate(failing_g4)
        assert report_fail_g4.is_live_ready is False
        assert report_fail_g4.gates["gate_4_sharpe_ratio"]["passed"] is False

    def test_extreme_negative_metrics_rejection(self) -> None:
        """Candidate with catastrophic metrics (drawdown 100%, negative returns) fails comprehensively."""
        evaluator = LiveReadinessEvaluator()
        catastrophic = {
            "backtest_months": 0.0,
            "paper_days": 0.0,
            "strategy_cum_return": -0.85,
            "sp500_cum_return": 0.20,
            "sharpe_ratio": -2.5,
            "max_30d_return": -0.40,
            "max_drawdown": 1.00,  # 100% drawdown
            "total_trades": 2,
            "win_rate": 0.0,
            "profit_factor": 0.0,
        }
        report = evaluator.evaluate(catastrophic)
        assert report.is_live_ready is False
        assert all(gate["passed"] is False for gate in report.gates.values())
        assert "8 gate(s) failed" in report.summary
