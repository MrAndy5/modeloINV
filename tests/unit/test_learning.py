"""Unit Tests for Continuous Learning, Model Registry, and Live Readiness (Milestone 4).

Tests:
1. RetrainingTriggerManager (daily schedule, trade count, drift, drawdown, cooldown).
2. TradeFeedbackRecord, TradeFeedbackIngestor, and PrioritizedExperienceReplayBuffer.
3. IncrementalSlidingWindowRetrainer (recency weighting, error weighting, ML/RL retraining).
4. ModelMetadata and ModelRegistry (semver, atomic persistence, artifact serialization, rollback).
5. ChampionChallengerGate (holdout evaluation, promotion, rejection, bootstrap).
6. LiveReadinessEvaluator (8 deterministic gates, checklist report).
7. LearningLoopBenchmark (30-day simulated feedback loop, v0 vs vN holdout improvement verification).
8. End-to-end continuous learning workflow.
"""

from __future__ import annotations

import datetime
import json
import os
import tempfile
from typing import Any, Dict, List

import numpy as np
import pytest

from modeloinv.learning.benchmark import BenchmarkResult, LearningLoopBenchmark
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
# 1. RetrainingTriggerManager Tests
# =====================================================================

class TestRetrainingTriggerManager:
    """Verifies all 4 retraining trigger conditions and cooldown rules."""

    def test_trade_count_trigger(self) -> None:
        mgr = RetrainingTriggerManager(trade_count_threshold=25, cooldown_minutes=60)
        now = datetime.datetime(2026, 10, 26, 12, 0, tzinfo=datetime.timezone.utc)

        # Record 24 trades -> should not trigger
        for _ in range(24):
            mgr.record_trade()
        should_retrain, reason, msg = mgr.check_triggers(current_time=now)
        assert not should_retrain
        assert reason is None

        # Record 25th trade -> should trigger TRADE_COUNT_MILESTONE
        mgr.record_trade()
        should_retrain, reason, msg = mgr.check_triggers(current_time=now)
        assert should_retrain
        assert reason == TriggerReason.TRADE_COUNT_MILESTONE
        assert "Trade count milestone" in msg

        # Retrain recorded -> resets trade count and engages cooldown
        mgr.record_retrain(timestamp=now, reason=reason)
        assert mgr.trades_since_last_retrain == 0
        assert mgr.is_in_cooldown(now + datetime.timedelta(minutes=30))

        # Check triggers during cooldown -> suppressed
        should_retrain, reason, msg = mgr.check_triggers(current_time=now + datetime.timedelta(minutes=30))
        assert not should_retrain
        assert "active cooldown" in msg

    def test_drawdown_breach_trigger(self) -> None:
        mgr = RetrainingTriggerManager(drawdown_threshold=0.05, cooldown_minutes=60)
        now = datetime.datetime(2026, 10, 26, 14, 0, tzinfo=datetime.timezone.utc)

        # 4% drawdown -> below threshold
        should_retrain, reason, _ = mgr.check_triggers(current_time=now, current_drawdown=0.04)
        assert not should_retrain

        # 5.5% drawdown -> triggers DRAWDOWN_BREACH
        should_retrain, reason, msg = mgr.check_triggers(current_time=now, current_drawdown=0.055)
        assert should_retrain
        assert reason == TriggerReason.DRAWDOWN_BREACH
        assert "Drawdown breach" in msg

    def test_prediction_drift_trigger(self) -> None:
        mgr = RetrainingTriggerManager(drift_brier_threshold=0.35, drift_window_size=10, cooldown_minutes=60)
        now = datetime.datetime(2026, 10, 26, 15, 0, tzinfo=datetime.timezone.utc)

        # Feed well-calibrated predictions: p_up=0.8, y=1.0 -> squared error = 0.04
        for _ in range(10):
            mgr.record_prediction(p_up=0.8, y_realized=1.0)
        brier = mgr.calculate_rolling_brier()
        assert brier is not None
        assert brier < 0.10
        should_retrain, reason, _ = mgr.check_triggers(current_time=now)
        assert not should_retrain

        # Feed inverted/miscalibrated predictions: p_up=0.9, y=0.0 -> squared error = 0.81
        for _ in range(10):
            mgr.record_prediction(p_up=0.9, y_realized=0.0)
        brier = mgr.calculate_rolling_brier()
        assert brier is not None
        assert brier > 0.35
        should_retrain, reason, msg = mgr.check_triggers(current_time=now)
        assert should_retrain
        assert reason == TriggerReason.PREDICTION_DRIFT
        assert "Prediction drift detected" in msg

    def test_scheduled_daily_trigger(self) -> None:
        mgr = RetrainingTriggerManager(scheduled_hour_utc=21, scheduled_minute_utc=0, cooldown_minutes=60)

        # 20:30 UTC -> before scheduled time
        t_before = datetime.datetime(2026, 10, 26, 20, 30, tzinfo=datetime.timezone.utc)
        should_retrain, reason, _ = mgr.check_triggers(current_time=t_before)
        assert not should_retrain

        # 21:05 UTC -> at or after scheduled time
        t_at = datetime.datetime(2026, 10, 26, 21, 5, tzinfo=datetime.timezone.utc)
        should_retrain, reason, msg = mgr.check_triggers(current_time=t_at)
        assert should_retrain
        assert reason == TriggerReason.SCHEDULED_DAILY

        # Record retrain on that date
        mgr.record_retrain(timestamp=t_at, reason=reason)

        # Check later on same day (post cooldown) -> should not re-trigger daily on same day
        t_later = datetime.datetime(2026, 10, 26, 23, 0, tzinfo=datetime.timezone.utc)
        should_retrain, reason, _ = mgr.check_triggers(current_time=t_later)
        assert not should_retrain

        # Next day at 21:05 UTC -> should trigger for the new day
        t_next_day = datetime.datetime(2026, 10, 27, 21, 5, tzinfo=datetime.timezone.utc)
        should_retrain, reason, _ = mgr.check_triggers(current_time=t_next_day)
        assert should_retrain
        assert reason == TriggerReason.SCHEDULED_DAILY

    def test_force_trigger_bypasses_cooldown(self) -> None:
        mgr = RetrainingTriggerManager(cooldown_minutes=60)
        now = datetime.datetime(2026, 10, 26, 12, 0, tzinfo=datetime.timezone.utc)
        mgr.record_retrain(timestamp=now)

        # Within cooldown, standard check is suppressed
        should_retrain, _, _ = mgr.check_triggers(current_time=now + datetime.timedelta(minutes=5))
        assert not should_retrain

        # Force check bypasses cooldown
        should_retrain, reason, msg = mgr.check_triggers(
            current_time=now + datetime.timedelta(minutes=5),
            force=True,
        )
        assert should_retrain
        assert reason == TriggerReason.MANUAL


# =====================================================================
# 2. Trade Feedback & Prioritized Replay Buffer Tests
# =====================================================================

class TestTradeFeedbackAndBuffer:
    """Verifies feedback logging, anomaly handling, and PER buffer math."""

    def test_feedback_record_validation_and_serialization(self) -> None:
        rec = TradeFeedbackRecord(
            trade_id="TRD-001",
            symbol="BTC/USDT",
            direction="LONG",
            entry_time=datetime.datetime(2026, 10, 26, 10, 0),
            exit_time=datetime.datetime(2026, 10, 26, 11, 0),
            entry_price=67000.0,
            exit_price=68000.0,
            quantity=0.01,
            realized_pnl=10.0,
            realized_return_pct=0.0149,
            fees_paid=0.5,
            expected_slippage=0.02,
            actual_slippage=0.025,
            predicted_p_up=0.75,
            predicted_confidence=0.60,
            prediction_error=0.05,
            holding_bars=12,
            model_version="v1.0.0",
        )
        d = rec.to_dict()
        assert d["trade_id"] == "TRD-001"
        assert d["realized_pnl"] == 10.0

        restored = TradeFeedbackRecord.from_dict(d)
        assert restored.trade_id == rec.trade_id
        assert restored.entry_price == rec.entry_price

        # Invariant checks: price <= 0 or invalid times
        with pytest.raises(ValueError):
            TradeFeedbackRecord(
                trade_id="TRD-BAD",
                symbol="AAPL",
                direction="LONG",
                entry_time=datetime.datetime(2026, 10, 26, 11, 0),
                exit_time=datetime.datetime(2026, 10, 26, 10, 0),  # Exit before entry!
                entry_price=100.0,
                exit_price=105.0,
                quantity=1.0,
                realized_pnl=5.0,
                realized_return_pct=0.05,
                fees_paid=0.1,
                expected_slippage=0.01,
                actual_slippage=0.01,
                predicted_p_up=0.5,
                predicted_confidence=0.5,
                prediction_error=0.0,
                holding_bars=5,
            )

    def test_slippage_outlier_clipping(self) -> None:
        ingestor = TradeFeedbackIngestor()
        rec = TradeFeedbackRecord(
            trade_id="TRD-SLIP-ANOMALY",
            symbol="SPY",
            direction="LONG",
            entry_time=datetime.datetime(2026, 10, 26, 10, 0),
            exit_time=datetime.datetime(2026, 10, 26, 10, 15),
            entry_price=580.0,
            exit_price=581.0,
            quantity=1.0,
            realized_pnl=1.0,
            realized_return_pct=0.0017,
            fees_paid=0.05,
            expected_slippage=0.02,
            actual_slippage=0.25,  # 12.5x expected slippage!
            predicted_p_up=0.6,
            predicted_confidence=0.5,
            prediction_error=0.1,
            holding_bars=3,
        )
        ingestor.ingest(rec)
        stored = ingestor.get_recent_records(1)[0]
        # Actual slippage must be capped at 5x expected slippage (5 * 0.02 = 0.10)
        assert stored.actual_slippage == pytest.approx(0.10, rel=1e-5)
        assert stored.metadata.get("slippage_anomaly") is True
        assert stored.metadata.get("uncapped_actual_slippage") == 0.25

    def test_ingestor_query_and_rolling_brier(self) -> None:
        ingestor = TradeFeedbackIngestor()
        base_time = datetime.datetime(2026, 10, 26, 9, 30)

        for i in range(10):
            ingestor.ingest(
                TradeFeedbackRecord(
                    trade_id=f"TRD-{i}",
                    symbol="QQQ",
                    direction="LONG",
                    entry_time=base_time + datetime.timedelta(minutes=i * 10),
                    exit_time=base_time + datetime.timedelta(minutes=(i + 1) * 10),
                    entry_price=490.0,
                    exit_price=491.0 if i % 2 == 0 else 489.0,
                    quantity=1.0,
                    realized_pnl=1.0 if i % 2 == 0 else -1.0,
                    realized_return_pct=0.002 if i % 2 == 0 else -0.002,
                    fees_paid=0.02,
                    expected_slippage=0.01,
                    actual_slippage=0.01,
                    predicted_p_up=0.8 if i % 2 == 0 else 0.2,
                    predicted_confidence=0.6,
                    prediction_error=0.2,
                    holding_bars=2,
                )
            )

        assert ingestor.count() == 10
        recent = ingestor.get_recent_records(5)
        assert len(recent) == 5

        # Window query
        window = ingestor.get_window(
            start_time=base_time + datetime.timedelta(minutes=20),
            end_time=base_time + datetime.timedelta(minutes=50),
        )
        assert len(window) > 0

        # Brier calculation over 10 trades
        brier = ingestor.calculate_rolling_brier(window=10)
        assert brier is not None
        assert brier < 0.10  # Well predicted (0.8 vs 1.0, 0.2 vs 0.0)

        df = ingestor.to_dataframe()
        assert len(df) == 10
        assert "realized_pnl" in df.columns

    def test_prioritized_experience_replay(self) -> None:
        buffer = PrioritizedExperienceReplayBuffer(capacity=100, alpha=0.6, beta_error=1.0)
        s1 = np.zeros(9, dtype=np.float32)
        s2 = np.ones(9, dtype=np.float32)

        # Push transitions with known TD errors and prediction errors
        # p = |delta_TD| + eps_p + beta * |pred_error|
        buffer.push(s1, action=1, reward=1.0, next_state=s2, done=False, td_error=0.1, prediction_error=0.0)
        buffer.push(s1, action=2, reward=2.0, next_state=s2, done=False, td_error=0.9, prediction_error=0.5)

        assert len(buffer) == 2

        # Sample batch
        states, actions, rewards, next_states, dones, indices, is_weights = buffer.sample(
            batch_size=2,
            beta_is=0.4,
        )
        assert len(states) == 2
        assert len(is_weights) == 2
        assert np.all(is_weights <= 1.0)
        assert np.all(is_weights > 0.0)

        # Priority update
        buffer.update_priorities(indices, td_errors=[0.5, 0.2])

        # Buffer capacity wrap-around
        small_buf = PrioritizedExperienceReplayBuffer(capacity=3)
        for i in range(5):
            small_buf.push(s1, action=i, reward=float(i), next_state=s2, done=False)
        assert len(small_buf) == 3


# =====================================================================
# 3. Incremental Sliding-Window Retrainer Tests
# =====================================================================

class TestIncrementalRetrainer:
    """Verifies recency decay, error sample weighting, and model training."""

    def test_sample_weight_calculation(self) -> None:
        retrainer = IncrementalSlidingWindowRetrainer(tau_decay_days=30.0)
        now = datetime.datetime(2026, 10, 26, 12, 0)
        t_recent = now - datetime.timedelta(days=1)
        t_old = now - datetime.timedelta(days=60)

        timestamps = [t_old, t_recent]
        errors = [0.0, 0.0]

        weights = retrainer.compute_sample_weights(timestamps, errors, current_time=now)
        # Recent sample must have higher weight than 60-day old sample
        assert weights[1] > weights[0]

        # Prediction error boost: w = recency * (1.0 + 2.0 * |error|)
        errors_boosted = [0.0, 0.5]  # recent sample gets 1 + 2*0.5 = 2x multiplier
        weights_boosted = retrainer.compute_sample_weights(timestamps, errors_boosted, current_time=now)
        assert weights_boosted[1] > weights[1]

    def test_retrain_ml_model(self) -> None:
        retrainer = IncrementalSlidingWindowRetrainer(min_samples=30, random_state=42)
        rng = np.random.RandomState(42)

        n = 100
        X = rng.normal(0, 1, size=(n, 32)).astype(np.float32)
        # Strong predictive relationship in feature 0
        y = np.where(X[:, 0] > 0.2, 1, np.where(X[:, 0] < -0.2, -1, 0))

        now = datetime.datetime.utcnow()
        timestamps = [now - datetime.timedelta(hours=i) for i in reversed(range(n))]

        candidate, metrics = retrainer.retrain_ml(X, y, timestamps=timestamps)
        assert candidate.is_fitted
        assert "accuracy" in metrics
        assert "roc_auc" in metrics
        assert "f1_score" in metrics
        assert "brier_score" in metrics
        assert metrics["accuracy"] >= 0.50

    def test_retrain_ml_insufficient_samples_raises(self) -> None:
        retrainer = IncrementalSlidingWindowRetrainer(min_samples=50)
        X = np.zeros((10, 32))
        y = np.zeros(10)
        with pytest.raises(ValueError, match="Insufficient training samples"):
            retrainer.retrain_ml(X, y)

    def test_retrain_rl_policy(self) -> None:
        retrainer = IncrementalSlidingWindowRetrainer()
        agent = PolicyAgent(state_dim=9, action_dim=7)
        buffer = PrioritizedExperienceReplayBuffer(capacity=500)

        s = np.zeros(9, dtype=np.float32)
        s_next = np.ones(9, dtype=np.float32)
        for i in range(70):
            buffer.push(s, action=i % 7, reward=1.0, next_state=s_next, done=False)

        metrics = retrainer.retrain_rl(agent, buffer, num_steps=5, batch_size=32)
        assert "value_loss" in metrics
        assert metrics["value_loss"] >= 0.0


# =====================================================================
# 4. Model Registry & Versioning Tests
# =====================================================================

class TestModelRegistry:
    """Verifies atomic persistence, semantic versioning, and rollback."""

    def test_semver_bumping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            reg_path = os.path.join(tmp_dir, "registry.json")
            reg = ModelRegistry(registry_path=reg_path, models_dir=tmp_dir)

            assert reg.next_version("patch") == "v1.0.0"

            # Register v1.0.0
            meta = ModelMetadata(
                version="v1.0.0",
                timestamp=datetime.datetime.utcnow(),
                trigger_reason="init",
                training_window={},
                status="CHAMPION",
            )
            reg.register_candidate(meta)

            assert reg.next_version("patch") == "v1.0.1"
            assert reg.next_version("minor") == "v1.1.0"
            assert reg.next_version("major") == "v2.0.0"

    def test_atomic_persistence_and_artifact_storage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            reg_path = os.path.join(tmp_dir, "registry.json")
            reg = ModelRegistry(registry_path=reg_path, models_dir=tmp_dir)

            # Create trained ML model and RL agent
            ml_model = HistGradientBoostingModel(max_iter=10)
            X = np.random.randn(20, 32)
            y = np.random.choice([-1, 0, 1], size=20)
            ml_model.fit(X, y)
            rl_agent = PolicyAgent()

            meta = ModelMetadata(
                version="v1.0.0",
                timestamp=datetime.datetime.utcnow(),
                trigger_reason="initial",
                training_window={"bars": 20},
                status="CHAMPION",
            )
            config = {"learning_rate": 0.05, "tau": 30}

            reg.register_candidate(meta, ml_model=ml_model, rl_agent=rl_agent, config=config)
            reg.promote_to_champion("v1.0.0")

            # Check registry file exists and is valid JSON
            assert os.path.exists(reg_path)
            with open(reg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["active_champion_version"] == "v1.0.0"

            # Check artifacts saved in models/v1.0.0/
            assert os.path.exists(meta.artifacts["ml_model_path"])
            assert os.path.exists(meta.artifacts["rl_policy_path"])
            assert os.path.exists(meta.artifacts["config_path"])

            # Reload into a new registry instance to verify persistence
            reg2 = ModelRegistry(registry_path=reg_path, models_dir=tmp_dir)
            assert reg2.active_champion_version == "v1.0.0"
            loaded_ml, loaded_rl, loaded_cfg = reg2.load_artifacts("v1.0.0")
            assert loaded_ml is not None
            assert loaded_rl is not None
            assert loaded_cfg["learning_rate"] == 0.05

    def test_promotion_rejection_and_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            reg_path = os.path.join(tmp_dir, "registry.json")
            reg = ModelRegistry(registry_path=reg_path, models_dir=tmp_dir)

            m1 = ModelMetadata(
                version="v1.0.0",
                timestamp=datetime.datetime.utcnow(),
                trigger_reason="init",
                training_window={},
                status="CHAMPION",
            )
            reg.register_candidate(m1)
            reg.promote_to_champion("v1.0.0")

            # Candidate v1.1.0
            m2 = ModelMetadata(
                version="v1.1.0",
                timestamp=datetime.datetime.utcnow(),
                trigger_reason="retrain",
                training_window={},
                status="CANDIDATE",
            )
            reg.register_candidate(m2)
            assert reg.active_champion_version == "v1.0.0"

            # Promote v1.1.0 to champion -> v1.0.0 becomes ARCHIVED
            reg.promote_to_champion("v1.1.0")
            assert reg.active_champion_version == "v1.1.0"
            assert reg.get_model("v1.0.0").status == "ARCHIVED"
            assert reg.get_model("v1.1.0").status == "CHAMPION"

            # Candidate v1.2.0 rejected
            m3 = ModelMetadata(
                version="v1.2.0",
                timestamp=datetime.datetime.utcnow(),
                trigger_reason="retrain",
                training_window={},
                status="CANDIDATE",
            )
            reg.register_candidate(m3)
            reg.reject_candidate("v1.2.0", reason="Sharpe degradation")
            assert reg.get_model("v1.2.0").status == "REJECTED"

            # Rollback: reverts to previous ARCHIVED champion (v1.0.0)
            rolled = reg.rollback()
            assert rolled is not None
            assert rolled.version == "v1.0.0"
            assert reg.active_champion_version == "v1.0.0"
            assert reg.get_model("v1.1.0").status == "ARCHIVED"


# =====================================================================
# 5. Champion-Challenger Gate Tests
# =====================================================================

class TestChampionChallengerGate:
    """Verifies holdout evaluation criteria: Sharpe, Drawdown, and WinRate thresholds."""

    def test_bootstrap_evaluation(self) -> None:
        gate = ChampionChallengerGate(bootstrap_min_sharpe=0.50, min_win_rate=0.50)

        # Passes bootstrap criteria
        res_pass = gate.evaluate_metrics(
            candidate_metrics={"sharpe_ratio": 1.2, "max_drawdown": 0.06, "win_rate": 0.54},
            champion_metrics=None,
        )
        assert res_pass.promoted
        assert "satisfied all bootstrap thresholds" in res_pass.reason

        # Fails bootstrap (low win rate)
        res_fail = gate.evaluate_metrics(
            candidate_metrics={"sharpe_ratio": 1.2, "max_drawdown": 0.06, "win_rate": 0.45},
            champion_metrics=None,
        )
        assert not res_fail.promoted
        assert "Win rate" in res_fail.reason

    def test_champion_challenger_promotion_and_rejection(self) -> None:
        gate = ChampionChallengerGate()
        champ_metrics = {
            "sharpe_ratio": 1.50,
            "max_drawdown": 0.06,
            "win_rate": 0.55,
        }

        # Candidate equal or better -> Promoted
        cand_good = {
            "sharpe_ratio": 1.55,
            "max_drawdown": 0.058,
            "win_rate": 0.56,
        }
        res_good = gate.evaluate_metrics(cand_good, champ_metrics)
        assert res_good.promoted

        # Candidate Sharpe within 0.10 tolerance -> Promoted
        cand_tolerated = {
            "sharpe_ratio": 1.42,  # 1.50 - 0.10 = 1.40 threshold
            "max_drawdown": 0.065, # <= 0.06 * 1.10 = 0.066
            "win_rate": 0.52,
        }
        res_tolerated = gate.evaluate_metrics(cand_tolerated, champ_metrics)
        assert res_tolerated.promoted

        # Candidate Sharpe drops below tolerance -> Rejected
        cand_bad_sharpe = {
            "sharpe_ratio": 1.35,  # < 1.40
            "max_drawdown": 0.06,
            "win_rate": 0.55,
        }
        res_bad_sharpe = gate.evaluate_metrics(cand_bad_sharpe, champ_metrics)
        assert not res_bad_sharpe.promoted
        assert "Sharpe" in res_bad_sharpe.reason

        # Candidate Drawdown explodes (> 10%) -> Rejected
        cand_bad_dd = {
            "sharpe_ratio": 1.80,
            "max_drawdown": 0.12,  # > 0.10 ceiling
            "win_rate": 0.60,
        }
        res_bad_dd = gate.evaluate_metrics(cand_bad_dd, champ_metrics)
        assert not res_bad_dd.promoted
        assert "Drawdown" in res_bad_dd.reason

        # Candidate Win Rate < 50% -> Rejected
        cand_bad_wr = {
            "sharpe_ratio": 1.60,
            "max_drawdown": 0.05,
            "win_rate": 0.48,  # < 0.50
        }
        res_bad_wr = gate.evaluate_metrics(cand_bad_wr, champ_metrics)
        assert not res_bad_wr.promoted
        assert "Win Rate" in res_bad_wr.reason


# =====================================================================
# 6. Live Readiness Evaluator Tests
# =====================================================================

class TestLiveReadinessEvaluator:
    """Verifies all 8 live readiness deployment gates."""

    @pytest.fixture
    def valid_readiness_metrics(self) -> ReadinessMetrics:
        return ReadinessMetrics(
            backtest_months=12.5,
            paper_days=31.0,
            strategy_cum_return=0.28,
            sp500_cum_return=0.14,
            sharpe_ratio=1.65,
            max_30d_return=0.185,
            max_drawdown=0.062,
            total_trades=74,
            win_rate=0.55,
            profit_factor=1.60,
        )

    def test_all_8_gates_pass(self, valid_readiness_metrics: ReadinessMetrics) -> None:
        evaluator = LiveReadinessEvaluator()
        report = evaluator.evaluate(valid_readiness_metrics)

        assert report.is_live_ready is True
        assert report.summary == "All 8 live deployment criteria satisfied."
        assert len(report.gates) == 8
        for gate_name, gate_data in report.gates.items():
            assert gate_data["passed"] is True, f"{gate_name} failed unexpectedly"

        # Check dictionary serialization
        d = report.to_dict()
        assert d["is_live_ready"] is True
        restored = LiveReadinessReport.from_dict(d)
        assert restored.is_live_ready is True

    @pytest.mark.parametrize(
        "metric_override,failed_gate_name",
        [
            ({"backtest_months": 11.5}, "gate_1_backtest_duration"),
            ({"paper_days": 29.0}, "gate_2_paper_duration"),
            ({"strategy_cum_return": 0.10, "sp500_cum_return": 0.12}, "gate_3_beat_sp500"),
            ({"sharpe_ratio": 0.95}, "gate_4_sharpe_ratio"),
            ({"max_30d_return": 0.12}, "gate_5_monthly_return"),
            ({"max_drawdown": 0.11}, "gate_6_max_drawdown"),
            ({"total_trades": 45}, "gate_7_min_trades"),
            ({"win_rate": 0.48}, "gate_8_execution_quality"),
            ({"profit_factor": 1.25}, "gate_8_execution_quality"),
        ],
    )
    def test_individual_gate_failures(
        self,
        valid_readiness_metrics: ReadinessMetrics,
        metric_override: Dict[str, Any],
        failed_gate_name: str,
    ) -> None:
        evaluator = LiveReadinessEvaluator()
        metrics_dict = {
            "backtest_months": valid_readiness_metrics.backtest_months,
            "paper_days": valid_readiness_metrics.paper_days,
            "strategy_cum_return": valid_readiness_metrics.strategy_cum_return,
            "sp500_cum_return": valid_readiness_metrics.sp500_cum_return,
            "sharpe_ratio": valid_readiness_metrics.sharpe_ratio,
            "max_30d_return": valid_readiness_metrics.max_30d_return,
            "max_drawdown": valid_readiness_metrics.max_drawdown,
            "total_trades": valid_readiness_metrics.total_trades,
            "win_rate": valid_readiness_metrics.win_rate,
            "profit_factor": valid_readiness_metrics.profit_factor,
        }
        metrics_dict.update(metric_override)

        report = evaluator.evaluate(metrics_dict)
        assert report.is_live_ready is False
        assert report.gates[failed_gate_name]["passed"] is False


# =====================================================================
# 7. 30-Day Learning Loop Verification Benchmark Tests
# =====================================================================

class TestLearningLoopBenchmark:
    """Verifies the automated 30-day continuous learning simulation protocol."""

    def test_benchmark_run_and_holdout_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = os.path.join(tmp_dir, "learning_progress_report.json")
            reg_path = os.path.join(tmp_dir, "registry.json")

            registry = ModelRegistry(registry_path=reg_path, models_dir=tmp_dir)
            benchmark = LearningLoopBenchmark(
                registry=registry,
                report_output_path=report_path,
                random_seed=42,
            )

            result = benchmark.run_benchmark(
                simulation_days=30,
                bars_per_day=30,
                retrain_every_days=10,
            )

            assert isinstance(result, BenchmarkResult)
            assert os.path.exists(report_path)

            # Check that holdout performance improved from v0 to vN:
            # Sharpe(vN) > Sharpe(v0), WinRate(vN) >= WinRate(v0), CumulativeReturn(vN) > CumulativeReturn(v0)
            assert result.delta_sharpe > 0.0, f"Sharpe did not improve: delta={result.delta_sharpe}"
            assert result.delta_win_rate >= 0.0, f"Win rate regressed: delta={result.delta_win_rate}"
            assert result.delta_return > 0.0, f"Cumulative return did not improve: delta={result.delta_return}"
            assert result.improvement_verified is True

            # Verify saved JSON report structure
            with open(report_path, "r", encoding="utf-8") as f:
                rep = json.load(f)

            assert rep["improvement_verified"] is True
            assert "metrics_comparison" in rep
            assert "version_history" in rep
            assert len(rep["version_history"]) >= 2


# =====================================================================
# 8. End-to-End Continuous Learning Workflow Integration Test
# =====================================================================

class TestEndToEndLearningWorkflow:
    """Validates complete feedback ingestion, triggering, retraining, and promotion flow."""

    def test_full_learning_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            reg_path = os.path.join(tmp_dir, "registry.json")
            registry = ModelRegistry(registry_path=reg_path, models_dir=tmp_dir)

            trigger_mgr = RetrainingTriggerManager(trade_count_threshold=5, cooldown_minutes=0)
            ingestor = TradeFeedbackIngestor()
            retrainer = IncrementalSlidingWindowRetrainer(min_samples=10, random_state=42)
            gate = ChampionChallengerGate(bootstrap_min_sharpe=0.1)

            # 1. Register baseline model v1.0.0
            rng = np.random.RandomState(42)
            X = rng.randn(40, 32).astype(np.float32)
            y = np.where(X[:, 0] > 0, 1, -1)
            v0_model = HistGradientBoostingModel(max_iter=10, random_state=42).fit(X[:20], y[:20])

            v0_meta = ModelMetadata(
                version="v1.0.0",
                timestamp=datetime.datetime.utcnow(),
                trigger_reason="bootstrap",
                training_window={"bars": 20},
                validation_metrics={"sharpe_ratio": 1.0, "max_drawdown": 0.05, "win_rate": 0.52},
                status="CHAMPION",
            )
            registry.register_candidate(v0_meta, ml_model=v0_model)
            registry.promote_to_champion("v1.0.0")

            # 2. Simulate 5 incoming trade results
            now = datetime.datetime.utcnow()
            for i in range(5):
                rec = TradeFeedbackRecord(
                    trade_id=f"TRD-{i}",
                    symbol="BTC/USDT",
                    direction="LONG",
                    entry_time=now - datetime.timedelta(minutes=(5 - i) * 10),
                    exit_time=now - datetime.timedelta(minutes=(4 - i) * 10),
                    entry_price=65000.0,
                    exit_price=65500.0,
                    quantity=0.01,
                    realized_pnl=5.0,
                    realized_return_pct=0.0076,
                    fees_paid=0.1,
                    expected_slippage=0.01,
                    actual_slippage=0.01,
                    predicted_p_up=0.7,
                    predicted_confidence=0.6,
                    prediction_error=0.1,
                    holding_bars=2,
                    model_version="v1.0.0",
                )
                ingestor.ingest(rec)
                trigger_mgr.record_trade()

            # 3. Check trigger -> trade count reached
            should_retrain, reason, _ = trigger_mgr.check_triggers(current_time=now)
            assert should_retrain
            assert reason == TriggerReason.TRADE_COUNT_MILESTONE

            # 4. Retrain candidate
            candidate_model, ml_metrics = retrainer.retrain_ml(
                X=X,
                y=y,
                base_model=HistGradientBoostingModel(max_iter=20, random_state=42),
            )

            cand_meta = ModelMetadata(
                version=registry.next_version("patch"),
                timestamp=now,
                trigger_reason=reason.value,
                training_window={"bars": 40},
                ml_metrics=ml_metrics,
                validation_metrics={"sharpe_ratio": 1.25, "max_drawdown": 0.045, "win_rate": 0.58},
                status="CANDIDATE",
            )
            registry.register_candidate(cand_meta, ml_model=candidate_model)

            # 5. Gate validation & promotion
            champ = registry.get_active_champion()
            res = gate.evaluate_and_promote(cand_meta, champ, registry)
            assert res.promoted

            # 6. Verify new champion active in registry
            new_champ = registry.get_active_champion()
            assert new_champ is not None
            assert new_champ.version == "v1.0.1"
            assert new_champ.status == "CHAMPION"
            assert registry.get_model("v1.0.0").status == "ARCHIVED"
