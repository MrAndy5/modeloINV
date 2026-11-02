"""Adversarial and Stress Test Suite for Milestone 4 (Learning Loop, Model Registry, Champion Gate & Live Readiness).

Targets:
1. ModelRegistry: Concurrent saves, corrupted registry.json recovery, invalid semantic versions, missing artifact paths, rollback lifecycle.
2. ChampionChallengerGate: Marginal candidates (exact -0.10 Sharpe diff, exact 10.0% drawdown, exact 0.50 win rate), cold-start bootstrapping, consecutive promotions/rejections.
3. LiveReadinessEvaluator: Boundary testing on all 8 gates (backtest 11.9 vs 12.0 mo, paper 29.9 vs 30.0 days, Sharpe 0.999 vs 1.000 vs 1.001, max 30d return 14.99% vs 15.00%, drawdown 10.01% vs 10.00%, trades 49 vs 50, win rate 49.9% vs 50.0%, profit factor 1.29 vs 1.30).
4. LearningLoopBenchmark: Default arguments execution, holdout metrics progression, and JSON report schema verification.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import datetime
import json
import os
import tempfile
from typing import Any, Dict, List

import numpy as np
import pytest

from modeloinv.learning.benchmark import BenchmarkResult, LearningLoopBenchmark
from modeloinv.learning.champion import ChampionChallengerGate, GateResult
from modeloinv.learning.readiness import (
    LiveReadinessEvaluator,
    LiveReadinessReport,
    ReadinessMetrics,
)
from modeloinv.learning.registry import ModelMetadata, ModelRegistry


# =====================================================================
# 1. ModelRegistry Adversarial Tests
# =====================================================================

class TestModelRegistryAdversarial:
    """Stress-tests atomic persistence, corrupted storage recovery, concurrency, and semver parsing."""

    def test_corrupted_json_recovery_truncated(self) -> None:
        """Verifies registry recovers cleanly when registry.json is truncated mid-write."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            reg_path = os.path.join(tmp_dir, "registry.json")
            # Write truncated / malformed JSON
            with open(reg_path, "w", encoding="utf-8") as f:
                f.write('{"active_champion_version": "v1.0.0", "models": [{"version": "v1.0.0"')

            registry = ModelRegistry(registry_path=reg_path, models_dir=tmp_dir)
            assert registry.models == []
            assert registry.active_champion_version is None

            # Can register new model cleanly after recovery
            meta = ModelMetadata(
                version="v1.0.0",
                timestamp=datetime.datetime.utcnow(),
                trigger_reason="manual",
                training_window={},
            )
            registry.register_candidate(meta)
            assert len(registry.models) == 1
            assert registry.models[0].version == "v1.0.0"

    def test_corrupted_json_recovery_invalid_structure(self) -> None:
        """Verifies registry recovers when registry.json contains valid JSON but invalid schema."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            reg_path = os.path.join(tmp_dir, "registry.json")
            with open(reg_path, "w", encoding="utf-8") as f:
                f.write('["not", "a", "dictionary"]')

            registry = ModelRegistry(registry_path=reg_path, models_dir=tmp_dir)
            assert registry.models == []
            assert registry.active_champion_version is None

    def test_non_existent_file_initialization(self) -> None:
        """Verifies registry initializes cleanly when file does not exist yet."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            reg_path = os.path.join(tmp_dir, "sub", "dir", "registry.json")
            registry = ModelRegistry(registry_path=reg_path, models_dir=tmp_dir)
            assert registry.models == []
            assert registry.active_champion_version is None
            assert registry.get_latest_version() is None
            assert registry.next_version() == "v1.0.0"

    def test_invalid_semantic_version_formats(self) -> None:
        """Verifies parse_semver strictly rejects invalid semantic versions."""
        invalid_versions = [
            "1.0",
            "v1.2",
            "v1.2.3.4",
            "version1.0.0",
            "1.0.0-rc1",
            "v-1.0.0",
            "",
            "   ",
            "v1.a.0",
        ]
        for ver in invalid_versions:
            with pytest.raises(ValueError, match="Invalid semantic version string"):
                ModelRegistry.parse_semver(ver)

    def test_invalid_bump_type_raises(self) -> None:
        """Verifies next_version raises ValueError on unrecognized bump types once models exist."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            reg_path = os.path.join(tmp_dir, "registry.json")
            registry = ModelRegistry(registry_path=reg_path, models_dir=tmp_dir)
            meta = ModelMetadata(
                version="v1.0.0",
                timestamp=datetime.datetime.utcnow(),
                trigger_reason="init",
                training_window={},
            )
            registry.register_candidate(meta)

            with pytest.raises(ValueError, match="Unknown bump_type"):
                registry.next_version(bump_type="invalid_type")

    def test_missing_artifact_files_handled_gracefully(self) -> None:
        """Verifies load_artifacts returns None for missing artifacts rather than crashing with unhandled error."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            reg_path = os.path.join(tmp_dir, "registry.json")
            registry = ModelRegistry(registry_path=reg_path, models_dir=tmp_dir)

            meta = ModelMetadata(
                version="v1.0.0",
                timestamp=datetime.datetime.utcnow(),
                trigger_reason="test",
                training_window={},
                artifacts={
                    "ml_model_path": os.path.join(tmp_dir, "non_existent.joblib"),
                    "rl_policy_path": os.path.join(tmp_dir, "non_existent.pt"),
                    "config_path": os.path.join(tmp_dir, "non_existent.json"),
                },
            )
            registry.register_candidate(meta)

            ml_m, rl_a, cfg = registry.load_artifacts("v1.0.0")
            assert ml_m is None
            assert rl_a is None
            assert cfg is None

    def test_load_artifacts_nonexistent_version_raises(self) -> None:
        """Verifies load_artifacts raises ValueError when requested version does not exist."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            reg_path = os.path.join(tmp_dir, "registry.json")
            registry = ModelRegistry(registry_path=reg_path, models_dir=tmp_dir)
            with pytest.raises(ValueError, match="not found in registry"):
                registry.load_artifacts("v9.9.9")

    def test_idempotent_candidate_registration(self) -> None:
        """Verifies registering the same version twice updates in-place without duplicating."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            reg_path = os.path.join(tmp_dir, "registry.json")
            registry = ModelRegistry(registry_path=reg_path, models_dir=tmp_dir)

            meta1 = ModelMetadata(
                version="v1.0.0",
                timestamp=datetime.datetime.utcnow(),
                trigger_reason="first_run",
                training_window={},
                description="Original description",
            )
            registry.register_candidate(meta1)
            assert len(registry.models) == 1

            meta2 = ModelMetadata(
                version="v1.0.0",
                timestamp=datetime.datetime.utcnow(),
                trigger_reason="first_run",
                training_window={},
                description="Updated description",
            )
            registry.register_candidate(meta2)
            assert len(registry.models) == 1
            assert registry.models[0].description == "Updated description"

    def test_promote_and_reject_nonexistent_version(self) -> None:
        """Verifies promote and reject return False safely when version does not exist."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            reg_path = os.path.join(tmp_dir, "registry.json")
            registry = ModelRegistry(registry_path=reg_path, models_dir=tmp_dir)
            assert registry.promote_to_champion("v9.9.9") is False
            assert registry.reject_candidate("v9.9.9") is False

    def test_rollback_edge_cases(self) -> None:
        """Verifies rollback returns None when no valid rollback target exists."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            reg_path = os.path.join(tmp_dir, "registry.json")
            registry = ModelRegistry(registry_path=reg_path, models_dir=tmp_dir)

            # Rollback with 0 models
            assert registry.rollback() is None

            # Single active champion, no archived models
            meta = ModelMetadata(
                version="v1.0.0",
                timestamp=datetime.datetime.utcnow(),
                trigger_reason="init",
                training_window={},
                status="CHAMPION",
            )
            registry.register_candidate(meta)
            registry.promote_to_champion("v1.0.0")

            assert registry.rollback() is None
            assert registry.rollback("v2.0.0") is None

    def test_concurrent_writes_thread_safety(self) -> None:
        """Adversarial stress test: verifies whether ModelRegistry handles concurrent multithreaded registrations.

        Under Windows, concurrent os.replace operations on the same target file
        raise PermissionError: [WinError 5] unless serialized via internal synchronization.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            reg_path = os.path.join(tmp_dir, "registry.json")
            registry = ModelRegistry(registry_path=reg_path, models_dir=tmp_dir)

            def register_worker(i: int) -> None:
                meta = ModelMetadata(
                    version=f"v1.{i}.0",
                    timestamp=datetime.datetime.utcnow(),
                    trigger_reason="concurrent_test",
                    training_window={},
                )
                registry.register_candidate(meta)

            errors: List[Exception] = []
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(register_worker, i) for i in range(20)]
                for fut in futures:
                    try:
                        fut.result()
                    except Exception as e:
                        errors.append(e)

            # Check if concurrency errors occurred
            assert len(errors) == 0, (
                f"Concurrent model registration raised {len(errors)} exception(s): "
                f"{[type(e).__name__ + ': ' + str(e) for e in errors]}"
            )


# =====================================================================
# 2. ChampionChallengerGate Adversarial Tests
# =====================================================================

class TestChampionChallengerGateAdversarial:
    """Stress-tests boundary conditions, floating-point precision, and promotion/rejection state transitions."""

    def test_marginal_candidate_sharpe_float_boundary(self) -> None:
        """Adversarial test: verifies candidate with Sharpe exactly equal to champion - 0.10.

        When champion Sharpe is 0.80 and candidate is 0.70:
        0.80 - 0.10 in IEEE 754 arithmetic produces 0.7000000000000001.
        Without epsilon tolerance (or rounding), 0.70 >= 0.7000000000000001 evaluates to False,
        erroneously rejecting a valid candidate with message:
        'Candidate Sharpe 0.70 < requirement 0.70 (Champion 0.80)'.
        """
        gate = ChampionChallengerGate(sharpe_tolerance=0.10)
        cand_metrics = {
            "sharpe_ratio": 0.70,
            "max_drawdown": 0.05,
            "win_rate": 0.55,
        }
        champ_metrics = {
            "sharpe_ratio": 0.80,
            "max_drawdown": 0.05,
            "win_rate": 0.55,
        }
        res = gate.evaluate_metrics(cand_metrics, champ_metrics, candidate_version="v1.1.0", champion_version="v1.0.0")
        assert res.promoted, f"Candidate was rejected at exact -0.10 Sharpe boundary: {res.reason}"

    def test_marginal_candidate_drawdown_boundary(self) -> None:
        """Verifies candidate with max_drawdown exactly equal to 0.10 ceiling passes."""
        gate = ChampionChallengerGate(max_drawdown_ceiling=0.10)
        cand_metrics = {
            "sharpe_ratio": 1.20,
            "max_drawdown": 0.10,
            "win_rate": 0.52,
        }
        champ_metrics = {
            "sharpe_ratio": 1.20,
            "max_drawdown": 0.10,
            "win_rate": 0.52,
        }
        res = gate.evaluate_metrics(cand_metrics, champ_metrics)
        assert res.promoted, f"Candidate was rejected at exact 10.0% drawdown ceiling: {res.reason}"

    def test_marginal_candidate_drawdown_expansion_boundary(self) -> None:
        """Verifies candidate at exactly champ_dd * 1.10 (e.g. 0.05 -> 0.055) passes."""
        gate = ChampionChallengerGate(drawdown_expansion_tolerance=1.10)
        cand_metrics = {
            "sharpe_ratio": 1.20,
            "max_drawdown": 0.055,
            "win_rate": 0.52,
        }
        champ_metrics = {
            "sharpe_ratio": 1.20,
            "max_drawdown": 0.05,
            "win_rate": 0.52,
        }
        res = gate.evaluate_metrics(cand_metrics, champ_metrics)
        assert res.promoted, f"Candidate was rejected at exact 1.10x drawdown expansion boundary: {res.reason}"

    def test_marginal_candidate_win_rate_boundary(self) -> None:
        """Verifies candidate with win_rate exactly equal to 0.50 (50.0%) passes."""
        gate = ChampionChallengerGate(min_win_rate=0.50)
        cand_metrics = {
            "sharpe_ratio": 1.20,
            "max_drawdown": 0.05,
            "win_rate": 0.50,
        }
        champ_metrics = {
            "sharpe_ratio": 1.20,
            "max_drawdown": 0.05,
            "win_rate": 0.52,
        }
        res = gate.evaluate_metrics(cand_metrics, champ_metrics)
        assert res.promoted, f"Candidate was rejected at exact 50.0% win rate boundary: {res.reason}"

    def test_cold_start_bootstrap_boundaries(self) -> None:
        """Verifies bootstrap evaluation behavior at exact thresholds."""
        gate = ChampionChallengerGate(
            bootstrap_min_sharpe=0.50,
            max_drawdown_ceiling=0.10,
            min_win_rate=0.50,
        )

        # Exact threshold passes
        res_exact = gate.evaluate_metrics(
            candidate_metrics={"sharpe_ratio": 0.50, "max_drawdown": 0.10, "win_rate": 0.50},
            champion_metrics=None,
        )
        assert res_exact.promoted

        # Marginally below Sharpe fails
        res_fail_sharpe = gate.evaluate_metrics(
            candidate_metrics={"sharpe_ratio": 0.499, "max_drawdown": 0.10, "win_rate": 0.50},
            champion_metrics=None,
        )
        assert not res_fail_sharpe.promoted
        assert "Sharpe" in res_fail_sharpe.reason

        # Marginally above drawdown fails
        res_fail_dd = gate.evaluate_metrics(
            candidate_metrics={"sharpe_ratio": 0.50, "max_drawdown": 0.1001, "win_rate": 0.50},
            champion_metrics=None,
        )
        assert not res_fail_dd.promoted
        assert "Drawdown" in res_fail_dd.reason

        # Marginally below win rate fails
        res_fail_wr = gate.evaluate_metrics(
            candidate_metrics={"sharpe_ratio": 0.50, "max_drawdown": 0.10, "win_rate": 0.499},
            champion_metrics=None,
        )
        assert not res_fail_wr.promoted
        assert "Win rate" in res_fail_wr.reason

    def test_consecutive_promotions_and_rejections_lifecycle(self) -> None:
        """Verifies state consistency across multi-stage promotion, rejection, and rollback."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            reg_path = os.path.join(tmp_dir, "registry.json")
            reg = ModelRegistry(registry_path=reg_path, models_dir=tmp_dir)
            gate = ChampionChallengerGate(sharpe_tolerance=0.10, min_win_rate=0.50)

            # 1. Bootstrap v1.0.0
            m1 = ModelMetadata(
                version="v1.0.0",
                timestamp=datetime.datetime.utcnow(),
                trigger_reason="init",
                training_window={},
                validation_metrics={"sharpe_ratio": 1.0, "max_drawdown": 0.05, "win_rate": 0.55},
                status="CANDIDATE",
            )
            reg.register_candidate(m1)
            res1 = gate.evaluate_and_promote(m1, None, reg)
            assert res1.promoted
            assert reg.active_champion_version == "v1.0.0"
            assert reg.get_model("v1.0.0").status == "CHAMPION"

            # 2. Reject v1.1.0 (win rate regression)
            m2 = ModelMetadata(
                version="v1.1.0",
                timestamp=datetime.datetime.utcnow(),
                trigger_reason="retrain",
                training_window={},
                validation_metrics={"sharpe_ratio": 1.1, "max_drawdown": 0.05, "win_rate": 0.45},
                status="CANDIDATE",
            )
            reg.register_candidate(m2)
            champ = reg.get_active_champion()
            res2 = gate.evaluate_and_promote(m2, champ, reg)
            assert not res2.promoted
            assert reg.active_champion_version == "v1.0.0"
            assert reg.get_model("v1.1.0").status == "REJECTED"

            # 3. Promote v1.2.0 (superior candidate)
            m3 = ModelMetadata(
                version="v1.2.0",
                timestamp=datetime.datetime.utcnow(),
                trigger_reason="retrain",
                training_window={},
                validation_metrics={"sharpe_ratio": 1.3, "max_drawdown": 0.04, "win_rate": 0.60},
                status="CANDIDATE",
            )
            reg.register_candidate(m3)
            champ = reg.get_active_champion()
            res3 = gate.evaluate_and_promote(m3, champ, reg)
            assert res3.promoted
            assert reg.active_champion_version == "v1.2.0"
            assert reg.get_model("v1.0.0").status == "ARCHIVED"
            assert reg.get_model("v1.2.0").status == "CHAMPION"

            # 4. Rollback to v1.0.0
            rolled = reg.rollback("v1.0.0")
            assert rolled is not None
            assert rolled.version == "v1.0.0"
            assert reg.active_champion_version == "v1.0.0"
            assert reg.get_model("v1.0.0").status == "CHAMPION"
            assert reg.get_model("v1.2.0").status == "ARCHIVED"


# =====================================================================
# 3. LiveReadinessEvaluator Boundary Tests (All 8 Gates)
# =====================================================================

class TestLiveReadinessEvaluatorAdversarial:
    """Stress-tests the exact boundary conditions of all 8 live readiness gates."""

    @pytest.fixture
    def evaluator(self) -> LiveReadinessEvaluator:
        return LiveReadinessEvaluator()

    @pytest.fixture
    def base_passing_metrics(self) -> Dict[str, Any]:
        return {
            "backtest_months": 12.0,
            "paper_days": 30.0,
            "strategy_cum_return": 0.20,
            "sp500_cum_return": 0.10,
            "sharpe_ratio": 1.50,
            "max_30d_return": 0.16,
            "max_drawdown": 0.08,
            "total_trades": 60,
            "win_rate": 0.55,
            "profit_factor": 1.40,
        }

    def test_all_8_gates_pass_at_exact_boundaries(self, evaluator: LiveReadinessEvaluator) -> None:
        """Verifies that when metrics are set exactly at passing thresholds, all 8 gates pass."""
        exact_metrics = {
            "backtest_months": 12.0,
            "paper_days": 30.0,
            "strategy_cum_return": 0.1001,
            "sp500_cum_return": 0.1000,
            "sharpe_ratio": 1.001,
            "max_30d_return": 0.1500,
            "max_drawdown": 0.1000,
            "total_trades": 50,
            "win_rate": 0.5000,
            "profit_factor": 1.30,
        }
        report = evaluator.evaluate(exact_metrics)
        assert report.is_live_ready is True
        for gate_name, gate_info in report.gates.items():
            assert gate_info["passed"] is True, f"{gate_name} failed unexpectedly at boundary"

    @pytest.mark.parametrize(
        "metric_key, fail_val, pass_val, gate_id",
        [
            ("backtest_months", 11.9, 12.0, "gate_1_backtest_duration"),
            ("paper_days", 29.9, 30.0, "gate_2_paper_duration"),
            ("sharpe_ratio", 0.999, 1.001, "gate_4_sharpe_ratio"),
            ("max_30d_return", 0.1499, 0.1501, "gate_5_monthly_return"),
            ("max_drawdown", 0.1001, 0.0999, "gate_6_max_drawdown"),
            ("total_trades", 49, 50, "gate_7_min_trades"),
            ("win_rate", 0.499, 0.501, "gate_8_execution_quality"),
            ("profit_factor", 1.29, 1.31, "gate_8_execution_quality"),
        ],
    )
    def test_gate_strict_boundary_pairs(
        self,
        evaluator: LiveReadinessEvaluator,
        base_passing_metrics: Dict[str, Any],
        metric_key: str,
        fail_val: Any,
        pass_val: Any,
        gate_id: str,
    ) -> None:
        """Tests marginal failure value and marginal passing value for each gate."""
        # Test marginal fail
        m_fail = dict(base_passing_metrics)
        m_fail[metric_key] = fail_val
        rep_fail = evaluator.evaluate(m_fail)
        assert rep_fail.is_live_ready is False
        assert rep_fail.gates[gate_id]["passed"] is False

        # Test marginal pass
        m_pass = dict(base_passing_metrics)
        m_pass[metric_key] = pass_val
        rep_pass = evaluator.evaluate(m_pass)
        assert rep_pass.is_live_ready is True
        assert rep_pass.gates[gate_id]["passed"] is True

    def test_gate_3_benchmark_equality_fails(
        self,
        evaluator: LiveReadinessEvaluator,
        base_passing_metrics: Dict[str, Any],
    ) -> None:
        """Verifies Gate 3 fails when strategy return is exactly equal to S&P 500 (must strictly beat)."""
        m = dict(base_passing_metrics)
        m["strategy_cum_return"] = 0.1200
        m["sp500_cum_return"] = 0.1200
        rep = evaluator.evaluate(m)
        assert rep.is_live_ready is False
        assert rep.gates["gate_3_beat_sp500"]["passed"] is False

    def test_gate_4_sharpe_equality_to_1_fails(
        self,
        evaluator: LiveReadinessEvaluator,
        base_passing_metrics: Dict[str, Any],
    ) -> None:
        """Verifies Gate 4 fails when Sharpe ratio is exactly 1.000 (must be strictly > 1.0)."""
        m = dict(base_passing_metrics)
        m["sharpe_ratio"] = 1.000
        rep = evaluator.evaluate(m)
        assert rep.is_live_ready is False
        assert rep.gates["gate_4_sharpe_ratio"]["passed"] is False

    def test_missing_keys_in_dictionary_graceful_defaults(
        self,
        evaluator: LiveReadinessEvaluator,
    ) -> None:
        """Verifies evaluator handles empty or incomplete dictionary gracefully without raising KeyError."""
        rep = evaluator.evaluate({})
        assert rep.is_live_ready is False
        assert len(rep.gates) == 8
        # Check that missing keys caused gates to fail
        assert rep.gates["gate_1_backtest_duration"]["passed"] is False
        assert rep.gates["gate_7_min_trades"]["passed"] is False

    def test_readiness_report_serialization_roundtrip(
        self,
        evaluator: LiveReadinessEvaluator,
        base_passing_metrics: Dict[str, Any],
    ) -> None:
        """Verifies LiveReadinessReport preserves fidelity across JSON / dict serialization."""
        rep = evaluator.evaluate(base_passing_metrics)
        d = rep.to_dict()
        rep_restored = LiveReadinessReport.from_dict(d)
        assert rep_restored.is_live_ready == rep.is_live_ready
        assert rep_restored.summary == rep.summary
        assert len(rep_restored.gates) == 8


# =====================================================================
# 4. LearningLoopBenchmark Adversarial Tests
# =====================================================================

class TestLearningLoopBenchmarkAdversarial:
    """Stress-tests the 30-day learning loop benchmark protocol and artifact generation."""

    def test_default_benchmark_arguments_improvement_verified(self) -> None:
        """Adversarial test: verifies whether running benchmark with default parameters succeeds.

        If run_benchmark() defaults to bars_per_day=40, it fails its own
        improvement_verified assertion out-of-the-box (Sharpe delta: -0.011).
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            rep_path = os.path.join(tmp_dir, "report.json")
            reg_path = os.path.join(tmp_dir, "registry.json")
            reg = ModelRegistry(registry_path=reg_path, models_dir=tmp_dir)
            benchmark = LearningLoopBenchmark(
                registry=reg,
                report_output_path=rep_path,
                random_seed=42,
            )

            # Call with default arguments
            res = benchmark.run_benchmark()

            assert os.path.exists(rep_path), "Benchmark failed to generate report JSON file."
            assert res.improvement_verified is True, (
                f"Default benchmark run failed verification: delta_sharpe={res.delta_sharpe}, "
                f"delta_win_rate={res.delta_win_rate}, delta_return={res.delta_return}"
            )

    def test_benchmark_report_json_schema(self) -> None:
        """Verifies benchmark generates a schema-compliant JSON report even when explicit parameters are provided."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            rep_path = os.path.join(tmp_dir, "report.json")
            reg_path = os.path.join(tmp_dir, "registry.json")
            reg = ModelRegistry(registry_path=reg_path, models_dir=tmp_dir)
            benchmark = LearningLoopBenchmark(
                registry=reg,
                report_output_path=rep_path,
                random_seed=42,
            )

            res = benchmark.run_benchmark(simulation_days=30, bars_per_day=30, retrain_every_days=10)

            with open(rep_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            assert "evaluation_timestamp" in data
            assert "improvement_verified" in data
            assert "v0_version" in data
            assert "vn_version" in data
            assert "holdout_samples" in data
            assert "metrics_comparison" in data
            assert "version_history" in data
            assert len(data["version_history"]) >= 2
