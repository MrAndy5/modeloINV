"""Champion-Challenger Promotion Gate for modeloINV.

Provides:
- GateResult: Detailed outcome of candidate vs champion holdout evaluation.
- ChampionChallengerGate: Out-of-sample promotion validator enforcing:
    1. Sharpe(Candidate) >= Sharpe(Champion) - 0.10
    2. MaxDrawdown(Candidate) <= min(MaxDrawdown(Champion) * 1.10, 0.10)
    3. WinRate(Candidate) >= 0.50
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from modeloinv.learning.registry import ModelMetadata, ModelRegistry


@dataclass
class GateResult:
    """Outcome of Champion-Challenger holdout validation."""
    promoted: bool
    reason: str
    candidate_version: str
    champion_version: Optional[str]
    checks: Dict[str, Dict[str, Any]]
    candidate_metrics: Dict[str, float]
    champion_metrics: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ChampionChallengerGate:
    """Evaluates retrained candidate models against the active champion on holdout data."""

    def __init__(
        self,
        sharpe_tolerance: float = 0.10,
        drawdown_expansion_tolerance: float = 1.10,
        max_drawdown_ceiling: float = 0.10,
        min_win_rate: float = 0.50,
        bootstrap_min_sharpe: float = 0.50,
    ) -> None:
        self.sharpe_tolerance = sharpe_tolerance
        self.drawdown_expansion_tolerance = drawdown_expansion_tolerance
        self.max_drawdown_ceiling = max_drawdown_ceiling
        self.min_win_rate = min_win_rate
        self.bootstrap_min_sharpe = bootstrap_min_sharpe

    def evaluate_metrics(
        self,
        candidate_metrics: Dict[str, float],
        champion_metrics: Optional[Dict[str, float]] = None,
        candidate_version: str = "candidate",
        champion_version: Optional[str] = None,
    ) -> GateResult:
        """Evaluates candidate performance metrics against champion metrics or bootstrap thresholds."""
        cand_sharpe = float(candidate_metrics.get("sharpe_ratio", 0.0))
        cand_dd = float(candidate_metrics.get("max_drawdown", 1.0))
        cand_wr = float(candidate_metrics.get("win_rate", 0.0))

        checks: Dict[str, Dict[str, Any]] = {}
        failure_reasons: List[str] = []

        if champion_metrics is None:
            # Bootstrap mode: no active champion exists yet
            sharpe_passed = cand_sharpe >= self.bootstrap_min_sharpe
            checks["sharpe_check"] = {
                "target": f">= {self.bootstrap_min_sharpe:.2f} (bootstrap)",
                "actual": cand_sharpe,
                "passed": sharpe_passed,
            }
            if not sharpe_passed:
                failure_reasons.append(f"Sharpe {cand_sharpe:.2f} < bootstrap threshold {self.bootstrap_min_sharpe:.2f}")

            dd_passed = cand_dd <= self.max_drawdown_ceiling
            checks["drawdown_check"] = {
                "target": f"<= {self.max_drawdown_ceiling:.2%}",
                "actual": cand_dd,
                "passed": dd_passed,
            }
            if not dd_passed:
                failure_reasons.append(f"Drawdown {cand_dd:.2%} > ceiling {self.max_drawdown_ceiling:.2%}")

            wr_passed = cand_wr >= self.min_win_rate
            checks["win_rate_check"] = {
                "target": f">= {self.min_win_rate:.2%}",
                "actual": cand_wr,
                "passed": wr_passed,
            }
            if not wr_passed:
                failure_reasons.append(f"Win rate {cand_wr:.2%} < threshold {self.min_win_rate:.2%}")

            promoted = len(failure_reasons) == 0
            reason = "Candidate satisfied all bootstrap thresholds." if promoted else f"Candidate rejected: {'; '.join(failure_reasons)}."

            return GateResult(
                promoted=promoted,
                reason=reason,
                candidate_version=candidate_version,
                champion_version=None,
                checks=checks,
                candidate_metrics=candidate_metrics,
                champion_metrics=None,
            )

        # Standard Champion-Challenger validation
        champ_sharpe = float(champion_metrics.get("sharpe_ratio", 0.0))
        champ_dd = float(champion_metrics.get("max_drawdown", 0.05))

        # Criterion 1: Sharpe(Candidate) >= Sharpe(Champion) - 0.10
        sharpe_req = champ_sharpe - self.sharpe_tolerance
        sharpe_passed = cand_sharpe >= (sharpe_req - 1e-6)
        checks["sharpe_check"] = {
            "target": f">= {sharpe_req:.2f} (Champion {champ_sharpe:.2f} - {self.sharpe_tolerance:.2f})",
            "actual": cand_sharpe,
            "passed": sharpe_passed,
        }
        if not sharpe_passed:
            failure_reasons.append(
                f"Candidate Sharpe {cand_sharpe:.2f} < requirement {sharpe_req:.2f} (Champion {champ_sharpe:.2f})"
            )

        # Criterion 2: MaxDrawdown(Candidate) <= min(MaxDrawdown(Champion) * 1.10, 0.10)
        dd_req = min(champ_dd * self.drawdown_expansion_tolerance, self.max_drawdown_ceiling)
        dd_passed = cand_dd <= (dd_req + 1e-6)
        checks["drawdown_check"] = {
            "target": f"<= {dd_req:.2%} (min(Champion {champ_dd:.2%} * {self.drawdown_expansion_tolerance:.2f}, {self.max_drawdown_ceiling:.2%}))",
            "actual": cand_dd,
            "passed": dd_passed,
        }
        if not dd_passed:
            failure_reasons.append(
                f"Candidate Max Drawdown {cand_dd:.2%} > limit {dd_req:.2%}"
            )

        # Criterion 3: WinRate(Candidate) >= 0.50
        wr_passed = cand_wr >= self.min_win_rate
        checks["win_rate_check"] = {
            "target": f">= {self.min_win_rate:.2%}",
            "actual": cand_wr,
            "passed": wr_passed,
        }
        if not wr_passed:
            failure_reasons.append(
                f"Candidate Win Rate {cand_wr:.2%} < threshold {self.min_win_rate:.2%}"
            )

        promoted = len(failure_reasons) == 0
        if promoted:
            reason = f"Candidate {candidate_version} outperformed or preserved champion risk-adjusted targets; promoted to CHAMPION."
        else:
            reason = f"Candidate {candidate_version} rejected: {'; '.join(failure_reasons)}."

        return GateResult(
            promoted=promoted,
            reason=reason,
            candidate_version=candidate_version,
            champion_version=champion_version,
            checks=checks,
            candidate_metrics=candidate_metrics,
            champion_metrics=champion_metrics,
        )

    def evaluate_and_promote(
        self,
        candidate_meta: ModelMetadata,
        champion_meta: Optional[ModelMetadata],
        registry: ModelRegistry,
    ) -> GateResult:
        """Evaluates candidate against champion and atomically updates registry status."""
        champ_metrics = champion_meta.validation_metrics if champion_meta else None
        champ_version = champion_meta.version if champion_meta else None

        result = self.evaluate_metrics(
            candidate_metrics=candidate_meta.validation_metrics,
            champion_metrics=champ_metrics,
            candidate_version=candidate_meta.version,
            champion_version=champ_version,
        )

        if result.promoted:
            registry.promote_to_champion(candidate_meta.version)
        else:
            registry.reject_candidate(candidate_meta.version, reason=result.reason)

        return result
