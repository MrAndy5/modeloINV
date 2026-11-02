"""Deterministic 8-Gate Live Readiness Engine for modeloINV.

Evaluates system eligibility to transition from paper trading to live broker execution:
    1. Backtest duration >= 12 months
    2. Paper trading duration >= 30 days
    3. Cumulative strategy return > S&P 500 benchmark return
    4. Annualized Sharpe ratio > 1.0
    5. Maximum 30-day window return >= 15.0%
    6. Maximum drawdown <= 10.0%
    7. Total executed trades >= 50
    8. Win rate >= 50% and Profit Factor >= 1.30
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import datetime
from typing import Any, Dict, List, Optional, Union


@dataclass
class ReadinessMetrics:
    """Quantitative performance telemetry required for live readiness gating."""
    backtest_months: float
    paper_days: float
    strategy_cum_return: float
    sp500_cum_return: float
    sharpe_ratio: float
    max_30d_return: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    profit_factor: float


@dataclass
class LiveReadinessReport:
    """Structured report assessing readiness across all 8 deployment gates."""
    is_live_ready: bool
    evaluation_timestamp: datetime.datetime
    summary: str
    gates: Dict[str, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes report to dictionary matching the canonical specification schema."""
        return {
            "is_live_ready": self.is_live_ready,
            "evaluation_timestamp": self.evaluation_timestamp.isoformat(),
            "summary": self.summary,
            "gates": self.gates,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LiveReadinessReport:
        """Constructs LiveReadinessReport from dictionary."""
        ts = data.get("evaluation_timestamp") or data.get("timestamp") or datetime.datetime.utcnow()
        if isinstance(ts, str):
            ts = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))

        return cls(
            is_live_ready=bool(data["is_live_ready"]),
            evaluation_timestamp=ts,
            summary=str(data.get("summary", "")),
            gates=dict(data.get("gates", {})),
        )


class LiveReadinessEvaluator:
    """Deterministic evaluator verifying all 8 quantitative live deployment criteria."""

    def __init__(
        self,
        min_backtest_months: float = 12.0,
        min_paper_days: float = 30.0,
        min_sharpe_ratio: float = 1.0,
        min_max_30d_return: float = 0.15,
        max_drawdown_ceiling: float = 0.10,
        min_total_trades: int = 50,
        min_win_rate: float = 0.50,
        min_profit_factor: float = 1.30,
    ) -> None:
        self.min_backtest_months = min_backtest_months
        self.min_paper_days = min_paper_days
        self.min_sharpe_ratio = min_sharpe_ratio
        self.min_max_30d_return = min_max_30d_return
        self.max_drawdown_ceiling = max_drawdown_ceiling
        self.min_total_trades = min_total_trades
        self.min_win_rate = min_win_rate
        self.min_profit_factor = min_profit_factor

    def evaluate(
        self,
        metrics: Union[ReadinessMetrics, Dict[str, Any]],
        evaluation_timestamp: Optional[datetime.datetime] = None,
    ) -> LiveReadinessReport:
        """Evaluates metrics against all 8 gates and generates an actionable readiness report."""
        if isinstance(metrics, dict):
            m = ReadinessMetrics(
                backtest_months=float(metrics.get("backtest_months", 0.0)),
                paper_days=float(metrics.get("paper_days", 0.0)),
                strategy_cum_return=float(metrics.get("strategy_cum_return", 0.0)),
                sp500_cum_return=float(metrics.get("sp500_cum_return", 0.0)),
                sharpe_ratio=float(metrics.get("sharpe_ratio", 0.0)),
                max_30d_return=float(metrics.get("max_30d_return", 0.0)),
                max_drawdown=float(metrics.get("max_drawdown", 1.0)),
                total_trades=int(metrics.get("total_trades", 0)),
                win_rate=float(metrics.get("win_rate", 0.0)),
                profit_factor=float(metrics.get("profit_factor", 0.0)),
            )
        else:
            m = metrics

        now = evaluation_timestamp or datetime.datetime.utcnow()
        failed_gates: List[str] = []
        gates: Dict[str, Dict[str, Any]] = {}

        # Gate 1: Backtest Duration >= 12 months
        g1_pass = m.backtest_months >= self.min_backtest_months
        gates["gate_1_backtest_duration"] = {
            "target": f">= {self.min_backtest_months:.1f} months",
            "actual": f"{m.backtest_months:.1f} months",
            "passed": g1_pass,
        }
        if not g1_pass:
            failed_gates.append(f"Gate 1 (Backtest Duration: {m.backtest_months:.1f}m < {self.min_backtest_months:.1f}m)")

        # Gate 2: Paper Trading Duration >= 30 days
        g2_pass = m.paper_days >= self.min_paper_days
        gates["gate_2_paper_duration"] = {
            "target": f">= {self.min_paper_days:.1f} days",
            "actual": f"{m.paper_days:.1f} days",
            "passed": g2_pass,
        }
        if not g2_pass:
            failed_gates.append(f"Gate 2 (Paper Trading Duration: {m.paper_days:.1f}d < {self.min_paper_days:.1f}d)")

        # Gate 3: Benchmark Outperformance (Return_strategy > Return_SP500)
        g3_pass = m.strategy_cum_return > m.sp500_cum_return
        gates["gate_3_beat_sp500"] = {
            "target": "Strategy Return > SP500 Return",
            "strategy_return": round(m.strategy_cum_return, 4),
            "sp500_return": round(m.sp500_cum_return, 4),
            "passed": g3_pass,
        }
        if not g3_pass:
            failed_gates.append(
                f"Gate 3 (Benchmark Outperformance: Strategy {m.strategy_cum_return:.2%} <= S&P 500 {m.sp500_cum_return:.2%})"
            )

        # Gate 4: Risk-Adjusted Return (Annualized Sharpe > 1.0)
        g4_pass = m.sharpe_ratio > self.min_sharpe_ratio
        gates["gate_4_sharpe_ratio"] = {
            "target": f"> {self.min_sharpe_ratio:.2f}",
            "actual": round(m.sharpe_ratio, 2),
            "passed": g4_pass,
        }
        if not g4_pass:
            failed_gates.append(f"Gate 4 (Sharpe Ratio: {m.sharpe_ratio:.2f} <= {self.min_sharpe_ratio:.2f})")

        # Gate 5: Monthly Return Window (>= 15.0% in at least one 30-day window)
        g5_pass = m.max_30d_return >= self.min_max_30d_return
        gates["gate_5_monthly_return"] = {
            "target": f">= {self.min_max_30d_return:.1%}",
            "actual": f"{m.max_30d_return:.1%}",
            "passed": g5_pass,
        }
        if not g5_pass:
            failed_gates.append(f"Gate 5 (Max 30d Return: {m.max_30d_return:.1%} < {self.min_max_30d_return:.1%})")

        # Gate 6: Capital Protection (Maximum Drawdown <= 10.0%)
        g6_pass = m.max_drawdown <= self.max_drawdown_ceiling
        gates["gate_6_max_drawdown"] = {
            "target": f"<= {self.max_drawdown_ceiling:.1%}",
            "actual": f"{m.max_drawdown:.1%}",
            "passed": g6_pass,
        }
        if not g6_pass:
            failed_gates.append(f"Gate 6 (Max Drawdown: {m.max_drawdown:.1%} > {self.max_drawdown_ceiling:.1%})")

        # Gate 7: Statistical Significance (Total Trades >= 50)
        g7_pass = m.total_trades >= self.min_total_trades
        gates["gate_7_min_trades"] = {
            "target": f">= {self.min_total_trades} trades",
            "actual": m.total_trades,
            "passed": g7_pass,
        }
        if not g7_pass:
            failed_gates.append(f"Gate 7 (Total Trades: {m.total_trades} < {self.min_total_trades})")

        # Gate 8: Execution Robustness (Win Rate >= 50% & Profit Factor >= 1.30)
        wr_pass = m.win_rate >= self.min_win_rate
        pf_pass = m.profit_factor >= self.min_profit_factor
        g8_pass = wr_pass and pf_pass
        gates["gate_8_execution_quality"] = {
            "target": f"Win Rate >= {self.min_win_rate:.0%} & Profit Factor >= {self.min_profit_factor:.2f}",
            "win_rate": round(m.win_rate, 4),
            "profit_factor": round(m.profit_factor, 2),
            "passed": g8_pass,
        }
        if not g8_pass:
            failed_gates.append(
                f"Gate 8 (Execution Quality: Win Rate {m.win_rate:.1%}, Profit Factor {m.profit_factor:.2f})"
            )

        is_live_ready = len(failed_gates) == 0
        if is_live_ready:
            summary = "All 8 live deployment criteria satisfied."
        else:
            summary = f"{len(failed_gates)} gate(s) failed: {'; '.join(failed_gates)}."

        return LiveReadinessReport(
            is_live_ready=is_live_ready,
            evaluation_timestamp=now,
            summary=summary,
            gates=gates,
        )
