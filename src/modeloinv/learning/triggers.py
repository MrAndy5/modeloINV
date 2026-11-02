"""Retraining Triggers and Trigger Management for modeloINV.

Provides:
- TriggerReason: Canonical enum of reasons that can trigger a model retraining.
- RetrainingTriggerManager: Multi-criteria trigger monitor evaluating:
    1. Daily schedule (e.g. 21:00 UTC post-market close)
    2. Trade count milestones (every N=25 closed trades)
    3. Prediction error drift (rolling Brier score drift > 0.35)
    4. Drawdown breach (> 5% from peak equity)
    5. Cooldown mechanism (minimum 60 minutes between retrainings)
"""

from __future__ import annotations

import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


class TriggerReason(str, Enum):
    """Reasons that initiate a continuous learning retraining cycle."""
    SCHEDULED_DAILY = "SCHEDULED_DAILY"
    TRADE_COUNT_MILESTONE = "TRADE_COUNT_MILESTONE"
    PREDICTION_DRIFT = "PREDICTION_DRIFT"
    DRAWDOWN_BREACH = "DRAWDOWN_BREACH"
    MANUAL = "MANUAL"


class RetrainingTriggerManager:
    """Evaluates multi-criteria retraining triggers with strict cooldown enforcement."""

    def __init__(
        self,
        scheduled_hour_utc: int = 21,
        scheduled_minute_utc: int = 0,
        trade_count_threshold: int = 25,
        drift_brier_threshold: float = 0.35,
        drift_window_size: int = 20,
        drawdown_threshold: float = 0.05,
        cooldown_minutes: int = 60,
    ) -> None:
        self.scheduled_hour_utc = scheduled_hour_utc
        self.scheduled_minute_utc = scheduled_minute_utc
        self.trade_count_threshold = trade_count_threshold
        self.drift_brier_threshold = drift_brier_threshold
        self.drift_window_size = drift_window_size
        self.drawdown_threshold = drawdown_threshold
        self.cooldown_minutes = cooldown_minutes

        self.last_retrain_time: Optional[datetime.datetime] = None
        self.last_scheduled_date: Optional[datetime.date] = None
        self.trades_since_last_retrain: int = 0
        self._prediction_history: List[Tuple[float, float]] = []  # List of (p_up, y_realized)

    def record_trade(self, count: int = 1) -> int:
        """Records completed trade(s) toward the trade count milestone."""
        if count > 0:
            self.trades_since_last_retrain += count
        return self.trades_since_last_retrain

    def record_prediction(self, p_up: float, y_realized: float) -> None:
        """Records directional prediction and actual realized binary outcome (1.0 or 0.0)."""
        p_clamped = max(0.0, min(1.0, float(p_up)))
        y_clamped = 1.0 if y_realized > 0.5 else 0.0
        self._prediction_history.append((p_clamped, y_clamped))
        if len(self._prediction_history) > self.drift_window_size * 2:
            self._prediction_history = self._prediction_history[-self.drift_window_size:]

    def calculate_rolling_brier(self) -> Optional[float]:
        """Calculates rolling Brier score over the most recent prediction window.

        Brier Score = (1 / K) * sum((p_up_i - y_realized_i) ** 2)
        """
        if len(self._prediction_history) < self.drift_window_size:
            return None
        window = self._prediction_history[-self.drift_window_size:]
        brier = sum((p - y) ** 2 for p, y in window) / len(window)
        return float(brier)

    def is_in_cooldown(self, current_time: datetime.datetime) -> bool:
        """Checks if current time falls within the post-retrain cooldown period."""
        if self.last_retrain_time is None:
            return False
        t_curr = current_time if current_time.tzinfo else current_time.replace(tzinfo=datetime.timezone.utc)
        t_last = self.last_retrain_time if self.last_retrain_time.tzinfo else self.last_retrain_time.replace(tzinfo=datetime.timezone.utc)
        t_curr = t_curr.astimezone(datetime.timezone.utc)
        t_last = t_last.astimezone(datetime.timezone.utc)
        delta_sec = (t_curr - t_last).total_seconds()
        return delta_sec < (self.cooldown_minutes * 60)

    def check_triggers(
        self,
        current_time: datetime.datetime,
        current_drawdown: float = 0.0,
        force: bool = False,
    ) -> Tuple[bool, Optional[TriggerReason], str]:
        """Evaluates whether any retraining trigger condition is met.

        Args:
            current_time: Current evaluation timestamp.
            current_drawdown: Current portfolio drawdown ratio (e.g. 0.06 for 6%).
            force: If True, bypasses cooldown and trigger conditions to force a retrain.

        Returns:
            Tuple of (should_retrain: bool, reason: Optional[TriggerReason], message: str)
        """
        if force:
            return True, TriggerReason.MANUAL, "Manual retraining triggered."

        # Cooldown check
        if self.is_in_cooldown(current_time):
            assert self.last_retrain_time is not None
            cooldown_end = self.last_retrain_time + datetime.timedelta(minutes=self.cooldown_minutes)
            return (
                False,
                None,
                f"Retraining suppressed: active cooldown until {cooldown_end.isoformat()}.",
            )

        # 1. Drawdown breach check (highest urgency for capital defense)
        if current_drawdown >= self.drawdown_threshold:
            return (
                True,
                TriggerReason.DRAWDOWN_BREACH,
                f"Drawdown breach: current drawdown {current_drawdown:.2%} >= threshold {self.drawdown_threshold:.2%}.",
            )

        # 2. Prediction error drift check
        brier = self.calculate_rolling_brier()
        if brier is not None and brier > self.drift_brier_threshold:
            return (
                True,
                TriggerReason.PREDICTION_DRIFT,
                f"Prediction drift detected: rolling Brier score {brier:.4f} > threshold {self.drift_brier_threshold:.4f}.",
            )

        # 3. Trade count milestone check
        if self.trades_since_last_retrain >= self.trade_count_threshold:
            return (
                True,
                TriggerReason.TRADE_COUNT_MILESTONE,
                f"Trade count milestone reached: {self.trades_since_last_retrain} >= threshold {self.trade_count_threshold}.",
            )

        # 4. Scheduled daily trigger check
        curr_utc = current_time.astimezone(datetime.timezone.utc) if current_time.tzinfo else current_time
        target_time_passed = (
            curr_utc.hour > self.scheduled_hour_utc
            or (curr_utc.hour == self.scheduled_hour_utc and curr_utc.minute >= self.scheduled_minute_utc)
        )
        curr_date = curr_utc.date()
        date_not_yet_run = (self.last_scheduled_date is None) or (self.last_scheduled_date < curr_date)

        if target_time_passed and date_not_yet_run:
            return (
                True,
                TriggerReason.SCHEDULED_DAILY,
                f"Daily scheduled retraining trigger at {self.scheduled_hour_utc:02d}:{self.scheduled_minute_utc:02d} UTC.",
            )

        return False, None, "No retraining trigger condition satisfied."

    def record_retrain(
        self,
        timestamp: datetime.datetime,
        reason: Optional[TriggerReason] = None,
    ) -> None:
        """Registers a completed retraining event, resetting counters and engaging cooldown."""
        self.last_retrain_time = timestamp
        self.trades_since_last_retrain = 0

        curr_utc = timestamp.astimezone(datetime.timezone.utc) if timestamp.tzinfo else timestamp
        if reason == TriggerReason.SCHEDULED_DAILY:
            self.last_scheduled_date = curr_utc.date()
