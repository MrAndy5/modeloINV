"""Trade Feedback Ingestion and Prioritized Experience Replay for modeloINV.

Provides:
- TradeFeedbackRecord: Canonical dataclass capturing complete execution and prediction telemetry.
- TradeFeedbackIngestor: Storage, filtering, outlier clipping, and query interface for trade feedback.
- PrioritizedExperienceReplayBuffer: RL replay buffer with priority:
    p_i = |delta_TD| + eps_p + beta * |y_realized - y_hat|
  and importance sampling weight correction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import datetime
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class TradeFeedbackRecord:
    """Telemetry record captured upon trade closure for feedback and retraining."""
    trade_id: str
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_time: datetime.datetime
    exit_time: datetime.datetime
    entry_price: float
    exit_price: float
    quantity: float
    realized_pnl: float
    realized_return_pct: float
    fees_paid: float
    expected_slippage: float
    actual_slippage: float
    predicted_p_up: float
    predicted_confidence: float
    prediction_error: float
    holding_bars: int
    model_version: str = "v1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.entry_price <= 0:
            raise ValueError(f"entry_price must be > 0, got {self.entry_price}")
        if self.exit_price <= 0:
            raise ValueError(f"exit_price must be > 0, got {self.exit_price}")
        if self.quantity <= 0:
            raise ValueError(f"quantity must be > 0, got {self.quantity}")
        if self.exit_time < self.entry_time:
            raise ValueError(f"exit_time ({self.exit_time}) cannot precede entry_time ({self.entry_time})")

    def to_dict(self) -> Dict[str, Any]:
        """Serializes record to standard dictionary format."""
        d = asdict(self)
        d["entry_time"] = self.entry_time.isoformat()
        d["exit_time"] = self.exit_time.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TradeFeedbackRecord:
        """Instantiates record from dictionary."""
        entry_t = data["entry_time"]
        if isinstance(entry_t, str):
            entry_t = datetime.datetime.fromisoformat(entry_t.replace("Z", "+00:00"))

        exit_t = data["exit_time"]
        if isinstance(exit_t, str):
            exit_t = datetime.datetime.fromisoformat(exit_t.replace("Z", "+00:00"))

        return cls(
            trade_id=str(data["trade_id"]),
            symbol=str(data["symbol"]),
            direction=str(data["direction"]),
            entry_time=entry_t,
            exit_time=exit_t,
            entry_price=float(data["entry_price"]),
            exit_price=float(data["exit_price"]),
            quantity=float(data["quantity"]),
            realized_pnl=float(data["realized_pnl"]),
            realized_return_pct=float(data["realized_return_pct"]),
            fees_paid=float(data["fees_paid"]),
            expected_slippage=float(data["expected_slippage"]),
            actual_slippage=float(data["actual_slippage"]),
            predicted_p_up=float(data["predicted_p_up"]),
            predicted_confidence=float(data["predicted_confidence"]),
            prediction_error=float(data["prediction_error"]),
            holding_bars=int(data["holding_bars"]),
            model_version=str(data.get("model_version", "v1.0.0")),
            metadata=dict(data.get("metadata", {})),
        )


class TradeFeedbackIngestor:
    """Manages trade feedback logging, anomaly detection, outlier filtering, and retrieval."""

    def __init__(self, max_records: int = 100_000) -> None:
        self.max_records = max_records
        self._records: List[TradeFeedbackRecord] = []

    def ingest(self, record: TradeFeedbackRecord) -> None:
        """Ingests a trade feedback record with outlier clipping for slippage anomalies."""
        # Spec edge case: If actual slippage > 5 * expected slippage, cap slippage and tag anomaly
        if record.expected_slippage > 0 and record.actual_slippage > 5.0 * record.expected_slippage:
            record.metadata["slippage_anomaly"] = True
            record.metadata["uncapped_actual_slippage"] = record.actual_slippage
            record.actual_slippage = 5.0 * record.expected_slippage
        elif record.expected_slippage <= 0 and abs(record.actual_slippage) > 0.05:
            record.metadata["slippage_anomaly"] = True
            record.metadata["uncapped_actual_slippage"] = record.actual_slippage
            record.actual_slippage = math.copysign(0.05, record.actual_slippage)

        self._records.append(record)
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records:]

    def count(self) -> int:
        """Returns total ingested records."""
        return len(self._records)

    def get_recent_records(self, n: int = 50) -> List[TradeFeedbackRecord]:
        """Retrieves n most recent records."""
        return list(self._records[-n:])

    def get_window(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> List[TradeFeedbackRecord]:
        """Retrieves records whose exit_time falls within [start_time, end_time]."""
        return [
            r for r in self._records
            if start_time <= r.exit_time <= end_time
        ]

    def calculate_rolling_brier(self, window: int = 20) -> Optional[float]:
        """Computes rolling Brier score over the last `window` trades.

        Binary outcome y_realized is 1.0 if realized_pnl > 0, else 0.0.
        """
        if len(self._records) < window or window <= 0:
            return None

        slice_records = self._records[-window:]
        brier_sum = 0.0
        for r in slice_records:
            y_realized = 1.0 if r.realized_pnl > 0 else 0.0
            p_up = max(0.0, min(1.0, r.predicted_p_up))
            brier_sum += (p_up - y_realized) ** 2

        return float(brier_sum / len(slice_records))

    def to_dataframe(self) -> pd.DataFrame:
        """Exports all records to a pandas DataFrame."""
        if not self._records:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self._records])


class PrioritizedExperienceReplayBuffer:
    """Prioritized Experience Replay (PER) for Reinforcement Learning policy refinement.

    Calculates priority:
        p_i = |delta_TD| + eps_p + beta * |y_realized - y_hat|
    Corrects bias using Importance Sampling weights.
    """

    def __init__(
        self,
        capacity: int = 50_000,
        alpha: float = 0.6,
        eps_p: float = 1e-5,
        beta_error: float = 1.0,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.eps_p = eps_p
        self.beta_error = beta_error

        self._states: List[np.ndarray] = []
        self._actions: List[int] = []
        self._rewards: List[float] = []
        self._next_states: List[np.ndarray] = []
        self._dones: List[bool] = []
        self._priorities: List[float] = []
        self._pos = 0

    def __len__(self) -> int:
        return len(self._states)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        td_error: Optional[float] = None,
        prediction_error: float = 0.0,
    ) -> None:
        """Inserts experience transition with calculated or default priority."""
        if td_error is not None:
            priority = abs(td_error) + self.eps_p + (self.beta_error * abs(prediction_error))
        else:
            # Default to maximum observed priority so new experiences are sampled promptly
            priority = max(self._priorities) if self._priorities else 1.0
        priority = float(min(max(priority, self.eps_p), 1e6))

        if len(self._states) < self.capacity:
            self._states.append(np.array(state, dtype=np.float32, copy=True))
            self._actions.append(int(action))
            self._rewards.append(float(reward))
            self._next_states.append(np.array(next_state, dtype=np.float32, copy=True))
            self._dones.append(bool(done))
            self._priorities.append(float(priority))
        else:
            self._states[self._pos] = np.array(state, dtype=np.float32, copy=True)
            self._actions[self._pos] = int(action)
            self._rewards[self._pos] = float(reward)
            self._next_states[self._pos] = np.array(next_state, dtype=np.float32, copy=True)
            self._dones[self._pos] = bool(done)
            self._priorities[self._pos] = float(priority)
            self._pos = (self._pos + 1) % self.capacity

    def sample(
        self,
        batch_size: int,
        beta_is: float = 0.4,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Samples a batch proportional to priority with importance sampling weights.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, is_weights)
        """
        n = len(self._states)
        if n == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        batch_size = min(batch_size, n)
        priorities = np.array(self._priorities, dtype=np.float64)
        scaled_priorities = priorities ** self.alpha
        sum_p = np.sum(scaled_priorities)
        if sum_p <= 0 or not np.isfinite(sum_p):
            probs = np.full(n, 1.0 / n)
        else:
            probs = scaled_priorities / sum_p

        indices = np.random.choice(n, size=batch_size, replace=False, p=probs)

        # Importance sampling weights: w_i = (1 / (N * P(i))) ** beta_is
        # Normalized by max(w) so weights are in [0, 1]
        p_sampled = probs[indices]
        weights = (1.0 / (n * p_sampled + 1e-12)) ** beta_is
        max_weight = np.max(weights) if len(weights) > 0 else 1.0
        weights = weights / (max_weight + 1e-12)

        states = np.array([self._states[i] for i in indices], dtype=np.float32)
        actions = np.array([self._actions[i] for i in indices], dtype=np.int64)
        rewards = np.array([self._rewards[i] for i in indices], dtype=np.float32)
        next_states = np.array([self._next_states[i] for i in indices], dtype=np.float32)
        dones = np.array([self._dones[i] for i in indices], dtype=np.bool_)

        return states, actions, rewards, next_states, dones, indices, weights.astype(np.float32)

    def update_priorities(
        self,
        indices: Sequence[int],
        td_errors: Sequence[float],
        prediction_errors: Optional[Sequence[float]] = None,
    ) -> None:
        """Updates priorities of sampled transitions."""
        for k, idx in enumerate(indices):
            if idx < 0 or idx >= len(self._priorities):
                continue
            td = abs(float(td_errors[k]))
            pe = abs(float(prediction_errors[k])) if prediction_errors is not None else 0.0
            new_p = td + self.eps_p + (self.beta_error * pe)
            self._priorities[idx] = float(min(max(new_p, self.eps_p), 1e6))
