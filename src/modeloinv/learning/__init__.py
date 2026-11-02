"""Continuous Learning, Model Version Registry, and Live Readiness for modeloINV.

Provides:
- Retraining triggers (Scheduled, Trade Count, Drift, Drawdown, Cooldown)
- Trade feedback ingestion and Prioritized Experience Replay buffer
- Incremental sliding-window retrainer with recency and error sample weighting
- Semantic model registry with atomic persistence and checkpoint serialization
- Champion-Challenger validation gate
- Deterministic 8-gate Live Readiness evaluation engine
- 30-Day learning loop verification benchmark
"""

from modeloinv.learning.benchmark import (
    BenchmarkResult,
    LearningLoopBenchmark,
)
from modeloinv.learning.champion import (
    ChampionChallengerGate,
    GateResult,
)
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
from modeloinv.learning.registry import (
    ModelMetadata,
    ModelRegistry,
)
from modeloinv.learning.retrainer import (
    IncrementalSlidingWindowRetrainer,
)
from modeloinv.learning.triggers import (
    RetrainingTriggerManager,
    TriggerReason,
)

__all__ = [
    # Triggers
    "TriggerReason",
    "RetrainingTriggerManager",
    # Feedback & RL Replay
    "TradeFeedbackRecord",
    "TradeFeedbackIngestor",
    "PrioritizedExperienceReplayBuffer",
    # Retraining
    "IncrementalSlidingWindowRetrainer",
    # Model Registry
    "ModelMetadata",
    "ModelRegistry",
    # Champion Gate
    "ChampionChallengerGate",
    "GateResult",
    # Live Readiness
    "ReadinessMetrics",
    "LiveReadinessReport",
    "LiveReadinessEvaluator",
    # Benchmark
    "BenchmarkResult",
    "LearningLoopBenchmark",
]
