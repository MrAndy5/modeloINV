"""Strategy Subsystem for modeloINV.

Exports the complete Hybrid AI Trading Strategy components:
- Technical Indicators (EMA, RSI, MACD, BB, ATR, VWAP, ADX)
- Market Regime Classification & Composite Quant Scoring
- Machine Learning (TripleBarrierLabeler, MLFeaturePipeline, HistGradientBoostingModel, DeepSequenceEmbedder)
- Reinforcement Learning (RLStateBuilder, PolicyAgent, CostPenalizedReward)
- Universe & Sizing (DynamicUniverseScreener, MultiArmedBanditAllocator, MicroCapitalFractionalAdapter)
- Engine & Timeframes (MultiTimeframeBarSynchronizer, HybridStrategyEngine)
"""

from modeloinv.strategy.engine import (
    HybridStrategyEngine,
    IntradayHybridSignalAggregator,
    MultiTimeframeBarSynchronizer,
)
from modeloinv.strategy.indicators import (
    TechnicalIndicatorSuite,
    calculate_adx,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    calculate_vwap,
)
from modeloinv.strategy.ml_model import (
    DeepSequenceEmbedder,
    HistGradientBoostingModel,
    MLFeaturePipeline,
    TripleBarrierLabeler,
)
from modeloinv.strategy.regime import (
    MarketRegimeClassifier,
    RegimeState,
    compute_composite_quant_score,
    get_regime_weights,
)
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

__all__ = [
    # Indicators
    "TechnicalIndicatorSuite",
    "calculate_ema",
    "calculate_rsi",
    "calculate_macd",
    "calculate_bollinger_bands",
    "calculate_atr",
    "calculate_vwap",
    "calculate_adx",
    # Regime
    "RegimeState",
    "MarketRegimeClassifier",
    "compute_composite_quant_score",
    "get_regime_weights",
    # ML
    "TripleBarrierLabeler",
    "MLFeaturePipeline",
    "HistGradientBoostingModel",
    "DeepSequenceEmbedder",
    # RL
    "RLStateBuilder",
    "PolicyNetwork",
    "PolicyAgent",
    "CostPenalizedReward",
    "ACTION_WEIGHT_MAP",
    # Universe & Allocation
    "CandidateAsset",
    "DynamicUniverseScreener",
    "MultiArmedBanditAllocator",
    "MicroCapitalFractionalAdapter",
    # Engine
    "MultiTimeframeBarSynchronizer",
    "HybridStrategyEngine",
    "IntradayHybridSignalAggregator",
]
