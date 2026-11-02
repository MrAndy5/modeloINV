"""Market Regime Classifier and Composite Quant Scoring for modeloINV.

Features:
- RegimeState enum: TRENDING_BULL, TRENDING_BEAR, MEAN_REVERTING, CHOPPY.
- Quantitative regime classification using ADX, Bollinger Bandwidth, ATR ratio,
  and moving average slopes.
- Scale-invariant Composite Quant Score S_quant in [-1.0, 1.0] dynamically
  weighting trend-following vs mean-reversion signals conditioned on the regime.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd


class RegimeState(str, Enum):
    """Market regime classification state."""
    TRENDING_BULL = "TRENDING_BULL"
    TRENDING_BEAR = "TRENDING_BEAR"
    MEAN_REVERTING = "MEAN_REVERTING"
    CHOPPY = "CHOPPY"


class MarketRegimeClassifier:
    """Classifies the market regime using ADX, volatility bandwidth, and trend direction."""

    def __init__(
        self,
        adx_trending_threshold: float = 25.0,
        adx_choppy_threshold: float = 20.0,
        bandwidth_mean_reverting_max: float = 0.05,
    ):
        self.adx_trending_threshold = adx_trending_threshold
        self.adx_choppy_threshold = adx_choppy_threshold
        self.bandwidth_mean_reverting_max = bandwidth_mean_reverting_max

    def classify_from_values(
        self,
        adx: float,
        trend_slope: Optional[float] = None,
        ema_9: Optional[float] = None,
        ema_21: Optional[float] = None,
        bandwidth: Optional[float] = None,
        atr_ratio: Optional[float] = None,
    ) -> RegimeState:
        """Determines market regime from scalar indicator values."""
        # Determine trend direction: prefer trend_slope if provided, otherwise ema_9 vs ema_21
        slope = 0.0
        if trend_slope is not None:
            slope = trend_slope
        elif ema_9 is not None and ema_21 is not None:
            slope = ema_9 - ema_21

        # Check for strong trend
        if adx >= self.adx_trending_threshold:
            if slope > 0:
                return RegimeState.TRENDING_BULL
            elif slope < 0:
                return RegimeState.TRENDING_BEAR
            else:
                return RegimeState.CHOPPY

        # Check for non-trending / low directional movement
        if adx < self.adx_choppy_threshold:
            # If bandwidth is moderate/low, it's classic mean-reverting consolidation
            if bandwidth is not None and bandwidth <= self.bandwidth_mean_reverting_max:
                return RegimeState.MEAN_REVERTING
            elif atr_ratio is not None and atr_ratio > 0.02:
                # High volatility without trend direction is choppy
                return RegimeState.CHOPPY
            return RegimeState.MEAN_REVERTING

        # Intermediate zone (20 <= ADX < 25)
        if bandwidth is not None and bandwidth > self.bandwidth_mean_reverting_max:
            return RegimeState.CHOPPY
        return RegimeState.MEAN_REVERTING

    def classify(self, indicators: Union[Dict[str, Any], pd.Series]) -> RegimeState:
        """Classify regime from an indicator dictionary or Series row."""
        if isinstance(indicators, pd.Series):
            ind = indicators.to_dict()
        else:
            ind = indicators

        adx = float(ind.get("adx_14", ind.get("adx", 20.0)))
        trend_slope = ind.get("trend_slope", None)
        if trend_slope is not None:
            trend_slope = float(trend_slope)

        ema_9 = ind.get("ema_9", None)
        if ema_9 is not None:
            ema_9 = float(ema_9)

        ema_21 = ind.get("ema_21", None)
        if ema_21 is not None:
            ema_21 = float(ema_21)

        bandwidth = ind.get("bb_bandwidth", ind.get("bandwidth", None))
        if bandwidth is not None:
            bandwidth = float(bandwidth)

        atr_ratio = ind.get("atr_ratio", None)
        if atr_ratio is None and "atr_14" in ind and "close" in ind:
            close_val = float(ind["close"])
            if close_val > 0:
                atr_ratio = float(ind["atr_14"]) / close_val

        return self.classify_from_values(
            adx=adx,
            trend_slope=trend_slope,
            ema_9=ema_9,
            ema_21=ema_21,
            bandwidth=bandwidth,
            atr_ratio=atr_ratio,
        )


def get_regime_weights(regime: RegimeState) -> Tuple[float, float, float]:
    """Returns (w_trend, w_mom, w_revert) summing to 1.0 conditioned on the regime."""
    if regime in (RegimeState.TRENDING_BULL, RegimeState.TRENDING_BEAR):
        return (0.50, 0.30, 0.20)
    elif regime == RegimeState.MEAN_REVERTING:
        return (0.10, 0.20, 0.70)
    else:  # CHOPPY
        return (0.25, 0.25, 0.50)


def compute_composite_quant_score(
    indicators: Union[Dict[str, Any], pd.Series],
    regime: Optional[RegimeState] = None,
    classifier: Optional[MarketRegimeClassifier] = None,
) -> float:
    """Computes scale-invariant composite quant score S_quant in [-1.0, 1.0].

    Formula:
        S_quant = w_trend * S_ema + w_mom * S_macd + w_revert * (0.5 * S_rsi + 0.5 * S_bb)
    where weights (w_trend, w_mom, w_revert) are dynamically selected based on market regime.
    """
    if isinstance(indicators, pd.Series):
        ind = indicators.to_dict()
    else:
        ind = indicators

    if regime is None:
        clf = classifier or MarketRegimeClassifier()
        regime = clf.classify(ind)

    w_trend, w_mom, w_revert = get_regime_weights(regime)

    # Extract or compute normalized scores
    # S_ema
    if "s_ema" in ind:
        s_ema = float(ind["s_ema"])
    else:
        ema_9 = float(ind.get("ema_9", 0.0))
        ema_21 = float(ind.get("ema_21", 0.0))
        atr = max(float(ind.get("atr_14", ind.get("atr", 1.0))), 1e-9)
        s_ema = float(np.tanh((ema_9 - ema_21) / atr))

    # S_macd
    if "s_macd" in ind:
        s_macd = float(ind["s_macd"])
    else:
        hist = float(ind.get("macd_hist", ind.get("hist", 0.0)))
        atr = max(float(ind.get("atr_14", ind.get("atr", 1.0))), 1e-9)
        s_macd = float(np.tanh(hist / (0.5 * atr)))

    # S_rsi
    if "s_rsi" in ind:
        s_rsi = float(ind["s_rsi"])
    else:
        rsi = float(ind.get("rsi_14", ind.get("rsi", 50.0)))
        s_rsi = float(np.clip((50.0 - rsi) / 25.0, -1.0, 1.0))

    # S_bb
    if "s_bb" in ind:
        s_bb = float(ind["s_bb"])
    else:
        pct_b = float(ind.get("bb_pct_b", ind.get("pct_b", 0.5)))
        s_bb = float(np.clip(1.0 - 2.0 * pct_b, -1.0, 1.0))

    reversion_score = 0.5 * s_rsi + 0.5 * s_bb

    score = (w_trend * s_ema) + (w_mom * s_macd) + (w_revert * reversion_score)
    # Strictly bound between -1.0 and 1.0
    return float(np.clip(score, -1.0, 1.0))
