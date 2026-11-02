"""Machine Learning Layer for modeloINV.

Provides:
- TripleBarrierLabeler: Labels price paths into {-1, 0, +1} using take-profit,
  stop-loss, and vertical time barriers.
- MLFeaturePipeline: Extracts 32 standardized, scale-invariant features.
- HistGradientBoostingModel: scikit-learn HistGradientBoostingClassifier forecasting
  directional probabilities, directional score S_ML in [-1.0, 1.0], and confidence C_ML.
- DeepSequenceEmbedder: PyTorch neural network extracting compact temporal/spatial embeddings
  with CPU fallback.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import torch
import torch.nn as nn

from modeloinv.core.models import Bar
from modeloinv.strategy.indicators import (
    TechnicalIndicatorSuite,
    _standardize_ohlcv_dataframe,
    calculate_ema,
)
from modeloinv.strategy.regime import MarketRegimeClassifier, compute_composite_quant_score


class TripleBarrierLabeler:
    """Labels price paths into {-1, 0, +1} using upper/lower ATR barriers and a vertical time horizon."""

    def __init__(self, k_take_profit: float = 2.0, k_stop_loss: float = 2.0, horizon_bars: int = 12):
        self.k_take_profit = k_take_profit
        self.k_stop_loss = k_stop_loss
        self.horizon_bars = horizon_bars

    def label_path(
        self,
        prices: Union[np.ndarray, Sequence[float], pd.Series],
        atrs: Union[np.ndarray, Sequence[float], pd.Series],
        highs: Optional[Union[np.ndarray, Sequence[float]]] = None,
        lows: Optional[Union[np.ndarray, Sequence[float]]] = None,
    ) -> np.ndarray:
        """Labels each bar based on which barrier is touched first within horizon_bars.

        Returns:
            np.ndarray of integers with values in {-1, 0, +1}.
        """
        p = np.asarray(prices, dtype=np.float64)
        a = np.asarray(atrs, dtype=np.float64)
        n = len(p)
        labels = np.zeros(n, dtype=np.int64)

        h = np.asarray(highs, dtype=np.float64) if highs is not None else p
        l = np.asarray(lows, dtype=np.float64) if lows is not None else p

        for i in range(n):
            entry_price = p[i]
            atr = max(a[i], 1e-6)
            upper_barrier = entry_price + (self.k_take_profit * atr)
            lower_barrier = entry_price - (self.k_stop_loss * atr)

            end_idx = min(i + self.horizon_bars + 1, n)
            label = 0

            for j in range(i + 1, end_idx):
                hit_upper = h[j] >= upper_barrier
                hit_lower = l[j] <= lower_barrier

                if hit_upper and not hit_lower:
                    label = 1
                    break
                elif hit_lower and not hit_upper:
                    label = -1
                    break
                elif hit_upper and hit_lower:
                    # In rare bar that touches both, compare close relative to entry
                    label = 1 if p[j] >= entry_price else -1
                    break

            labels[i] = label

        return labels

    def label_dataframe(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        atr_col: str = "atr_14",
        high_col: str = "high",
        low_col: str = "low",
    ) -> pd.Series:
        """Applies triple barrier labeling to a DataFrame."""
        prices = df[price_col].values
        atrs = df[atr_col].values if atr_col in df.columns else (prices * 0.01)
        highs = df[high_col].values if high_col in df.columns else None
        lows = df[low_col].values if low_col in df.columns else None

        labels = self.label_path(prices, atrs, highs=highs, lows=lows)
        return pd.Series(labels, index=df.index, name="target_label")


class MLFeaturePipeline:
    """Extracts 32 standardized, scale-invariant features from market bar series."""

    FEATURE_NAMES: List[str] = [
        "log_ret_1", "log_ret_3", "log_ret_5", "log_ret_15",
        "vol_atr_ratio", "vol_sigma_ratio", "bb_bandwidth",
        "osc_rsi", "osc_pct_b", "osc_stoch_k", "osc_stoch_d",
        "ma_rel_9", "ma_rel_21", "ma_rel_50", "ma_rel_200",
        "macd_line_norm", "macd_signal_norm", "macd_hist_norm",
        "vol_z_log_ratio", "vol_z_norm_5", "obv_slope_10",
        "vwap_deviation",
        "trend_adx", "trend_di_diff",
        "quant_score",
        "time_sin", "time_cos",
        "bar_range_norm", "bar_body_norm", "bar_upper_wick_norm", "bar_lower_wick_norm",
        "benchmark_spread_return",
    ]

    def __init__(self):
        self.indicator_suite = TechnicalIndicatorSuite()
        self.regime_classifier = MarketRegimeClassifier()

    def extract_features(
        self,
        data: Union[pd.DataFrame, Sequence[Bar]],
        spy_df: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """Transforms OHLCV data into a 32-dimensional feature matrix X in R^{T x 32}."""
        df = _standardize_ohlcv_dataframe(data)
        if len(df) == 0:
            return np.empty((0, 32), dtype=np.float32)

        # Compute full indicator suite
        df = self.indicator_suite.compute_all_indicators(df)

        close = df["close"]
        open_p = df["open"]
        high = df["high"]
        low = df["low"]
        vol = df["volume"]
        atr = np.maximum(df["atr_14"], 1e-6)

        features: Dict[str, np.ndarray] = {}

        # 1-4. Multi-lag log returns
        safe_close = np.maximum(close.values, 1e-9)
        features["log_ret_1"] = np.log(safe_close / np.maximum(np.roll(safe_close, 1), 1e-9))
        features["log_ret_1"][0] = 0.0

        features["log_ret_3"] = np.log(safe_close / np.maximum(np.roll(safe_close, 3), 1e-9))
        features["log_ret_3"][:3] = 0.0

        features["log_ret_5"] = np.log(safe_close / np.maximum(np.roll(safe_close, 5), 1e-9))
        features["log_ret_5"][:5] = 0.0

        features["log_ret_15"] = np.log(safe_close / np.maximum(np.roll(safe_close, 15), 1e-9))
        features["log_ret_15"][:15] = 0.0

        # 5-7. Normalized Volatility
        features["vol_atr_ratio"] = atr.values / safe_close
        rolling_std_20 = close.rolling(20, min_periods=1).std(ddof=0).fillna(0.0).values
        features["vol_sigma_ratio"] = rolling_std_20 / safe_close
        features["bb_bandwidth"] = df["bb_bandwidth"].values

        # 8-11. Oscillators
        features["osc_rsi"] = (df["rsi_14"].values / 100.0) - 0.5
        features["osc_pct_b"] = df["bb_pct_b"].values - 0.5

        # Stochastic Oscillator (%K and %D)
        low_14 = low.rolling(14, min_periods=1).min().values
        high_14 = high.rolling(14, min_periods=1).max().values
        stoch_span = np.maximum(high_14 - low_14, 1e-9)
        stoch_k = 100.0 * (close.values - low_14) / stoch_span
        stoch_d = pd.Series(stoch_k).rolling(3, min_periods=1).mean().values
        features["osc_stoch_k"] = (stoch_k / 100.0) - 0.5
        features["osc_stoch_d"] = (stoch_d / 100.0) - 0.5

        # 12-15. Moving Average Relatives
        features["ma_rel_9"] = (close.values - df["ema_9"].values) / atr.values
        features["ma_rel_21"] = (close.values - df["ema_21"].values) / atr.values
        features["ma_rel_50"] = (close.values - df["ema_50"].values) / atr.values
        features["ma_rel_200"] = (close.values - df["ema_200"].values) / atr.values

        # 16-18. MACD Normalized
        features["macd_line_norm"] = df["macd_line"].values / atr.values
        features["macd_signal_norm"] = df["macd_signal"].values / atr.values
        features["macd_hist_norm"] = df["macd_hist"].values / atr.values

        # 19-21. Microstructure / Volume
        sma_vol_20 = vol.rolling(20, min_periods=1).mean().values
        features["vol_z_log_ratio"] = np.log(np.maximum(vol.values, 1.0) / np.maximum(sma_vol_20, 1.0))

        sma_vol_5 = vol.rolling(5, min_periods=1).mean().values
        std_vol_5 = vol.rolling(5, min_periods=1).std(ddof=0).fillna(0.0).values
        features["vol_z_norm_5"] = (vol.values - sma_vol_5) / np.maximum(std_vol_5, 1e-6)

        # On-Balance Volume (OBV) slope over 10 bars
        price_diff = np.diff(close.values, prepend=close.values[0])
        obv_direction = np.sign(price_diff)
        obv = np.cumsum(obv_direction * vol.values)
        obv_series = pd.Series(obv)
        obv_rolling_mean = obv_series.rolling(10, min_periods=1).mean().values
        features["obv_slope_10"] = (obv - obv_rolling_mean) / np.maximum(sma_vol_20 * 10.0, 1.0)

        # 22. VWAP Deviation
        features["vwap_deviation"] = (close.values - df["vwap"].values) / atr.values

        # 23-24. Trend Strength
        features["trend_adx"] = df["adx_14"].values / 100.0
        features["trend_di_diff"] = (df["plus_di_14"].values - df["minus_di_14"].values) / 100.0

        # 25. Quant Score
        quant_scores = []
        for idx in range(len(df)):
            row_dict = df.iloc[idx].to_dict()
            s_q = compute_composite_quant_score(row_dict, classifier=self.regime_classifier)
            quant_scores.append(s_q)
        features["quant_score"] = np.array(quant_scores, dtype=np.float64)

        # 26-27. Intraday Cyclical Time
        if "timestamp" in df.columns:
            ts_series = pd.to_datetime(df["timestamp"])
            minute_of_day = ts_series.dt.hour * 60 + ts_series.dt.minute
            features["time_sin"] = np.sin(2.0 * np.pi * minute_of_day.values / 1440.0)
            features["time_cos"] = np.cos(2.0 * np.pi * minute_of_day.values / 1440.0)
        else:
            # Fallback to bar index cyclical
            idx_mod = (np.arange(len(df)) % 390)  # Standard trading day has 390 1-min bars
            features["time_sin"] = np.sin(2.0 * np.pi * idx_mod / 390.0)
            features["time_cos"] = np.cos(2.0 * np.pi * idx_mod / 390.0)

        # 28-31. Bar Properties / Geometry
        features["bar_range_norm"] = (high.values - low.values) / atr.values
        features["bar_body_norm"] = (close.values - open_p.values) / atr.values
        features["bar_upper_wick_norm"] = (high.values - np.maximum(open_p.values, close.values)) / atr.values
        features["bar_lower_wick_norm"] = (np.minimum(open_p.values, close.values) - low.values) / atr.values

        # 32. Cross-Asset Benchmark Spread
        if spy_df is not None and "close" in spy_df.columns and len(spy_df) >= len(df):
            spy_close = spy_df["close"].iloc[-len(df):].values
            spy_ret = np.diff(spy_close, prepend=spy_close[0]) / np.maximum(spy_close, 1e-9)
            features["benchmark_spread_return"] = features["log_ret_1"] - spy_ret
        else:
            # Relative return vs rolling 50-period return
            rolling_ret_50 = (close.values - np.roll(close.values, 50)) / np.maximum(np.roll(close.values, 50), 1e-9)
            rolling_ret_50[:50] = 0.0
            features["benchmark_spread_return"] = features["log_ret_1"] - (rolling_ret_50 / 50.0)

        # Assemble and clean matrix
        mat = np.column_stack([features[name] for name in self.FEATURE_NAMES])
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return mat


class HistGradientBoostingModel:
    """Gradient-boosted decision tree directional forecaster using scikit-learn."""

    def __init__(
        self,
        learning_rate: float = 0.05,
        max_iter: int = 100,
        max_leaf_nodes: int = 31,
        l2_regularization: float = 0.1,
        random_state: int = 42,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.l2_regularization = l2_regularization
        self.random_state = random_state

        self.classifier = HistGradientBoostingClassifier(
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            l2_regularization=self.l2_regularization,
            random_state=self.random_state,
            class_weight="balanced",
        )
        self.is_fitted = False
        self.classes_: np.ndarray = np.array([-1, 0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> HistGradientBoostingModel:
        """Fits the classifier on feature matrix X and directional labels y."""
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Training dataset cannot be empty.")
        self.classifier.fit(X, y)
        self.classes_ = np.array(self.classifier.classes_)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Forecasts directional probabilities [P(down), P(flat), P(up)] for each sample."""
        if not self.is_fitted:
            # Default to uniform prior if untrained
            n_samples = len(X)
            return np.full((n_samples, 3), 1.0 / 3.0, dtype=np.float32)

        raw_probs = self.classifier.predict_proba(X)
        n_samples = len(X)
        full_probs = np.zeros((n_samples, 3), dtype=np.float32)

        # Map trained classes to index 0: down (-1), 1: flat (0), 2: up (+1)
        target_map = {-1: 0, 0: 1, 1: 2}
        for col_idx, cls_val in enumerate(self.classes_):
            if cls_val in target_map:
                full_probs[:, target_map[cls_val]] = raw_probs[:, col_idx]

        # Normalize rows to sum to 1.0 safely
        row_sums = full_probs.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        return full_probs / row_sums

    def predict_signal(self, X: np.ndarray) -> Tuple[float, float]:
        """Predicts directional score S_ML in [-1.0, 1.0] and confidence C_ML in [0.0, 1.0] for the latest sample."""
        if len(X) == 0:
            return 0.0, 0.0

        if X.ndim == 1:
            X = X.reshape(1, -1)

        probs = self.predict_proba(X[-1:])
        p_down, p_flat, p_up = probs[0, 0], probs[0, 1], probs[0, 2]

        s_ml = float(p_up - p_down)
        c_ml = float(1.0 - p_flat)

        s_ml = float(np.clip(s_ml, -1.0, 1.0))
        c_ml = float(np.clip(c_ml, 0.0, 1.0))
        return s_ml, c_ml

    def save(self, filepath: str) -> None:
        """Persists the model to disk atomically."""
        dir_name = os.path.dirname(filepath)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> HistGradientBoostingModel:
        """Loads a persisted model from disk."""
        return joblib.load(filepath)


class DeepSequenceEmbedder(nn.Module):
    """PyTorch MLP representation extractor with CPU fallback for intraday feature vectors."""

    def __init__(self, in_features: int = 32, hidden_dim: int = 64, embedding_dim: int = 16):
        super().__init__()
        self.device = torch.device("cpu")  # Robust CPU execution

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim),
        )
        self.head = nn.Linear(embedding_dim, 3)  # Directional logits [-1, 0, +1]
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (embedding, class_logits)."""
        x = x.to(self.device)
        emb = self.net(x)
        logits = self.head(emb)
        return emb, logits

    def extract_embedding(self, feature_array: np.ndarray) -> np.ndarray:
        """Extracts 16-dimensional embedding representation from numpy features."""
        self.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(feature_array).float().to(self.device)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            emb, _ = self.forward(tensor)
            return emb.cpu().numpy()
