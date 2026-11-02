"""High-Performance Technical Indicator Suite for modeloINV.

Provides vectorized, scale-invariant technical indicator calculations including:
- Exponential Moving Averages (EMA 9, 21, 50, 200)
- Relative Strength Index (RSI 14 with Wilder smoothing)
- Moving Average Convergence Divergence (MACD 12, 26, 9)
- Bollinger Bands (20, 2 std, %B, Bandwidth)
- Average True Range (ATR 14 with flash-crash clipping)
- Volume Weighted Average Price (VWAP with zero-volume protection)
- Average Directional Index (ADX 14 with +DI / -DI)
- Scale-invariant normalized scoring for quantitative signals.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from modeloinv.core.models import Bar


def _standardize_ohlcv_dataframe(data: Union[pd.DataFrame, Sequence[Bar]]) -> pd.DataFrame:
    """Normalize input into a DataFrame with lowercase columns: open, high, low, close, volume."""
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        col_map = {col: col.lower() for col in df.columns}
        df = df.rename(columns=col_map)
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"DataFrame missing required column: '{col}'")
        return df

    # Sequence of Bar objects
    rows = []
    for b in data:
        rows.append({
            "timestamp": b.timestamp,
            "open": float(b.open),
            "high": float(b.high),
            "low": float(b.low),
            "close": float(b.close),
            "volume": float(b.volume),
        })
    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculate Exponential Moving Average (EMA) with span N.

    Formula:
        alpha = 2 / (span + 1)
        EMA_t = alpha * P_t + (1 - alpha) * EMA_{t-1}
    """
    if len(series) == 0:
        return pd.Series(dtype=np.float64)
    return series.ewm(span=span, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI) using Wilder's smoothing.

    Formula:
        delta = P_t - P_{t-1}
        U = max(delta, 0), D = max(-delta, 0)
        smooth_U = Wilder(U, period), smooth_D = Wilder(D, period)
        RS = smooth_U / (smooth_D + 1e-9)
        RSI = 100 - (100 / (1 + RS))
    """
    if len(series) == 0:
        return pd.Series(dtype=np.float64)
    if len(series) == 1:
        return pd.Series([50.0], index=series.index, dtype=np.float64)

    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder smoothing corresponds to ewm with alpha = 1 / period
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    # Backfill warm-up period safely with simple mean or first valid
    simple_gain = gain.expanding().mean()
    simple_loss = loss.expanding().mean()
    avg_gain = avg_gain.fillna(simple_gain).fillna(0.0)
    avg_loss = avg_loss.fillna(simple_loss).fillna(0.0)

    rs = avg_gain / np.maximum(avg_loss, 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.clip(0.0, 100.0)


def calculate_macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Dict[str, pd.Series]:
    """Calculate Moving Average Convergence Divergence (MACD).

    Returns:
        Dict containing:
            'macd_line': EMA_fast - EMA_slow
            'signal_line': EMA(macd_line, signal_period)
            'hist': macd_line - signal_line
    """
    if len(series) == 0:
        empty = pd.Series(dtype=np.float64)
        return {"macd_line": empty, "signal_line": empty, "hist": empty}

    ema_fast = calculate_ema(series, fast_period)
    ema_slow = calculate_ema(series, slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal_period)
    hist = macd_line - signal_line

    return {
        "macd_line": macd_line,
        "signal_line": signal_line,
        "hist": hist,
    }


def calculate_bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> Dict[str, pd.Series]:
    """Calculate Bollinger Bands (Middle, Upper, Lower, %B, Bandwidth).

    Formula:
        SMA = rolling_mean(series, window)
        std = rolling_std(series, window)
        Upper = SMA + num_std * std
        Lower = SMA - num_std * std
        %B = (series - Lower) / (Upper - Lower + 1e-9)
        Bandwidth = (Upper - Lower) / (SMA + 1e-9)
    """
    if len(series) == 0:
        empty = pd.Series(dtype=np.float64)
        return {
            "middle": empty,
            "upper": empty,
            "lower": empty,
            "pct_b": empty,
            "bandwidth": empty,
        }

    middle = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std(ddof=0).fillna(0.0)

    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    span_width = upper - lower
    pct_b = (series - lower) / np.maximum(span_width, 1e-9)
    bandwidth = span_width / np.maximum(middle.abs(), 1e-9)

    return {
        "middle": middle,
        "upper": upper,
        "lower": lower,
        "pct_b": pct_b,
        "bandwidth": bandwidth,
    }


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    clip_multiplier: float = 3.0,
) -> pd.Series:
    """Calculate Average True Range (ATR) with flash crash clipping.

    Formula:
        TR_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)
        ATR_t = Wilder(TR, period)
        ATR clipped to clip_multiplier * rolling_median(ATR, 50) if available.
    """
    if len(close) == 0:
        return pd.Series(dtype=np.float64)

    prev_close = close.shift(1).fillna(close)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder smoothing (alpha = 1 / period)
    atr = tr.ewm(alpha=1.0 / period, min_periods=1, adjust=False).mean()

    # Flash crash clipping: protect stops against extreme price shocks
    if len(atr) >= period:
        rolling_median = atr.rolling(window=max(period * 2, 30), min_periods=period).median()
        upper_limit = rolling_median * clip_multiplier
        # Only clip where rolling median is well defined and positive
        valid_mask = upper_limit.notna() & (upper_limit > 0)
        atr = pd.Series(
            np.where(valid_mask & (atr > upper_limit), upper_limit, atr),
            index=atr.index,
        )

    # Invariant: ATR must be strictly positive
    return np.maximum(atr, 1e-6)


def calculate_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Calculate Volume-Weighted Average Price (VWAP) with zero-volume protection.

    Formula:
        P_typical = (H + L + C) / 3
        VWAP = cumsum(P_typical * V) / cumsum(V)
    If cumsum(V) == 0, falls back to P_typical.
    """
    if len(close) == 0:
        return pd.Series(dtype=np.float64)

    typical_price = (high + low + close) / 3.0
    vol = volume.clip(lower=0.0).fillna(0.0)

    cum_vol_price = (typical_price * vol).cumsum()
    cum_vol = vol.cumsum()

    # Guard division by zero volume: fallback to typical price where volume is 0
    vwap = np.where(cum_vol > 0, cum_vol_price / cum_vol, typical_price)
    return pd.Series(vwap, index=close.index, dtype=np.float64)


def calculate_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> Dict[str, pd.Series]:
    """Calculate Average Directional Index (ADX) along with +DI and -DI.

    Returns:
        Dict containing:
            'adx': ADX line in [0, 100]
            'plus_di': +DI in [0, 100]
            'minus_di': -DI in [0, 100]
    """
    if len(close) == 0:
        empty = pd.Series(dtype=np.float64)
        return {"adx": empty, "plus_di": empty, "minus_di": empty}

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_series = pd.Series(plus_dm, index=high.index)
    minus_dm_series = pd.Series(minus_dm, index=low.index)

    # Wilder smoothing of DM and TR
    atr = calculate_atr(high, low, close, period=period, clip_multiplier=10.0)
    smooth_plus_dm = plus_dm_series.ewm(alpha=1.0 / period, min_periods=1, adjust=False).mean()
    smooth_minus_dm = minus_dm_series.ewm(alpha=1.0 / period, min_periods=1, adjust=False).mean()

    safe_atr = np.maximum(atr, 1e-9)
    plus_di = (100.0 * smooth_plus_dm / safe_atr).clip(0.0, 100.0)
    minus_di = (100.0 * smooth_minus_dm / safe_atr).clip(0.0, 100.0)

    di_sum = np.maximum(plus_di + minus_di, 1e-9)
    dx = (100.0 * (plus_di - minus_di).abs() / di_sum).clip(0.0, 100.0)

    adx = dx.ewm(alpha=1.0 / period, min_periods=1, adjust=False).mean().clip(0.0, 100.0)

    return {
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
    }


class TechnicalIndicatorSuite:
    """Consolidated Technical Indicator Suite for high-performance intraday analytics."""

    def __init__(self, atr_period: int = 14, rsi_period: int = 14, adx_period: int = 14):
        self.atr_period = atr_period
        self.rsi_period = rsi_period
        self.adx_period = adx_period

    def compute_all_indicators(
        self,
        data: Union[pd.DataFrame, Sequence[Bar]],
    ) -> pd.DataFrame:
        """Computes all technical indicators and normalized scores into a unified DataFrame.

        Columns output:
            - open, high, low, close, volume
            - ema_9, ema_21, ema_50, ema_200
            - rsi_14
            - macd_line, macd_signal, macd_hist
            - bb_middle, bb_upper, bb_lower, bb_pct_b, bb_bandwidth
            - atr_14
            - vwap
            - adx_14, plus_di_14, minus_di_14
            - s_ema, s_rsi, s_macd, s_bb, s_vwap (scale-invariant normalized scores in [-1.0, 1.0])
        """
        df = _standardize_ohlcv_dataframe(data)
        if len(df) == 0:
            return df

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # 1. EMAs
        df["ema_9"] = calculate_ema(close, 9)
        df["ema_21"] = calculate_ema(close, 21)
        df["ema_50"] = calculate_ema(close, 50)
        df["ema_200"] = calculate_ema(close, 200)

        # 2. RSI
        df["rsi_14"] = calculate_rsi(close, self.rsi_period)

        # 3. MACD
        macd_dict = calculate_macd(close, fast_period=12, slow_period=26, signal_period=9)
        df["macd_line"] = macd_dict["macd_line"]
        df["macd_signal"] = macd_dict["signal_line"]
        df["macd_hist"] = macd_dict["hist"]

        # 4. Bollinger Bands
        bb_dict = calculate_bollinger_bands(close, window=20, num_std=2.0)
        df["bb_middle"] = bb_dict["middle"]
        df["bb_upper"] = bb_dict["upper"]
        df["bb_lower"] = bb_dict["lower"]
        df["bb_pct_b"] = bb_dict["pct_b"]
        df["bb_bandwidth"] = bb_dict["bandwidth"]

        # 5. ATR
        df["atr_14"] = calculate_atr(high, low, close, period=self.atr_period)

        # 6. VWAP
        df["vwap"] = calculate_vwap(high, low, close, volume)

        # 7. ADX
        adx_dict = calculate_adx(high, low, close, period=self.adx_period)
        df["adx_14"] = adx_dict["adx"]
        df["plus_di_14"] = adx_dict["plus_di"]
        df["minus_di_14"] = adx_dict["minus_di"]

        # 8. Scale-Invariant Normalized Scores in [-1.0, 1.0]
        safe_atr = np.maximum(df["atr_14"], 1e-9)

        # S_ema: tanh of EMA 9 - EMA 21 normalized by ATR
        df["s_ema"] = np.tanh((df["ema_9"] - df["ema_21"]) / safe_atr).clip(-1.0, 1.0)

        # S_rsi: (50 - RSI) / 25 clipped to [-1, 1] (oversold < 30 gives positive score; overbought > 70 gives negative)
        df["s_rsi"] = ((50.0 - df["rsi_14"]) / 25.0).clip(-1.0, 1.0)

        # S_macd: tanh of Hist normalized by (0.5 * ATR)
        df["s_macd"] = np.tanh(df["macd_hist"] / (0.5 * safe_atr)).clip(-1.0, 1.0)

        # S_bb: 1.0 - 2.0 * %B clipped to [-1, 1]
        df["s_bb"] = (1.0 - 2.0 * df["bb_pct_b"]).clip(-1.0, 1.0)

        # S_vwap: tanh of (VWAP - Close) / ATR
        df["s_vwap"] = np.tanh((df["vwap"] - close) / safe_atr).clip(-1.0, 1.0)

        # Clean any remaining NaNs or Infs safely
        df = df.bfill().fillna(0.0)
        return df

    def get_latest_indicators(self, data: Union[pd.DataFrame, Sequence[Bar]]) -> Dict[str, float]:
        """Calculates indicators and returns the most recent bar's values as a dictionary."""
        df = self.compute_all_indicators(data)
        if len(df) == 0:
            return {}
        last_row = df.iloc[-1].to_dict()
        return {k: float(v) for k, v in last_row.items()}
