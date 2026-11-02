"""Hybrid AI Strategy Engine and Multi-Timeframe Synchronizer for modeloINV.

Combines:
- TechnicalIndicatorSuite (EMA, RSI, MACD, BB, ATR, VWAP, ADX)
- MarketRegimeClassifier (Trending Bull/Bear, Mean Reverting, Choppy)
- MLFeaturePipeline & HistGradientBoostingModel (32 features, directional probability)
- PyTorch PolicyAgent (DDQN target weight selection)
- MultiTimeframeBarSynchronizer (1m, 5m, 15m, 1h lookahead-free bar aggregation)
- MicroCapitalFractionalAdapter (€100 virtual capital execution)
- Generates canonical TradingSignal dataclass.
"""

from __future__ import annotations

import datetime
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from modeloinv.core.models import AssetClass, Bar, TradingSignal
from modeloinv.strategy.indicators import TechnicalIndicatorSuite
from modeloinv.strategy.ml_model import HistGradientBoostingModel, MLFeaturePipeline
from modeloinv.strategy.regime import MarketRegimeClassifier, RegimeState, compute_composite_quant_score
from modeloinv.strategy.rl_agent import PolicyAgent, RLStateBuilder
from modeloinv.strategy.universe import (
    DynamicUniverseScreener,
    MicroCapitalFractionalAdapter,
    MultiArmedBanditAllocator,
)


class MultiTimeframeBarSynchronizer:
    """Aggregates high-frequency 1-minute candlestick bars into 5m, 15m, and 1h intervals without lookahead bias."""

    def __init__(self, supported_timeframes: Sequence[str] = ("5m", "15m", "1h")):
        self.supported_timeframes = list(supported_timeframes)
        # symbol -> timeframe -> list of completed bars
        self.completed_bars: Dict[str, Dict[str, List[Bar]]] = defaultdict(lambda: defaultdict(list))
        # symbol -> timeframe -> ongoing accumulator {start_ts, open, high, low, close, volume, asset_class, currency}
        self.accumulators: Dict[str, Dict[str, Optional[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(lambda: None))

    @staticmethod
    def _get_timeframe_minutes(timeframe: str) -> int:
        tf = timeframe.lower()
        if tf == "1m":
            return 1
        elif tf == "5m":
            return 5
        elif tf == "15m":
            return 15
        elif tf == "1h":
            return 60
        return 5

    def ingest_bar(self, bar: Bar) -> Dict[str, Optional[Bar]]:
        """Ingests a new 1m bar and emits completed higher timeframe bars if window closed.

        Returns:
            Dict mapping timeframe (e.g. '5m') to completed Bar, or None if still accumulating.
        """
        emitted: Dict[str, Optional[Bar]] = {}
        sym = bar.symbol

        for tf in self.supported_timeframes:
            minutes = self._get_timeframe_minutes(tf)
            acc = self.accumulators[sym][tf]

            bar_ts = bar.timestamp
            # Calculate interval start timestamp (floored to window boundary)
            minute_mod = bar_ts.minute % minutes
            window_start = bar_ts.replace(
                minute=bar_ts.minute - minute_mod,
                second=0,
                microsecond=0,
            )

            if acc is None:
                # Start new accumulation window
                self.accumulators[sym][tf] = {
                    "start_ts": window_start,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "asset_class": bar.asset_class,
                    "currency": bar.currency,
                }
                emitted[tf] = None
            else:
                # Check if current bar belongs to the ongoing accumulator
                if window_start == acc["start_ts"]:
                    acc["high"] = max(acc["high"], bar.high)
                    acc["low"] = min(acc["low"], bar.low)
                    acc["close"] = bar.close
                    acc["volume"] += bar.volume
                    emitted[tf] = None
                else:
                    # Previous window is complete! Emit it.
                    completed_bar = Bar(
                        timestamp=acc["start_ts"] + datetime.timedelta(minutes=minutes),
                        symbol=sym,
                        open=acc["open"],
                        high=acc["high"],
                        low=acc["low"],
                        close=acc["close"],
                        volume=acc["volume"],
                        asset_class=acc["asset_class"],
                        currency=acc["currency"],
                    )
                    self.completed_bars[sym][tf].append(completed_bar)
                    emitted[tf] = completed_bar

                    # Start next accumulation window with current bar
                    self.accumulators[sym][tf] = {
                        "start_ts": window_start,
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close": bar.close,
                        "volume": bar.volume,
                        "asset_class": bar.asset_class,
                        "currency": bar.currency,
                    }

        return emitted

    def get_bars(self, symbol: str, timeframe: str) -> List[Bar]:
        """Returns completed historical bars for symbol and timeframe."""
        return list(self.completed_bars[symbol][timeframe])


class HybridStrategyEngine:
    """Production Hybrid AI Trading Strategy Engine combining Quant, ML, and RL layers."""

    def __init__(
        self,
        ml_model: Optional[HistGradientBoostingModel] = None,
        rl_agent: Optional[PolicyAgent] = None,
        k_take_profit: float = 3.0,
        k_stop_loss: float = 2.0,
        default_account_equity_eur: float = 100.0,
    ):
        self.k_take_profit = k_take_profit
        self.k_stop_loss = k_stop_loss
        self.account_equity_eur = default_account_equity_eur

        # Subsystems
        self.indicator_suite = TechnicalIndicatorSuite()
        self.regime_classifier = MarketRegimeClassifier()
        self.feature_pipeline = MLFeaturePipeline()
        self.ml_model = ml_model or HistGradientBoostingModel()
        self.rl_agent = rl_agent or PolicyAgent()
        self.universe_screener = DynamicUniverseScreener()
        self.bandit_allocator = MultiArmedBanditAllocator()
        self.fractional_adapter = MicroCapitalFractionalAdapter()
        self.synchronizer = MultiTimeframeBarSynchronizer()

        # Rolling in-memory history: symbol -> list of Bar
        self.bar_history: Dict[str, List[Bar]] = defaultdict(list)

    def update_history(self, bar: Bar) -> None:
        """Appends a new bar to in-memory symbol history and synchronizes timeframes."""
        self.bar_history[bar.symbol].append(bar)
        if len(self.bar_history[bar.symbol]) > 100:
            self.bar_history[bar.symbol] = self.bar_history[bar.symbol][-100:]
        self.synchronizer.ingest_bar(bar)

    def generate_signal(
        self,
        symbol: str,
        bars: Optional[Sequence[Bar]] = None,
        account_equity_eur: Optional[float] = None,
        current_weight: float = 0.0,
        unrealized_pnl_pct: float = 0.0,
        drawdown_pct: float = 0.0,
    ) -> TradingSignal:
        """Generates a complete, actionable TradingSignal combining Quant, ML, and RL layers."""
        equity = account_equity_eur or self.account_equity_eur
        active_bars = list(bars) if bars is not None else list(self.bar_history[symbol])

        if not active_bars:
            now = datetime.datetime.utcnow()
            return TradingSignal(
                timestamp=now,
                symbol=symbol,
                target_weight=0.0,
                action="HOLD",
                confidence=0.0,
                quant_score=0.0,
                ml_score=0.0,
                rl_action=0,
                metadata={"reason": "no_bars_available"},
            )

        latest_bar = active_bars[-1]
        timestamp = latest_bar.timestamp
        current_price = latest_bar.close

        # Step 1: Classical Quant Layer
        df_indicators = self.indicator_suite.compute_all_indicators(active_bars)
        latest_ind = df_indicators.iloc[-1].to_dict()

        regime = self.regime_classifier.classify(latest_ind)
        quant_score = compute_composite_quant_score(latest_ind, regime=regime)
        atr = max(float(latest_ind.get("atr_14", current_price * 0.01)), 1e-6)
        adx = float(latest_ind.get("adx_14", 20.0))

        # Step 2: Machine Learning Layer
        feature_matrix = self.feature_pipeline.extract_features(df_indicators)
        if len(feature_matrix) > 0 and self.ml_model.is_fitted:
            ml_score, ml_confidence = self.ml_model.predict_signal(feature_matrix)
        else:
            # Baseline proxy when ML is warm-up/unfitted
            ml_score = float(quant_score)
            ml_confidence = float(np.clip(abs(quant_score), 0.1, 0.9))

        # Step 3: Reinforcement Learning Layer
        rl_state = RLStateBuilder.build_state(
            quant_score=quant_score,
            ml_score=ml_score,
            ml_confidence=ml_confidence,
            adx=adx,
            atr=atr,
            price=current_price,
            current_weight=current_weight,
            unrealized_pnl_pct=unrealized_pnl_pct,
            drawdown_pct=drawdown_pct,
        )

        rl_action, target_weight = self.rl_agent.act(
            rl_state,
            epsilon=0.0,
            current_weight=current_weight,
        )

        # Step 4: Dynamic Stop-Loss & Take-Profit calculation
        if target_weight > 0:
            stop_loss = round(current_price - (self.k_stop_loss * atr), 4)
            take_profit = round(current_price + (self.k_take_profit * atr), 4)
        elif target_weight < 0:
            stop_loss = round(current_price + (self.k_stop_loss * atr), 4)
            take_profit = round(current_price - (self.k_take_profit * atr), 4)
        else:
            stop_loss = None
            take_profit = None

        # Step 5: Action Synthesis
        # Determine BUY / SELL / HOLD based on target weight delta and signal confidence
        delta_weight = target_weight - current_weight
        if target_weight > 0.05 and delta_weight > 0.01:
            action = "BUY"
        elif target_weight < -0.05 and delta_weight < -0.01:
            action = "SELL"
        elif abs(target_weight) <= 0.05 and abs(current_weight) > 0.05:
            # Closing open position
            action = "SELL" if current_weight > 0 else "BUY"
        else:
            action = "HOLD"

        # Fractional order feasibility check for €100 virtual capital
        order_qty = self.fractional_adapter.calculate_order_quantity(
            symbol=symbol,
            target_weight=target_weight,
            account_equity_eur=equity,
            price=current_price,
            asset_class=str(latest_bar.asset_class.value if isinstance(latest_bar.asset_class, AssetClass) else latest_bar.asset_class),
            currency=latest_bar.currency,
        )

        signal = TradingSignal(
            timestamp=timestamp,
            symbol=symbol,
            target_weight=round(target_weight, 4),
            action=action,
            confidence=round(ml_confidence, 4),
            quant_score=round(quant_score, 4),
            ml_score=round(ml_score, 4),
            rl_action=rl_action,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "regime": regime.value,
                "atr": round(atr, 4),
                "current_price": round(current_price, 4),
                "fractional_order_quantity": order_qty,
                "account_equity_eur": equity,
            },
        )
        return signal

    def process_bar(
        self,
        bar: Bar,
        account_equity_eur: Optional[float] = None,
        current_weight: float = 0.0,
        unrealized_pnl_pct: float = 0.0,
        drawdown_pct: float = 0.0,
    ) -> TradingSignal:
        """Processes an incoming bar, updates history, and emits a TradingSignal."""
        self.update_history(bar)
        return self.generate_signal(
            symbol=bar.symbol,
            bars=self.bar_history[bar.symbol],
            account_equity_eur=account_equity_eur,
            current_weight=current_weight,
            unrealized_pnl_pct=unrealized_pnl_pct,
            drawdown_pct=drawdown_pct,
        )


# Canonical alias adhering to Project Specification (Feature R1)
IntradayHybridSignalAggregator = HybridStrategyEngine
