"""Adversarial Stress Test Suite for Milestone 3 (Quant Strategy Engine).

Targets:
1. Zero-volume periods, single-price flat bars, infinite/extreme returns.
2. Flash crash spikes (huge wicks) to verify ATR clipping and VWAP stability.
3. Regime transitions under noisy/choppy conditions to verify hysteresis and score bounds in [-1.0, 1.0].
4. Multi-Armed Bandit allocation under negative, zero, and extreme Sharpe regimes.
5. Micro-capital fractional sizing under boundary conditions (€100 account constraints).
6. Reinforcement learning state robustness and cost-penalized reward bounds.
7. Multi-timeframe bar synchronizer lookahead-free integrity under irregular timestamps.
"""

from __future__ import annotations

import datetime
import math
import numpy as np
import pandas as pd
import pytest
import torch

from modeloinv.core.models import AssetClass, Bar, TradingSignal
from modeloinv.strategy import (
    CandidateAsset,
    CostPenalizedReward,
    DynamicUniverseScreener,
    HistGradientBoostingModel,
    HybridStrategyEngine,
    MarketRegimeClassifier,
    MicroCapitalFractionalAdapter,
    MLFeaturePipeline,
    MultiArmedBanditAllocator,
    MultiTimeframeBarSynchronizer,
    PolicyAgent,
    RegimeState,
    RLStateBuilder,
    TechnicalIndicatorSuite,
    TripleBarrierLabeler,
    calculate_adx,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    calculate_vwap,
    compute_composite_quant_score,
    get_regime_weights,
)


# =====================================================================
# Helpers & Fixtures
# =====================================================================

def make_test_bar(
    timestamp: datetime.datetime,
    open_p: float,
    high_p: float,
    low_p: float,
    close_p: float,
    volume: float,
    symbol: str = "TEST",
    currency: str = "USD",
    asset_class: AssetClass = AssetClass.EQUITY,
) -> Bar:
    return Bar(
        timestamp=timestamp,
        symbol=symbol,
        open=float(open_p),
        high=float(high_p),
        low=float(low_p),
        close=float(close_p),
        volume=float(volume),
        asset_class=asset_class,
        currency=currency,
    )


# =====================================================================
# 1. Zero-Volume Periods and Degenerate Candlesticks
# =====================================================================

class TestZeroVolumeAndFlatBarsStress:
    """Stress tests covering zero-volume streams and completely flat candlesticks."""

    def test_complete_zero_volume_stream_vwap_and_indicators(self):
        """Verify 100 consecutive bars with 0 volume do not crash VWAP or generate NaNs."""
        base_ts = datetime.datetime(2026, 1, 15, 9, 30)
        bars = [
            make_test_bar(
                timestamp=base_ts + datetime.timedelta(minutes=i),
                open_p=100.0 + (i * 0.1),
                high_p=100.5 + (i * 0.1),
                low_p=99.5 + (i * 0.1),
                close_p=100.2 + (i * 0.1),
                volume=0.0,  # Strict zero volume
            )
            for i in range(100)
        ]

        suite = TechnicalIndicatorSuite()
        df = suite.compute_all_indicators(bars)

        assert len(df) == 100
        # VWAP must fall back to typical price without crashing or emitting NaNs
        assert not df["vwap"].isna().any()
        assert not np.isinf(df["vwap"]).any()
        expected_last_typical = (df["high"].iloc[-1] + df["low"].iloc[-1] + df["close"].iloc[-1]) / 3.0
        assert np.isclose(df["vwap"].iloc[-1], expected_last_typical)

        # Scale-invariant scores must all be finite and within [-1.0, 1.0]
        for score_col in ["s_ema", "s_rsi", "s_macd", "s_bb", "s_vwap"]:
            vals = df[score_col].values
            assert not np.isnan(vals).any(), f"NaN found in {score_col}"
            assert not np.isinf(vals).any(), f"Inf found in {score_col}"
            assert (vals >= -1.0).all() and (vals <= 1.0).all(), f"{score_col} out of [-1.0, 1.0] bounds"

    def test_single_price_flat_bars_zero_variance(self):
        """Verify 100 consecutive bars with Open=High=Low=Close=100.0 (zero variance)."""
        base_ts = datetime.datetime(2026, 1, 15, 9, 30)
        bars = [
            make_test_bar(
                timestamp=base_ts + datetime.timedelta(minutes=i),
                open_p=100.0,
                high_p=100.0,
                low_p=100.0,
                close_p=100.0,
                volume=100.0,
            )
            for i in range(100)
        ]

        suite = TechnicalIndicatorSuite()
        df = suite.compute_all_indicators(bars)

        # ATR must maintain strictly positive floor (>= 1e-6)
        assert (df["atr_14"] >= 1e-6).all()

        # Bollinger bandwidth is 0, middle == upper == lower == 100.0
        assert (df["bb_bandwidth"] == 0.0).all()
        assert (df["bb_middle"] == 100.0).all()

        # RSI must be finite in [0.0, 100.0]
        assert not df["rsi_14"].isna().any()
        assert (df["rsi_14"] >= 0.0).all() and (df["rsi_14"] <= 100.0).all()

        # ADX must be finite in [0.0, 100.0]
        assert not df["adx_14"].isna().any()
        assert (df["adx_14"] >= 0.0).all() and (df["adx_14"] <= 100.0).all()

        # ML Feature extraction on flat bars must yield 32 finite columns with zero NaNs
        pipeline = MLFeaturePipeline()
        features = pipeline.extract_features(df)
        assert features.shape == (100, 32)
        assert not np.isnan(features).any()
        assert not np.isinf(features).any()

    def test_extreme_gap_and_astronomical_log_returns(self):
        """Verify massive 1000x price explosion and 99.9% crash are handled cleanly without Inf."""
        base_ts = datetime.datetime(2026, 1, 15, 9, 30)
        # 10 bars at 10.0, then instant jump to 100,000.0, then crash to 0.01
        prices = [10.0] * 10 + [100_000.0] * 10 + [0.01] * 10
        bars = [
            make_test_bar(
                timestamp=base_ts + datetime.timedelta(minutes=i),
                open_p=p,
                high_p=p * 1.01,
                low_p=p * 0.99,
                close_p=p,
                volume=5000.0,
            )
            for i, p in enumerate(prices)
        ]

        pipeline = MLFeaturePipeline()
        features = pipeline.extract_features(bars)
        assert features.shape == (30, 32)
        assert not np.isnan(features).any()
        assert not np.isinf(features).any()

    def test_end_to_end_zero_volume_engine_processing(self):
        """Verify HybridStrategyEngine processes zero-volume incoming stream smoothly."""
        engine = HybridStrategyEngine()
        base_ts = datetime.datetime(2026, 1, 15, 9, 30)

        for i in range(25):
            bar = make_test_bar(
                timestamp=base_ts + datetime.timedelta(minutes=i),
                open_p=50.0,
                high_p=50.2,
                low_p=49.8,
                close_p=50.1,
                volume=0.0,  # Zero volume
            )
            sig = engine.process_bar(bar, account_equity_eur=100.0)
            assert isinstance(sig, TradingSignal)
            assert sig.action in ("BUY", "SELL", "HOLD")
            assert -1.0 <= sig.target_weight <= 1.0
            assert -1.0 <= sig.quant_score <= 1.0
            assert 0.0 <= sig.confidence <= 1.0


# =====================================================================
# 2. Flash Crash Spikes & Tail Shocks
# =====================================================================

class TestFlashCrashAndTailShocksStress:
    """Stress tests for flash crashes, massive wicks, and ATR clipping."""

    def test_flash_crash_spike_atr_clipping_bounds(self):
        """Verify a massive lower wick (flash crash) clips ATR against 3x rolling median."""
        base_ts = datetime.datetime(2026, 1, 15, 9, 30)
        bars = []
        # 60 normal bars with ATR ~ 1.0 around price 100.0
        for i in range(60):
            bars.append(
                make_test_bar(
                    timestamp=base_ts + datetime.timedelta(minutes=i),
                    open_p=100.0,
                    high_p=100.5,
                    low_p=99.5,
                    close_p=100.0,
                    volume=1000.0,
                )
            )

        # Bar 60: Flash crash spike down to 20.0 (80-point drop), closing back up at 98.0
        bars.append(
            make_test_bar(
                timestamp=base_ts + datetime.timedelta(minutes=60),
                open_p=100.0,
                high_p=100.0,
                low_p=20.0,  # 80-point drop!
                close_p=98.0,
                volume=50000.0,
            )
        )

        df = pd.DataFrame([
            {"high": b.high, "low": b.low, "close": b.close}
            for b in bars
        ])

        atr_series = calculate_atr(df["high"], df["low"], df["close"], period=14, clip_multiplier=3.0)
        flash_atr = atr_series.iloc[-1]

        # Prior ATR was ~1.0. With 3.0 clip multiplier, flash_atr must be bounded <= 3.5
        # (Without clipping, an 80 point TR would spike Wilder ATR to > 6.0)
        assert flash_atr < 4.0, f"ATR failed to clip on flash crash! Spike value: {flash_atr}"
        assert flash_atr > 0.0

    def test_flash_rally_spike_take_profit_and_stop_loss_sanity(self):
        """Verify massive upper wick does not produce inverted or nonsensical TP/SL levels."""
        engine = HybridStrategyEngine(k_take_profit=3.0, k_stop_loss=2.0)
        base_ts = datetime.datetime(2026, 1, 15, 9, 30)

        # 50 normal bars
        for i in range(50):
            engine.process_bar(
                make_test_bar(
                    timestamp=base_ts + datetime.timedelta(minutes=i),
                    open_p=100.0,
                    high_p=101.0,
                    low_p=99.0,
                    close_p=100.0,
                    volume=1000.0,
                )
            )

        # Bar 50: Flash rally up to 300.0 (+200% wick), closes at 102.0
        flash_bar = make_test_bar(
            timestamp=base_ts + datetime.timedelta(minutes=50),
            open_p=100.0,
            high_p=300.0,
            low_p=99.0,
            close_p=102.0,
            volume=20000.0,
        )
        sig = engine.process_bar(flash_bar, account_equity_eur=100.0)

        assert isinstance(sig, TradingSignal)
        if sig.target_weight > 0:
            assert sig.stop_loss is not None and sig.take_profit is not None
            assert sig.stop_loss < flash_bar.close < sig.take_profit
            # SL must be positive price
            assert sig.stop_loss > 0.0
        elif sig.target_weight < 0:
            assert sig.stop_loss is not None and sig.take_profit is not None
            assert sig.take_profit < flash_bar.close < sig.stop_loss

    def test_triple_barrier_labeler_on_flash_crash(self):
        """Verify TripleBarrierLabeler cleanly detects downside breach during flash crash."""
        labeler = TripleBarrierLabeler(k_take_profit=2.0, k_stop_loss=2.0, horizon_bars=10)
        prices = np.array([100.0, 100.0, 100.0, 20.0, 100.0], dtype=np.float64)
        atrs = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        highs = prices + 0.5
        lows = prices - 0.5

        labels = labeler.label_path(prices, atrs, highs=highs, lows=lows)
        # Entry at index 0, 1, 2 should see the flash crash at index 3 as a downside hit (-1)
        assert labels[0] == -1
        assert labels[1] == -1
        assert labels[2] == -1


# =====================================================================
# 3. Market Regime Transitions & Hysteresis Stress
# =====================================================================

class TestRegimeTransitionsAndQuantScoreBounds:
    """Stress tests for regime classification stability and composite quant score bounds."""

    def test_noisy_choppy_sawtooth_score_invariants(self):
        """Simulate high-frequency oscillating price series under choppy/noisy conditions."""
        classifier = MarketRegimeClassifier(
            adx_trending_threshold=25.0,
            adx_choppy_threshold=20.0,
            bandwidth_mean_reverting_max=0.05,
        )

        np.random.seed(1234)
        # Adversarial indicator values fluctuating right around thresholds
        for _ in range(500):
            adx_noisy = float(np.random.uniform(15.0, 30.0))
            slope_noisy = float(np.random.choice([-1.5, 0.0, 1.5]))
            bandwidth_noisy = float(np.random.uniform(0.01, 0.10))
            atr_ratio_noisy = float(np.random.uniform(0.005, 0.05))

            regime = classifier.classify_from_values(
                adx=adx_noisy,
                trend_slope=slope_noisy,
                bandwidth=bandwidth_noisy,
                atr_ratio=atr_ratio_noisy,
            )
            assert isinstance(regime, RegimeState)

            # Test composite score with arbitrary indicator signals
            test_ind = {
                "adx_14": adx_noisy,
                "s_ema": np.random.uniform(-1.0, 1.0),
                "s_macd": np.random.uniform(-1.0, 1.0),
                "s_rsi": np.random.uniform(-1.0, 1.0),
                "s_bb": np.random.uniform(-1.0, 1.0),
                "bb_bandwidth": bandwidth_noisy,
                "atr_ratio": atr_ratio_noisy,
                "trend_slope": slope_noisy,
            }

            score = compute_composite_quant_score(test_ind, regime=regime, classifier=classifier)
            assert not math.isnan(score)
            assert -1.0 <= score <= 1.0, f"Composite score {score} violated [-1.0, 1.0]"

    def test_instant_bull_to_bear_waterfall_regime_flip(self):
        """Verify immediate transition from sustained bull trend to panic bear trend."""
        suite = TechnicalIndicatorSuite()
        classifier = MarketRegimeClassifier()
        base_ts = datetime.datetime(2026, 1, 15, 9, 30)

        # 40 bars of steady bull trend: +1.0 per bar
        bull_bars = [
            make_test_bar(
                timestamp=base_ts + datetime.timedelta(minutes=i),
                open_p=100.0 + i,
                high_p=101.0 + i,
                low_p=99.8 + i,
                close_p=100.9 + i,
                volume=1000.0,
            )
            for i in range(40)
        ]

        # 40 bars of waterfall crash: -2.0 per bar
        bear_bars = [
            make_test_bar(
                timestamp=base_ts + datetime.timedelta(minutes=40 + i),
                open_p=140.0 - (i * 2.0),
                high_p=140.2 - (i * 2.0),
                low_p=137.5 - (i * 2.0),
                close_p=138.0 - (i * 2.0),
                volume=5000.0,
            )
            for i in range(40)
        ]

        all_bars = bull_bars + bear_bars
        df = suite.compute_all_indicators(all_bars)

        # Near end of bull segment (bar 38): should be TRENDING_BULL
        row_bull = df.iloc[38].to_dict()
        regime_bull = classifier.classify(row_bull)
        score_bull = compute_composite_quant_score(row_bull, regime=regime_bull)
        assert regime_bull == RegimeState.TRENDING_BULL
        assert score_bull > 0.0

        # Near end of bear segment (bar 78): should be TRENDING_BEAR
        row_bear = df.iloc[78].to_dict()
        regime_bear = classifier.classify(row_bear)
        score_bear = compute_composite_quant_score(row_bear, regime=regime_bear)
        assert regime_bear == RegimeState.TRENDING_BEAR
        assert score_bear < 0.0

    def test_regime_weights_sum_to_one_exhaustively(self):
        """Verify weights strictly sum to 1.0 across all enum states."""
        for state in RegimeState:
            w_trend, w_mom, w_revert = get_regime_weights(state)
            assert np.isclose(w_trend + w_mom + w_revert, 1.0)
            assert w_trend >= 0 and w_mom >= 0 and w_revert >= 0


# =====================================================================
# 4. Multi-Armed Bandit Dynamic Allocation Stress
# =====================================================================

class TestBanditAllocationAdversarialStress:
    """Stress tests for MultiArmedBanditAllocator under degenerate Sharpe ratios."""

    def test_all_negative_sharpe_ratios_allocation(self):
        """All assets losing money: verify allocation assigns highest weight to least-negative asset."""
        allocator = MultiArmedBanditAllocator(temperature=1.0, max_concentration=0.50)
        sharpes = {
            "BTC": -4.5,
            "ETH": -6.2,
            "SOL": -10.0,
            "BNB": -1.8,  # Least negative asset
            "ADA": -7.1,
        }

        weights = allocator.allocate(sharpes)

        # Invariant 1: All assets present
        assert set(weights.keys()) == set(sharpes.keys())
        # Invariant 2: Sum of weights == 1.0
        assert np.isclose(sum(weights.values()), 1.0, atol=1e-5)
        # Invariant 3: Single asset concentration cap <= 0.50
        for sym, w in weights.items():
            assert w <= 0.50 + 1e-5, f"{sym} weight {w} exceeded 50% cap"
            assert w > 0.0, f"{sym} weight {w} was non-positive"

        # Invariant 4: BNB must have strictly highest weight
        assert weights["BNB"] == max(weights.values())
        # Invariant 5: SOL (worst performer) must have lowest weight
        assert weights["SOL"] == min(weights.values())

    def test_all_zero_sharpe_ratios_uniform_distribution(self):
        """Uninformative stagnant assets (Sharpe = 0.0): verify exact uniform distribution."""
        allocator = MultiArmedBanditAllocator(temperature=1.0, max_concentration=0.50)
        sharpes = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
        weights = allocator.allocate(sharpes)

        assert np.isclose(sum(weights.values()), 1.0, atol=1e-5)
        for sym, w in weights.items():
            assert np.isclose(w, 0.25, atol=1e-4)

    def test_dominant_monopolistic_asset_concentration_clipping(self):
        """One asset overwhelmingly outperforms (Sharpe +15 vs -5): verify 50% cap holds and remainder redistributes."""
        allocator = MultiArmedBanditAllocator(temperature=1.0, max_concentration=0.50)
        sharpes = {
            "SUPER_STAR": 15.0,
            "DOG_1": -5.0,
            "DOG_2": -5.0,
            "DOG_3": -5.0,
        }

        weights = allocator.allocate(sharpes)

        assert np.isclose(sum(weights.values()), 1.0, atol=1e-5)
        # SUPER_STAR must be exactly capped at 50%
        assert np.isclose(weights["SUPER_STAR"], 0.50, atol=1e-4)

        # The other 50% must be evenly distributed among the 3 DOG assets (~16.66% each)
        for dog in ("DOG_1", "DOG_2", "DOG_3"):
            assert np.isclose(weights[dog], 0.50 / 3.0, atol=1e-4)

    def test_single_asset_and_two_asset_boundary(self):
        """Edge cases: universe of size 1 and size 2."""
        allocator = MultiArmedBanditAllocator(max_concentration=0.50)

        # 1 asset: capped at 0.50
        w1 = allocator.allocate({"ONLY_ONE": 2.5})
        assert w1 == {"ONLY_ONE": 0.50}

        # 2 assets: both capped at 0.50
        w2 = allocator.allocate({"ASSET_A": 10.0, "ASSET_B": -10.0})
        assert np.isclose(w2["ASSET_A"], 0.50, atol=1e-4)
        assert np.isclose(w2["ASSET_B"], 0.50, atol=1e-4)
        assert np.isclose(sum(w2.values()), 1.0)

    def test_empty_asset_map(self):
        """Empty input dict returns empty dict."""
        allocator = MultiArmedBanditAllocator()
        assert allocator.allocate({}) == {}


# =====================================================================
# 5. Micro-Capital Fractional Adapter & Screener Stress
# =====================================================================

class TestMicroCapitalAndScreenerStress:
    """Stress tests for fractional sizing on €100 capital and dynamic screening."""

    def test_micro_capital_below_one_euro_threshold_rejection(self):
        """Orders below €1.00 minimum notional must be rejected (return None) to prevent fee thrashing."""
        adapter = MicroCapitalFractionalAdapter(min_order_notional_eur=1.00)

        # €100 capital * 0.005 target weight = €0.50 -> Below threshold -> None
        qty = adapter.calculate_order_quantity(
            symbol="AAPL",
            target_weight=0.005,
            account_equity_eur=100.0,
            price=150.0,
        )
        assert qty is None

        # €100 capital * 0.009 target weight = €0.90 -> Below threshold -> None
        qty2 = adapter.calculate_order_quantity(
            symbol="AAPL",
            target_weight=0.009,
            account_equity_eur=100.0,
            price=150.0,
        )
        assert qty2 is None

    def test_micro_capital_valid_fractional_shares(self):
        """Orders >= €1.00 must compute correct decimal quantity with appropriate precision."""
        adapter = MicroCapitalFractionalAdapter(min_order_notional_eur=1.00, default_eur_usd_rate=1.08)

        # €100 capital * 0.05 weight = €5.00 notional.
        # USD notional = 5.00 * 1.08 = $5.40.
        # Price = $65,000 (BTC).
        # Expected quantity = 5.40 / 65,000 = 0.000083 BTC (6 decimal precision)
        qty = adapter.calculate_order_quantity(
            symbol="BTC/USD",
            target_weight=0.05,
            account_equity_eur=100.0,
            price=65000.0,
            asset_class="CRYPTO",
        )
        assert qty is not None
        assert np.isclose(qty, 5.40 / 65000.0, atol=1e-6)
        assert qty > 0.0

    def test_screener_rejects_illiquid_and_low_volatility_assets(self):
        """Verify DynamicUniverseScreener strictly filters out illiquid (< $5M) and low-beta (< 0.5% ATR) assets."""
        screener = DynamicUniverseScreener(
            min_volume_24h_usd=5_000_000.0,
            min_volatility_ratio=0.005,  # 0.5%
            max_selected_assets=3,
        )

        candidates = [
            # Candidate 1: High volume ($10M), High volatility (2.0% ATR) -> ELIGIBLE
            CandidateAsset(
                symbol="GOOD_1", asset_class="EQUITY", price=100.0,
                volume_24h_usd=10_000_000.0, atr_14=2.0, sharpe_30d=1.5,
            ),
            # Candidate 2: Low volume ($2M < $5M), High volatility (3.0% ATR) -> REJECTED (Illiquid)
            CandidateAsset(
                symbol="BAD_VOL", asset_class="CRYPTO", price=50.0,
                volume_24h_usd=2_000_000.0, atr_14=1.5, sharpe_30d=2.0,
            ),
            # Candidate 3: High volume ($50M), Low volatility (0.2% ATR < 0.5%) -> REJECTED (Dead stock)
            CandidateAsset(
                symbol="BAD_ATR", asset_class="ETF", price=200.0,
                volume_24h_usd=50_000_000.0, atr_14=0.4, sharpe_30d=1.0,
            ),
            # Candidate 4: High volume ($20M), High volatility (1.5% ATR) -> ELIGIBLE
            CandidateAsset(
                symbol="GOOD_2", asset_class="EQUITY", price=80.0,
                volume_24h_usd=20_000_000.0, atr_14=1.2, sharpe_30d=0.8,
            ),
        ]

        selected = screener.filter_and_rank(candidates)
        selected_symbols = [a.symbol for a in selected]

        assert "GOOD_1" in selected_symbols
        assert "GOOD_2" in selected_symbols
        assert "BAD_VOL" not in selected_symbols
        assert "BAD_ATR" not in selected_symbols


# =====================================================================
# 6. Reinforcement Learning State & Reward Penalties
# =====================================================================

class TestReinforcementLearningStateAndRewardStress:
    """Stress tests for RL state clipping, cost penalties, and Huber DDQN stability."""

    def test_rl_state_builder_clamps_out_of_bounds_inputs(self):
        """Verify RLStateBuilder strictly clips extreme inputs and replaces NaNs."""
        state = RLStateBuilder.build_state(
            quant_score=999.0,        # Should clip to 1.0
            ml_score=-500.0,          # Should clip to -1.0
            ml_confidence=2.5,        # Should clip to 1.0
            adx=150.0,                # Should clip to 1.0
            atr=50.0, price=100.0,    # Ratio 0.5 should clip to 0.1
            current_weight=-5.0,      # Should clip to -1.0
            unrealized_pnl_pct=250.0, # >1.0 treated as % -> 2.5 -> clip 1.0
            drawdown_pct=85.0,        # >1.0 treated as % -> 0.85
            holding_time_bars=120.0,  # 120/60 = 2.0 -> clip 1.0
        )

        assert state.shape == (9,)
        assert state[0] == 1.0
        assert state[1] == -1.0
        assert state[2] == 1.0
        assert state[3] == 1.0
        assert state[4] == 0.1
        assert state[5] == -1.0
        assert state[6] == 1.0
        assert state[7] == 0.85
        assert state[8] == 1.0

    def test_cost_penalized_reward_catastrophic_drawdown_clipping(self):
        """Verify quadratic drawdown penalty and [-10.0, 10.0] clipping."""
        reward_fn = CostPenalizedReward(
            lambda_fee=1.0,
            lambda_slippage=1.0,
            lambda_drawdown=10.0,
            drawdown_threshold=0.05,
        )

        # Case 1: Huge loss (-€50) with heavy drawdown (40%)
        r_neg = reward_fn.calculate_reward(
            delta_pnl=-50.0,
            fee=2.0,
            slippage=3.0,
            drawdown=0.40,
        )
        # Raw reward is deeply negative, but clipped to -10.0
        assert r_neg == -10.0

        # Case 2: Astronomical win (+€100) -> clipped to +10.0
        r_pos = reward_fn.calculate_reward(
            delta_pnl=100.0,
            fee=0.1,
            slippage=0.1,
            drawdown=0.01,
        )
        assert r_pos == 10.0

        # Case 3: Zero PnL with fee and slippage -> strictly negative
        r_cost = reward_fn.calculate_reward(
            delta_pnl=0.0,
            fee=0.5,
            slippage=0.3,
            drawdown=0.0,
        )
        assert np.isclose(r_cost, -0.8)


# =====================================================================
# 7. Multi-Timeframe Synchronizer Integrity & Lookahead Stress
# =====================================================================

class TestMultiTimeframeSynchronizerStress:
    """Stress tests for multi-timeframe bar aggregation and lookahead bias prevention."""

    def test_synchronizer_lookahead_free_5m_aggregation(self):
        """Verify 5m bars are ONLY emitted when the 5m window has fully closed."""
        sync = MultiTimeframeBarSynchronizer(supported_timeframes=("5m",))
        base_ts = datetime.datetime(2026, 1, 15, 9, 30)

        # Bar 1 (09:30): Window starts -> emitted should be None
        bar1 = make_test_bar(base_ts, open_p=10.0, high_p=11.0, low_p=9.0, close_p=10.5, volume=100)
        out1 = sync.ingest_bar(bar1)
        assert out1["5m"] is None

        # Bars 2..4 (09:31, 09:32, 09:33): Within same window -> emitted None
        for m in (1, 2, 3, 4):
            bar = make_test_bar(
                base_ts + datetime.timedelta(minutes=m),
                open_p=10.5, high_p=12.0, low_p=10.0, close_p=11.5, volume=100,
            )
            out = sync.ingest_bar(bar)
            assert out["5m"] is None

        # Bar 6 (09:35): First bar of the NEXT 5m window!
        # The 09:30..09:35 window is now closed and MUST be emitted
        bar6 = make_test_bar(
            base_ts + datetime.timedelta(minutes=5),
            open_p=11.5, high_p=13.0, low_p=11.0, close_p=12.0, volume=200,
        )
        out6 = sync.ingest_bar(bar6)

        assert out6["5m"] is not None
        completed_5m = out6["5m"]
        assert completed_5m.timestamp == base_ts + datetime.timedelta(minutes=5)  # 09:35:00
        assert completed_5m.open == 10.0   # Open of first bar
        assert completed_5m.high == 12.0   # Max high of bars 09:30..09:34
        assert completed_5m.low == 9.0     # Min low of bars 09:30..09:34
        assert completed_5m.close == 11.5  # Close of bar 09:34
        assert completed_5m.volume == 500  # 5 * 100
