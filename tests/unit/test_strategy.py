"""Comprehensive Unit Tests for Milestone 3: Hybrid AI Trading Strategy Engine.

Tests:
- TechnicalIndicatorSuite (EMA, RSI, MACD, BB, ATR, VWAP, ADX, normalized scoring)
- MarketRegimeClassifier & Composite Quant Score S_quant in [-1.0, 1.0]
- TripleBarrierLabeler, MLFeaturePipeline (32 features), HistGradientBoostingModel, DeepSequenceEmbedder
- RLStateBuilder (9-dim vector), PolicyAgent (DDQN), CostPenalizedReward
- DynamicUniverseScreener, MultiArmedBanditAllocator, MicroCapitalFractionalAdapter (€100 account)
- MultiTimeframeBarSynchronizer (no lookahead bias)
- HybridStrategyEngine (end-to-end signal synthesis)
"""

import datetime
import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch

from modeloinv.core.models import AssetClass, Bar, TradingSignal
from modeloinv.strategy import (
    CandidateAsset,
    CostPenalizedReward,
    DeepSequenceEmbedder,
    DynamicUniverseScreener,
    HistGradientBoostingModel,
    HybridStrategyEngine,
    MarketRegimeClassifier,
    MicroCapitalFractionalAdapter,
    MLFeaturePipeline,
    MultiArmedBanditAllocator,
    MultiTimeframeBarSynchronizer,
    PolicyAgent,
    PolicyNetwork,
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


@pytest.fixture
def sample_bars() -> list[Bar]:
    """Generates 100 deterministic 1-minute candlestick bars for testing."""
    start_time = datetime.datetime(2026, 1, 15, 9, 30)
    bars = []
    price = 100.0

    # Deterministic price series with trend and pullbacks
    np.random.seed(42)
    shocks = np.sin(np.linspace(0, 4 * np.pi, 100)) * 2.0 + np.random.normal(0, 0.3, 100)

    for i in range(100):
        price += shocks[i]
        price = max(price, 10.0)
        open_p = price - 0.2
        high_p = price + 0.8
        low_p = price - 0.7
        close_p = price + 0.3
        vol = 1000.0 + (i * 20.0)

        bar = Bar(
            timestamp=start_time + datetime.timedelta(minutes=i),
            symbol="AAPL",
            open=round(open_p, 4),
            high=round(high_p, 4),
            low=round(low_p, 4),
            close=round(close_p, 4),
            volume=round(vol, 2),
            asset_class=AssetClass.EQUITY,
            currency="USD",
        )
        bars.append(bar)

    return bars


@pytest.fixture
def sample_df(sample_bars) -> pd.DataFrame:
    """Converts sample bars into a pandas DataFrame."""
    return pd.DataFrame([
        {
            "timestamp": b.timestamp,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
        }
        for b in sample_bars
    ])


# =====================================================================
# 1. Technical Indicators Tests
# =====================================================================

class TestTechnicalIndicators:
    """Unit tests for technical indicator suite and vectorized math."""

    def test_ema_calculation(self):
        prices = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        ema9 = calculate_ema(prices, span=9)
        assert len(ema9) == len(prices)
        assert ema9.iloc[-1] > ema9.iloc[0]
        # Mathematical formula: alpha = 2 / (9 + 1) = 0.2
        alpha = 0.2
        expected_1 = alpha * 11.0 + (1 - alpha) * 10.0
        assert np.isclose(ema9.iloc[1], expected_1)

    def test_rsi_wilder_smoothing_and_bounds(self):
        # Monotonically increasing prices -> RSI should be high (> 70)
        rising = pd.Series([100.0 + i * 2.0 for i in range(30)])
        rsi_rising = calculate_rsi(rising, period=14)
        assert len(rsi_rising) == 30
        assert rsi_rising.iloc[-1] > 70.0
        assert (rsi_rising >= 0.0).all() and (rsi_rising <= 100.0).all()

        # Monotonically decreasing prices -> RSI should be low (< 30)
        falling = pd.Series([200.0 - i * 2.0 for i in range(30)])
        rsi_falling = calculate_rsi(falling, period=14)
        assert rsi_falling.iloc[-1] < 30.0
        assert (rsi_falling >= 0.0).all() and (rsi_falling <= 100.0).all()

    def test_macd_calculation(self):
        prices = pd.Series(np.linspace(100, 150, 50))
        macd = calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9)
        assert "macd_line" in macd and "signal_line" in macd and "hist" in macd
        assert len(macd["macd_line"]) == 50
        assert np.allclose(macd["hist"], macd["macd_line"] - macd["signal_line"])
        # For a steady upward trend, macd_line should be positive
        assert macd["macd_line"].iloc[-1] > 0

    def test_bollinger_bands_invariants(self):
        prices = pd.Series([100.0, 102.0, 98.0, 101.0, 103.0, 99.0] * 5)
        bb = calculate_bollinger_bands(prices, window=20, num_std=2.0)
        assert (bb["upper"] >= bb["middle"]).all()
        assert (bb["middle"] >= bb["lower"]).all()
        assert (bb["bandwidth"] >= 0.0).all()

    def test_atr_and_flash_crash_clipping(self):
        high = pd.Series([105.0] * 40 + [160.0])  # Bar 40 has a huge flash spike
        low = pd.Series([95.0] * 40 + [90.0])
        close = pd.Series([100.0] * 41)

        atr = calculate_atr(high, low, close, period=14, clip_multiplier=3.0)
        assert len(atr) == 41
        assert (atr > 0).all()
        # Normal ATR was around 10.0; spike bar should be capped around 30.0 rather than exploding to 70
        assert atr.iloc[-1] <= 35.0

    def test_vwap_zero_volume_protection(self):
        high = pd.Series([105.0, 106.0, 107.0])
        low = pd.Series([95.0, 96.0, 97.0])
        close = pd.Series([100.0, 101.0, 102.0])
        volume = pd.Series([0.0, 0.0, 0.0])  # Zero volume

        vwap = calculate_vwap(high, low, close, volume)
        assert len(vwap) == 3
        # In zero volume, VWAP should fallback to typical price rather than dividing by zero
        assert np.isclose(vwap.iloc[0], (105.0 + 95.0 + 100.0) / 3.0)
        assert not np.isnan(vwap).any()

    def test_adx_and_directional_indicators(self, sample_df):
        adx_dict = calculate_adx(sample_df["high"], sample_df["low"], sample_df["close"], period=14)
        assert "adx" in adx_dict and "plus_di" in adx_dict and "minus_di" in adx_dict
        adx = adx_dict["adx"]
        assert (adx >= 0.0).all() and (adx <= 100.0).all()
        assert (adx_dict["plus_di"] >= 0.0).all() and (adx_dict["plus_di"] <= 100.0).all()

    def test_indicator_suite_dataframe_pipeline(self, sample_df):
        suite = TechnicalIndicatorSuite()
        df_ind = suite.compute_all_indicators(sample_df)

        expected_cols = [
            "ema_9", "ema_21", "ema_50", "ema_200", "rsi_14",
            "macd_line", "macd_signal", "macd_hist",
            "bb_middle", "bb_upper", "bb_lower", "bb_pct_b", "bb_bandwidth",
            "atr_14", "vwap", "adx_14", "plus_di_14", "minus_di_14",
            "s_ema", "s_rsi", "s_macd", "s_bb", "s_vwap",
        ]
        for col in expected_cols:
            assert col in df_ind.columns, f"Missing column {col} in indicator dataframe"
            assert not df_ind[col].isna().any(), f"Column {col} contains NaN values"

        # Check scale-invariant scores are strictly in [-1.0, 1.0]
        for s_col in ["s_ema", "s_rsi", "s_macd", "s_bb", "s_vwap"]:
            assert (df_ind[s_col] >= -1.0).all() and (df_ind[s_col] <= 1.0).all()


# =====================================================================
# 2. Market Regime & Composite Quant Score Tests
# =====================================================================

class TestMarketRegimeAndQuantScore:
    """Tests for RegimeState classification and dynamic composite quant score."""

    def test_regime_classification_rules(self):
        clf = MarketRegimeClassifier(adx_trending_threshold=25.0, adx_choppy_threshold=20.0)

        # Bull trend: ADX >= 25 and slope > 0
        assert clf.classify_from_values(adx=30.0, trend_slope=0.05) == RegimeState.TRENDING_BULL
        assert clf.classify_from_values(adx=26.0, ema_9=105.0, ema_21=100.0) == RegimeState.TRENDING_BULL

        # Bear trend: ADX >= 25 and slope < 0
        assert clf.classify_from_values(adx=32.0, trend_slope=-0.03) == RegimeState.TRENDING_BEAR
        assert clf.classify_from_values(adx=28.0, ema_9=95.0, ema_21=100.0) == RegimeState.TRENDING_BEAR

        # Mean Reverting: ADX < 20 and low bandwidth
        assert clf.classify_from_values(adx=15.0, bandwidth=0.03) == RegimeState.MEAN_REVERTING

        # Choppy / Volatile: ADX < 20 but high ATR ratio
        assert clf.classify_from_values(adx=18.0, bandwidth=0.08, atr_ratio=0.03) == RegimeState.CHOPPY

    def test_regime_weights_sum_to_one(self):
        for state in RegimeState:
            w_trend, w_mom, w_revert = get_regime_weights(state)
            assert np.isclose(w_trend + w_mom + w_revert, 1.0)
            assert w_trend >= 0.0 and w_mom >= 0.0 and w_revert >= 0.0

    def test_composite_quant_score_bounds_and_dynamics(self):
        # Bullish indicators
        bull_ind = {
            "s_ema": 0.8,
            "s_macd": 0.7,
            "s_rsi": 0.5,
            "s_bb": 0.6,
            "adx_14": 30.0,
            "ema_9": 105.0,
            "ema_21": 100.0,
        }
        score_bull = compute_composite_quant_score(bull_ind)
        assert -1.0 <= score_bull <= 1.0
        assert score_bull > 0.5

        # Extreme clipping verification
        extreme_ind = {"s_ema": 2.5, "s_macd": 3.0, "s_rsi": 1.5, "s_bb": 2.0}
        score_extreme = compute_composite_quant_score(extreme_ind, regime=RegimeState.TRENDING_BULL)
        assert score_extreme == 1.0


# =====================================================================
# 3. Machine Learning Layer Tests
# =====================================================================

class TestMachineLearningLayer:
    """Tests for Triple-Barrier labeling, 32-dim Feature Pipeline, and ML models."""

    def test_triple_barrier_labeler(self):
        labeler = TripleBarrierLabeler(k_take_profit=2.0, k_stop_loss=2.0, horizon_bars=5)

        # Case 1: Upper take-profit touched first -> label +1
        prices_up = np.array([100.0, 101.0, 102.5, 105.0, 104.0, 103.0])
        atrs = np.array([2.0] * len(prices_up))  # Upper = 100 + 2*2 = 104
        labels_up = labeler.label_path(prices_up, atrs)
        assert labels_up[0] == 1

        # Case 2: Lower stop-loss touched first -> label -1
        prices_down = np.array([100.0, 99.0, 97.0, 95.0, 98.0, 99.0])  # Lower = 100 - 2*2 = 96
        labels_down = labeler.label_path(prices_down, atrs)
        assert labels_down[0] == -1

        # Case 3: Flat prices within horizon -> label 0
        prices_flat = np.array([100.0, 100.1, 100.2, 100.1, 100.0, 100.2])
        labels_flat = labeler.label_path(prices_flat, atrs)
        assert labels_flat[0] == 0

    def test_ml_feature_pipeline_32_dimensions(self, sample_df):
        pipeline = MLFeaturePipeline()
        assert len(pipeline.FEATURE_NAMES) == 32

        features = pipeline.extract_features(sample_df)
        assert isinstance(features, np.ndarray)
        assert features.shape == (len(sample_df), 32)
        assert np.isfinite(features).all()
        assert not np.isnan(features).any()

    def test_hist_gradient_boosting_model_training_and_inference(self):
        np.random.seed(42)
        X = np.random.randn(150, 32).astype(np.float32)
        # 3 classes: -1 (down), 0 (flat), 1 (up)
        y = np.random.choice([-1, 0, 1], size=150)

        model = HistGradientBoostingModel(max_iter=30, random_state=42)
        model.fit(X, y)
        assert model.is_fitted

        # Probabilities
        probs = model.predict_proba(X[:10])
        assert probs.shape == (10, 3)
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

        # Signal score and confidence
        s_ml, c_ml = model.predict_signal(X[-1])
        assert -1.0 <= s_ml <= 1.0
        assert 0.0 <= c_ml <= 1.0

        # Persistence save / load test
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.joblib")
            model.save(save_path)
            assert os.path.exists(save_path)

            loaded = HistGradientBoostingModel.load(save_path)
            s_loaded, c_loaded = loaded.predict_signal(X[-1])
            assert np.isclose(s_ml, s_loaded)
            assert np.isclose(c_ml, c_loaded)

    def test_pytorch_deep_sequence_embedder(self):
        embedder = DeepSequenceEmbedder(in_features=32, hidden_dim=64, embedding_dim=16)

        # PyTorch CPU forward pass
        x = torch.randn(8, 32)
        emb, logits = embedder(x)
        assert emb.shape == (8, 16)
        assert logits.shape == (8, 3)

        # Numpy interface
        np_x = np.random.randn(4, 32).astype(np.float32)
        np_emb = embedder.extract_embedding(np_x)
        assert np_emb.shape == (4, 16)


# =====================================================================
# 4. Reinforcement Learning Layer Tests
# =====================================================================

class TestReinforcementLearningLayer:
    """Tests for RLStateBuilder, DDQN PolicyAgent, and CostPenalizedReward."""

    def test_rl_state_builder(self):
        state = RLStateBuilder.build_state(
            quant_score=0.65,
            ml_score=-0.40,
            ml_confidence=0.85,
            adx=32.0,
            atr=1.5,
            price=150.0,
            current_weight=0.50,
            unrealized_pnl_pct=2.5,
            drawdown_pct=0.03,
            holding_time_bars=15.0,
        )
        assert isinstance(state, np.ndarray)
        assert state.shape == (9,)
        assert state.dtype == np.float32
        assert np.isfinite(state).all()

        # Check bounds
        assert -1.0 <= state[0] <= 1.0  # quant_score
        assert -1.0 <= state[1] <= 1.0  # ml_score
        assert 0.0 <= state[2] <= 1.0   # confidence
        assert 0.0 <= state[3] <= 1.0   # adx
        assert 0.0 <= state[4] <= 0.1   # atr / price
        assert -1.0 <= state[5] <= 1.0  # weight

    def test_cost_penalized_reward_function(self):
        reward_fn = CostPenalizedReward(
            lambda_fee=1.0,
            lambda_slippage=1.0,
            lambda_drawdown=10.0,
            drawdown_threshold=0.05,
        )

        # Case 1: Positive PnL with low drawdown -> positive reward
        r1 = reward_fn.calculate_reward(delta_pnl=5.0, fee=0.1, slippage=0.05, drawdown=0.02)
        assert np.isclose(r1, 5.0 - 0.1 - 0.05)

        # Case 2: Drawdown exceeding 5% incurs quadratic penalty
        r2 = reward_fn.calculate_reward(delta_pnl=1.0, fee=0.0, slippage=0.0, drawdown=0.08)
        # Excess dd = 0.03 -> penalty = 10 * 0.0009 = 0.009
        assert r2 < 1.0

        # Case 3: Extreme negative move is clipped to [-10, 10]
        r3 = reward_fn.calculate_reward(delta_pnl=-50.0, fee=10.0, slippage=10.0, drawdown=0.5)
        assert r3 == -10.0

    def test_ddqn_policy_agent_act_and_update(self):
        np.random.seed(42)
        torch.manual_seed(42)
        agent = PolicyAgent(state_dim=9, action_dim=7, lr=1e-3)

        state = np.random.randn(9).astype(np.float32)
        action, weight = agent.act(state, epsilon=0.0, current_weight=0.25)
        assert 0 <= action <= 6
        assert -1.0 <= weight <= 1.0

        # Batch update test
        batch_size = 16
        states = np.random.randn(batch_size, 9).astype(np.float32)
        actions = np.random.randint(0, 7, size=batch_size).astype(np.int64)
        rewards = np.random.randn(batch_size).astype(np.float32)
        next_states = np.random.randn(batch_size, 9).astype(np.float32)
        dones = np.zeros(batch_size, dtype=np.float32)

        loss = agent.update(states, actions, rewards, next_states, dones)
        assert isinstance(loss, float)
        assert loss >= 0.0

        # Save and load checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "policy.pt")
            agent.save(ckpt_path)
            assert os.path.exists(ckpt_path)

            loaded_agent = PolicyAgent(state_dim=9, action_dim=7)
            loaded_agent.load(ckpt_path)

            act_post, weight_post = agent.act(state, epsilon=0.0, current_weight=0.25)
            loaded_act, loaded_weight = loaded_agent.act(state, epsilon=0.0, current_weight=0.25)
            assert act_post == loaded_act
            assert np.isclose(weight_post, loaded_weight)


# =====================================================================
# 5. Universe Screener, Bandit Allocator & Fractional Sizing Tests
# =====================================================================

class TestUniverseAndFractionalSizing:
    """Tests for DynamicUniverseScreener, MultiArmedBanditAllocator, and €100 capital sizing."""

    def test_dynamic_universe_screener_filtering(self):
        screener = DynamicUniverseScreener(
            min_volume_24h_usd=5_000_000.0,
            min_volatility_ratio=0.005,
            max_selected_assets=3,
        )

        candidates = [
            {"symbol": "SPY", "volume_24h_usd": 20_000_000, "price": 450.0, "atr_14": 4.5, "sharpe_30d": 1.5},   # ATR ratio 0.010 -> pass
            {"symbol": "BTC/USDT", "volume_24h_usd": 50_000_000, "price": 60_000.0, "atr_14": 1200.0, "sharpe_30d": 2.2}, # ATR ratio 0.020 -> pass
            {"symbol": "EUR/USD", "volume_24h_usd": 15_000_000, "price": 1.08, "atr_14": 0.008, "sharpe_30d": 0.8},   # ATR ratio 0.0074 -> pass
            {"symbol": "LOW_VOL", "volume_24h_usd": 10_000_000, "price": 100.0, "atr_14": 0.1, "sharpe_30d": 1.0},    # ATR ratio 0.001 -> FAIL (< 0.5%)
            {"symbol": "ILLIQUID", "volume_24h_usd": 100_000, "price": 50.0, "atr_14": 1.5, "sharpe_30d": 3.0},       # Volume 100k -> FAIL (< 5M)
        ]

        selected = screener.filter_and_rank(candidates)
        assert len(selected) == 3
        symbols = [a.symbol for a in selected]
        assert "SPY" in symbols
        assert "BTC/USDT" in symbols
        assert "EUR/USD" in symbols
        assert "LOW_VOL" not in symbols
        assert "ILLIQUID" not in symbols

    def test_multi_armed_bandit_allocator_with_concentration_cap(self):
        allocator = MultiArmedBanditAllocator(temperature=1.0, max_concentration=0.50)
        sharpes = {"BTC/USDT": 3.5, "SPY": 1.5, "AAPL": 1.0}

        weights = allocator.allocate(sharpes)
        assert len(weights) == 3
        # Invariant 1: Top asset has highest weight
        assert weights["BTC/USDT"] > weights["SPY"] > weights["AAPL"]
        # Invariant 2: Concentration cap (50%) strictly enforced
        assert weights["BTC/USDT"] <= 0.500001
        # Invariant 3: Sum of weights is 1.0
        assert np.isclose(sum(weights.values()), 1.0, atol=1e-4)

    def test_micro_capital_fractional_adapter_for_100_eur(self):
        adapter = MicroCapitalFractionalAdapter(min_order_notional_eur=1.00, default_eur_usd_rate=1.08)

        # 20% allocation of €100 = €20 -> ~$21.60 USD -> share price $180 -> ~0.12 shares
        qty_aapl = adapter.calculate_order_quantity(
            symbol="AAPL",
            target_weight=0.20,
            account_equity_eur=100.0,
            price=180.0,
            asset_class="EQUITY",
            currency="USD",
        )
        assert qty_aapl is not None
        assert 0.0 < qty_aapl < 1.0
        assert round(qty_aapl, 6) == qty_aapl

        # High-value asset: BTC at $65,000 with €30 allocation -> ~0.000498 BTC
        qty_btc = adapter.calculate_order_quantity(
            symbol="BTC/USDT",
            target_weight=0.30,
            account_equity_eur=100.0,
            price=65_000.0,
            asset_class="CRYPTO",
            currency="USD",
        )
        assert qty_btc is not None
        assert 0.0 < qty_btc < 0.01

        # Minimum notional threshold rejection: €0.50 allocation (< €1.00 minimum)
        qty_submin = adapter.calculate_order_quantity(
            symbol="AAPL",
            target_weight=0.005,  # 0.5% of €100 = €0.50
            account_equity_eur=100.0,
            price=180.0,
        )
        assert qty_submin is None


# =====================================================================
# 6. Multi-Timeframe Synchronizer & Hybrid Engine Tests
# =====================================================================

class TestMultiTimeframeSynchronizer:
    """Tests for lookahead-free bar aggregation across 1m, 5m, 15m, 1h."""

    def test_synchronizer_aggregates_5m_bars(self, sample_bars):
        sync = MultiTimeframeBarSynchronizer(supported_timeframes=["5m"])

        completed_5m: list[Bar] = []
        for bar in sample_bars[:15]:
            emitted = sync.ingest_bar(bar)
            if emitted.get("5m") is not None:
                completed_5m.append(emitted["5m"])

        # In 15 1-minute bars, exactly two 5-minute bars complete (at min 5 and min 10)
        assert len(completed_5m) == 2
        first_5m = completed_5m[0]
        assert first_5m.symbol == "AAPL"
        # First 5m bar open should match 1st 1m bar open
        assert np.isclose(first_5m.open, sample_bars[0].open)
        # Volume should equal sum of first 5 bars
        expected_vol = sum(b.volume for b in sample_bars[:5])
        assert np.isclose(first_5m.volume, expected_vol)


class TestHybridStrategyEngine:
    """Tests for complete Hybrid AI Strategy Engine generating TradingSignals."""

    def test_hybrid_engine_signal_generation(self, sample_bars):
        engine = HybridStrategyEngine(k_take_profit=3.0, k_stop_loss=2.0)

        # Ingest first 30 bars to build indicator history
        for b in sample_bars[:29]:
            engine.update_history(b)

        # Generate signal for 30th bar
        signal = engine.process_bar(sample_bars[29], account_equity_eur=100.0)

        assert isinstance(signal, TradingSignal)
        assert signal.symbol == "AAPL"
        assert signal.action in ("BUY", "SELL", "HOLD")
        assert -1.0 <= signal.target_weight <= 1.0
        assert 0.0 <= signal.confidence <= 1.0
        assert -1.0 <= signal.quant_score <= 1.0
        assert -1.0 <= signal.ml_score <= 1.0

        # Verify stop loss and take profit invariants when position is active
        if signal.target_weight > 0:
            assert signal.stop_loss is not None
            assert signal.take_profit is not None
            assert signal.stop_loss < sample_bars[29].close
            assert signal.take_profit > sample_bars[29].close
        elif signal.target_weight < 0:
            assert signal.stop_loss is not None
            assert signal.take_profit is not None
            assert signal.stop_loss > sample_bars[29].close
            assert signal.take_profit < sample_bars[29].close

        # Metadata checks
        assert "regime" in signal.metadata
        assert "atr" in signal.metadata
        assert signal.metadata["account_equity_eur"] == 100.0
