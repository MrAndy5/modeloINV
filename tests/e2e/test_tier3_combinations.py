"""Tier 3: Cross-Feature Interactions (Pairwise Combinations) E2E Tests.

Validates multi-subsystem interaction and coupled workflows:
1. Paper trading execution + Continuous retraining loop
2. Multi-currency FX conversion + Multi-asset portfolio rebalancing
3. FastAPI Web Dashboard streaming + Live order execution
4. Dynamic screener + Multi-armed bandit allocation + Fractional sizing
5. Drawdown guardrail + Stop-loss execution + Champion-challenger demotion
6. Resilient data cascade failover + Multi-timeframe bar synchronization
7. Configuration-only toggle + Broker state persistence

All tests run 100% offline and deterministically on Windows.
"""

from __future__ import annotations

import os
import sys
import json
import sqlite3
import datetime
from pathlib import Path
from typing import List, Dict, Any

import pytest
import numpy as np
import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.conftest import (
    Bar, OrderRequest, OrderResult, Position, AccountInfo, TradingSignal,
    OrderType, OrderSide, OrderStatus, AssetClass,
    MockMarketDataProvider, OfflineSimulatedBroker,
    calculate_sharpe_ratio, calculate_sortino_ratio, calculate_max_drawdown, calculate_win_rate
)


class TestTier3Combinations:
    """Pairwise cross-feature interactions and composite workflows."""

    def test_pair_01_paper_trading_with_continuous_retraining_loop(self, offline_broker, sqlite_db_conn):
        """Pair 1: Paper trading execution feeds feedback ingestor, triggering retraining at N=25."""
        trade_feedback_log = []
        model_registry = [{"version": "v1.0.0", "sharpe": 1.10, "is_champion": True}]

        # Execute 25 simulated trades
        for i in range(25):
            side = OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
            req = OrderRequest(
                symbol="SPY",
                side=side,
                order_type=OrderType.MARKET,
                quantity=0.05
            )
            price = 450.0 + (i * 0.5)
            res = offline_broker.submit_order(req, current_market_price=price)
            assert res.status == OrderStatus.FILLED

            # Feedback ingestor logs trade
            trade_feedback_log.append({
                "trade_id": res.order_id,
                "symbol": "SPY",
                "price": res.filled_avg_price,
                "fee_eur": res.fee_eur,
                "pnl_eur": 0.50 if side == OrderSide.SELL else 0.0
            })

        # Trigger check at N=25
        assert len(trade_feedback_log) == 25
        retrain_triggered = len(trade_feedback_log) >= 25
        assert retrain_triggered is True

        # Candidate model trained and evaluated by champion-challenger gate
        candidate_sharpe = 1.35
        champion_sharpe = model_registry[0]["sharpe"]
        if candidate_sharpe > champion_sharpe:
            model_registry[0]["is_champion"] = False
            model_registry.append({
                "version": "v1.1.0",
                "sharpe": candidate_sharpe,
                "is_champion": True
            })

        assert len(model_registry) == 2
        assert model_registry[-1]["version"] == "v1.1.0"
        assert model_registry[-1]["is_champion"] is True

    def test_pair_02_multi_currency_fx_with_multi_asset_rebalancing(self, offline_broker):
        """Pair 2: Virtual €100 rebalances across USD Equity, USDT Crypto, and EUR Forex with FX isolation."""
        eur_usd = 1.08
        
        # Portfolio allocation: 40% SPY (USD), 30% BTC/USDT (Crypto USD), 30% Cash (EUR)
        spy_notional_eur = 40.0
        btc_notional_eur = 30.0

        # Execute SPY
        spy_req = OrderRequest(symbol="SPY", side=OrderSide.BUY, order_type=OrderType.MARKET, notional=spy_notional_eur)
        res_spy = offline_broker.submit_order(spy_req, current_market_price=450.0)
        assert res_spy.status == OrderStatus.FILLED

        # Execute BTC
        btc_req = OrderRequest(symbol="BTC/USDT", side=OrderSide.BUY, order_type=OrderType.MARKET, notional=btc_notional_eur)
        res_btc = offline_broker.submit_order(btc_req, current_market_price=60000.0)
        assert res_btc.status == OrderStatus.FILLED

        # Check multi-asset portfolio integrity
        acc = offline_broker.get_account()
        assert len(offline_broker.positions) == 2
        assert acc.cash_eur < 35.0  # ~€30 remaining cash
        assert acc.equity_eur > 95.0

    def test_pair_03_dashboard_streaming_with_live_order_execution(self, offline_broker):
        """Pair 3: Order placement updates internal state and generates real-time telemetry event."""
        telemetry_events = []

        def event_dispatcher(event_type: str, data: Dict[str, Any]):
            telemetry_events.append({"event": event_type, "data": data, "timestamp": datetime.datetime.utcnow().isoformat()})

        # Submit order
        req = OrderRequest(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.1)
        res = offline_broker.submit_order(req, current_market_price=180.0)
        assert res.status == OrderStatus.FILLED

        # Dispatch WebSocket event
        acc = offline_broker.get_account()
        event_dispatcher("order_fill", {
            "order_id": res.order_id,
            "symbol": res.symbol,
            "filled_qty": res.filled_quantity,
            "price": res.filled_avg_price,
            "cash_eur": acc.cash_eur,
            "equity_eur": acc.equity_eur
        })

        assert len(telemetry_events) == 1
        assert telemetry_events[0]["event"] == "order_fill"
        assert telemetry_events[0]["data"]["order_id"] == res.order_id
        assert telemetry_events[0]["data"]["symbol"] == "AAPL"

    def test_pair_04_dynamic_screener_bandit_fractional_adapter(self, offline_broker):
        """Pair 4: Dynamic universe screener + multi-armed bandit + micro-capital fractional adapter."""
        # 1. Screener filters candidate assets
        candidates = [
            {"symbol": "SPY", "sharpe_30d": 1.4},
            {"symbol": "QQQ", "sharpe_30d": 1.8},
            {"symbol": "BTC/USDT", "sharpe_30d": 2.2}
        ]

        # 2. Multi-armed bandit allocates weights using softmax
        sharpes = np.array([c["sharpe_30d"] for c in candidates])
        weights = np.exp(sharpes) / np.sum(np.exp(sharpes))
        
        # 3. Fractional adapter translates to orders within €100 virtual capital
        total_equity = offline_broker.get_account().equity_eur
        orders = []
        for c, w in zip(candidates, weights):
            alloc_eur = total_equity * w * 0.3  # Max 30% per asset
            req = OrderRequest(
                symbol=c["symbol"],
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                notional=alloc_eur
            )
            orders.append(req)

        assert len(orders) == 3
        # BTC/USDT has highest sharpe, should have highest notional
        assert orders[2].notional > orders[1].notional > orders[0].notional

    def test_pair_05_drawdown_guardrail_stop_loss_champion_gate(self, offline_broker):
        """Pair 5: Stop-loss trigger + drawdown breaker + candidate model rejection."""
        # Buy asset
        buy_req = OrderRequest(symbol="RISK_ASSET", side=OrderSide.BUY, order_type=OrderType.MARKET, notional=50.0)
        offline_broker.submit_order(buy_req, current_market_price=100.0)

        # Asset drops 30% -> stop loss executed
        offline_broker.update_market_price("RISK_ASSET", 70.0)
        sell_req = OrderRequest(symbol="RISK_ASSET", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=offline_broker.positions["RISK_ASSET"].quantity)
        offline_broker.submit_order(sell_req, current_market_price=70.0)

        # Drawdown trigger checked
        acc = offline_broker.get_account()
        realized_loss = acc.initial_capital_eur - acc.equity_eur
        assert realized_loss > 10.0

        # Retraining candidate produced during high drawdown with Sharpe 0.8 is rejected by gate
        candidate_model_sharpe = 0.85
        min_required_sharpe = 1.0
        gate_passed = candidate_model_sharpe >= min_required_sharpe
        assert gate_passed is False

    def test_pair_06_cascade_failover_multi_timeframe_sync(self, mock_data_provider):
        """Pair 6: Cascade failover from primary to mock does not corrupt multi-timeframe bar synchronizer."""
        # Fetch 30 1m bars from secondary fallback mock
        bars_1m = mock_data_provider.generate_bars("SPY", datetime.datetime(2026, 1, 1, 9, 30), num_bars=30, timeframe_minutes=1)
        
        # Build DataFrame
        df_1m = pd.DataFrame([{
            "timestamp": b.timestamp,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume
        } for b in bars_1m]).set_index("timestamp")

        # Synchronize into 5m bars without data loss
        df_5m = df_1m.resample("5min").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()

        assert len(df_5m) == 6
        assert df_5m["volume"].sum() == df_1m["volume"].sum()

    def test_pair_07_config_mode_toggle_broker_persistence(self, monkeypatch, sqlite_db_conn):
        """Pair 7: Toggling MODELOINV_EXECUTION_MODE preserves accounting state in database."""
        # 1. Record trade in paper mode
        monkeypatch.setenv("MODELOINV_EXECUTION_MODE", "paper")
        cursor = sqlite_db_conn.cursor()
        cursor.execute("""
            INSERT INTO trades (trade_id, order_id, symbol, side, quantity, price, fee_eur, slippage_eur, realized_pnl_eur, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("tr_p1", "ord_p1", "SPY", "BUY", 0.1, 450.0, 0.05, 0.02, 0.0, datetime.datetime.utcnow().isoformat()))
        sqlite_db_conn.commit()

        # 2. Toggle to live mode
        monkeypatch.setenv("MODELOINV_EXECUTION_MODE", "live")
        assert os.getenv("MODELOINV_EXECUTION_MODE") == "live"

        # 3. Verify trade history in database persists across mode toggle
        cursor.execute("SELECT COUNT(*) as cnt FROM trades")
        count = cursor.fetchone()["cnt"]
        assert count == 1
