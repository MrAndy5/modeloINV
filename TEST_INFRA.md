# Test Infrastructure & Specification: modeloINV

## 1. Test Philosophy & Architecture

modeloINV employs an opaque-box, requirement-driven 4-tier E2E testing framework designed to rigorously validate all functional requirements, mathematical invariants, execution guarantees, and live-readiness gates defined in `ORIGINAL_REQUEST.md` and `PROJECT.md`.

All tests are guaranteed to run **100% offline**, **deterministically**, and **natively on Windows** without external network dependencies, live broker credentials, or stochastic test flakiness.

```
+----------------------------------------------------------------------------------------------------+
|                                 modeloINV 4-Tier E2E Test Suite                                    |
+----------------------------------------------------------------------------------------------------+
  │
  ├─► Tier 1: Feature Coverage (Category-Partition)
  │   └── Primary happy-path coverage across all 61 features (≥5 tests per functional domain)
  │
  ├─► Tier 2: Boundary Value Analysis (BVA) & Corner Cases
  │   └── Stress testing: zero-volume, extreme slippage, €0 capital exhaustion, float edge cases
  │
  ├─► Tier 3: Cross-Feature Interactions (Pairwise Combinations)
  │   └── Pairwise interaction testing: paper trading + retraining, FX + multi-asset, dashboard + orders
  │
  └─► Tier 4: Real-World Workload Scenarios
      └── 5 end-to-end application workloads validating all Acceptance Criteria from ORIGINAL_REQUEST.md
```

### 1.1 Methodology

1. **Opaque-Box Requirement Verification**:
   Tests interact strictly via public interfaces, canonical data models (`Bar`, `OrderRequest`, `OrderResult`, `Position`, `AccountInfo`, `TradingSignal`), abstract broker interfaces (`BaseBroker`), REST/WebSocket endpoints, and CLI commands. Internal implementation details are decoupled from test assertions.

2. **Category-Partitioning & Equivalence Classes (Tier 1)**:
   Every feature domain is decomposed into discrete functional partitions with representative inputs exercising core operational paths.

3. **Boundary Value Analysis & Stress Invariants (Tier 2)**:
   Rigorous boundary probing: zero quantities, micro-fractional shares (0.000001 BTC), negative buying power, €0.00 cash exhaustion, zero-variance Sharpe calculations, sudden 50% flash crashes, and rapid order-cancellation races.

4. **Pairwise Cross-Feature Interactions (Tier 3)**:
   Verification of multi-subsystem coupling: continuous retraining triggered during active paper trading, multi-currency FX mark-to-market during portfolio rebalancing, real-time WebSocket event dissemination during live order execution, and pre-trade risk breakers halting training updates.

5. **Real-World Application Workloads (Tier 4)**:
   Macro-level operational scenarios executing complete lifecycle workflows matching user acceptance criteria: 12-month backtest beating buy-and-hold S&P 500 with Sharpe > 1.0, 30-day learning loop with measurable improvement ($v_N > v_0$), zero-code broker switching, full-fidelity FastAPI dashboard data flow, and complete CLI/documentation compliance.

---

## 2. Feature Inventory Coverage Matrix (Features 1–61)

| # | Feature Name | Domain | Tier 1 | Tier 2 | Tier 3 | Tier 4 | Verification Target |
|---|--------------|--------|:------:|:------:|:------:|:------:|---------------------|
| 1 | Configuration & Environment Manager | Core | ✓ | ✓ | ✓ | ✓ | YAML/Env loading, validation, paper/live mode |
| 2 | Canonical Data Models | Core | ✓ | ✓ | ✓ | ✓ | Strongly typed dataclasses, serialization, frozen bars |
| 3 | SQLite WAL Persistence Engine | Storage | ✓ | ✓ | ✓ | ✓ | In-memory/file SQLite, concurrent reads/writes |
| 4 | Parquet Historical Bar Cache | Storage | ✓ | ✓ | ✓ | ✓ | Columnar OHLCV storage, timestamp indexing |
| 5 | Multi-Currency FX Engine | Core | ✓ | ✓ | ✓ | ✓ | Continuous mark-to-market EUR/USD, FX P&L isolation |
| 6 | Yahoo Finance Data Client | Data | ✓ | ✓ | ✓ | ✓ | Equities/ETFs/FX data client, rate limit handling |
| 7 | Binance Public Data Client | Data | ✓ | ✓ | ✓ | ✓ | Free crypto klines & depth without API keys |
| 8 | Alpaca Market Data Client | Data | ✓ | ✓ | ✓ | ✓ | Alpaca v2 market data client with fallback |
| 9 | Resilient Data Provider Cascade | Data | ✓ | ✓ | ✓ | ✓ | Failover across providers and local cache |
| 10 | Offline Mock Data Provider | Data | ✓ | ✓ | ✓ | ✓ | Deterministic synthetic OHLCV generator |
| 11 | Virtual €100 Account Ledger | Execution | ✓ | ✓ | ✓ | ✓ | Base EUR balance, equity tracking, invariants |
| 12 | Fractional Share Sizing Engine | Execution | ✓ | ✓ | ✓ | ✓ | Micro-shares (6 decimals), cash notional sizing |
| 13 | Market Order Simulation | Execution | ✓ | ✓ | ✓ | ✓ | Immediate execution with spread, volume impact, slippage |
| 14 | Limit Order Simulation | Execution | ✓ | ✓ | ✓ | ✓ | Price-contingent queue matching, Time-in-Force |
| 15 | Stop-Loss & Stop-Limit Orders | Execution | ✓ | ✓ | ✓ | ✓ | Adverse breach triggers, gap slippage modeling |
| 16 | Trailing Stop Simulation | Execution | ✓ | ✓ | ✓ | ✓ | Dynamic ratcheting tracking favorable high-water marks |
| 17 | Fixed BPS & Bid-Ask Spread Slippage | Execution | ✓ | ✓ | ✓ | ✓ | Corwin-Schultz / bid-ask half-spread penalties |
| 18 | Volume Impact Slippage Model | Execution | ✓ | ✓ | ✓ | ✓ | Almgren-Chriss square-root price impact ($Q/V$) |
| 19 | Transaction Fee Simulator | Execution | ✓ | ✓ | ✓ | ✓ | SEC, FINRA, exchange commission, crypto maker/taker |
| 20 | Performance & Metrics Calculator | Execution | ✓ | ✓ | ✓ | ✓ | Cumulative return, Sharpe, Sortino, Max DD, Win Rate |
| 21 | S&P 500 Benchmark Synchronizer | Execution | ✓ | ✓ | ✓ | ✓ | Time-aligned SPY tracking, Alpha, Beta calculation |
| 22 | BaseBroker Abstraction Interface | Execution | ✓ | ✓ | ✓ | ✓ | Unified methods: get_account, submit_order, cancel |
| 23 | SimulatedPaperBroker | Execution | ✓ | ✓ | ✓ | ✓ | Full-fidelity in-memory paper broker |
| 24 | AlpacaPaperBroker | Execution | ✓ | ✓ | ✓ | ✓ | Alpaca REST API v2 sandbox with €100 partition |
| 25 | MockBroker | Execution | ✓ | ✓ | ✓ | ✓ | Scriptable deterministic broker with error injection |
| 26 | Configuration-Only Toggle | Execution | ✓ | ✓ | ✓ | ✓ | Zero-code switching via MODELOINV_EXECUTION_MODE |
| 27 | Pre-Trade Risk Guardrails | Execution | ✓ | ✓ | ✓ | ✓ | Drawdown circuit breaker, position caps, kill switch |
| 28 | Live Broker Transition Framework | Execution | ✓ | ✓ | ✓ | ✓ | Hooks and docs for myInvestor (MCP), IBKR, CCXT |
| 29 | Technical Indicator Suite | Strategy | ✓ | ✓ | ✓ | ✓ | EMA, RSI, MACD, Bollinger Bands, ATR, VWAP, ADX |
| 30 | Market Regime Classifier | Strategy | ✓ | ✓ | ✓ | ✓ | ADX and volatility-based regime identification |
| 31 | Scale-Invariant Composite Quant Score| Strategy | ✓ | ✓ | ✓ | ✓ | Normalized score $S_{quant} \in [-1.0, 1.0]$ |
| 32 | Triple-Barrier Labeling Engine | Strategy | ✓ | ✓ | ✓ | ✓ | Upper TP, lower SL, and time barriers |
| 33 | ML Feature Engineering Pipeline | Strategy | ✓ | ✓ | ✓ | ✓ | 32-dimensional standardized feature matrix |
| 34 | HistGradientBoosting ML Directional | Strategy | ✓ | ✓ | ✓ | ✓ | Scikit-learn directional probability & confidence |
| 35 | PyTorch Sequence Embedding / MLP | Strategy | ✓ | ✓ | ✓ | ✓ | Neural network representation of intraday patterns |
| 36 | RL MDP State Vector Builder | Strategy | ✓ | ✓ | ✓ | ✓ | 9-dimensional state vector (regime, quant, ML, equity) |
| 37 | PyTorch RL Policy Agent | Strategy | ✓ | ✓ | ✓ | ✓ | DDQN/Actor-Critic optimal weight $w_t \in [-1.0, 1.0]$ |
| 38 | Cost/Slippage Penalized Reward Func | Strategy | ✓ | ✓ | ✓ | ✓ | Reward penalizing slippage, fees, and drawdown |
| 39 | Intraday Hybrid Signal Aggregator | Strategy | ✓ | ✓ | ✓ | ✓ | Multi-signal fusion with dynamic TP/SL levels |
| 40 | Dynamic Cross-Asset Screener | Strategy | ✓ | ✓ | ✓ | ✓ | Liquidity/volatility scanner selecting top tradable assets |
| 41 | Multi-Armed Bandit Dynamic Weighting | Strategy | ✓ | ✓ | ✓ | ✓ | Rolling Sharpe allocation to top-performing assets |
| 42 | Micro-Capital Fractional Adapter | Strategy | ✓ | ✓ | ✓ | ✓ | Translation for €100 enforcing min contract constraints |
| 43 | Multi-Timeframe Bar Synchronizer | Strategy | ✓ | ✓ | ✓ | ✓ | Lookahead-bias-free alignment of 1m, 5m, 15m, 1h bars |
| 44 | Multi-Criteria Retraining Trigger | Learning | ✓ | ✓ | ✓ | ✓ | Triggers: Schedule, N=25 trades, Drift, Drawdown |
| 45 | Realized Trade Feedback Ingestor | Learning | ✓ | ✓ | ✓ | ✓ | Closed-trade execution telemetry capture |
| 46 | Prioritized Experience Replay Buffer | Learning | ✓ | ✓ | ✓ | ✓ | TD-error prioritized buffer for online RL updates |
| 47 | Incremental Sliding-Window Retrainer | Learning | ✓ | ✓ | ✓ | ✓ | Rolling 60-day retraining with recency weighting |
| 48 | Semantic Model Version Registry | Learning | ✓ | ✓ | ✓ | ✓ | Model metadata, hyperparameters, metrics in JSON |
| 49 | Champion-Challenger Validation Gate | Learning | ✓ | ✓ | ✓ | ✓ | Out-of-sample promotion preventing degraded models |
| 50 | 8-Gate Live Readiness Engine | Learning | ✓ | ✓ | ✓ | ✓ | Evaluates `is_live_ready` across 8 strict criteria |
| 51 | 30-Day Learning Loop Benchmark | Learning | ✓ | ✓ | ✓ | ✓ | Measurable performance improvement ($v_N > v_0$) |
| 52 | FastAPI Web Backend | Dashboard | ✓ | ✓ | ✓ | ✓ | Asynchronous REST and WebSocket server |
| 53 | WebSocket Real-Time Streamer | Dashboard | ✓ | ✓ | ✓ | ✓ | Sub-millisecond tick, order fill, equity push |
| 54 | REST API Endpoints | Dashboard | ✓ | ✓ | ✓ | ✓ | /portfolio, /positions, /trades, /performance, etc. |
| 55 | Responsive Dark-Theme SPA | Dashboard | ✓ | ✓ | ✓ | ✓ | Glassmorphism HTML5/JS UI frontend assets |
| 56 | Chart.js Real-Time Visualizations | Dashboard | ✓ | ✓ | ✓ | ✓ | Equity curve vs S&P 500, drawdown, asset allocation |
| 57 | Paper/Live Mode UI Toggle | Dashboard | ✓ | ✓ | ✓ | ✓ | Authenticated toggle switching execution mode |
| 58 | Command Line Interface (CLI) | Interface | ✓ | ✓ | ✓ | ✓ | `modeloinv` commands (backtest, paper, live, etc.) |
| 59 | Comprehensive System Documentation | Interface | ✓ | ✓ | ✓ | ✓ | README, BROKER_SETUP.md, ARCHITECTURE.md |
| 60 | 100% E2E Test Suite Verification | Acceptance | ✓ | ✓ | ✓ | ✓ | All Tiers 1-4 passing cleanly and published |
| 61 | Adversarial Coverage Hardening | Acceptance | ✓ | ✓ | ✓ | ✓ | Zero integrity bypasses, robust offline guarantees |

---

## 3. Test Runner Instructions

### 3.1 Prerequisites
- Python 3.11+
- Dependencies installed: `pytest`, `pandas`, `numpy`, `scikit-learn`, `torch`, `fastapi`, `uvicorn`, `httpx`
- Operating System: Windows (powershell / cmd compatible)
- Network: **0% external network required**; fully offline test execution

### 3.2 Execution Commands

```powershell
# Run the complete E2E test suite (all 4 tiers)
pytest tests/e2e -v

# Run individual tiers
pytest tests/e2e/test_tier1_features.py -v
pytest tests/e2e/test_tier2_boundaries.py -v
pytest tests/e2e/test_tier3_combinations.py -v
pytest tests/e2e/test_tier4_scenarios.py -v

# Run with coverage report
pytest tests/e2e --cov=modeloinv --cov-report=term-missing
```

### 3.3 Acceptance Criteria Thresholds

1. **Simulation Correctness**: Slippage, spread, and commissions modeled; starting balance of €100.00 tracked with zero floating-point divergence.
2. **Strategy Performance**: 12-month backtest simulation achieves return beating buy-and-hold S&P 500, Sharpe ratio > 1.0, and at least one 30-day window with ≥15% return.
3. **Live Readiness**: Working broker abstraction with Alpaca paper sandbox support; config-only toggle (`MODELOINV_EXECUTION_MODE=paper|live`) with zero code modification.
4. **Dashboard**: REST endpoints and `/ws/stream` WebSocket delivering real-time portfolio, positions, trades, S&P 500 comparison, and learning curves.
5. **Learning Loop**: Semantic version registry tracking checkpoints; measurable performance improvement ($v_N > v_0$) over 30 simulated trading days.
6. **Documentation & CLI**: Complete documentation (`README.md`, `BROKER_SETUP.md`, `ARCHITECTURE.md`) and executable CLI entry points.
