# Project: modeloINV — Algorithmic Trading System

## Architecture
modeloINV is an event-driven, production-grade algorithmic trading system designed to trade intraday on €100 starting capital, learn continuously to beat the S&P 500 (targeting ≥20% monthly return), and transition smoothly between realistic paper trading and live broker execution via configuration only.

```
+---------------------------------------------------------------------------------------+
|                                    User Interface                                     |
|  - Real-time Web Dashboard (FastAPI + WebSockets + Dark-Theme HTML5/Chart.js SPA)     |
|  - Command Line Interface (CLI: run, backtest, dashboard, retrain, evaluate)          |
+---------------------------------------------------------------------------------------+
                                           │
                                           ▼
+───────────────────────────────────────────────────────────────────────────────────────+
|                         Continuous Learning & Lifecycle (R5)                          |
|  - Retraining Triggers (Daily, N=25 trades, Drift, Drawdown)                          |
|  - Trade Feedback Ingestor & Prioritized Experience Replay Buffer                     |
|  - Model Version Registry (Semantic vMAJOR.MINOR.PATCH, atomic checkpointing)         |
|  - Champion-Challenger Promotion Gate & 8-Gate Live Readiness Engine                  |
+───────────────────────────────────────────────────────────────────────────────────────+
                                           │
                                           ▼
+───────────────────────────────────────────────────────────────────────────────────────+
|                           Hybrid AI Strategy Engine (R1)                              |
|  - Classical Quant Layer (EMA, RSI, MACD, BB, ATR, VWAP, ADX, Regime Classifier)       |
|  - Machine Learning Layer (HistGradientBoosting + Triple-Barrier Labeling + Confs)    |
|  - Reinforcement Learning Layer (PyTorch DDQN/Actor-Critic, Cost/Slippage Penalty)    |
|  - Dynamic Asset Universe Screener & Multi-Armed Bandit Dynamic Weighting             |
|  - Fractional Sizing Adapter (€100 account constraints)                               |
+───────────────────────────────────────────────────────────────────────────────────────+
                                           │
                                           ▼
+───────────────────────────────────────────────────────────────────────────────────────+
|                     Execution Engine & Broker Abstraction (R2 & R3)                   |
|  - BaseBroker Interface (get_account, get_positions, submit_order, cancel_order)       |
|  - SimulatedPaperBroker (Realistic Slippage, Spread, Commission/SEC/FINRA/Crypto Fees)|
|  - AlpacaPaperBroker (Live Sandbox API with €100 Virtual Partition)                   |
|  - MockBroker (Deterministic offline execution for unit & E2E testing)                |
|  - Configuration-Only Toggle (MODELOINV_EXECUTION_MODE=paper|live)                    |
|  - Live Risk Guardrails (Drawdown stop, Kill Switch)                                  |
+───────────────────────────────────────────────────────────────────────────────────────+
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     ▼                                           ▼
+─────────────────────────────────────────+ +───────────────────────────────────────────+
|        Data Infrastructure (R2)         | |       Storage & Persistence Engine        |
| - Yahoo Finance (Equities, ETFs, FX)    | | - SQLite (WAL mode, concurrent reads/writes|
| - Binance Public API (Crypto klines)    | |   for trades, account snapshots, metrics)   |
| - Alpaca Market Data v2 (Fallback)      | | - Parquet Cache (Local compressed OHLCV)   |
| - Offline MockDataProvider (Zero network| | - Model Artifacts (.joblib, .pt, JSON)    |
+─────────────────────────────────────────+ +───────────────────────────────────────────+
```

---

## Feature Inventory
Every feature identified during the Survey phase is mapped to a definitive milestone.

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Configuration & Environment Manager | Central YAML/Env loader with validation for API keys, trading modes, and parameters | M1 | explorer_survey_1 |
| 2 | Canonical Data Models | Strongly typed dataclasses for Bar, OrderRequest, OrderResult, Trade, Position, AccountInfo | M1 | spec_miner_survey_2 |
| 3 | SQLite WAL Persistence Engine | High-concurrency thread-safe SQLite storage with WAL mode for trades, equity, and logs | M1 | explorer_survey_1 |
| 4 | Parquet Historical Bar Cache | Compressed columnar storage for historical OHLCV bars with timestamp indexing | M1 | explorer_survey_1 |
| 5 | Multi-Currency FX Engine | Continuous mark-to-market EUR/USD tracking, quote normalization, and FX P&L isolation | M1 | spec_miner_survey_2 |
| 6 | Yahoo Finance Data Client | Free equity/ETF/FX historical and intraday data fetching with rate-limit backoff | M1 | spec_miner_survey_2 |
| 7 | Binance Public Data Client | Free crypto klines and depth fetching without API keys | M1 | spec_miner_survey_2 |
| 8 | Alpaca Market Data Client | Official Alpaca v2 market data client with fallback to Yahoo | M1 | spec_miner_survey_2 |
| 9 | Resilient Data Provider Cascade | Failover pipeline across Yahoo, Binance, Alpaca, and local cache | M1 | spec_miner_survey_2 |
| 10 | Offline Mock Data Provider | Deterministic offline market data generator for reproducible tests | M1 | explorer_survey_1 |
| 11 | Virtual €100 Account Ledger | Base EUR cash and equity tracking starting at exactly €100.00 with invariant checks | M2 | spec_miner_survey_2 |
| 12 | Fractional Share Sizing Engine | Decimal share calculation (up to 6 decimals) and cash-notional ordering for micro-capital | M2 | spec_miner_survey_2 |
| 13 | Market Order Simulation | Immediate execution with spread, volume impact, and price slippage | M2 | spec_miner_survey_2 |
| 14 | Limit Order Simulation | Price-contingent queue matching at limit price or better with Time-in-Force | M2 | spec_miner_survey_2 |
| 15 | Stop-Loss & Stop-Limit Orders | Adverse price breach triggers with gap slippage modeling | M2 | spec_miner_survey_2 |
| 16 | Trailing Stop Simulation | Dynamic stop price ratcheting tracking favorable high-water marks | M2 | spec_miner_survey_2 |
| 17 | Fixed BPS & Bid-Ask Spread Slippage | Basis points penalty plus Corwin-Schultz / bid-ask half-spread penalty | M2 | spec_miner_survey_2 |
| 18 | Volume Impact Slippage Model | Almgren-Chriss square-root price impact model ($Q/V$) | M2 | spec_miner_survey_2 |
| 19 | Transaction Fee Simulator | Regulatory fees (SEC, FINRA), exchange commission, and crypto maker/taker fees | M2 | spec_miner_survey_2 |
| 20 | Performance & Metrics Calculator | Real-time calculation of Cumulative Return, Sharpe, Sortino, Max Drawdown, Win Rate | M2 | spec_miner_survey_2 |
| 21 | S&P 500 Benchmark Synchronizer | Time-aligned tracking of SPY buy-and-hold returns, Alpha, and Beta | M2 | spec_miner_survey_2 |
| 22 | BaseBroker Abstraction Interface | Abstract base class defining uniform methods for paper and live order execution | M2 | spec_miner_survey_2 |
| 23 | SimulatedPaperBroker | Full-fidelity in-memory paper broker implementing BaseBroker | M2 | spec_miner_survey_2 |
| 24 | AlpacaPaperBroker | Integration with Alpaca REST API v2 paper sandbox with €100 virtual partition | M2 | spec_miner_survey_2 |
| 25 | MockBroker | Deterministic scriptable broker for offline testing with error injection | M2 | spec_miner_survey_2 |
| 26 | Configuration-Only Toggle | Zero-code switching between paper and live trading via configuration/environment | M2 | spec_miner_survey_2 |
| 27 | Pre-Trade Risk Guardrails | Drawdown circuit breaker, position size limits, and emergency kill switch | M2 | spec_miner_survey_2 |
| 28 | Live Broker Transition Framework | Documentation and extension hooks for myInvestor (via MCP), IBKR, and CCXT | M2 | spec_miner_survey_2 |
| 29 | Technical Indicator Suite | Optimized calculation of EMA (9, 21, 50, 200), RSI, MACD, BB, ATR, VWAP, ADX | M3 | spec_miner_survey_1 |
| 30 | Market Regime Classifier | ADX and volatility-based classification (Trending Bull/Bear, Mean-Revert, Choppy) | M3 | spec_miner_survey_1 |
| 31 | Scale-Invariant Composite Quant Score | Normalized score $S_{quant} \in [-1.0, 1.0]$ across all timeframes | M3 | spec_miner_survey_1 |
| 32 | Triple-Barrier Labeling Engine | Price path labeling with upper take-profit, lower stop-loss, and time barriers | M3 | spec_miner_survey_1 |
| 33 | ML Feature Engineering Pipeline | 32-dimensional standardized feature matrix from multi-lag returns and indicators | M3 | spec_miner_survey_1 |
| 34 | HistGradientBoosting ML Directional Engine | Directional probability forecast and confidence score using scikit-learn | M3 | spec_miner_survey_1 |
| 35 | PyTorch Sequence Embedding / MLP | Neural network representation extractor for complex intraday patterns | M3 | spec_miner_survey_1 |
| 36 | RL MDP State Vector Builder | Compact 9-dimensional state vector (regime, quant score, ML signal, equity, drawdown) | M3 | spec_miner_survey_1 |
| 37 | PyTorch RL Policy Agent | DDQN / Actor-Critic agent selecting optimal position weight $w_t \in [-1.0, 1.0]$ | M3 | spec_miner_survey_1 |
| 38 | Cost/Slippage Penalized Reward Function | Custom RL reward accounting for realized P&L, fees, slippage, and drawdown penalties | M3 | spec_miner_survey_1 |
| 39 | Intraday Hybrid Signal Aggregator | Fusion of Quant, ML, and RL signals into actionable TradingSignals with TP/SL | M3 | spec_miner_survey_1 |
| 40 | Dynamic Cross-Asset Screener | Multi-asset liquidity/volatility scanner selecting top tradable assets on free feeds | M3 | spec_miner_survey_1 |
| 41 | Multi-Armed Bandit Dynamic Weighting | Rolling Sharpe allocation dynamically shifting weights to best-performing assets | M3 | spec_miner_survey_1 |
| 42 | Micro-Capital Fractional Adapter | Position translation for €100 account enforcing minimum contract constraints | M3 | spec_miner_survey_1 |
| 43 | Multi-Timeframe Bar Synchronizer | Real-time alignment of 1m, 5m, 15m, and 1h bars without lookahead bias | M3 | spec_miner_survey_1 |
| 44 | Multi-Criteria Retraining Trigger Manager | Automated retraining on schedules, trade count (N=25), prediction drift, and drawdown | M4 | spec_miner_survey_1 |
| 45 | Realized Trade Feedback Ingestor | Captures closed-trade execution telemetry (P&L, slippage, holding time) for retraining | M4 | spec_miner_survey_1 |
| 46 | Prioritized Experience Replay Buffer | TD-error prioritized experience buffer for online RL training updates | M4 | spec_miner_survey_1 |
| 47 | Incremental Sliding-Window Retrainer | Rolling 60-day retraining of ML and RL models with recency weighting | M4 | spec_miner_survey_1 |
| 48 | Semantic Model Version Registry | Model checkpointing, metadata, hyperparameters, and metrics in models/registry.json | M4 | spec_miner_survey_1 |
| 49 | Champion-Challenger Validation Gate | Out-of-sample promotion gate preventing deployment of degraded candidate models | M4 | spec_miner_survey_1 |
| 50 | 8-Gate Live Readiness Engine | Deterministic evaluator for `is_live_ready: bool` (12mo backtest, Sharpe>1, etc.) | M4 | spec_miner_survey_1 |
| 51 | 30-Day Learning Loop Benchmark | Automated evaluation protocol proving measurable improvement from $v_0$ to $v_N$ | M4 | spec_miner_survey_1 |
| 52 | FastAPI Web Backend | High-performance asynchronous API server serving REST and WebSocket endpoints | M5 | explorer_survey_1 |
| 53 | WebSocket Real-Time Streamer | Sub-millisecond tick, trade fill, and equity push via `/ws/stream` | M5 | explorer_survey_1 |
| 54 | REST API Endpoints | Portfolio, trades, performance, benchmark, learning curves, mode toggle, readiness | M5 | explorer_survey_1 |
| 55 | Responsive Dark-Theme SPA | Zero-build HTML5/Vanilla JS frontend with glassmorphism institutional design | M5 | explorer_survey_1 |
| 56 | Chart.js Real-Time Visualizations | Equity curve vs S&P 500, drawdown chart, asset allocation, learning curve graphs | M5 | explorer_survey_1 |
| 57 | Paper/Live Mode UI Toggle | Authenticated toggle button switching execution mode with instant confirmation | M5 | explorer_survey_1 |
| 58 | Command Line Interface (CLI) | Console entry point `modeloinv` with subcommands: backtest, paper, live, dashboard | M5 | explorer_survey_1 |
| 59 | Comprehensive System Documentation | README with installation, paper trading guide, live broker setup, and dashboard guide | M5 | explorer_survey_1 |
| 60 | 100% E2E Test Suite Verification | Verification of all acceptance criteria across Tiers 1-4 published in TEST_READY.md | M6 | orchestrator |
| 61 | Adversarial Coverage Hardening (Tier 5) | White-box adversarial testing, edge-case stress verification, zero integrity gaps | M6 | orchestrator |

---

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Core Engine, Data Infrastructure & Storage | Config, Data Models, SQLite WAL, Parquet, Free Data Clients (Yahoo, Binance, Alpaca), Resilient Cascade, Mock Data Provider, FX Engine (Features 1-10) | None | DONE |
| M2 | Paper-Trading Simulation Engine & Broker Abstraction (R2 & R3) | Virtual €100 Ledger, Fractional Sizing, Order Simulation (Market, Limit, Stop, Trailing), Slippage & Fee Models, Metrics Engine, S&P 500 Tracker, BaseBroker, SimulatedPaperBroker, AlpacaPaperBroker, MockBroker, Config Toggle, Risk Guardrails, Broker Docs (Features 11-28) | M1 | DONE |
| M3 | Hybrid AI Trading Strategy Engine (R1) | Technical Indicators, Regime Gate, Composite Score, Triple-Barrier ML, HistGradientBoosting, PyTorch RL Policy & Cost-Penalized Reward, Signal Aggregator, Dynamic Universe Screener, Bandit Allocation, Fractional Adapter, Multi-Timeframe Sync (Features 29-43) | M1, M2 | DONE |
| M4 | Continuous Learning Loop, Model Registry & Live Readiness (R5) | Retraining Triggers, Trade Feedback Ingestor, Prioritized Replay, Incremental Retrainer, Model Version Registry, Champion-Challenger Gate, 8-Gate Live Readiness Engine, 30-Day Learning Loop Verification (Features 44-51) | M1, M2, M3 | DONE |
| M5 | Live Performance Dashboard Web UI & CLI (R4) | FastAPI Server, WebSocket Streamer, REST API, Dark-Theme SPA UI, Chart.js Visualizations, Mode Toggle, CLI runner, README & User Documentation (Features 52-59) | M1, M2, M3, M4 | IN_PROGRESS |
| M6 | Final Acceptance & Verification (E2E & Adversarial) | Phase 1: Pass 100% of E2E Test Suite (Tiers 1-4) in TEST_READY.md.<br>Phase 2: Adversarial Coverage Hardening (Tier 5). Formal Forensic Audit and final sign-off (Features 60-61). | M1, M2, M3, M4, M5, TEST_READY.md | PLANNED |

---

## Interface Contracts

### 1. `modeloinv.core` ↔ All Subsystems
```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any

class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    TRAILING_STOP = "TRAILING_STOP"

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class AssetClass(str, Enum):
    EQUITY = "EQUITY"
    ETF = "ETF"
    CRYPTO = "CRYPTO"
    FOREX = "FOREX"

@dataclass(frozen=True)
class Bar:
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    asset_class: AssetClass = AssetClass.EQUITY
    currency: str = "USD"

@dataclass
class OrderRequest:
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Optional[float] = None
    notional: Optional[float] = None
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trail_percent: Optional[float] = None
    client_order_id: str = field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d%H%M%S%f"))

@dataclass
class OrderResult:
    order_id: str
    client_order_id: str
    symbol: str
    status: OrderStatus
    side: OrderSide
    filled_quantity: float
    filled_avg_price: float
    fee_eur: float
    slippage_eur: float
    timestamp: datetime
    message: str = ""

@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    currency: str = "USD"
    unrealized_pnl_eur: float = 0.0

@dataclass
class AccountInfo:
    currency: str
    cash_eur: float
    equity_eur: float
    initial_capital_eur: float
    realized_pnl_eur: float
    unrealized_pnl_eur: float
    buying_power_eur: float
    is_live: bool
```

### 2. `modeloinv.execution` ↔ `modeloinv.strategy` & `modeloinv.dashboard` (BaseBroker)
```python
from abc import ABC, abstractmethod

class BaseBroker(ABC):
    @abstractmethod
    def get_account(self) -> AccountInfo:
        """Returns current account balance, equity, and P&L in EUR."""
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Returns list of open positions."""
        pass

    @abstractmethod
    def submit_order(self, request: OrderRequest) -> OrderResult:
        """Submits an order and returns the execution result."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancels an active pending order."""
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderResult:
        """Retrieves status of a previously submitted order."""
        pass
```

### 3. `modeloinv.strategy` ↔ `modeloinv.learning` (Hybrid Strategy & Signals)
```python
@dataclass
class TradingSignal:
    timestamp: datetime
    symbol: str
    target_weight: float  # -1.0 to 1.0
    action: str           # BUY, SELL, HOLD
    confidence: float     # 0.0 to 1.0
    quant_score: float    # -1.0 to 1.0
    ml_score: float       # -1.0 to 1.0
    rl_action: int
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### 4. `modeloinv.dashboard` REST & WebSocket Endpoints
- `GET /api/v1/portfolio` -> `AccountInfo`
- `GET /api/v1/positions` -> `List[Position]`
- `GET /api/v1/trades` -> `List[TradeRecord]`
- `GET /api/v1/performance` -> Performance metrics (Sharpe, Drawdown, Cumulative Return, Win Rate)
- `GET /api/v1/benchmark` -> Comparative equity series (modeloINV vs S&P 500)
- `GET /api/v1/learning` -> Model registry history, metrics ($v_0$ vs $v_N$), training loss curves
- `GET /api/v1/readiness` -> 8-gate `is_live_ready` report with gate checklist
- `POST /api/v1/mode` -> Toggle paper/live mode (`{"mode": "paper" | "live"}`)
- `WebSocket /ws/stream` -> Real-time streaming push of ticks, order fills, equity updates

---

## Code Layout
```
modeloINV/
├── src/
│   └── modeloinv/
│       ├── __init__.py
│       ├── config.py                 # Configuration settings, defaults, env loading
│       ├── core/                     # Canonical models, currency converter, events
│       │   ├── __init__.py
│       │   ├── models.py
│       │   ├── currency.py
│       │   └── events.py
│       ├── storage/                  # SQLite WAL and Parquet database managers
│       │   ├── __init__.py
│       │   ├── database.py
│       │   └── cache.py
│       ├── data/                     # Market data providers and cascade
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── yahoo.py
│       │   ├── binance.py
│       │   ├── alpaca.py
│       │   ├── cascade.py
│       │   └── mock.py
│       ├── execution/                # Brokers, slippage, fees, and risk guardrails
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── paper.py
│       │   ├── alpaca.py
│       │   ├── mock.py
│       │   ├── slippage.py
│       │   ├── fees.py
│       │   └── risk.py
│       ├── strategy/                 # Hybrid AI strategy engine
│       │   ├── __init__.py
│       │   ├── indicators.py
│       │   ├── regime.py
│       │   ├── ml_model.py
│       │   ├── rl_agent.py
│       │   ├── universe.py
│       │   └── engine.py
│       ├── learning/                 # Continuous learning and model management
│       │   ├── __init__.py
│       │   ├── triggers.py
│       │   ├── feedback.py
│       │   ├── registry.py
│       │   ├── champion.py
│       │   └── readiness.py
│       ├── dashboard/                # FastAPI web dashboard and static assets
│       │   ├── __init__.py
│       │   ├── app.py
│       │   ├── routes.py
│       │   ├── websockets.py
│       │   └── static/
│       │       ├── index.html
│       │       ├── css/style.css
│       │       └── js/app.js
│       └── cli.py                    # Console CLI entrypoint
├── tests/
│   ├── conftest.py                   # Pytest fixtures and in-memory test databases
│   ├── unit/                         # Unit tests for each module
│   ├── integration/                  # Cross-module integration tests
│   └── e2e/                          # 4-tier requirement-driven E2E tests
├── docs/                             # Documentation for live broker setup & architecture
│   ├── BROKER_SETUP.md
│   └── ARCHITECTURE.md
├── config/
│   └── default.yaml                  # Default configuration parameters
├── models/                           # Model registry and saved checkpoints
│   └── registry.json
├── data/                             # Local SQLite database and Parquet bar cache
│   └── .gitkeep
├── pyproject.toml                    # Package definition and dependencies
├── README.md                         # Complete project documentation
├── TEST_INFRA.md                     # E2E Test Suite specifications
└── TEST_READY.md                     # E2E Test Suite readiness publication
```

---

## Test & Acceptance Strategy

1. **Dual Track Execution**:
   - **E2E Testing Track**: Dispatched concurrently to build `TEST_INFRA.md` and implement the 4-tier requirement-driven test suite (≥11×N tests). Publishes `TEST_READY.md`.
   - **Implementation Track**: Executes M1 through M5 per dependency sequence.
2. **Acceptance Criteria Verification (M6)**:
   - **Simulation Correctness**: Slippage and transaction costs verified against reference baselines. Initial €100 balance tracked with zero divergence.
   - **Strategy Performance**: 12-month backtest beats buy-and-hold S&P 500 with Sharpe ratio > 1.0 and at least one 30-day window with ≥15% simulated return.
   - **Live Readiness**: Broker abstraction layer documented with working Alpaca paper API; config-only paper/live toggle.
   - **Dashboard**: Browser-accessible web UI showing real-time P&L, trades, S&P 500 comparison, and learning curves in institutional dark-theme glassmorphism.
   - **Learning Loop**: Version registry tracking timestamps & metrics; automated 30-day simulated run showing measurable improvement ($v_N$ vs $v_0$).
   - **Documentation**: Comprehensive README explaining installation, paper trading, live broker connection, and dashboard operation.
3. **Forensic Integrity Verification**:
   - Every milestone undergoes independent Forensic Audit (`teamwork_preview_auditor`) with ZERO TOLERANCE for dummy implementations, hardcoded values, or facade bypasses.
