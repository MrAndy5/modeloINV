# Original User Request

## Initial Request — 2026-10-26T18:51:28Z

Build a polished, production-quality algorithmic trading system called **modeloINV** that learns to beat the S&P 500 by targeting ≥20% monthly return on a starting capital of €100. The system operates in two phases: (1) paper-trading simulation with a self-improving hybrid AI strategy, and (2) live trading via a broker-agnostic API layer that can connect to real funds later (initially targeting myInvestor or any MCP-controllable broker). The codebase must be well-documented, presentable, and designed to handle the transition from paper to live without architectural changes.

Working directory: `C:\Users\andre\Desktop\modeloINV`
Integrity mode: benchmark

---

## Requirements

### R1. Hybrid AI trading strategy (intraday focus)

Build a hybrid strategy engine that combines classical quant signals (momentum, mean-reversion, trend-following), a machine-learning layer (e.g. gradient boosting or neural network trained on OHLCV + derived features), and a reinforcement learning agent that continuously improves its execution policy in simulation. The system must trade short-term / intraday timeframes (minutes to hours) across any mix of asset classes (equities, ETFs, crypto, forex) where free data is available. The strategy selection and asset universe should be dynamic — the model should discover what is most profitable.

### R2. Paper-trading simulation engine

Implement a realistic paper-trading engine that simulates order execution, slippage, and transaction costs starting from a €100 virtual account. The engine must use the best freely available historical and/or live data (e.g. Yahoo Finance, Alpha Vantage free tier, Binance public API, Alpaca free data) — the team should select and combine sources for the richest coverage. The simulation must track all trades, P&L, and model learning progress over time.

### R3. Broker-agnostic live trading interface

Implement a clean abstraction layer for live order execution so the system can be connected to a real broker (initially myInvestor, or any broker with an API or MCP hook) without changing the strategy or simulation code. Document clearly what needs to be configured to go live with a new broker.

### R4. Live performance dashboard (web UI)

Build a real-time web dashboard showing: current portfolio value and P&L, open/closed positions, trade history, strategy performance vs S&P 500 benchmark, model learning curves, and a toggle to switch between paper and live mode. The dashboard must be polished enough to demonstrate the system to a third party.

### R5. Continuous learning and self-improvement loop

The system must implement an online or periodic retraining loop: as paper trading produces new trade outcomes, the ML and RL components must incorporate that feedback and update their models. The system should log model versions, performance metrics over time, and flag when it believes it is ready for live deployment (e.g. sustained X months of simulated ≥15% monthly return with acceptable drawdown).

---

## Acceptance Criteria

### Simulation correctness
- [ ] Paper-trading engine correctly accounts for slippage and transaction costs (verifiable by replaying a fixed historical period and comparing to a reference baseline)
- [ ] Starting balance of €100 is accurately tracked across all simulated trades

### Strategy performance (paper trading)
- [ ] After backtesting on at least 12 months of historical data, the strategy achieves a cumulative return that beats buy-and-hold S&P 500 over the same period
- [ ] The system produces a Sharpe ratio > 1.0 in backtesting
- [ ] Monthly P&L reports show at least one 30-day window with ≥15% simulated return

### Live readiness
- [ ] The broker abstraction layer has a documented interface and at least one working implementation (e.g. Alpaca paper API or similar free sandbox)
- [ ] Switching from paper to live mode requires only a configuration change, not code changes

### Dashboard
- [ ] The web dashboard loads in a browser and displays real-time P&L, trade history, and strategy vs S&P 500 comparison
- [ ] Dashboard is visually polished and navigable without instructions

### Learning loop
- [ ] The system logs model versions with timestamps and performance metrics
- [ ] After a simulated 30-day run, measurable improvement in a performance metric (e.g. win rate, Sharpe ratio) is visible between the first and last model version

### Documentation
- [ ] README explains: how to install, how to run paper trading, how to connect a live broker, and how to read the dashboard
- [ ] Code is well-commented throughout
