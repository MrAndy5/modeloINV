# Broker Setup & Live Transition Guide

**Project**: modeloINV — Algorithmic Trading System  
**Component**: Execution Engine & Broker Abstraction (R2 & R3)  
**Version**: 1.0.0  

---

## 1. Architecture Overview

`modeloINV` is designed with an institutional, broker-agnostic architecture. All execution components (Strategy Engine, Dashboard, Risk Manager) interact strictly with the canonical `BaseBroker` abstraction interface. The underlying execution backend can be switched dynamically through configuration or environment variables with **zero application code modifications**.

```
+─────────────────────────────────────────────────────────────+
│                       Strategy Engine                       │
+─────────────────────────────────────────────────────────────+
                               │
                               ▼  (OrderRequest)
+─────────────────────────────────────────────────────────────+
│                        Risk Manager                         │
│   (Max position size, daily drawdown stop, kill switch)     │
+─────────────────────────────────────────────────────────────+
                               │
                               ▼  (BaseBroker interface)
+─────────────────────────────────────────────────────────────+
│                        BrokerFactory                        │
+─────────────────────────────────────────────────────────────+
         │                     │                     │
         ▼                     ▼                     ▼
+─────────────────+   +─────────────────+   +─────────────────+
│ SimulatedPaper  │   │   AlpacaPaper   │   │   MockBroker    │
│     Broker      │   │     Broker      │   │   (Unit/E2E)    │
│  (€100 Ledger)  │   │  (€100 Virtual  │   +─────────────────+
+─────────────────+   │    Partition)   │
                      +─────────────────+
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
       Alpaca Paper REST API          Alpaca Live REST API
   (https://paper-api.alpaca.markets) (https://api.alpaca.markets)
```

---

## 2. Configuration-Only Mode Switching

Switching between local simulation, Alpaca paper trading, and live execution requires **no code changes**. You simply set environment variables or edit `config/default.yaml`.

### 2.1 Mode Toggle via Environment Variables

To run in **Paper Trading Mode** (Local Virtual Ledger):
```bash
export MODELOINV_EXECUTION_MODE=paper
export MODELOINV_EXECUTION__BROKER=simulated
export MODELOINV_STARTING_CAPITAL_EUR=100.00
```

To run in **Alpaca Paper Sandbox Mode** (Alpaca API with Virtual €100 Partition):
```bash
export MODELOINV_EXECUTION_MODE=paper
export MODELOINV_EXECUTION__BROKER=alpaca
export MODELOINV_ALPACA_API_KEY_ID="your_paper_key_id"
export MODELOINV_ALPACA_API_SECRET_KEY="your_paper_secret_key"
export MODELOINV_ALPACA_BASE_URL="https://paper-api.alpaca.markets"
```

To run in **Alpaca Live Mode** (Real Capital):
```bash
export MODELOINV_EXECUTION_MODE=live
export MODELOINV_EXECUTION__BROKER=alpaca
export MODELOINV_ALPACA_API_KEY_ID="your_live_key_id"
export MODELOINV_ALPACA_API_SECRET_KEY="your_live_secret_key"
export MODELOINV_ALPACA_BASE_URL="https://api.alpaca.markets"
```

### 2.2 Configuration Schema (`config/default.yaml`)

```yaml
system:
  project_name: "modeloINV"
  execution_mode: "paper"  # Options: "paper" | "live" | "backtest"
  log_level: "INFO"

account:
  base_currency: "EUR"
  starting_capital_eur: 100.00
  enable_fractional: true
  min_order_notional_eur: 1.00
  max_portfolio_positions: 5
  max_position_weight: 0.35

execution:
  broker: "simulated"      # Options: "simulated" | "alpaca" | "mock"
  slippage_model: "fixed_bps"
  slippage_bps: 5.0
  fee_model: "zero_commission"

alpaca:
  api_key_id: "${MODELOINV_ALPACA_API_KEY_ID:-}"
  api_secret_key: "${MODELOINV_ALPACA_API_SECRET_KEY:-}"
  base_url: "https://paper-api.alpaca.markets"
  live_base_url: "https://api.alpaca.markets"
  virtual_capital_eur: 100.00

risk:
  daily_drawdown_limit: 0.10     # 10% daily drawdown circuit breaker
  max_drawdown_limit: 0.15       # 15% overall peak-to-trough stop
  kill_switch_drawdown: 0.25     # 25% emergency kill switch
```

---

## 3. Alpaca Integration & Virtual Sub-Account Partition

### 3.1 The Sandbox Capital Problem
By default, Alpaca initializes paper trading accounts with **$100,000 USD** in buying power. However, `modeloINV` is mathematically architected to trade and compound from an initial virtual capital of exactly **€100.00 EUR**.

### 3.2 The Virtual Sub-Account Partition Solution
`AlpacaPaperBroker` implements an enforced software partition:
1. **Capital Ceiling**: Caps effective portfolio capital at `allocated_capital_eur = 100.00`.
2. **Pre-Trade Guardrail**: Every `submit_order` validates that:
   $$\text{CurrentPositionValue}_{EUR} + \text{OrderNotional}_{EUR} \le \text{VirtualCeiling}_{EUR}$$
   If an order would breach the €100 limit, it is immediately rejected locally before transmitting to Alpaca.
3. **Multi-Currency Normalization**: Continuous EUR/USD rate conversion marks positions and cash to base EUR.

---

## 4. Connecting to myInvestor (Retail Broker via MCP Bridge)

### 4.1 Overview
`myInvestor` is a premier Spanish retail investment platform and neobank offering index funds, ETFs, and equities. Because myInvestor does not provide an official public OpenAPI or FIX retail endpoint, integration is achieved through a **Model Context Protocol (MCP) Execution Bridge**.

### 4.2 MCP Bridge Architecture

```
+───────────────────────+          +───────────────────────+
│   MyInvestorBroker    │  JSON-   │    myinvestor-mcp     │
│     (BaseBroker)      │ ───────> │      Tool Server      │
+───────────────────────+   RPC    +───────────────────────+
                                               │
                                     (Playwright / Private)
                                               ▼
                                   +───────────────────────+
                                   │   myInvestor Portal   │
                                   +───────────────────────+
```

### 4.3 Exposed MCP Tool Interface
The `myinvestor-mcp` service exposes the following standard tools:
- `myinvestor_get_balances() -> {"cash_eur": float, "portfolio_eur": float}`
- `myinvestor_get_positions() -> List[Dict[str, Any]]`
- `myinvestor_place_order(symbol: str, isin: str, notional_eur: float, order_type: str) -> Dict`
- `myinvestor_cancel_order(order_id: str) -> bool`

### 4.4 Specific Considerations for myInvestor
1. **ISIN Mapping**: Retail European exchanges operate via ISIN codes rather than US ticker symbols:
   - S&P 500 ETF (SPY): `US78462F1030`
   - Vanguard S&P 500 UCITS ETF (VUSA): `IE00B3XXRP09`
2. **Spanish Capital Gains Tax Retention**: 19% withholding on capital gains up to €6,000 is tracked by the performance accounting ledger.
3. **No Direct Fractional Shares on Madrid / BME**: For micro-capital (€100), automated fractional index fund plans or Vanguard UCITS ETFs should be selected.

---

## 5. Connecting to Interactive Brokers (IBKR)

Interactive Brokers supports automated retail trading via the **Client Portal Web API** or **TWS / IB Gateway** (`ib_insync`):

1. **Fractional US Equities**: IBKR natively supports decimal fractional share trading down to 4 decimal places.
2. **Multi-Currency Ledger**: Allows holding EUR cash and USD balances simultaneously, avoiding FX conversion on every trade.
3. **Integration Steps**:
   - Start the IBKR Client Portal Gateway: `https://localhost:5000/v1/api/`.
   - Inherit from `BaseBroker` as `InteractiveBrokersBroker`.
   - Implement `submit_order` routing to `POST /v1/api/iserver/account/{accountId}/orders`.

---

## 6. Connecting to CCXT (Crypto Exchanges)

For 24/7 crypto market execution (Binance, Kraken, Coinbase):
1. Install CCXT: `pip install ccxt`.
2. Native EUR trading pairs: `BTCEUR` and `ETHEUR` eliminate USD conversion friction.
3. Zero network latency: Sub-second WebSocket order routing and live order book streaming.

---

## 7. Pre-Flight Live Readiness Checklist

Before transitioning `MODELOINV_EXECUTION_MODE` from `paper` to `live`, verify all safety gates:

- [ ] **Gate 1**: 12-Month Out-of-Sample Backtest beats S&P 500 with Sharpe ratio > 1.0.
- [ ] **Gate 2**: Maximum historical drawdown < 15.0%.
- [ ] **Gate 3**: Kill switch and emergency liquidation independently verified.
- [ ] **Gate 4**: Account credentials stored in environment variables (never in source code).
- [ ] **Gate 5**: API keys verified with read/trade permissions (withdrawal permissions strictly disabled).
- [ ] **Gate 6**: Daily loss limit circuit breaker configured (default €10.00 / 10%).
- [ ] **Gate 7**: Multi-currency EUR/USD rate feed active and fresh (< 60 seconds).
- [ ] **Gate 8**: Dry-run order placed and canceled in Paper mode within last 24 hours.
