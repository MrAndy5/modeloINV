"""Root pytest fixtures and test doubles for modeloINV E2E test suite.

All fixtures and doubles operate 100% offline, deterministically, and without
external network or live broker dependencies.
"""

from __future__ import annotations

import os
import sys
import random
import sqlite3
import datetime
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Generator

import pytest
import numpy as np

# Ensure project root and src/ are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Set test environment flags
os.environ["MODELOINV_ENV"] = "test"
os.environ["MODELOINV_EXECUTION_MODE"] = "paper"
os.environ["MODELOINV_DATA_DIR"] = str(PROJECT_ROOT / "data")
os.environ["MODELOINV_STARTING_CAPITAL"] = "100.0"

# Fix seeds for 100% deterministic reproducibility
random.seed(42)
np.random.seed(42)
try:
    import torch
    torch.manual_seed(42)
except ImportError:
    pass

# =====================================================================
# Canonical Data Contracts (Adhering to PROJECT.md § Interface Contracts)
# =====================================================================

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
    timestamp: datetime.datetime
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
    client_order_id: str = field(default_factory=lambda: datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S%f"))

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
    timestamp: datetime.datetime
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

@dataclass
class TradingSignal:
    timestamp: datetime.datetime
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


# Try to export or alias from modeloinv if already implemented
try:
    import modeloinv.core.models as core_models
    # Validate core models match expectations
except ImportError:
    pass


# =====================================================================
# In-Memory SQLite Fixture
# =====================================================================

@pytest.fixture
def sqlite_db_conn() -> Generator[sqlite3.Connection, None, None]:
    """In-memory SQLite database simulating WAL-mode transactional storage."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")
    
    # Initialize canonical tables
    cursor.executescript("""
        CREATE TABLE trades (
            trade_id TEXT PRIMARY KEY,
            order_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity REAL NOT NULL,
            price REAL NOT NULL,
            fee_eur REAL NOT NULL,
            slippage_eur REAL NOT NULL,
            realized_pnl_eur REAL NOT NULL,
            timestamp TEXT NOT NULL
        );
        CREATE TABLE account_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            cash_eur REAL NOT NULL,
            equity_eur REAL NOT NULL,
            realized_pnl_eur REAL NOT NULL,
            unrealized_pnl_eur REAL NOT NULL,
            is_live INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE model_registry (
            version TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            model_type TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            is_champion INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE readiness_gates (
            gate_name TEXT PRIMARY KEY,
            passed INTEGER NOT NULL,
            details TEXT NOT NULL,
            checked_at TEXT NOT NULL
        );
    """)
    conn.commit()
    yield conn
    conn.close()


# =====================================================================
# Mock Market Data Provider
# =====================================================================

class MockMarketDataProvider:
    """Deterministic offline market data generator across equities, ETFs, crypto, and FX."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate_bars(
        self,
        symbol: str,
        start_dt: datetime.datetime,
        num_bars: int = 100,
        timeframe_minutes: int = 5,
        start_price: float = 100.0,
        volatility: float = 0.002,
        drift: float = 0.0001,
        asset_class: AssetClass = AssetClass.EQUITY,
        currency: str = "USD"
    ) -> List[Bar]:
        bars = []
        current_price = start_price
        current_time = start_dt

        for _ in range(num_bars):
            ret = self.rng.normal(drift, volatility)
            open_p = current_price
            close_p = open_p * (1.0 + ret)
            high_p = max(open_p, close_p) * (1.0 + abs(self.rng.normal(0, volatility * 0.5)))
            low_p = min(open_p, close_p) * (1.0 - abs(self.rng.normal(0, volatility * 0.5)))
            volume = float(self.rng.randint(1000, 50000))

            bars.append(Bar(
                timestamp=current_time,
                symbol=symbol,
                open=round(open_p, 4),
                high=round(high_p, 4),
                low=round(low_p, 4),
                close=round(close_p, 4),
                volume=volume,
                asset_class=asset_class,
                currency=currency
            ))
            current_price = close_p
            current_time += datetime.timedelta(minutes=timeframe_minutes)

        return bars

@pytest.fixture
def mock_data_provider() -> MockMarketDataProvider:
    return MockMarketDataProvider(seed=42)


# =====================================================================
# Mock Simulated Paper Broker
# =====================================================================

class OfflineSimulatedBroker:
    """Full-fidelity deterministic offline paper broker tracking €100 virtual ledger."""

    def __init__(self, initial_capital_eur: float = 100.0, eur_usd_rate: float = 1.08):
        self.initial_capital_eur = initial_capital_eur
        self.cash_eur = initial_capital_eur
        self.eur_usd_rate = eur_usd_rate
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, OrderResult] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self.is_live = False
        self.max_drawdown_limit = 0.20  # 20% circuit breaker
        self.is_halted = False

    def get_account(self) -> AccountInfo:
        positions_market_value_eur = sum((p.quantity * p.current_price) / self.eur_usd_rate for p in self.positions.values())
        equity = self.cash_eur + positions_market_value_eur
        unrealized = sum(p.unrealized_pnl_eur for p in self.positions.values())
        realized = equity - self.initial_capital_eur - unrealized
        return AccountInfo(
            currency="EUR",
            cash_eur=round(self.cash_eur, 4),
            equity_eur=round(equity, 4),
            initial_capital_eur=self.initial_capital_eur,
            realized_pnl_eur=round(realized, 4),
            unrealized_pnl_eur=round(unrealized, 4),
            buying_power_eur=round(max(0.0, self.cash_eur), 4),
            is_live=self.is_live
        )

    def get_positions(self) -> List[Position]:
        return list(self.positions.values())

    def update_market_price(self, symbol: str, current_price: float):
        if symbol in self.positions:
            pos = self.positions[symbol]
            pos.current_price = current_price
            # Update unrealized P&L in EUR
            price_diff = current_price - pos.entry_price
            val_usd = price_diff * pos.quantity
            pos.unrealized_pnl_eur = round(val_usd / self.eur_usd_rate, 4)

    def submit_order(self, request: OrderRequest, current_market_price: Optional[float] = None) -> OrderResult:
        if self.is_halted:
            return OrderResult(
                order_id=f"ord_{len(self.orders)+1}",
                client_order_id=request.client_order_id,
                symbol=request.symbol,
                status=OrderStatus.REJECTED,
                side=request.side,
                filled_quantity=0.0,
                filled_avg_price=0.0,
                fee_eur=0.0,
                slippage_eur=0.0,
                timestamp=datetime.datetime.utcnow(),
                message="Trading halted due to risk circuit breaker"
            )

        ref_price = current_market_price if current_market_price is not None else 100.0
        
        # Calculate quantity
        if request.quantity is not None:
            qty = request.quantity
        elif request.notional is not None:
            # Notional EUR converted to USD shares
            notional_usd = request.notional * self.eur_usd_rate
            qty = round(notional_usd / ref_price, 6)
        else:
            qty = 1.0

        # Slippage: 2 bps + bid-ask spread
        slippage_rate = 0.0002
        if request.side == OrderSide.BUY:
            exec_price = ref_price * (1.0 + slippage_rate)
        else:
            exec_price = ref_price * (1.0 - slippage_rate)

        # Fees: SEC ($0.0000278) + FINRA TAF ($0.000166) + Exchange (0.001) min $0.01
        trade_val_usd = qty * exec_price
        fee_usd = max(0.01, trade_val_usd * 0.0005)
        fee_eur = round(fee_usd / self.eur_usd_rate, 4)
        slippage_eur = round((abs(exec_price - ref_price) * qty) / self.eur_usd_rate, 4)

        trade_cost_eur = (trade_val_usd / self.eur_usd_rate) + fee_eur

        if request.side == OrderSide.BUY and trade_cost_eur > self.cash_eur:
            return OrderResult(
                order_id=f"ord_{len(self.orders)+1}",
                client_order_id=request.client_order_id,
                symbol=request.symbol,
                status=OrderStatus.REJECTED,
                side=request.side,
                filled_quantity=0.0,
                filled_avg_price=0.0,
                fee_eur=0.0,
                slippage_eur=0.0,
                timestamp=datetime.datetime.utcnow(),
                message=f"Insufficient funds: required {trade_cost_eur:.2f} EUR, available {self.cash_eur:.2f} EUR"
            )

        # Execute order
        order_id = f"ord_{len(self.orders)+1:04d}"
        if request.side == OrderSide.BUY:
            self.cash_eur -= trade_cost_eur
            if request.symbol in self.positions:
                curr = self.positions[request.symbol]
                new_qty = curr.quantity + qty
                curr.entry_price = (curr.entry_price * curr.quantity + exec_price * qty) / new_qty
                curr.quantity = new_qty
                curr.current_price = exec_price
            else:
                self.positions[request.symbol] = Position(
                    symbol=request.symbol,
                    quantity=qty,
                    entry_price=exec_price,
                    current_price=exec_price,
                    currency="USD",
                    unrealized_pnl_eur=0.0
                )
        else:  # SELL
            if request.symbol not in self.positions or self.positions[request.symbol].quantity < qty:
                return OrderResult(
                    order_id=order_id,
                    client_order_id=request.client_order_id,
                    symbol=request.symbol,
                    status=OrderStatus.REJECTED,
                    side=request.side,
                    filled_quantity=0.0,
                    filled_avg_price=0.0,
                    fee_eur=0.0,
                    slippage_eur=0.0,
                    timestamp=datetime.datetime.utcnow(),
                    message="Insufficient position quantity to sell"
                )
            pos = self.positions[request.symbol]
            proceeds_usd = qty * exec_price
            proceeds_eur = (proceeds_usd / self.eur_usd_rate) - fee_eur
            self.cash_eur += proceeds_eur
            pos.quantity -= qty
            if pos.quantity <= 1e-6:
                del self.positions[request.symbol]

        result = OrderResult(
            order_id=order_id,
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            status=OrderStatus.FILLED,
            side=request.side,
            filled_quantity=qty,
            filled_avg_price=round(exec_price, 4),
            fee_eur=fee_eur,
            slippage_eur=slippage_eur,
            timestamp=datetime.datetime.utcnow(),
            message="Executed successfully"
        )
        self.orders[order_id] = result
        self.trade_history.append({
            "order_id": order_id,
            "symbol": request.symbol,
            "side": request.side.value,
            "quantity": qty,
            "price": exec_price,
            "fee_eur": fee_eur,
            "slippage_eur": slippage_eur,
            "timestamp": result.timestamp.isoformat()
        })
        return result

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self.orders and self.orders[order_id].status == OrderStatus.PENDING:
            self.orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False

    def get_order_status(self, order_id: str) -> OrderResult:
        if order_id in self.orders:
            return self.orders[order_id]
        raise KeyError(f"Order {order_id} not found")

@pytest.fixture
def offline_broker() -> OfflineSimulatedBroker:
    return OfflineSimulatedBroker(initial_capital_eur=100.0)


# =====================================================================
# Financial Metrics Utilities
# =====================================================================

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    rf_per_period = risk_free_rate / periods_per_year
    excess_returns = returns - rf_per_period
    std = float(np.std(excess_returns, ddof=1))
    if std <= 1e-12:
        return 0.0
    return float(np.mean(excess_returns) / std * np.sqrt(periods_per_year))

def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    rf_per_period = risk_free_rate / periods_per_year
    excess_returns = returns - rf_per_period
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return 10.0  # Zero downside risk
    downside_std = float(np.sqrt(np.mean(downside_returns ** 2)))
    if downside_std <= 1e-12:
        return 0.0
    return float(np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year))

def calculate_max_drawdown(equity_series: np.ndarray) -> float:
    if len(equity_series) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity_series)
    drawdowns = (peak - equity_series) / np.maximum(peak, 1e-6)
    return float(np.max(drawdowns))

def calculate_win_rate(trade_pnls: List[float]) -> float:
    if not trade_pnls:
        return 0.0
    wins = sum(1 for pnl in trade_pnls if pnl > 0)
    return wins / len(trade_pnls)
