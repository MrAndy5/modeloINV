"""Unit tests for the modeloINV Command Line Interface (CLI).

Validates:
- Parser construction and argument parsing across all subcommands
- Backtest runner execution and performance table formatting
- Paper trading loop with initial virtual capital
- Live execution safety guardrails (confirm flag requirement)
- Live Readiness 8-gate evaluation command output
- Retrain command execution and model registry version bump
- Dashboard command launch and configuration
"""

from __future__ import annotations

import datetime
import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modeloinv.cli import build_parser, main


class TestCLIParser:
    """Validates CLI argument parsing and subcommand configuration."""

    def test_parser_has_all_required_subcommands(self):
        parser = build_parser()
        # Verify subcommands exist
        subparsers_action = next(
            action for action in parser._actions if action.dest == "command"
        )
        subcommands = subparsers_action.choices.keys()
        for cmd in ["backtest", "paper", "live", "dashboard", "retrain", "evaluate"]:
            assert cmd in subcommands

    def test_backtest_arguments(self):
        parser = build_parser()
        args = parser.parse_args(["backtest", "--months", "6", "--capital", "250.0", "--universe", "AAPL,MSFT"])
        assert args.command == "backtest"
        assert args.months == 6
        assert args.capital == 250.0
        assert args.universe == "AAPL,MSFT"

    def test_paper_arguments(self):
        parser = build_parser()
        args = parser.parse_args(["paper", "--capital", "150.0", "--duration", "5"])
        assert args.command == "paper"
        assert args.capital == 150.0
        assert args.duration == 5

    def test_live_arguments(self):
        parser = build_parser()
        args = parser.parse_args(["live", "--broker", "alpaca_paper", "--confirm"])
        assert args.command == "live"
        assert args.broker == "alpaca_paper"
        assert args.confirm is True

    def test_dashboard_arguments(self):
        parser = build_parser()
        args = parser.parse_args(["dashboard", "--host", "0.0.0.0", "--port", "9000"])
        assert args.command == "dashboard"
        assert args.host == "0.0.0.0"
        assert args.port == 9000

    def test_retrain_arguments(self):
        parser = build_parser()
        args = parser.parse_args(["retrain", "--force"])
        assert args.command == "retrain"
        assert args.force is True

    def test_evaluate_arguments(self):
        parser = build_parser()
        args = parser.parse_args(["evaluate", "--db", "custom.db", "--simulate"])
        assert args.command == "evaluate"
        assert args.db == "custom.db"
        assert args.simulate is True


class TestCLIExecution:
    """Validates execution behavior of each CLI subcommand."""

    def test_main_no_arguments_prints_help(self, capsys):
        exit_code = main([])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "usage: modeloinv" in captured.out or "usage: modeloinv" in captured.err or "modeloINV" in captured.out

    def test_backtest_execution(self, capsys):
        exit_code = main(["backtest", "--months", "3", "--capital", "100.0"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "modeloINV BACKTEST SIMULATION" in captured.out
        assert "Starting Capital" in captured.out
        assert "Annualized Sharpe" in captured.out
        assert "Max Portfolio Drawdown" in captured.out

    def test_paper_execution(self, capsys):
        exit_code = main(["paper", "--capital", "100.0", "--duration", "1"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "REAL-TIME PAPER TRADING ENGINE" in captured.out
        assert "€100.00 EUR" in captured.out
        assert "[PAPER]" in captured.out

    def test_live_without_confirm_fails_with_guardrail(self, capsys):
        exit_code = main(["live", "--broker", "alpaca_paper"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "[GUARDRAIL ERROR]" in captured.out
        assert "--confirm" in captured.out

    def test_live_with_confirm_succeeds(self, capsys):
        exit_code = main(["live", "--broker", "alpaca_paper", "--confirm"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Safety confirmation received" in captured.out
        assert "Broker connection established" in captured.out

    def test_evaluate_execution(self, capsys):
        """On an unseeded system without prior sustained live-certification history,
        evaluate honestly evaluates baseline simulation, reports Gate 5 incomplete,
        and returns exit code 1 (PAPER TRADING ONLY).
        """
        exit_code = main(["evaluate"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "8-GATE LIVE READINESS EVALUATION" in captured.out
        assert "Gate 1" in captured.out
        assert "Gate 8" in captured.out
        assert "OVERALL STATUS: PAPER TRADING ONLY" in captured.out

    def test_evaluate_execution_with_certified_telemetry(self, tmp_path, capsys):
        """When supplied with certified database telemetry, evaluate satisfies all 8 gates,
        reports LIVE READY, and exits with code 0.
        """
        from modeloinv.storage.database import DatabaseManager

        db_file = tmp_path / "certified_telemetry.db"
        db = DatabaseManager(db_path=str(db_file))
        start_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=60)
        eq1 = np.linspace(100.0, 102.0, 10)
        eq2 = np.linspace(102.0, 119.0, 22)[1:]
        eq3 = np.linspace(119.0, 114.0, 10)[1:]
        eq4 = np.linspace(114.0, 122.40, 21)[1:]
        equities = np.concatenate([eq1, eq2, eq3, eq4])
        spy_prices = [450.0 * (1.0 + (i * (0.098 / 59.0))) for i in range(60)]

        for i in range(60):
            ts = start_date + datetime.timedelta(days=i)
            eq = float(equities[i])
            db.record_portfolio_snapshot({
                "timestamp": ts.isoformat(),
                "cash_eur": round(eq * 0.4, 2),
                "equity_eur": round(eq, 2),
                "realized_pnl_eur": round(eq - 100.0, 2),
                "unrealized_pnl_eur": 0.0,
                "cumulative_return_pct": round(eq - 100.0, 2),
                "drawdown_pct": 0.035,
                "benchmark_spy_price": round(spy_prices[i], 2),
                "benchmark_cumulative_return_pct": round((spy_prices[i] - 450.0) / 450.0 * 100.0, 2),
                "open_positions_count": 0,
                "positions": [],
                "execution_mode": "paper",
            })
        for i in range(64):
            is_win = (i % 8) not in (3, 7)
            db.record_trade({
                "trade_id": f"tr_cert_{i}",
                "order_id": None,
                "symbol": "SPY",
                "side": "SELL",
                "quantity": 0.05,
                "price": 450.0 + i,
                "fee_eur": 0.015,
                "slippage_eur": 0.005,
                "realized_pnl_eur": 1.25 if is_win else -0.85,
                "pnl_percent": 5.0 if is_win else -3.5,
                "holding_period_seconds": 3600.0,
                "strategy_name": "hybrid_ai",
                "model_version": "v1.1.0",
                "execution_mode": "paper",
                "timestamp": (start_date + datetime.timedelta(hours=i * 20)).isoformat(),
            })

        exit_code = main(["evaluate", "--db", str(db_file)])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "OVERALL STATUS: LIVE READY (CERTIFIED FOR REAL-CAPITAL TRADING)" in captured.out

    def test_retrain_execution(self, capsys):
        exit_code = main(["retrain", "--force"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "CONTINUOUS LEARNING RETRAINING" in captured.out
        assert "New Champion Promoted" in captured.out

    @patch("uvicorn.run")
    def test_dashboard_launch(self, mock_uvicorn_run, capsys):
        exit_code = main(["dashboard", "--host", "127.0.0.1", "--port", "8000"])
        assert exit_code == 0
        assert mock_uvicorn_run.called
        call_kwargs = mock_uvicorn_run.call_args[1]
        assert call_kwargs["host"] == "127.0.0.1"
        assert call_kwargs["port"] == 8000

    def test_backtest_invalid_capital_rejected(self, capsys):
        exit_code = main(["backtest", "--capital", "0.0"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "[ERROR] Starting capital must be strictly positive" in captured.out

    def test_backtest_invalid_months_rejected(self, capsys):
        exit_code = main(["backtest", "--months", "-1"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "[ERROR] Simulation duration must be positive" in captured.out

    def test_dashboard_invalid_port_rejected(self, capsys):
        exit_code = main(["dashboard", "--port", "999999"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "[ERROR] Invalid port 999999" in captured.out

    def test_paper_invalid_capital_rejected(self, capsys):
        exit_code = main(["paper", "--capital", "-50.0"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "[ERROR] Starting ledger capital must be strictly positive" in captured.out
