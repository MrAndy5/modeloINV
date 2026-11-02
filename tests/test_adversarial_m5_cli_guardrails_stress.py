"""Adversarial and Stress Test Suite for Milestone 5: CLI, Guardrails & Integration.

Mandate Coverage:
1. Execute CLI subcommands with extreme and invalid arguments (--months -1, --capital 0.0, --port 999999, missing required flags).
2. Adversarial Live Guardrail Verification: Attempt to execute live mode without --confirm and verify non-zero exit code and zero order submissions.
3. Test end-to-end execution of backtest, evaluate, and retrain CLI commands, capturing stdout and confirming output format, exit codes, and no uncaught tracebacks.
4. Verify documentation command examples in README.md: run exact CLI commands and confirm clean execution.
5. Verify static asset integrity: check that index.html references only local CSS/JS files that exist on disk, with zero broken HTTP 404 links.
"""

from __future__ import annotations

import argparse
import datetime
import io
import os
from pathlib import Path
import re
import sys
from typing import List
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import numpy as np
import pytest

from modeloinv.cli import _run_backtest_simulation, build_parser, main
from modeloinv.dashboard.app import create_app


# =============================================================================
# 1. Extreme & Invalid Arguments Stress (Mandate 1)
# =============================================================================

class TestCLIInvalidAndExtremeArguments:
    """Stress tests CLI subcommands with boundary, extreme, and malformed arguments."""

    def test_backtest_capital_zero_division_bug(self, capsys):
        """Verifies CLI backtest with --capital 0.0 returns exit code 1 with clean error message."""
        exit_code = main(["backtest", "--capital", "0.0"])
        captured = capsys.readouterr()
        assert exit_code == 1
        assert "[ERROR] Starting capital must be strictly positive" in captured.out

    def test_backtest_negative_capital_unvalidated(self, capsys):
        """Verifies CLI backtest with --capital -100.0 returns exit code 1 with validation error."""
        exit_code = main(["backtest", "--capital", "-100.0"])
        captured = capsys.readouterr()
        assert exit_code == 1
        assert "[ERROR] Starting capital must be strictly positive" in captured.out

    def test_backtest_negative_months_silent_fallback(self, capsys):
        """Verifies CLI backtest with --months -1 returns exit code 1 with validation error."""
        exit_code = main(["backtest", "--months", "-1"])
        captured = capsys.readouterr()
        assert exit_code == 1
        assert "[ERROR] Simulation duration must be positive" in captured.out

    def test_dashboard_port_overflow_bug(self, capsys):
        """Verifies CLI dashboard with --port 999999 returns exit code 1 without socket crash."""
        exit_code = main(["dashboard", "--host", "127.0.0.1", "--port", "999999"])
        captured = capsys.readouterr()
        assert exit_code == 1
        assert "[ERROR] Invalid port 999999" in captured.out

    def test_dashboard_negative_port_overflow_bug(self, capsys):
        """Verifies CLI dashboard with --port -1 returns exit code 1 without socket crash."""
        exit_code = main(["dashboard", "--host", "127.0.0.1", "--port", "-1"])
        captured = capsys.readouterr()
        assert exit_code == 1
        assert "[ERROR] Invalid port -1" in captured.out

    def test_type_error_non_numeric_months(self):
        """Verifies CLI rejects non-integer --months with standard exit code 2 or 1."""
        with pytest.raises(SystemExit) as exc_info:
            main(["backtest", "--months", "not_a_number"])
        assert exc_info.value.code in [1, 2]

    def test_type_error_non_numeric_capital(self):
        """Verifies CLI rejects non-float --capital with standard exit code."""
        with pytest.raises(SystemExit) as exc_info:
            main(["backtest", "--capital", "abc_eur"])
        assert exc_info.value.code in [1, 2]

    def test_type_error_non_numeric_port(self):
        """Verifies CLI rejects non-integer --port with standard exit code."""
        with pytest.raises(SystemExit) as exc_info:
            main(["dashboard", "--port", "eight_thousand"])
        assert exc_info.value.code in [1, 2]

    def test_unknown_subcommand_rejection(self):
        """Verifies CLI rejects unknown subcommand with standard usage exit."""
        with pytest.raises(SystemExit) as exc_info:
            main(["unsupported_action"])
        assert exc_info.value.code in [1, 2]

    def test_no_arguments_shows_help_and_exits_cleanly(self, capsys):
        """Verifies running CLI with no arguments exits cleanly with 0 and shows help."""
        exit_code = main([])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "usage: modeloinv" in captured.out or "usage: modeloinv" in captured.err


# =============================================================================
# 2. Adversarial Live Guardrail Verification (Mandate 2)
# =============================================================================

class TestLiveGuardrailVerification:
    """Stress tests live trading subcommand safety guardrails."""

    def test_live_without_confirm_strictly_blocked(self, capsys):
        """Attempt to execute live mode without --confirm: must exit non-zero and block orders."""
        exit_code = main(["live", "--broker", "alpaca_paper"])
        captured = capsys.readouterr()

        assert exit_code == 1, "Live command without --confirm must return non-zero exit code"
        assert "[GUARDRAIL ERROR]" in captured.out
        assert "Live execution blocked" in captured.out
        assert "--confirm" in captured.out
        assert "Broker connection established" not in captured.out

    def test_live_default_broker_without_confirm_blocked(self, capsys):
        """Attempt to execute live mode with default broker without --confirm."""
        exit_code = main(["live"])
        captured = capsys.readouterr()

        assert exit_code == 1
        assert "[GUARDRAIL ERROR]" in captured.out
        assert "--confirm" in captured.out

    def test_live_real_broker_without_confirm_blocked(self, capsys):
        """Attempt to execute live mode with real alpaca broker without --confirm."""
        exit_code = main(["live", "--broker", "alpaca"])
        captured = capsys.readouterr()

        assert exit_code == 1
        assert "[GUARDRAIL ERROR]" in captured.out
        assert "Broker connection established" not in captured.out

    def test_live_with_confirm_succeeds_and_initializes_guardrails(self, capsys):
        """Verifies that when --confirm is supplied, safety checks and risk guardrails are displayed."""
        exit_code = main(["live", "--broker", "alpaca_paper", "--confirm"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "[LIVE] Safety confirmation received." in captured.out
        assert "Pre-trade risk guardrails initialized:" in captured.out
        assert "Daily Drawdown Stop: 15.0%" in captured.out
        assert "Emergency Kill Switch: 25.0%" in captured.out
        assert "Broker connection established." in captured.out


# =============================================================================
# 3. End-to-End Execution of CLI Subcommands (Mandate 3)
# =============================================================================

class TestEndToEndSubcommands:
    """End-to-end execution of backtest, evaluate, and retrain CLI commands."""

    def test_e2e_backtest_execution_and_format(self, capsys):
        """Executes backtest, checks stdout format, metrics, and no uncaught tracebacks."""
        exit_code = main(["backtest", "--months", "12", "--capital", "100.0"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "modeloINV BACKTEST SIMULATION" in captured.out
        assert "Starting Capital" in captured.out
        assert "Final Account Equity" in captured.out
        assert "Total Net Return" in captured.out
        assert "S&P 500 Buy-and-Hold" in captured.out
        assert "Annualized Sharpe" in captured.out
        assert "Max Portfolio Drawdown" in captured.out
        assert "Win Rate" in captured.out
        assert "Best 30-Day Window Ret" in captured.out
        assert "Total Execution Fees" in captured.out
        assert "[SUCCESS] Strategy BEATS S&P 500 benchmark with Sharpe > 1.0." in captured.out

    def test_e2e_evaluate_execution_and_gates(self, capsys):
        """Executes evaluate subcommand on unseeded system, confirms 8 gates output, format, and honest exit code."""
        exit_code = main(["evaluate"])
        captured = capsys.readouterr()

        # On an unseeded system, baseline simulation yields max 30d return < 15%, so Gate 5 fails
        assert exit_code == 1
        assert "8-GATE LIVE READINESS EVALUATION" in captured.out
        for gate_num in range(1, 9):
            assert f"Gate {gate_num}" in captured.out
        assert "OVERALL STATUS: PAPER TRADING ONLY (Readiness criteria incomplete)" in captured.out

    def test_e2e_evaluate_with_certified_database(self, tmp_path, capsys):
        """Executes evaluate against an isolated temporary database with certified telemetry, verifying LIVE READY."""
        from modeloinv.storage.database import DatabaseManager

        db_file = tmp_path / "certified_e2e.db"
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
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "OVERALL STATUS: LIVE READY (CERTIFIED FOR REAL-CAPITAL TRADING)" in captured.out

    def test_e2e_retrain_execution_and_version_bump(self, capsys):
        """Executes retrain subcommand, confirms champion promotion and model registry bump."""
        exit_code = main(["retrain", "--force"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "modeloINV CONTINUOUS LEARNING RETRAINING" in captured.out
        assert "Force Bypassing Cooldown: True" in captured.out
        assert "New Champion Promoted" in captured.out
        assert "Validation Accuracy" in captured.out
        assert "Validation ROC-AUC" in captured.out
        assert "Artifacts Saved To" in captured.out

    def test_e2e_paper_trading_execution(self, capsys):
        """Executes bounded paper trading subcommand, confirms output and virtual capital tracking."""
        exit_code = main(["paper", "--capital", "100.0", "--duration", "2"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "REAL-TIME PAPER TRADING ENGINE" in captured.out
        assert "Initial Ledger: €100.00 EUR" in captured.out or "100.00 EUR" in captured.out
        assert "[TRADE] Filled" in captured.out
        assert "[PAPER] Status: ACTIVE" in captured.out


# =============================================================================
# 4. Documentation Examples in README.md (Mandate 4)
# =============================================================================

class TestReadmeDocumentationCommands:
    """Runs the exact CLI commands shown in README.md and confirms clean execution."""

    def test_readme_quickstart_step1_backtest(self, capsys):
        """Exact command from README line 151: python -m modeloinv.cli backtest --months 12 --capital 100.0"""
        exit_code = main(["backtest", "--months", "12", "--capital", "100.0"])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "modeloINV BACKTEST SIMULATION" in captured.out

    def test_readme_quickstart_step2_evaluate(self, capsys):
        """Exact command from README line 157: python -m modeloinv.cli evaluate"""
        exit_code = main(["evaluate"])
        captured = capsys.readouterr()
        # On a fresh installation, evaluating live readiness reports paper trading status with exit code 1
        assert exit_code == 1
        assert "8-GATE LIVE READINESS EVALUATION" in captured.out

    def test_readme_section6_paper_defaults(self, capsys):
        """Command from README line 193: python -m modeloinv.cli paper --capital 100.0 --duration 1"""
        exit_code = main(["paper", "--capital", "100.0", "--duration", "1"])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "REAL-TIME PAPER TRADING ENGINE" in captured.out

    def test_readme_section6_retrain_force(self, capsys):
        """Command from README line 223: python -m modeloinv.cli retrain --force"""
        exit_code = main(["retrain", "--force"])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "CONTINUOUS LEARNING RETRAINING" in captured.out

    def test_readme_section6_live_with_confirm(self, capsys):
        """Command from README line 203: python -m modeloinv.cli live --broker alpaca_paper --confirm"""
        exit_code = main(["live", "--broker", "alpaca_paper", "--confirm"])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "Broker connection established" in captured.out

    def test_readme_section6_live_without_confirm_fails_safely(self, capsys):
        """Command from README line 203 without safety flag: must fail safely."""
        exit_code = main(["live", "--broker", "alpaca_paper"])
        captured = capsys.readouterr()
        assert exit_code == 1
        assert "[GUARDRAIL ERROR]" in captured.out

    @patch("uvicorn.run")
    def test_readme_quickstart_step3_dashboard(self, mock_uvicorn_run, capsys):
        """Exact command from README line 163: python -m modeloinv.cli dashboard --host 127.0.0.1 --port 8000"""
        exit_code = main(["dashboard", "--host", "127.0.0.1", "--port", "8000"])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert mock_uvicorn_run.called
        assert mock_uvicorn_run.call_args[1]["host"] == "127.0.0.1"
        assert mock_uvicorn_run.call_args[1]["port"] == 8000


# =============================================================================
# 5. Static Asset Integrity & Offline Verification (Mandate 5)
# =============================================================================

class TestStaticAssetIntegrity:
    """Verifies static asset integrity, local filesystem existence, and zero 404 links."""

    @pytest.fixture
    def static_dir(self) -> Path:
        base = Path(__file__).parent.parent / "src" / "modeloinv" / "dashboard" / "static"
        return base.resolve()

    def test_index_html_exists(self, static_dir: Path):
        """Verifies index.html exists in dashboard/static directory."""
        index_file = static_dir / "index.html"
        assert index_file.exists(), f"index.html not found at {index_file}"
        assert index_file.stat().st_size > 0

    def test_all_referenced_assets_exist_on_disk(self, static_dir: Path):
        """Parses index.html and asserts every link/script exists locally on disk."""
        index_file = static_dir / "index.html"
        content = index_file.read_text(encoding="utf-8")

        # Find all src and href attributes
        pattern = re.compile(r'(?:src|href)=["\']([^"\']+)["\']')
        links = pattern.findall(content)

        assert len(links) >= 3, f"Expected at least 3 static asset links, found {links}"

        for link in links:
            # Check zero external remote CDNs or HTTP(S) links in index.html
            assert not link.startswith("http://"), f"External remote link found: {link}"
            assert not link.startswith("https://"), f"External remote link found: {link}"
            assert not link.startswith("//"), f"Protocol-relative remote link found: {link}"

            # Strip leading /static/ or /
            rel_path = link.lstrip("/")
            if rel_path.startswith("static/"):
                rel_path = rel_path[len("static/"):]

            disk_path = static_dir / rel_path
            assert disk_path.exists(), f"Referenced asset {link} does not exist on disk at {disk_path}"
            assert disk_path.stat().st_size > 0, f"Referenced asset {disk_path} is empty (0 bytes)"

    def test_dashboard_serves_all_assets_with_http_200_and_no_404(self):
        """Tests FastAPI app directly via TestClient: GET / and all static assets return 200."""
        app = create_app()
        client = TestClient(app)

        # 1. Root index.html
        res_root = client.get("/")
        assert res_root.status_code == 200
        html = res_root.text

        # 2. Extract asset links from served HTML
        links = re.findall(r'(?:src|href)=["\'](/static/[^"\']+)["\']', html)
        assert len(links) >= 3, f"Found links in served HTML: {links}"

        # 3. Assert each asset returns HTTP 200 (ZERO 404s)
        for link in links:
            res_asset = client.get(link)
            assert res_asset.status_code == 200, f"Asset {link} failed with status {res_asset.status_code}"
            assert len(res_asset.content) > 0, f"Asset {link} returned empty body"

        # Explicitly verify required CSS and JS files
        css_res = client.get("/static/css/style.css")
        assert css_res.status_code == 200
        assert "glass-panel" in css_res.text

        chart_res = client.get("/static/js/chart.min.js")
        assert chart_res.status_code == 200
        assert "Chart" in chart_res.text

        app_res = client.get("/static/js/app.js")
        assert app_res.status_code == 200
        assert "initWebSocket" in app_res.text

    def test_css_offline_integrity_no_external_font_imports(self, static_dir: Path):
        """Verifies style.css contains zero @import url(...) to external CDNs or Google Fonts."""
        css_file = static_dir / "css" / "style.css"
        assert css_file.exists()
        css_text = css_file.read_text(encoding="utf-8")

        # Check for remote imports
        remote_imports = re.findall(r'@import\s+url\(["\']?(https?://[^"\'\)]+)["\']?\)', css_text, re.IGNORECASE)
        assert len(remote_imports) == 0, f"Found external remote CSS @import: {remote_imports}"

        # Check for remote url() references in css
        remote_urls = re.findall(r'url\(["\']?(https?://[^"\'\)]+)["\']?\)', css_text, re.IGNORECASE)
        assert len(remote_urls) == 0, f"Found external remote url() references in CSS: {remote_urls}"
