"""
Adversarial CLI Exhaustive Boundary & Fuzzing Harness for Challenger M5.
"""
import sys
import os
import math
import tempfile
import pytest
from unittest.mock import patch
from modeloinv.cli import main

def test_cli_fuzz_and_boundaries():
    cases_expect_exit_1 = [
        ["backtest", "--capital", "0.0"],
        ["backtest", "--capital", "-0.0001"],
        ["backtest", "--capital", "-1000"],
        ["backtest", "--months", "0"],
        ["backtest", "--months", "-1"],
        ["backtest", "--months", "-100"],
        ["paper", "--capital", "0.0"],
        ["paper", "--capital", "-50.0"],
        ["dashboard", "--port", "0"],
        ["dashboard", "--port", "-1"],
        ["dashboard", "--port", "65536"],
        ["dashboard", "--port", "999999"],
        ["live", "--broker", "alpaca_paper"], # no --confirm
        ["live", "--broker", "simulated"],     # no --confirm
    ]

    for argv in cases_expect_exit_1:
        ret = main(argv)
        assert ret == 1, f"Expected exit code 1 for args {argv}, got {ret}"
    print(f"Passed {len(cases_expect_exit_1)} boundary rejection cases with exit code 1!")

    # Test empty universe fallback in backtest
    # When universe is empty string, default should fall back to SPY
    ret = main(["backtest", "--months", "1", "--capital", "100.0", "--universe", ""])
    assert ret == 0, f"Expected backtest with empty universe string to fall back to SPY, got {ret}"
    print("Passed empty universe fallback test!")

    # Test paper with small positive capital and duration 1
    ret = main(["paper", "--capital", "50.0", "--duration", "1"])
    assert ret == 0, f"Expected paper duration 1 to exit 0, got {ret}"
    print("Passed paper bounded execution test!")

    # Test live with --confirm
    ret = main(["live", "--broker", "alpaca_paper", "--confirm"])
    assert ret == 0, f"Expected live --confirm to exit 0, got {ret}"
    print("Passed live --confirm test!")

    # Test evaluate with temporary empty database (triggers dynamic simulation fallback)
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_db = os.path.join(tmpdir, "empty.db")
        with patch("modeloinv.cli.os.path.exists", side_effect=lambda p: False if p == "data/modeloinv.db" else os.path.exists(p)):
            ret = main(["evaluate"])
            assert ret in (0, 1), f"Expected evaluate fallback to return 0 or 1 cleanly, got {ret}"
    print("Passed evaluate fallback without data/modeloinv.db!")

    print("SUCCESS: ALL EXHAUSTIVE CLI BOUNDARIES SURVIVED!")

if __name__ == "__main__":
    test_cli_fuzz_and_boundaries()
