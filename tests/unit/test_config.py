"""Unit tests for configuration manager (modeloinv.config)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from modeloinv.config import (
    AppConfig,
    ConfigurationError,
    get_config,
    load_config,
    reset_config,
    set_config,
)


@pytest.fixture(autouse=True)
def clean_config_env(monkeypatch):
    """Ensure clean environment and reset singleton for each test."""
    reset_config()
    for key in list(os.environ.keys()):
        if key.startswith("MODELOINV_"):
            monkeypatch.delenv(key, raising=False)
    yield
    reset_config()


def test_default_config_loading():
    """Verify default config loads successfully from default.yaml."""
    cfg = load_config()
    assert cfg.system.project_name == "modeloINV"
    assert cfg.system.execution_mode == "paper"
    assert cfg.account.starting_capital_eur == 100.00
    assert cfg.account.base_currency == "EUR"
    assert cfg.risk.max_drawdown_limit == 0.15
    assert cfg.risk.kill_switch_drawdown == 0.25
    assert cfg.execution.broker == "simulated"
    assert cfg.currency.fallback_rate == 1.0800
    assert "SPY" in cfg.data.symbols.equities
    assert cfg.is_paper() is True
    assert cfg.is_live() is False
    assert cfg.is_backtest() is False


def test_config_explicit_overrides():
    """Verify explicit override dictionary updates configuration."""
    overrides = {
        "account": {"starting_capital_eur": 250.0},
        "system": {"execution_mode": "backtest"},
        "risk": {"max_drawdown_limit": 0.20, "kill_switch_drawdown": 0.30},
    }
    cfg = load_config(overrides=overrides)
    assert cfg.account.starting_capital_eur == 250.0
    assert cfg.system.execution_mode == "backtest"
    assert cfg.is_backtest() is True
    assert cfg.risk.max_drawdown_limit == 0.20


def test_env_var_convenience_aliases(monkeypatch):
    """Verify MODELOINV_* convenience environment variables override config."""
    monkeypatch.setenv("MODELOINV_EXECUTION_MODE", "paper")
    monkeypatch.setenv("MODELOINV_STARTING_CAPITAL", "500.0")
    monkeypatch.setenv("MODELOINV_BASE_CURRENCY", "EUR")
    monkeypatch.setenv("MODELOINV_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("MODELOINV_DATA_DIR", "custom_data")

    cfg = load_config()
    assert cfg.system.execution_mode == "paper"
    assert cfg.account.starting_capital_eur == 500.0
    assert cfg.account.base_currency == "EUR"
    assert cfg.system.log_level == "DEBUG"
    assert cfg.system.data_dir == "custom_data"


def test_env_var_double_underscore_mapping(monkeypatch):
    """Verify hierarchical double-underscore environment variable overrides."""
    monkeypatch.setenv("MODELOINV_RISK__MAX_DRAWDOWN_LIMIT", "0.10")
    monkeypatch.setenv("MODELOINV_RISK__KILL_SWITCH_DRAWDOWN", "0.20")
    monkeypatch.setenv("MODELOINV_EXECUTION__SLIPPAGE_BPS", "12.5")
    monkeypatch.setenv("MODELOINV_ACCOUNT__ENABLE_FRACTIONAL", "false")

    cfg = load_config()
    assert cfg.risk.max_drawdown_limit == 0.10
    assert cfg.risk.kill_switch_drawdown == 0.20
    assert cfg.execution.slippage_bps == 12.5
    assert cfg.account.enable_fractional is False


def test_validation_rejects_negative_or_zero_starting_capital():
    """Verify validation rejects non-positive starting capital."""
    with pytest.raises(ConfigurationError, match="Starting capital must be strictly positive"):
        load_config(overrides={"account": {"starting_capital_eur": -10.0}})

    with pytest.raises(ConfigurationError, match="Starting capital must be strictly positive"):
        load_config(overrides={"account": {"starting_capital_eur": 0.0}})


def test_validation_rejects_invalid_execution_mode():
    """Verify validation rejects unapproved execution modes."""
    with pytest.raises(ConfigurationError, match="Invalid execution_mode 'invalid_mode'"):
        load_config(overrides={"system": {"execution_mode": "invalid_mode"}})


def test_validation_rejects_invalid_drawdowns():
    """Verify validation enforces drawdown range and ordering constraints."""
    with pytest.raises(ConfigurationError, match="max_drawdown_limit must be in range"):
        load_config(overrides={"risk": {"max_drawdown_limit": 0.005}})

    with pytest.raises(ConfigurationError, match="max_drawdown_limit must be in range"):
        load_config(overrides={"risk": {"max_drawdown_limit": 0.60}})

    with pytest.raises(ConfigurationError, match="kill_switch_drawdown .* must be greater"):
        load_config(overrides={"risk": {"max_drawdown_limit": 0.20, "kill_switch_drawdown": 0.15}})


def test_validation_rejects_negative_slippage():
    """Verify validation rejects negative slippage basis points."""
    with pytest.raises(ConfigurationError, match="slippage_bps cannot be negative"):
        load_config(overrides={"execution": {"slippage_bps": -2.0}})


def test_validation_live_mode_requires_alpaca_credentials():
    """Verify live trading on Alpaca requires valid API credentials."""
    with pytest.raises(ConfigurationError, match="Live trading with Alpaca requires valid API keys"):
        load_config(overrides={
            "system": {"execution_mode": "live"},
            "execution": {"broker": "alpaca"},
            "alpaca": {"api_key_id": "", "api_secret_key": ""},
        })

    # When credentials are provided, validation passes
    cfg = load_config(overrides={
        "system": {"execution_mode": "live"},
        "execution": {"broker": "alpaca"},
        "alpaca": {"api_key_id": "test_key", "api_secret_key": "test_secret"},
    })
    assert cfg.is_live() is True


def test_to_dict_masks_secrets():
    """Verify to_dict masks Alpaca secret key by default."""
    cfg = load_config(overrides={
        "alpaca": {"api_key_id": "AK_TEST", "api_secret_key": "SUPER_SECRET"}
    })
    masked = cfg.to_dict(mask_secrets=True)
    assert masked["alpaca"]["api_secret_key"] == "********"
    assert masked["alpaca"]["api_key_id"] == "AK_TEST"

    unmasked = cfg.to_dict(mask_secrets=False)
    assert unmasked["alpaca"]["api_secret_key"] == "SUPER_SECRET"


def test_global_singleton_getter_setter():
    """Verify global config getter and setter lifecycle."""
    cfg1 = get_config()
    assert cfg1 is not None

    custom = load_config(overrides={"account": {"starting_capital_eur": 777.0}})
    set_config(custom)
    assert get_config().account.starting_capital_eur == 777.0

    reset_config()
    assert get_config().account.starting_capital_eur == 100.00
