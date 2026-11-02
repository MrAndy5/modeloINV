"""Configuration manager for modeloINV.

Provides type-safe dataclasses, YAML defaults loading, environment variable overrides,
and live readiness validation.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, cast, get_type_hints

import yaml

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ConfigurationError(ValueError):
    """Raised when configuration validation fails."""
    pass


@dataclass
class SystemConfig:
    project_name: str = "modeloINV"
    execution_mode: str = "paper"  # "paper" | "live" | "backtest"
    log_level: str = "INFO"
    data_dir: str = "data"
    cache_dir: str = "data/cache"
    models_dir: str = "models"
    database_url: str = "sqlite:///data/modeloinv.db"


@dataclass
class AccountConfig:
    base_currency: str = "EUR"
    starting_capital_eur: float = 100.00
    enable_fractional: bool = True
    min_order_notional_eur: float = 1.00
    max_portfolio_positions: int = 5
    max_position_weight: float = 0.30


@dataclass
class RiskConfig:
    max_drawdown_limit: float = 0.15
    kill_switch_drawdown: float = 0.25
    daily_loss_limit_eur: float = 10.00
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.06


@dataclass
class ExecutionConfig:
    broker: str = "simulated"  # "simulated" | "alpaca" | "mock"
    slippage_model: str = "fixed_bps"
    slippage_bps: float = 5.0
    fee_model: str = "alpaca"
    fill_delay_ms: int = 0


@dataclass
class AlpacaConfig:
    api_key_id: str = ""
    api_secret_key: str = ""
    base_url: str = "https://paper-api.alpaca.markets"
    data_url: str = "https://data.alpaca.markets"
    virtual_capital_eur: float = 100.00


@dataclass
class CurrencyConfig:
    default_pair: str = "EURUSD"
    fallback_rate: float = 1.0800
    cache_ttl_seconds: int = 3600
    fx_fee_pct: float = 0.0010


@dataclass
class DataSymbolsConfig:
    equities: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "AAPL", "NVDA", "MSFT"])
    crypto: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    fx: List[str] = field(default_factory=lambda: ["EURUSD=X"])


@dataclass
class DataConfig:
    default_provider: str = "yahoo"
    secondary_provider: str = "binance"
    symbols: DataSymbolsConfig = field(default_factory=DataSymbolsConfig)
    benchmark_symbol: str = "SPY"
    cache_enabled: bool = True


@dataclass
class StrategyConfig:
    target_monthly_return: float = 0.20
    timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h"])
    quant_weight: float = 0.35
    ml_weight: float = 0.35
    rl_weight: float = 0.30
    regime_detection: bool = True


@dataclass
class LearningConfig:
    auto_retrain_trades: int = 25
    retrain_schedule_cron: str = "0 0 * * *"
    min_eval_sharpe: float = 1.0
    promotion_gate_enabled: bool = True


@dataclass
class DashboardConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class AppConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    account: AccountConfig = field(default_factory=AccountConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    alpaca: AlpacaConfig = field(default_factory=AlpacaConfig)
    currency: CurrencyConfig = field(default_factory=CurrencyConfig)
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)

    def validate(self) -> None:
        """Validate logical invariants across configuration parameters."""
        if self.account.starting_capital_eur <= 0:
            raise ConfigurationError(
                f"Starting capital must be strictly positive, got {self.account.starting_capital_eur}"
            )
        valid_modes = {"paper", "live", "backtest"}
        if self.system.execution_mode not in valid_modes:
            raise ConfigurationError(
                f"Invalid execution_mode '{self.system.execution_mode}'. Expected one of {valid_modes}."
            )
        if not (0.01 <= self.risk.max_drawdown_limit <= 0.50):
            raise ConfigurationError(
                f"max_drawdown_limit must be in range [0.01, 0.50], got {self.risk.max_drawdown_limit}"
            )
        if self.risk.kill_switch_drawdown <= self.risk.max_drawdown_limit:
            raise ConfigurationError(
                f"kill_switch_drawdown ({self.risk.kill_switch_drawdown}) must be greater than "
                f"max_drawdown_limit ({self.risk.max_drawdown_limit})"
            )
        if self.execution.slippage_bps < 0:
            raise ConfigurationError(
                f"slippage_bps cannot be negative, got {self.execution.slippage_bps}"
            )

        # Pre-flight credential check for Live mode
        if self.is_live():
            if self.execution.broker == "alpaca":
                if not self.alpaca.api_key_id or not self.alpaca.api_secret_key:
                    raise ConfigurationError(
                        "Live trading with Alpaca requires valid API keys. "
                        "Set MODELOINV_ALPACA_API_KEY_ID and MODELOINV_ALPACA_API_SECRET_KEY."
                    )

    def is_live(self) -> bool:
        return self.system.execution_mode == "live"

    def is_paper(self) -> bool:
        return self.system.execution_mode == "paper"

    def is_backtest(self) -> bool:
        return self.system.execution_mode == "backtest"

    def to_dict(self, mask_secrets: bool = True) -> Dict[str, Any]:
        """Convert to dictionary representation, optionally masking API credentials."""
        d = asdict(self)
        if mask_secrets:
            if d.get("alpaca", {}).get("api_secret_key"):
                d["alpaca"]["api_secret_key"] = "********"
        return d


# Global singleton cache
_CONFIG_INSTANCE: Optional[AppConfig] = None


def _coerce_value(val: str) -> Any:
    """Coerce string environment variable to appropriate primitive type."""
    val_lower = val.strip().lower()
    if val_lower in ("true", "yes", "1"):
        return True
    if val_lower in ("false", "no", "0"):
        return False
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    if "," in val:
        return [item.strip() for item in val.split(",") if item.strip()]
    return val


def _apply_env_overrides(cfg_dict: Dict[str, Any]) -> None:
    """Scan and apply MODELOINV_* environment variable overrides."""
    # 1. Direct convenience aliases
    alias_map = {
        "MODELOINV_EXECUTION_MODE": ("system", "execution_mode"),
        "MODELOINV_STARTING_CAPITAL_EUR": ("account", "starting_capital_eur"),
        "MODELOINV_STARTING_CAPITAL": ("account", "starting_capital_eur"),
        "MODELOINV_BASE_CURRENCY": ("account", "base_currency"),
        "MODELOINV_LOG_LEVEL": ("system", "log_level"),
        "MODELOINV_DATA_DIR": ("system", "data_dir"),
        "MODELOINV_DATABASE_URL": ("system", "database_url"),
        "MODELOINV_ALPACA_API_KEY_ID": ("alpaca", "api_key_id"),
        "MODELOINV_ALPACA_API_KEY": ("alpaca", "api_key_id"),
        "MODELOINV_ALPACA_API_SECRET_KEY": ("alpaca", "api_secret_key"),
        "MODELOINV_ALPACA_SECRET_KEY": ("alpaca", "api_secret_key"),
        "MODELOINV_ALPACA_BASE_URL": ("alpaca", "base_url"),
    }
    for env_var, (section, key) in alias_map.items():
        if env_var in os.environ:
            val = _coerce_value(os.environ[env_var])
            cfg_dict.setdefault(section, {})[key] = val

    # 2. General double-underscore overrides (e.g. MODELOINV_RISK__MAX_DRAWDOWN_LIMIT)
    prefix = "MODELOINV_"
    for env_key, env_val in os.environ.items():
        if env_key.startswith(prefix) and "__" in env_key:
            path_part = env_key[len(prefix):].lower()
            sections = path_part.split("__")
            target = cfg_dict
            for p in sections[:-1]:
                target = target.setdefault(p, {})
            target[sections[-1]] = _coerce_value(env_val)


def _dict_to_dataclass(cls: Type[T], data: Dict[str, Any]) -> T:
    """Recursively populate dataclass fields from dictionary."""
    if not isinstance(data, dict):
        return cast(T, data)
    type_hints = get_type_hints(cls)
    init_kwargs = {}
    for f in cls.__dataclass_fields__.values():
        name = f.name
        if name in data:
            val = data[name]
            resolved_type = type_hints.get(name, f.type)
            if hasattr(resolved_type, "__dataclass_fields__") and isinstance(val, dict):
                init_kwargs[name] = _dict_to_dataclass(resolved_type, val)
            else:
                init_kwargs[name] = val
    return cls(**init_kwargs)


def load_config(
    config_path: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> AppConfig:
    """
    Load configuration from YAML file, apply environment variable overrides,
    validate invariants, and return AppConfig instance.
    """
    cfg_dict: Dict[str, Any] = {}

    # Locate default YAML file
    target_path = None
    if config_path is not None:
        target_path = Path(config_path)
    else:
        possible_paths = [
            Path.cwd() / "config" / "default.yaml",
            Path(__file__).resolve().parent.parent.parent / "config" / "default.yaml",
        ]
        for p in possible_paths:
            if p.is_file():
                target_path = p
                break

    if target_path and target_path.is_file():
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    cfg_dict = loaded
        except Exception as e:
            logger.warning(f"Failed to parse config file at {target_path}: {e}")

    # Apply environment variable overrides
    _apply_env_overrides(cfg_dict)

    # Apply explicit runtime overrides
    if overrides:
        for k, v in overrides.items():
            if isinstance(v, dict) and k in cfg_dict and isinstance(cfg_dict[k], dict):
                cfg_dict[k].update(v)
            else:
                cfg_dict[k] = v

    # Build AppConfig object
    config = _dict_to_dataclass(AppConfig, cfg_dict)
    config.validate()
    return config


def get_config() -> AppConfig:
    """Retrieve global cached configuration or load default."""
    global _CONFIG_INSTANCE
    if _CONFIG_INSTANCE is None:
        _CONFIG_INSTANCE = load_config()
    return _CONFIG_INSTANCE


def set_config(config: AppConfig) -> None:
    """Set the global configuration singleton."""
    global _CONFIG_INSTANCE
    config.validate()
    _CONFIG_INSTANCE = config


def reset_config() -> None:
    """Reset global configuration singleton to uninitialized state."""
    global _CONFIG_INSTANCE
    _CONFIG_INSTANCE = None
