"""modeloINV - Algorithmic Trading System.

Targets >=20% monthly return on €100 starting capital.
Transitions from paper-trading simulation to live broker execution via configuration only.
"""

__version__ = "0.1.0"

from modeloinv.config import AppConfig, get_config, load_config

__all__ = ["__version__", "AppConfig", "get_config", "load_config"]
