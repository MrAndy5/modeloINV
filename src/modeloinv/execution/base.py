"""Abstract Broker Contract and Factory for modeloINV.

Defines the universal BaseBroker interface adhering to PROJECT.md and DISPATCH.md,
ensuring seamless, configuration-only transitions across local simulation,
Alpaca paper trading, and live execution interfaces.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from modeloinv.core.models import (
    AccountInfo,
    OrderRequest,
    OrderResult,
    OrderStatus,
    Position,
)

logger = logging.getLogger(__name__)


class BrokerError(Exception):
    """Base exception for all broker operations."""
    pass


class InsufficientFundsError(BrokerError):
    """Raised when an order exceeds available cash or buying power."""
    pass


class OrderRejectedError(BrokerError):
    """Raised when an order is rejected by broker or exchange."""
    pass


class BrokerConnectionError(BrokerError):
    """Raised when broker connectivity fails or credentials are invalid."""
    pass


class OrderNotFoundError(BrokerError):
    """Raised when an order lookup fails."""
    pass


class InvalidOrderError(BrokerError):
    """Raised when order parameters violate exchange or broker rules."""
    pass


class BaseBroker(ABC):
    """Universal abstract base class for all trading brokers."""

    def connect(self) -> bool:
        """Establish connection or validate credentials. Default returns True."""
        return True

    def disconnect(self) -> None:
        """Gracefully terminate connection and clean up resources."""
        pass

    def is_connected(self) -> bool:
        """Check if broker connection is active."""
        return True

    @abstractmethod
    def get_account(self) -> AccountInfo:
        """Retrieve current cash, equity, and purchasing power in EUR."""
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Retrieve all currently open positions."""
        pass

    def get_position(self, symbol: str) -> Optional[Position]:
        """Retrieve open position for a specific symbol, or None if not held."""
        for pos in self.get_positions():
            if pos.symbol == symbol:
                return pos
        return None

    @abstractmethod
    def submit_order(
        self,
        request: OrderRequest,
        current_market_price: Optional[float] = None,
    ) -> OrderResult:
        """Submit an order request for execution.

        Parameters:
            request: Canonical OrderRequest specification.
            current_market_price: Optional explicit market price override for testing
                                  or simulation ticks.
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order by ID. Returns True if successfully canceled."""
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderResult:
        """Retrieve the current execution status of an order."""
        pass

    def get_open_orders(self) -> List[OrderResult]:
        """List all active/pending orders. Default returns empty list."""
        return []

    def get_latest_price(self, symbol: str) -> float:
        """Retrieve latest observed market price for symbol."""
        pos = self.get_position(symbol)
        if pos is not None and pos.current_price > 0:
            return pos.current_price
        return 0.0

    def is_market_open(self, symbol: str) -> bool:
        """Check if market is currently open for symbol. Default returns True."""
        return True


class BrokerFactory:
    """Factory to instantiate broker backends dynamically based on configuration."""

    @staticmethod
    def create_broker(config: Any) -> BaseBroker:
        """Create a BaseBroker instance from AppConfig or configuration dict."""
        broker_type = "simulated"
        is_live = False

        if hasattr(config, "execution") and hasattr(config.execution, "broker"):
            broker_type = str(config.execution.broker).lower()
            if hasattr(config, "is_live"):
                is_live = config.is_live()
        elif isinstance(config, dict):
            # Check broker.backend or execution.broker
            broker_sec = config.get("broker", {})
            exec_sec = config.get("execution", {})
            system_sec = config.get("system", {})
            broker_type = (
                broker_sec.get("backend")
                or exec_sec.get("broker")
                or config.get("backend")
                or "simulated"
            ).lower()
            is_live = (
                system_sec.get("execution_mode") == "live"
                or config.get("execution_mode") == "live"
            )
        elif isinstance(config, str):
            broker_type = config.lower()

        logger.info(f"Instantiating broker '{broker_type}' (is_live={is_live})")

        if broker_type in ("simulated", "paper", "simulatedpaperbroker"):
            from modeloinv.execution.paper import SimulatedPaperBroker
            return SimulatedPaperBroker(config)
        elif broker_type in ("alpaca", "alpaca_paper", "alpacapaperbroker"):
            from modeloinv.execution.alpaca import AlpacaPaperBroker
            return AlpacaPaperBroker(config, live=False)
        elif broker_type in ("alpaca_live", "alpaca_real"):
            from modeloinv.execution.alpaca import AlpacaPaperBroker
            return AlpacaPaperBroker(config, live=True)
        elif broker_type in ("mock", "mockbroker"):
            from modeloinv.execution.mock import MockBroker
            return MockBroker(config)
        else:
            raise ValueError(f"Unsupported broker backend: '{broker_type}'")

    @classmethod
    def create(cls, config: Any) -> BaseBroker:
        """Alias for create_broker to adhere to alternative factory conventions."""
        return cls.create_broker(config)
