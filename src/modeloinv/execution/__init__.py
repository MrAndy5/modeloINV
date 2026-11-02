"""Execution subsystem for modeloINV.

Provides broker abstraction, simulation engine, slippage and fee models,
and pre-trade risk guardrails.
"""

from modeloinv.execution.alpaca import AlpacaPaperBroker
from modeloinv.execution.base import (
    BaseBroker,
    BrokerConnectionError,
    BrokerError,
    BrokerFactory,
    InsufficientFundsError,
    InvalidOrderError,
    OrderNotFoundError,
    OrderRejectedError,
)
from modeloinv.execution.fees import (
    CompositeFeeModel,
    CryptoPercentageFeeModel,
    EuropeanRetailFeeModel,
    FeeModel,
    ZeroCommissionFeeModel,
)
from modeloinv.execution.mock import MockBroker
from modeloinv.execution.paper import SimulatedPaperBroker
from modeloinv.execution.risk import RiskManager, RiskViolationError
from modeloinv.execution.slippage import (
    BidAskSpreadModel,
    CompositeSlippageModel,
    FixedBpsSlippageModel,
    SlippageModel,
    VolumeImpactSlippageModel,
)

__all__ = [
    # Base and Factory
    "BaseBroker",
    "BrokerFactory",
    "BrokerError",
    "InsufficientFundsError",
    "OrderRejectedError",
    "BrokerConnectionError",
    "OrderNotFoundError",
    "InvalidOrderError",
    # Brokers
    "SimulatedPaperBroker",
    "AlpacaPaperBroker",
    "MockBroker",
    # Slippage
    "SlippageModel",
    "FixedBpsSlippageModel",
    "BidAskSpreadModel",
    "VolumeImpactSlippageModel",
    "CompositeSlippageModel",
    # Fees
    "FeeModel",
    "ZeroCommissionFeeModel",
    "CryptoPercentageFeeModel",
    "EuropeanRetailFeeModel",
    "CompositeFeeModel",
    # Risk
    "RiskManager",
    "RiskViolationError",
]
