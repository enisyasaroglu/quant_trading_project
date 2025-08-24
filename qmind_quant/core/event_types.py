# qmind_quant/core/event_types.py

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Event:
    """Base class for all events."""

    # This field is intentionally left without a default in the base class
    event_type: str


@dataclass
class MarketEvent(Event):
    """
    Handles the event of receiving new market data (a new bar).
    """

    timestamp: datetime
    ticker: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    event_type: str = field(default="MARKET", init=False)


@dataclass
class SignalEvent(Event):
    """
    Handles the event of a strategy generating a signal.
    """

    timestamp: datetime
    ticker: str
    signal_type: str  # 'LONG', 'SHORT', 'EXIT'
    strength: float = 1.0  # Represents the confidence in the signal
    event_type: str = field(default="SIGNAL", init=False)


@dataclass
class OrderEvent(Event):
    """
    Handles the event of sending an Order to an execution system.
    """

    timestamp: datetime
    ticker: str
    order_type: str  # 'MKT' (Market), 'LMT' (Limit)
    direction: str  # 'BUY' or 'SELL'
    quantity: int
    event_type: str = field(default="ORDER", init=False)


@dataclass
class FillEvent(Event):
    """
    Represents a filled order, as returned from a broker.
    """

    timestamp: datetime
    ticker: str
    direction: str  # 'BUY' or 'SELL'
    quantity: int
    fill_price: float
    commission: float = 0.0
    event_type: str = field(default="FILL", init=False)
