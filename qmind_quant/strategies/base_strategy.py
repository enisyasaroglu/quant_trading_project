# qmind_quant/strategies/base_strategy.py

from abc import ABC, abstractmethod
from qmind_quant.core.event_types import MarketEvent


class BaseStrategy(ABC):
    """
    BaseStrategy is an abstract base class providing an interface for all subsequent
    strategy handling objects.
    """

    def __init__(self, tickers: list[str], event_manager):
        self.tickers = tickers
        self.event_manager = event_manager

    @abstractmethod
    def on_market_event(self, event: MarketEvent):
        """
        Actions to be taken on receipt of a MarketEvent.
        This method must be implemented by all subclasses.
        """
        raise NotImplementedError("Should implement on_market_event()")
