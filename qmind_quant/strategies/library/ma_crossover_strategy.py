# qmind_quant/strategies/library/ma_crossover_strategy.py

import pandas as pd
from collections import deque
from qmind_quant.strategies.base_strategy import BaseStrategy
from qmind_quant.core.event_types import MarketEvent, SignalEvent


class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    A simple moving average (MA) crossover strategy.
    """

    def __init__(
        self, tickers: list[str], event_manager, short_window=10, long_window=30
    ):
        super().__init__(tickers, event_manager)
        self.short_window = short_window
        self.long_window = long_window

        # Dictionary to store recent close prices for each ticker
        self.prices = {
            ticker: deque(maxlen=self.long_window) for ticker in self.tickers
        }
        # Dictionary to store the last calculated moving averages for each ticker
        self.invested = {ticker: "NONE" for ticker in self.tickers}

    def on_market_event(self, event: MarketEvent):
        """
        On a new market event, update the price deque and check for a crossover.
        """
        if event.ticker not in self.tickers:
            return

        # Append the new closing price
        self.prices[event.ticker].append(event.close)

        # Wait until we have enough data to calculate the long window MA
        if len(self.prices[event.ticker]) < self.long_window:
            return

        # Calculate the short and long SMAs
        short_sma = (
            pd.Series(self.prices[event.ticker])
            .rolling(self.short_window)
            .mean()
            .iloc[-1]
        )
        long_sma = (
            pd.Series(self.prices[event.ticker])
            .rolling(self.long_window)
            .mean()
            .iloc[-1]
        )

        # Check for crossover conditions
        ticker = event.ticker
        if short_sma > long_sma and self.invested[ticker] != "LONG":
            signal = SignalEvent(
                timestamp=event.timestamp, ticker=ticker, signal_type="LONG"
            )
            self.event_manager.put(signal)
            self.invested[ticker] = "LONG"
        elif short_sma < long_sma and self.invested[ticker] != "SHORT":
            # Signal a short or exit long position
            signal = SignalEvent(
                timestamp=event.timestamp, ticker=ticker, signal_type="SHORT"
            )
            self.event_manager.put(signal)
            self.invested[ticker] = "SHORT"
