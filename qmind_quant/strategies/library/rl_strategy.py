# qmind_quant/strategies/library/rl_strategy.py

import numpy as np
import pandas as pd
import pandas_ta as ta  # noqa: F401
from collections import deque
from stable_baselines3.common.base_class import BaseAlgorithm

from qmind_quant.strategies.base_strategy import BaseStrategy
from qmind_quant.core.event_types import MarketEvent, SignalEvent, FillEvent


class RLStrategy(BaseStrategy):
    """
    A strategy that uses a pre-trained Stable-Baselines3 reinforcement learning agent
    to generate trading signals.
    """

    def __init__(
        self,
        tickers: list[str],
        event_manager,
        agent: BaseAlgorithm,
        lookback_window=30,
    ):
        """
        Initializes the RL Strategy.

        Args:
            tickers (list[str]): The list of tickers this strategy will trade.
            event_manager: The central event manager instance.
            agent (BaseAlgorithm): The pre-trained Stable-Baselines3 agent.
            lookback_window (int): The number of historical bars to use for observations.
        """
        super().__init__(tickers, event_manager)
        self.agent = agent
        self.lookback_window = lookback_window

        # Per-ticker dataframes for efficient feature calculation
        self.data_frames = {ticker: pd.DataFrame() for ticker in self.tickers}
        # Per-ticker position tracking (shares)
        self.positions = dict.fromkeys(self.tickers, 0.0)
        # Simplified cash tracking for constructing the agent's observation state
        self.cash = 100000.0

    def _get_observation(self, ticker: str) -> np.ndarray | None:
        """Constructs the observation numpy array that the agent expects."""
        df = self.data_frames[ticker]
        if len(df) < self.lookback_window:
            return None

        frame = df.tail(self.lookback_window)
        # Ensure the features match exactly what the agent was trained on
        feature_columns = [
            "returns_1d",
            "returns_5d",
            "returns_21d",
            "volatility_21d",
            "rsi_14d",
        ]
        features = frame[feature_columns].values

        position_value = self.positions[ticker] * frame["close"].iloc[-1]
        portfolio_info = np.array(
            [[self.cash, position_value] for _ in range(self.lookback_window)]
        )

        # Combine into the final observation array
        obs = np.concatenate([portfolio_info, features], axis=1)
        return obs.astype(np.float32)

    def on_market_event(self, event: MarketEvent):
        """On a new bar, update data, calculate features, and get an agent action."""
        if event.ticker not in self.tickers:
            return

        # Append new bar data and recalculate features
        new_bar = pd.DataFrame([event.__dict__])
        self.data_frames[event.ticker] = pd.concat(
            [self.data_frames[event.ticker], new_bar], ignore_index=True
        )
        df = self.data_frames[event.ticker]

        df["returns_1d"] = df["close"].pct_change(1)
        df["returns_5d"] = df["close"].pct_change(5)
        df["returns_21d"] = df["close"].pct_change(21)
        df["volatility_21d"] = df["returns_1d"].rolling(window=21).std()
        df["rsi_14d"] = df["close"].ta.rsi(length=14)
        df.ffill(inplace=True)  # Fill NaNs from rolling calculations

        observation = self._get_observation(event.ticker)
        if observation is None or np.isnan(observation).any():
            return

        # Get a deterministic action from the trained agent
        action, _ = self.agent.predict(observation, deterministic=True)

        current_position = self.positions[event.ticker]
        # Action space: 0=Hold, 1=Buy, 2=Sell
        if action == 1 and current_position == 0:  # Buy signal and not invested
            signal = SignalEvent(event.timestamp, event.ticker, "LONG")
            self.event_manager.put(signal)
        elif action == 2 and current_position > 0:  # Sell signal and invested
            signal = SignalEvent(event.timestamp, event.ticker, "SHORT")
            self.event_manager.put(signal)

    def on_fill_event(self, event: FillEvent):
        """Update the strategy's internal state after a trade is executed."""
        if event.ticker not in self.tickers:
            return

        fill_cost = event.fill_price * event.quantity
        if event.direction == "BUY":
            self.positions[event.ticker] += event.quantity
            self.cash -= fill_cost
        elif event.direction == "SELL":
            self.positions[event.ticker] -= event.quantity
            self.cash += fill_cost
