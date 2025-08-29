# qmind_quant/strategies/library/rl_strategy.py

import numpy as np
import pandas as pd
from collections import deque
from stable_baselines3.common.base_class import BaseAlgorithm

from qmind_quant.strategies.base_strategy import BaseStrategy
from qmind_quant.core.event_types import MarketEvent, SignalEvent, FillEvent
from qmind_quant.analytics.technical_indicators import (
    calculate_ema,
    calculate_macd,
    calculate_adx,
    calculate_rsi,
    calculate_stochastic_oscillator,
    calculate_bollinger_bands,
    calculate_atr,
    calculate_obv,
    calculate_vwap,
)


class RLStrategy(BaseStrategy):
    """
    A refactored, robust RL strategy with a single source of truth for state.
    """

    # --- THIS IS THE FIX ---
    # The lookback window must match the training environment's default (30).
    def __init__(
        self,
        tickers: list[str],
        event_manager,
        agent: BaseAlgorithm,
        lookback_window=30,
    ):
        super().__init__(tickers, event_manager)
        self.agent = agent
        self.lookback_window = lookback_window

        self.data_frames = {ticker: pd.DataFrame() for ticker in self.tickers}
        self.positions = dict.fromkeys(self.tickers, 0.0)
        self.cash = 100000.0
        self.trading_halted = False

    def _get_observation(self, ticker: str) -> np.ndarray | None:
        df = self.data_frames[ticker]
        if len(df) < self.lookback_window:
            return None
        frame = df.tail(self.lookback_window)
        feature_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "ema_12",
            "ema_26",
            "macd",
            "adx_14",
            "rsi_14",
            "stoch_k_14",
            "bb_width",
            "atr_14",
            "obv",
            "vwap",
            "target",
        ]
        if not all(col in frame.columns for col in feature_columns):
            return None
        features = frame[feature_columns].values
        position_value = self.positions[ticker] * frame["close"].iloc[-1]
        portfolio_info = np.array(
            [[self.cash, position_value] for _ in range(self.lookback_window)]
        )
        obs = np.concatenate([portfolio_info, features], axis=1)
        return obs.astype(np.float32)

    def on_market_event(self, event: MarketEvent):
        if self.trading_halted:
            return
        if event.ticker not in self.tickers:
            return

        new_bar = pd.DataFrame([event.__dict__])
        self.data_frames[event.ticker] = pd.concat(
            [self.data_frames[event.ticker], new_bar], ignore_index=True
        )
        df = self.data_frames[event.ticker]
        if len(df) < self.lookback_window:
            return

        df["ema_12"] = calculate_ema(df["close"], window=12)
        df["ema_26"] = calculate_ema(df["close"], window=26)
        macd_df = calculate_macd(df["close"])
        df["macd"] = macd_df["macd"]
        df["adx_14"] = calculate_adx(df["high"], df["low"], df["close"], window=14)
        df["rsi_14"] = calculate_rsi(df["close"], window=14)
        df["stoch_k_14"] = calculate_stochastic_oscillator(
            df["high"], df["low"], df["close"], window=14
        )
        bbands_df = calculate_bollinger_bands(df["close"], window=20)
        df["bb_width"] = (bbands_df["bb_upper"] - bbands_df["bb_lower"]) / bbands_df[
            "bb_middle"
        ]
        df["atr_14"] = calculate_atr(df["high"], df["low"], df["close"], window=14)
        df["obv"] = calculate_obv(df["close"], df["volume"])
        df["vwap"] = calculate_vwap(df["close"], df["volume"])
        if "target" not in df.columns:
            df["target"] = 0
        df.ffill(inplace=True)

        observation = self._get_observation(event.ticker)
        if observation is None or np.isnan(observation).any():
            return

        action, _ = self.agent.predict(observation, deterministic=True)

        current_position = self.positions[event.ticker]
        if action == 1 and current_position == 0:
            signal = SignalEvent(event.timestamp, event.ticker, "LONG")
            self.event_manager.put(signal)
        elif action == 2 and current_position > 0:
            signal = SignalEvent(event.timestamp, event.ticker, "SHORT")
            self.event_manager.put(signal)

    def on_fill_event(self, event: FillEvent):
        if event.ticker not in self.tickers:
            return
        fill_cost = event.fill_price * event.quantity
        if event.direction == "BUY":
            self.positions[event.ticker] += event.quantity
            self.cash -= fill_cost
        elif event.direction == "SELL":
            self.positions[event.ticker] -= event.quantity
            self.cash += fill_cost
