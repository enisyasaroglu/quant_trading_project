# qmind_quant/strategies/library/ml_strategy.py

import joblib
import pandas as pd
import pandas_ta as ta
from collections import deque
from qmind_quant.strategies.base_strategy import BaseStrategy
from qmind_quant.core.event_types import MarketEvent, SignalEvent


class MLStrategy(BaseStrategy):
    """
    A strategy that uses a pre-trained machine learning model to generate signals.
    """

    def __init__(
        self, tickers: list[str], event_manager, model_path: str, data_window=30
    ):
        super().__init__(tickers, event_manager)
        self.model = joblib.load(model_path)
        self.data_window = data_window

        # Store recent close prices to calculate features
        self.prices = {
            ticker: deque(maxlen=self.data_window) for ticker in self.tickers
        }
        self.invested = {ticker: "NONE" for ticker in self.tickers}

    def _calculate_features(self, ticker: str) -> pd.DataFrame | None:
        """
        Calculates the features for a single ticker based on the current price deque.
        """
        # Create a DataFrame from the deque of prices
        price_series = pd.Series(list(self.prices[ticker]), name="close")
        if len(price_series) < self.data_window:
            return None

        # --- Recreate the *exact same* features used in training ---
        features = pd.DataFrame(index=[price_series.index[-1]])

        returns_1d = price_series.pct_change(1)
        features["returns_1d"] = returns_1d.iloc[-1]
        features["returns_5d"] = price_series.pct_change(5).iloc[-1]
        features["returns_21d"] = price_series.pct_change(21).iloc[-1]
        features["volatility_21d"] = returns_1d.rolling(window=21).std().iloc[-1]
        features["rsi_14d"] = ta.rsi(price_series, length=14).iloc[-1]

        # Drop any potential NaN values from the single row
        features.dropna(inplace=True)
        if features.empty:
            return None

        return features

    def on_market_event(self, event: MarketEvent):
        """
        On a new market event, calculate features and generate a signal if the model predicts one.
        """
        if event.ticker not in self.tickers:
            return

        self.prices[event.ticker].append(event.close)

        # Wait until we have enough data to calculate all features
        if len(self.prices[event.ticker]) < self.data_window:
            return

        # Calculate features for the current bar
        features = self._calculate_features(event.ticker)
        if features is None:
            return

        # Make a prediction (model expects a 2D array)
        prediction = self.model.predict(features)[0]

        ticker = event.ticker
        if prediction == 1 and self.invested[ticker] != "LONG":
            signal = SignalEvent(event.timestamp, ticker, "LONG")
            self.event_manager.put(signal)
            self.invested[ticker] = "LONG"
        elif prediction == 0 and self.invested[ticker] == "LONG":
            # If the model predicts down, we exit our long position
            signal = SignalEvent(
                event.timestamp, ticker, "SHORT"
            )  # 'SHORT' signal is used to exit
            self.event_manager.put(signal)
            self.invested[ticker] = "NONE"
