# qmind_quant/strategies/library/ml_strategy.py

import joblib
import pandas as pd
from collections import deque
from qmind_quant.strategies.base_strategy import BaseStrategy
from qmind_quant.core.event_types import MarketEvent, SignalEvent
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


class MLStrategy(BaseStrategy):
    """
    A strategy that uses a pre-trained supervised learning model to generate signals.
    """

    def __init__(
        self,
        tickers: list[str],
        event_manager,
        model_path: str = None,
        model=None,
        data_window=50,
    ):
        super().__init__(tickers, event_manager)
        if model_path is None and model is None:
            raise ValueError("Either 'model_path' or 'model' must be provided.")

        self.model = model if model is not None else joblib.load(model_path)
        self.data_window = data_window
        self.bars = {ticker: deque(maxlen=self.data_window) for ticker in self.tickers}
        self.invested = dict.fromkeys(self.tickers, "NONE")

    # ... The rest of the methods (_calculate_features, on_market_event) remain unchanged ...
    def _calculate_features(self, ticker: str) -> pd.DataFrame | None:
        if len(self.bars[ticker]) < self.data_window:
            return None
        df = pd.DataFrame(list(self.bars[ticker]))
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

        latest_features = df.tail(1)
        feature_names = [
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
        ]
        return latest_features[feature_names]

    def on_market_event(self, event: MarketEvent):
        if event.ticker not in self.tickers:
            return
        self.bars[event.ticker].append(event)
        features = self._calculate_features(event.ticker)
        if features is None or features.isnull().values.any():
            return

        prediction = self.model.predict(features)[0]
        ticker = event.ticker
        if prediction == 1 and self.invested[ticker] != "LONG":
            signal = SignalEvent(event.timestamp, ticker, "LONG")
            self.event_manager.put(signal)
            self.invested[ticker] = "LONG"
        elif prediction == 0 and self.invested[ticker] == "LONG":
            signal = SignalEvent(event.timestamp, ticker, "SHORT")
            self.event_manager.put(signal)
            self.invested[ticker] = "NONE"
