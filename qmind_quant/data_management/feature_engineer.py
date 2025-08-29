# qmind_quant/data_management/feature_engineer.py

import pandas as pd

# Import all the custom indicator functions from your new module
from qmind_quant.analytics.technical_indicators import (
    calculate_sma,
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


class FeatureEngineer:
    """
    A class to engineer a rich set of features for a given OHLCV dataset
    using a custom-built technical indicator library.
    """

    def create_features(self, df: pd.DataFrame, ticker_col="ticker") -> pd.DataFrame:
        """
        Adds a curated set of features and the target variable to the OHLCV data.

        Args:
            df (pd.DataFrame): DataFrame with multi-ticker OHLCV data.
            ticker_col (str): The name of the column that identifies the ticker.

        Returns:
            pd.DataFrame: The DataFrame with added features and target.
        """
        all_features = []

        # Process each stock's data individually to prevent look-ahead bias
        for ticker, group in df.groupby(ticker_col):
            # Ensure the group is a DataFrame
            group = group.copy()

            # --- A. Trend Indicators ---
            group["ema_12"] = calculate_ema(group["close"], window=12)
            group["ema_26"] = calculate_ema(group["close"], window=26)
            macd_df = calculate_macd(group["close"])
            group["macd"] = macd_df["macd"]
            group["adx_14"] = calculate_adx(
                group["high"], group["low"], group["close"], window=14
            )

            # --- B. Momentum Indicators ---
            group["rsi_14"] = calculate_rsi(group["close"], window=14)
            group["stoch_k_14"] = calculate_stochastic_oscillator(
                group["high"], group["low"], group["close"], window=14
            )

            # --- C. Volatility Indicators ---
            bbands_df = calculate_bollinger_bands(group["close"], window=20)
            group["bb_width"] = (
                bbands_df["bb_upper"] - bbands_df["bb_lower"]
            ) / bbands_df["bb_middle"]
            group["atr_14"] = calculate_atr(
                group["high"], group["low"], group["close"], window=14
            )

            # --- D. Volume Indicators ---
            group["obv"] = calculate_obv(group["close"], group["volume"])
            group["vwap"] = calculate_vwap(group["close"], group["volume"])

            # --- Target Variable Creation ---
            # Predict if the price will be higher in 5 days (1 for up, 0 for down/same)
            future_returns = group["close"].shift(-5).pct_change(5, fill_method=None)
            group["target"] = (future_returns > 0).astype(int)

            all_features.append(group)

        # Combine the feature-engineered data for all tickers
        feature_df = pd.concat(all_features)

        # Drop rows with NaN values created by rolling calculations and shifting
        feature_df.dropna(inplace=True)

        return feature_df
