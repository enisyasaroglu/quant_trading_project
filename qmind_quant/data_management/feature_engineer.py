# qmind_quant/data_management/feature_engineer.py

import pandas as pd

# Import all the custom indicator functions from your new module
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


class FeatureEngineer:
    """
    A class to engineer a rich set of features for a given OHLCV dataset
    using a custom-built technical indicator library.

    This class is the core of the data preparation pipeline for all machine
    learning models.
    """

    def create_features(self, df: pd.DataFrame, ticker_col="ticker") -> pd.DataFrame:
        """
        Adds a curated set of features and the target variable to the OHLCV data.

        Args:
            df (pd.DataFrame): DataFrame with multi-ticker OHLCV data.
            ticker_col (str): The name of the column that identifies the ticker.

        Returns:
            pd.DataFrame: The DataFrame with added features and target, with any
                          rows containing NaN values (from the initial warm-up
                          period of the indicators) removed.
        """
        all_features = []

        # We process each stock's data individually. This is a critical step
        # to prevent data from one stock's indicators "leaking" into another's.
        for ticker, group in df.groupby(ticker_col):
            # --- This is a professional safety improvement ---
            # Using .copy() prevents a common pandas warning ('SettingWithCopyWarning').
            # It ensures that we are working on an independent copy of the data for
            # each stock, not a "view" of the original DataFrame.
            group = group.copy()

            # --- A. Trend Indicators ---
            # These indicators help the model understand the market's direction.
            group["ema_12"] = calculate_ema(group["close"], window=12)
            group["ema_26"] = calculate_ema(group["close"], window=26)
            macd_df = calculate_macd(group["close"])
            group["macd"] = macd_df["macd"]
            group["adx_14"] = calculate_adx(
                group["high"], group["low"], group["close"], window=14
            )

            # --- B. Momentum Indicators ---
            # These oscillators help the model identify overbought or oversold conditions.
            group["rsi_14"] = calculate_rsi(group["close"], window=14)
            group["stoch_k_14"] = calculate_stochastic_oscillator(
                group["high"], group["low"], group["close"], window=14
            )

            # --- C. Volatility Indicators ---
            # These features help the model understand the level of market risk and turbulence.
            bbands_df = calculate_bollinger_bands(group["close"], window=20)
            group["bb_width"] = (
                bbands_df["bb_upper"] - bbands_df["bb_lower"]
            ) / bbands_df["bb_middle"]
            group["atr_14"] = calculate_atr(
                group["high"], group["low"], group["close"], window=14
            )

            # --- D. Volume Indicators ---
            # These features help the model confirm the strength behind a price move.
            group["obv"] = calculate_obv(group["close"], group["volume"])
            group["vwap"] = calculate_vwap(group["close"], group["volume"])

            # --- Target Variable Creation ---
            # This is what we are trying to predict: will the price be higher
            # in 5 days? (1 for 'Up', 0 for 'Down' or 'Same').
            future_returns = group["close"].shift(-5).pct_change(5, fill_method=None)
            group["target"] = (future_returns > 0).astype(int)

            all_features.append(group)

        # Combine the feature-engineered data for all tickers into a single DataFrame.
        feature_df = pd.concat(all_features)

        # Drop any rows that have missing values (NaNs). This happens naturally
        # at the beginning of the dataset during the "warm-up" period for indicators
        # like moving averages that need a certain amount of prior data.
        feature_df.dropna(inplace=True)

        return feature_df
