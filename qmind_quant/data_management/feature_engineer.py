# qmind_quant/data_management/feature_engineer.py

import pandas as pd
import pandas_ta as ta


class FeatureEngineer:
    """
    A class to engineer features for a given OHLCV dataset.
    """

    def create_features(self, df: pd.DataFrame, ticker_col="ticker") -> pd.DataFrame:
        """
        Adds features and the target variable to the OHLCV data.

        Args:
            df (pd.DataFrame): DataFrame with multi-ticker OHLCV data.
            ticker_col (str): The name of the column that identifies the ticker.

        Returns:
            pd.DataFrame: The DataFrame with added features and target.
        """
        # To calculate features correctly for each stock, we group by the ticker.
        # We'll store the results for each ticker in a list and concatenate at the end.
        all_features = []

        for ticker, group in df.groupby(ticker_col):
            # --- Feature Creation ---

            # 1. Momentum Features (Returns over different periods)
            group["returns_1d"] = group["close"].pct_change(1)
            group["returns_5d"] = group["close"].pct_change(5)
            group["returns_21d"] = group["close"].pct_change(
                21
            )  # Approx. 1 trading month

            # 2. Volatility Feature (Rolling standard deviation of returns)
            group["volatility_21d"] = group["returns_1d"].rolling(window=21).std()

            # 3. Technical Indicator (RSI - Relative Strength Index)
            group["rsi_14d"] = ta.rsi(group["close"], length=14)

            # --- Target Variable Creation ---
            # We want to predict if the price will go up or down in the next 5 days.
            # 1 = Price went up, 0 = Price went down or stayed the same.
            future_returns = group["close"].shift(-5).pct_change(5)
            group["target"] = (future_returns > 0).astype(int)

            all_features.append(group)

        # Combine the feature-engineered data for all tickers
        feature_df = pd.concat(all_features)

        # Drop rows with NaN values created by rolling calculations and shifting
        feature_df.dropna(inplace=True)

        return feature_df
