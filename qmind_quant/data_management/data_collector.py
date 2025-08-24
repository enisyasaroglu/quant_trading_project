# qmind_quant/data_management/data_collector.py

import os
import pandas as pd
import yfinance as yf


class DataCollector:
    """
    A class to collect historical market data using yfinance.
    """

    def fetch_daily_data(
        self, tickers: list[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Fetches historical daily OHLCV data for a list of tickers from Yahoo Finance.

        Args:
            tickers (list[str]): A list of stock tickers.
            start_date (str): The start date in 'YYYY-MM-DD' format.
            end_date (str): The end date in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: A DataFrame containing the adjusted daily data for all tickers.
        """
        print(
            f"Fetching daily data for {tickers} from {start_date} to {end_date} using yfinance..."
        )

        # yfinance download is powerful. It can fetch multiple tickers at once.
        # auto_adjust=True automatically handles splits and dividends.
        df = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            # Group by ticker for a multi-index DataFrame, similar to the Alpaca output
            group_by="ticker",
        )

        # A DataFrame can arrive in two formats:
        # 1. Single-ticker: Columns are simple ['Open', 'High', 'Low', 'Close', ...].
        # 2. Multi-ticker: Columns are nested, like [('AAPL', 'Open'), ('GOOG', 'Open'), ...].
        # This logic standardizes both formats into a single, consistent "long" format.

        if isinstance(df.columns, pd.MultiIndex):
            # This block handles the multi-ticker case by reshaping the DataFrame.
            # The goal is to turn the ticker names from column headers into a new 'Ticker' column.
            #
            # .stack(level=0): Pivots the top level of columns (the tickers) into rows.
            # .rename_axis(...): Names the new index levels created by stacking.
            # .reset_index(): Converts these index levels into regular columns.
            df = (
                df.stack(level=0, future_stack=True)
                .rename_axis(["Date", "Ticker"])
                .reset_index()
            )

        # Standardize column names to be lowercase with underscores (e.g., "Adj Close" -> "adj_close").
        # This makes accessing columns more predictable and consistent.
        df.columns = [col.lower().replace(" ", "_") for col in df.columns]

        # Optional: For time-series analysis, you might want the Date as the index.
        # You can uncomment the line below if needed for subsequent steps.
        # df.set_index('Date', inplace=True)

        print(f"Successfully processed {len(df)} rows of data.")
        return df

    def save_to_parquet(self, data: pd.DataFrame, file_path: str):
        """
        Saves a DataFrame to a Parquet file.

        Args:
            data (pd.DataFrame): The data to save.
            file_path (str): The path to the output Parquet file.
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        data.to_parquet(file_path, index=False)
        print(f"Data successfully saved to {file_path}")
