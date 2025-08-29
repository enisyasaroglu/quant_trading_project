# qmind_quant/data_management/data_handler.py

import pandas as pd
from qmind_quant.core.event_types import MarketEvent


class HistoricalDataHandler:
    """
    Reads historical data and provides it as a stream of MarketEvents.
    Can be initialized from a Parquet file path or an in-memory DataFrame.
    """

    def __init__(
        self, tickers: list[str], file_path: str = None, data_df: pd.DataFrame = None
    ):
        """
        Initializes the data handler.

        Args:
            tickers (list[str]): A list of tickers to include in the backtest.
            file_path (str, optional): The path to the Parquet file.
            data_df (pd.DataFrame, optional): An in-memory DataFrame with OHLCV data.
        """
        if file_path is None and data_df is None:
            raise ValueError("Either 'file_path' or 'data_df' must be provided.")

        self.tickers = tickers
        self._all_data = self._load_data(file_path, data_df)
        self._bar_generator = self._create_bar_generator()
        self.continue_backtest = True
        self.latest_bars = {ticker: None for ticker in self.tickers}
        self.start_date = (
            self._all_data["date"].min() if not self._all_data.empty else None
        )

    def _load_data(self, file_path: str, data_df: pd.DataFrame) -> pd.DataFrame:
        """Loads and prepares the data from the specified source."""
        if data_df is not None:
            df = data_df.copy()
        else:
            df = pd.read_parquet(file_path)

        df = df[df["ticker"].isin(self.tickers)]
        df.sort_values(by=["date", "ticker"], inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        return df

    # ... The rest of the methods (_create_bar_generator, get_latest_close_price, stream_next_bar) remain unchanged ...
    def _create_bar_generator(self):
        for index, row in self._all_data.iterrows():
            yield row

    def get_latest_close_price(self, ticker: str) -> float | None:
        if self.latest_bars.get(ticker):
            return self.latest_bars[ticker].close
        return None

    def stream_next_bar(self) -> MarketEvent | None:
        try:
            bar = next(self._bar_generator)
            event = MarketEvent(
                timestamp=bar["date"],
                ticker=bar["ticker"],
                open=bar["open"],
                high=bar["high"],
                low=bar["low"],
                close=bar["close"],
                volume=bar["volume"],
            )
            self.latest_bars[event.ticker] = event
            return event
        except StopIteration:
            self.continue_backtest = False
            return None
