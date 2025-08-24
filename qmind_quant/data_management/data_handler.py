# qmind_quant/data_management/data_handler.py

import pandas as pd
from qmind_quant.core.event_types import MarketEvent


class HistoricalDataHandler:
    def __init__(self, file_path: str, tickers: list[str]):
        self.file_path = file_path
        self.tickers = tickers
        self._all_data = self._load_data()
        self._bar_generator = self._create_bar_generator()
        self.continue_backtest = True
        self.latest_bars = {ticker: None for ticker in self.tickers}
        self.start_date = (
            self._all_data["date"].min() if not self._all_data.empty else None
        )

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_parquet(self.file_path)
        df = df[df["ticker"].isin(self.tickers)]
        df.sort_values(by=["date", "ticker"], inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def _create_bar_generator(self):
        for index, row in self._all_data.iterrows():
            yield row

    def get_latest_close_price(self, ticker: str) -> float | None:
        """
        Returns the most recent closing price for a given ticker.
        """
        if self.latest_bars.get(ticker):
            return self.latest_bars[ticker].close
        return None

    def stream_next_bar(self) -> MarketEvent | None:
        try:
            bar = next(self._bar_generator)

            # New: Update the latest bar for the corresponding ticker
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
