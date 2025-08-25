# qmind_quant/data_management/live_data_handler.py

import os
from alpaca.data.live import StockDataStream
from qmind_quant.core.event_manager import EventManager
from qmind_quant.core.event_types import MarketEvent


class LiveDataHandler:
    """
    Receives live market data from Alpaca's WebSocket stream.
    """

    def __init__(self, event_manager: EventManager, tickers: list[str]):
        api_key = os.getenv("APCA_API_KEY_ID")
        secret_key = os.getenv("APCA_API_SECRET_KEY")

        self.event_manager = event_manager
        self.tickers = tickers
        self.stream = StockDataStream(api_key, secret_key)

    async def _bar_handler(self, data):
        """
        This is called every time a new bar is received from the stream.
        """
        # Create a MarketEvent from the live bar data
        event = MarketEvent(
            timestamp=data.timestamp,
            ticker=data.symbol,
            open=data.open,
            high=data.high,
            low=data.low,
            close=data.close,
            volume=data.volume,
        )
        # Put the event into the main queue for the engine to process
        self.event_manager.put(event)
        print(f".", end="", flush=True)  # Print a dot to show data is flowing

    def run(self):
        """
        Starts the data stream.
        """
        print("--- Starting Live Data Stream ---")
        # Subscribe to minute bars for our tickers and run the stream
        self.stream.subscribe_bars(self._bar_handler, *self.tickers)
        self.stream.run()
