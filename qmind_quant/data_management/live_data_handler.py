# qmind_quant/data_management/live_data_handler.py

import os
import redis
from alpaca.data.live import StockDataStream

from qmind_quant.core.event_manager import EventManager
from qmind_quant.core.event_types import MarketEvent

# --- This is a professional improvement for maintainability ---
# We define our Redis keys as constants in one place. This prevents typos
# and makes it easy to change them later if we need to.
LIVE_PRICES_KEY = "qmind:live_prices"


class LiveDataHandler:
    """
    Connects to the Alpaca real-time data stream, receives live market bars,
    and performs two key actions:
    1. Puts a MarketEvent onto the central event queue for the trading engine to process.
    2. Publishes the latest price to a Redis cache for the dashboard to read.
    """

    def __init__(self, event_manager: EventManager, tickers: list[str]):
        """
        Initializes the LiveDataHandler.

        Args:
            event_manager (EventManager): The central event queue for the system.
            tickers (list[str]): The list of stock symbols to subscribe to.
        """
        api_key = os.getenv("APCA_API_KEY_ID")
        secret_key = os.getenv("APCA_API_SECRET_KEY")

        self.event_manager = event_manager
        self.tickers = tickers

        # Initialize the connection to Alpaca's data stream.
        self.stream = StockDataStream(api_key, secret_key)

        # --- This is a professional improvement for flexibility ---
        # Instead of hardcoding "localhost", we get the Redis host from an
        # environment variable. This allows the code to work both locally and
        # in a different environment (like Docker or the cloud) without changes.
        # If the variable isn't set, it defaults to 'localhost'.
        redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_client = redis.StrictRedis(
            host=redis_host, port=6379, db=0, decode_responses=True
        )
        # Clear any old prices from a previous run to ensure a clean start.
        self.redis_client.delete(LIVE_PRICES_KEY)

    async def _bar_handler(self, data):
        """
        This is the main callback function that gets executed every time a new
        market bar is received from the Alpaca WebSocket.
        The 'async' keyword means this function is designed to run asynchronously,
        handling network events efficiently without blocking the rest of the program.
        """
        # Publish the latest close price to a Redis hash for the dashboard to read.
        self.redis_client.hset(LIVE_PRICES_KEY, data.symbol, data.close)

        # Create a MarketEvent for the trading engine.
        event = MarketEvent(
            timestamp=data.timestamp,
            ticker=data.symbol,
            open=data.open,
            high=data.high,
            low=data.low,
            close=data.close,
            volume=data.volume,
        )
        # Put the event onto the central queue for the other modules to process.
        self.event_manager.put(event)

        # Print a dot to the console as a simple "heartbeat" to show that
        # live data is flowing.
        print(f".", end="", flush=True)

    def run(self):
        """
        Starts the data stream. This is a blocking call that will run forever,
        listening for new market data until the program is stopped.
        """
        print("--- Starting Live Data Stream ---")
        # Subscribe to the one-minute bar data for our list of tickers.
        self.stream.subscribe_bars(self._bar_handler, *self.tickers)
        # Start the main event loop for the data stream.
        self.stream.run()
