# qmind_quant/execution/live_execution.py

import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from qmind_quant.core.event_types import OrderEvent


class LiveExecutionHandler:
    """
    Handles the execution of orders via the Alpaca API.
    """

    def __init__(self):
        api_key = os.getenv("APCA_API_KEY_ID")
        secret_key = os.getenv("APCA_API_SECRET_KEY")

        # Connect to the paper trading endpoint
        self.client = TradingClient(api_key, secret_key, paper=True)

    def on_order(self, event: OrderEvent):
        """
        Takes an OrderEvent and executes it via the Alpaca API.
        Note: This does not listen for fill confirmations, for simplicity.
        """
        print(f"--- LIVE EXECUTION: Received OrderEvent for {event.ticker} ---")

        order_request = MarketOrderRequest(
            symbol=event.ticker,
            qty=event.quantity,
            side=OrderSide.BUY if event.direction == "BUY" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )

        try:
            order = self.client.submit_order(order_data=order_request)
            print(f"--- Submitted Order: {order.id} for {order.symbol} ---")
        except Exception as e:
            print(f"!!! ERROR: Failed to submit order for {event.ticker}: {e} !!!")
