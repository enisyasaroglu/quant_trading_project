# qmind_quant/execution/execution.py

from qmind_quant.core.event_types import OrderEvent, FillEvent
from qmind_quant.data_management.data_handler import HistoricalDataHandler


class SimulatedExecutionHandler:
    def __init__(self, event_manager, data_handler: HistoricalDataHandler):
        self.event_manager = event_manager
        self.data_handler = data_handler

    def on_order(self, event: OrderEvent):
        """
        Handles an OrderEvent, creating a FillEvent with a realistic price.
        """
        fill_price = self.data_handler.get_latest_close_price(event.ticker)
        if fill_price is None:
            print(
                f"  EXECUTION: Could not get price for {event.ticker}. Order cancelled."
            )
            return

        # Simulate a simple fixed commission
        commission = 1.0

        fill_event = FillEvent(
            timestamp=event.timestamp,
            ticker=event.ticker,
            direction=event.direction,
            quantity=event.quantity,
            fill_price=fill_price,
            commission=commission,
        )
        self.event_manager.put(fill_event)
