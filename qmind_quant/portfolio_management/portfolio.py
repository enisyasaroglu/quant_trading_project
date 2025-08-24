# qmind_quant/portfolio_management/portfolio.py

import pandas as pd
from qmind_quant.core.event_types import SignalEvent, OrderEvent, FillEvent, MarketEvent
from qmind_quant.data_management.data_handler import HistoricalDataHandler


class Portfolio:
    def __init__(
        self,
        event_manager,
        data_handler: HistoricalDataHandler,
        initial_capital=100000.0,
    ):
        self.event_manager = event_manager
        self.data_handler = data_handler
        self.initial_capital = initial_capital
        self.cash = initial_capital

        # {ticker: quantity}
        self.current_holdings = {}
        # List of dictionaries to track portfolio value over time
        self.all_holdings = []
        initial_timestamp = self.data_handler.start_date
        self._update_holdings_for_timestamp(initial_timestamp)  # Record initial state

    def on_market_event(self, event: MarketEvent):
        """
        On a new market bar, update the portfolio's total value.
        This is how we generate the equity curve.
        """
        self._update_holdings_for_timestamp(event.timestamp)

    def _update_holdings_for_timestamp(self, timestamp):
        """Records the current cash and market value of all holdings."""
        record = {"timestamp": timestamp, "cash": self.cash, "total_value": self.cash}

        market_value = 0
        for ticker, quantity in self.current_holdings.items():
            if quantity != 0:
                price = self.data_handler.get_latest_close_price(ticker)
                if price:
                    market_value += price * quantity

        record["market_value"] = market_value
        record["total_value"] += market_value
        self.all_holdings.append(record)

    def on_signal(self, event: SignalEvent):
        """
        On a SignalEvent, generate a new OrderEvent based on position sizing.
        """
        price = self.data_handler.get_latest_close_price(event.ticker)
        if price is None:
            return

        fixed_investment_amount = 10000.0
        quantity = int(fixed_investment_amount / price)
        if quantity == 0:
            return  # Not enough capital to buy even one share

        if event.signal_type == "LONG":
            order = OrderEvent(event.timestamp, event.ticker, "MKT", "BUY", quantity)
            self.event_manager.put(order)

        elif event.signal_type == "SHORT":
            current_quantity = self.current_holdings.get(event.ticker, 0)
            if current_quantity > 0:
                order = OrderEvent(
                    event.timestamp, event.ticker, "MKT", "SELL", current_quantity
                )
                self.event_manager.put(order)

    def on_fill(self, event: FillEvent):
        """
        On a FillEvent, update the portfolio's cash and holdings.
        """
        cost = event.fill_price * event.quantity

        if event.direction == "BUY":
            self.cash -= cost
            self.cash -= event.commission
            current_qty = self.current_holdings.get(event.ticker, 0)
            self.current_holdings[event.ticker] = current_qty + event.quantity
        elif event.direction == "SELL":
            self.cash += cost
            self.cash -= event.commission
            current_qty = self.current_holdings.get(event.ticker, 0)
            self.current_holdings[event.ticker] = current_qty - event.quantity

    def get_equity_curve(self) -> pd.DataFrame:
        """
        Returns a DataFrame of the portfolio's value over time with a proper DatetimeIndex.
        """
        curve = pd.DataFrame(self.all_holdings)
        curve.set_index("timestamp", inplace=True)

        # FIX: Convert the index to a proper DatetimeIndex and handle the initial 'None' timestamp
        curve.index = pd.to_datetime(curve.index, utc=True)
        curve.dropna(inplace=True)  # Drops the initial row where timestamp was None

        # Calculate returns for quantstats. The Series must have a DatetimeIndex.
        curve["returns"] = curve["total_value"].pct_change().fillna(0.0)
        return curve
