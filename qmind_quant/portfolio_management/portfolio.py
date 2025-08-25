# qmind_quant/portfolio_management/portfolio.py

import pandas as pd
from qmind_quant.core.event_types import SignalEvent, OrderEvent, FillEvent, MarketEvent
from qmind_quant.data_management.data_handler import HistoricalDataHandler


class Portfolio:
    def __init__(
        self,
        event_manager,
        data_handler: HistoricalDataHandler
        | None = None,  # Allow data_handler to be None
        initial_capital=100000.0,
        max_drawdown_pct=0.15,
    ):
        self.event_manager = event_manager
        self.data_handler = data_handler
        self.initial_capital = initial_capital
        self.cash = initial_capital

        self.current_holdings = {}
        self.all_holdings = []

        self.max_drawdown_pct = max_drawdown_pct
        self.high_water_mark = initial_capital
        self.is_risk_managed = False

        # --- THIS IS THE FIX ---
        # Handle initialization for both backtesting and live trading
        if self.data_handler:
            # Backtesting mode: Use the start date from the historical data
            initial_timestamp = self.data_handler.start_date
        else:
            # Live trading mode: Use the current time (with a relevant timezone)
            initial_timestamp = pd.Timestamp.now(tz="America/New_York")

        self._update_holdings_for_timestamp(initial_timestamp)

    def _update_holdings_for_timestamp(self, timestamp):
        """Records the current cash and market value, and checks for max drawdown."""
        # For live trading, we need a valid price source to do this,
        # which is a more advanced topic. We'll skip this logic if no data_handler.
        if self.data_handler is None:
            return

        record = {"timestamp": timestamp, "cash": self.cash}

        market_value = 0
        for ticker, quantity in self.current_holdings.items():
            if quantity != 0:
                price = self.data_handler.get_latest_close_price(ticker)
                if price:
                    market_value += price * quantity

        total_value = self.cash + market_value
        record["market_value"] = market_value
        record["total_value"] = total_value
        self.all_holdings.append(record)

        if not self.is_risk_managed:
            self.high_water_mark = max(self.high_water_mark, total_value)
            drawdown = (self.high_water_mark - total_value) / self.high_water_mark

            if drawdown > self.max_drawdown_pct:
                print(
                    f"!!! RISK TRIGGERED: Max drawdown of {self.max_drawdown_pct:.2%} exceeded. !!!"
                )
                self._liquidate_all_positions(timestamp)

    def _liquidate_all_positions(self, timestamp):
        """Generates SELL orders for all current holdings."""
        print("--- LIQUIDATING ALL POSITIONS ---")
        for ticker, quantity in list(self.current_holdings.items()):
            if quantity > 0:
                order = OrderEvent(timestamp, ticker, "MKT", "SELL", quantity)
                self.event_manager.put(order)

    def on_market_event(self, event: MarketEvent):
        # This is primarily for backtesting to update the equity curve daily
        self._update_holdings_for_timestamp(event.timestamp)

    def on_signal(self, event: SignalEvent):
        """On a SignalEvent, generate a new OrderEvent if not risk-managed."""
        if self.is_risk_managed:
            return

        # For live trading, we can't get a historical price easily,
        # so we will use a simplified position sizing logic.
        # A production system would use the latest tick/quote price.
        quantity = 10  # Simplified fixed quantity for live trading

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
        cost = event.fill_price * event.quantity

        if event.direction == "BUY":
            self.cash -= cost
            self.cash -= event.commission
            self.current_holdings[event.ticker] = (
                self.current_holdings.get(event.ticker, 0) + event.quantity
            )
        elif event.direction == "SELL":
            self.cash += cost
            self.cash -= event.commission
            self.current_holdings[event.ticker] = (
                self.current_holdings.get(event.ticker, 0) - event.quantity
            )

    def get_equity_curve(self) -> pd.DataFrame:
        curve = pd.DataFrame(self.all_holdings)
        curve.set_index("timestamp", inplace=True)
        curve["returns"] = curve["total_value"].pct_change().fillna(0.0)
        return curve
