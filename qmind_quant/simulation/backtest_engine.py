# qmind_quant/simulation/backtest_engine.py

from qmind_quant.core.event_manager import EventManager
from qmind_quant.data_management.data_handler import HistoricalDataHandler
from qmind_quant.strategies.base_strategy import BaseStrategy
from qmind_quant.portfolio_management.portfolio import Portfolio
from qmind_quant.execution.execution import SimulatedExecutionHandler


class BacktestEngine:
    def __init__(
        self,
        event_manager: EventManager,
        data_handler: HistoricalDataHandler,
        strategy: BaseStrategy,
        portfolio: Portfolio,
        execution_handler: SimulatedExecutionHandler,
    ):
        self.event_manager = event_manager
        self.data_handler = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution_handler = execution_handler

    def run_backtest(self):
        print("Starting backtest...")
        while self.data_handler.continue_backtest:
            market_event = self.data_handler.stream_next_bar()
            if market_event is not None:
                self.event_manager.put(market_event)

            while not self.event_manager.empty():
                event = self.event_manager.get()

                if event.event_type == "MARKET":
                    self.strategy.on_market_event(event)
                    self.portfolio.on_market_event(event)

                elif event.event_type == "SIGNAL":
                    self.portfolio.on_signal(event)

                elif event.event_type == "ORDER":
                    print(
                        f"  ORDER: {event.timestamp} - {event.ticker} - {event.direction} {event.quantity} shares"
                    )
                    self.execution_handler.on_order(event)

                elif event.event_type == "FILL":
                    self.portfolio.on_fill(event)
        print("Backtest finished.")
