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
            # --- THIS IS THE FIX ---
            # Check for risk management halt at the start of each bar
            if self.portfolio.is_risk_managed:
                if hasattr(self.strategy, "trading_halted"):
                    self.strategy.trading_halted = True
                print(
                    "--- Trading halted by Portfolio Risk Manager. Ending backtest. ---"
                )
                break  # Exit the main loop

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
                    self.execution_handler.on_order(event)
                elif event.event_type == "FILL":
                    self.portfolio.on_fill(event)
                    self.strategy.on_fill_event(event)
        print("Backtest finished.")
