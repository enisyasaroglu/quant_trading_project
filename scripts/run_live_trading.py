# scripts/run_live_trading.py

import time
import threading
from dotenv import load_dotenv
from qmind_quant.core.event_manager import EventManager
from qmind_quant.data_management.live_data_handler import LiveDataHandler
from qmind_quant.execution.live_execution import LiveExecutionHandler
from qmind_quant.strategies.library.ml_strategy import (
    MLStrategy,
)  # Using our ML strategy
from qmind_quant.portfolio_management.portfolio import (
    Portfolio,
)  # We'll need to adapt this slightly

# Load environment variables
load_dotenv()


def main():
    # --- Configuration ---
    tickers = ["AAPL", "GOOG"]
    model_path = "qmind_quant/ml_models/models/random_forest_v1.joblib"

    # --- Initialization ---
    event_manager = EventManager()

    # We use live components now
    data_handler = LiveDataHandler(event_manager, tickers)
    execution_handler = LiveExecutionHandler()

    # The Strategy and Portfolio remain largely the same
    # NOTE: The portfolio's data_handler dependency for price is a challenge in live trading.
    # A more advanced system would have a dedicated price cache. For now, we omit it.
    # This means our live position sizing and PnL tracking will be simplified.
    portfolio = Portfolio(event_manager, data_handler=None, initial_capital=100000.0)
    strategy = MLStrategy(tickers, event_manager, model_path)

    # --- Start Data Stream in a Separate Thread ---
    data_thread = threading.Thread(target=data_handler.run)
    data_thread.start()

    print("--- Starting Live Trading Engine ---")

    # --- Main Event Loop ---
    while True:
        try:
            event = event_manager.get()  # This will wait until an event is available

            if event.event_type == "MARKET":
                strategy.on_market_event(event)
                # In a live system, portfolio updates are usually driven by fills, not market data
                # portfolio.on_market_event(event)

            elif event.event_type == "SIGNAL":
                portfolio.on_signal(event)

            elif event.event_type == "ORDER":
                execution_handler.on_order(event)

            # Note: We are not handling FILL events from the broker in this version

        except KeyboardInterrupt:
            print("\n--- Halting Live Trading Engine ---")
            data_handler.stream.stop()
            data_thread.join()
            break
        except Exception as e:
            print(f"!!! An error occurred: {e} !!!")
            time.sleep(1)


if __name__ == "__main__":
    main()
