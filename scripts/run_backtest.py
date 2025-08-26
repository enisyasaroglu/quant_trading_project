# scripts/run_backtest.py

import os
import quantstats as qs
from qmind_quant.core.event_manager import EventManager
from qmind_quant.data_management.data_handler import HistoricalDataHandler
from qmind_quant.simulation.backtest_engine import BacktestEngine
from qmind_quant.strategies.library.ml_strategy import MLStrategy
from qmind_quant.portfolio_management.portfolio import Portfolio
from qmind_quant.execution.execution import SimulatedExecutionHandler
from qmind_quant.config.paths import (
    DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
)  # Import REPORTS_DIR


def main():
    """
    Main function to configure and run a backtest for an ML strategy.
    """
    # --- Configuration ---
    data_file = DATA_DIR / "processed" / "us_equities_daily.parquet"

    # --- CHOOSE YOUR MODEL ---
    model_name = "xgboost_v1.joblib"
    model_file = MODELS_DIR / model_name

    tickers = ["AAPL", "GOOG"]
    initial_capital = 100000.0
    max_drawdown_pct = 0.20

    # --- Initialization ---
    event_manager = EventManager()
    data_handler = HistoricalDataHandler(
        file_path=str(data_file), tickers=tickers
    )  # Convert Path to string
    strategy = MLStrategy(tickers, event_manager, model_path=str(model_file))
    portfolio = Portfolio(
        event_manager, data_handler, initial_capital, max_drawdown_pct
    )
    execution_handler = SimulatedExecutionHandler(event_manager, data_handler)

    engine = BacktestEngine(
        event_manager, data_handler, strategy, portfolio, execution_handler
    )

    # --- Run ---
    engine.run_backtest()

    # --- Performance Reporting ---
    print("Generating performance report...")
    equity_curve = portfolio.get_equity_curve()

    # Ensure the reports directory exists
    os.makedirs(REPORTS_DIR, exist_ok=True)

    report_name = model_name.replace(".joblib", "_report.html")
    output_path = REPORTS_DIR / report_name

    qs.reports.html(
        equity_curve["returns"], output=str(output_path), title=f"{model_name} Strategy"
    )
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    main()
