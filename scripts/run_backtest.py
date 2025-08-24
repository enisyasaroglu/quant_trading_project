# scripts/run_backtest.py

import os
import quantstats as qs
from qmind_quant.core.event_manager import EventManager
from qmind_quant.data_management.data_handler import HistoricalDataHandler
from qmind_quant.simulation.backtest_engine import BacktestEngine
from qmind_quant.strategies.library.ml_strategy import MLStrategy
from qmind_quant.portfolio_management.portfolio import Portfolio
from qmind_quant.execution.execution import SimulatedExecutionHandler


def main():
    # --- Configuration ---
    project_root = os.path.join(os.path.dirname(__file__), "..")
    data_file = os.path.join(project_root, "data/processed/us_equities_daily.parquet")
    model_file = os.path.join(
        project_root, "qmind_quant/ml_models/models/random_forest_v1.joblib"
    )

    tickers = ["AAPL", "GOOG"]
    initial_capital = 100000.0

    # --- Initialization ---
    event_manager = EventManager()
    data_handler = HistoricalDataHandler(file_path=data_file, tickers=tickers)

    # Use the MLStrategy
    strategy = MLStrategy(
        tickers=tickers, event_manager=event_manager, model_path=model_file
    )

    portfolio = Portfolio(event_manager, data_handler, initial_capital)
    execution_handler = SimulatedExecutionHandler(event_manager, data_handler)

    engine = BacktestEngine(
        event_manager, data_handler, strategy, portfolio, execution_handler
    )

    # --- Run ---
    engine.run_backtest()

    # --- Performance Reporting ---
    print("Generating performance report...")
    equity_curve = portfolio.get_equity_curve()

    qs.reports.html(
        equity_curve["returns"], output="ml_strategy_report.html", title="ML Strategy"
    )
    print("Report saved to ml_strategy_report.html")


if __name__ == "__main__":
    main()
