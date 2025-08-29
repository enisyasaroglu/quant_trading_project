# scripts/run_backtest.py

import os
import quantstats as qs
import pandas as pd
from qmind_quant.core.event_manager import EventManager
from qmind_quant.data_management.data_handler import HistoricalDataHandler
from qmind_quant.simulation.backtest_engine import BacktestEngine
from qmind_quant.strategies.library.ml_strategy import MLStrategy
from qmind_quant.portfolio_management.portfolio import Portfolio
from qmind_quant.execution.execution import SimulatedExecutionHandler
from qmind_quant.config.paths import DATA_DIR, MODELS_DIR, REPORTS_DIR
from qmind_quant.ml_models.model_trainer import train_xgboost_model
from qmind_quant.data_management.feature_engineer import FeatureEngineer


def run_backtest_for_model(
    model, data_df: pd.DataFrame, tickers: list[str], initial_capital: float
) -> pd.DataFrame:
    """
    Runs a backtest for a given trained model on a specific dataset.

    Args:
        model: The pre-trained machine learning model.
        data_df (pd.DataFrame): The OHLCV data for the backtest period.
        tickers (list[str]): The tickers to backtest on.
        initial_capital (float): The starting capital.

    Returns:
        pd.DataFrame: The equity curve DataFrame from the backtest.
    """
    event_manager = EventManager()
    # The data handler now works with an in-memory DataFrame instead of a file
    data_handler = HistoricalDataHandler(data_df=data_df, tickers=tickers)

    strategy = MLStrategy(
        tickers=tickers,
        event_manager=event_manager,
        model=model,  # Pass the trained model object directly
    )

    portfolio = Portfolio(event_manager, data_handler, initial_capital)
    execution_handler = SimulatedExecutionHandler(event_manager, data_handler)

    engine = BacktestEngine(
        event_manager, data_handler, strategy, portfolio, execution_handler
    )
    engine.run_backtest()

    return portfolio.get_equity_curve()


def main():
    """
    Main function to run a single, standard backtest for demonstration.
    """
    # --- This main block now demonstrates a single run ---
    print("--- Running a single demonstration backtest ---")

    # 1. Load all data
    full_data_file = DATA_DIR / "processed" / "us_equities_daily.parquet"
    full_df = pd.read_parquet(full_data_file)

    # 2. Engineer features on all data
    engineer = FeatureEngineer()
    feature_df = engineer.create_features(full_df)

    # 3. Train the model on all feature data
    print("Training model for single run...")
    model = train_xgboost_model(feature_df)

    # 4. Run the backtest on the original OHLCV data
    print("Running backtest...")
    equity_curve = run_backtest_for_model(
        model=model,
        data_df=full_df,  # Backtest runs on raw data, strategy calculates features internally
        tickers=["AAPL", "GOOG"],
        initial_capital=100000.0,
    )

    # 5. Generate Report
    print("Generating performance report...")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_name = "single_xgboost_report.html"
    output_path = REPORTS_DIR / report_name
    qs.reports.html(
        equity_curve["returns"], output=str(output_path), title="Single XGBoost Run"
    )
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    main()
