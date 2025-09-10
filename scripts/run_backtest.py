# scripts/run_backtest.py

import pandas as pd
import quantstats as qs
from qmind_quant.core.event_manager import EventManager
from qmind_quant.data_management.data_handler import HistoricalDataHandler
from qmind_quant.simulation.backtest_engine import BacktestEngine
from qmind_quant.strategies.library.ml_strategy import MLStrategy
from qmind_quant.portfolio_management.portfolio import Portfolio
from qmind_quant.execution.execution import SimulatedExecutionHandler
from qmind_quant.ml_models.model_trainer import train_xgboost_model
from qmind_quant.config.paths import DATA_DIR, REPORTS_DIR, FEATURES_DATA_DIR


def run_backtest_and_get_curve(
    model, data_df: pd.DataFrame, tickers: list[str], initial_capital: float
) -> pd.DataFrame:
    """
    A reusable function to run a backtest and return the full equity curve.
    """
    event_manager = EventManager()
    data_handler = HistoricalDataHandler(tickers=tickers, data_df=data_df)
    strategy = MLStrategy(tickers=tickers, event_manager=event_manager, model=model)
    portfolio = Portfolio(event_manager, data_handler, initial_capital)
    execution_handler = SimulatedExecutionHandler(event_manager, data_handler)

    engine = BacktestEngine(
        event_manager, data_handler, strategy, portfolio, execution_handler
    )
    engine.run_backtest()
    return portfolio.get_equity_curve()


def main():
    """
    Main function to run a single demonstration backtest and generate a report.
    This demonstrates that the script can still be run on its own.
    """
    print("--- Running a single demonstration backtest ---")

    full_feature_df = pd.read_parquet(FEATURES_DATA_DIR / "ml_feature_data.parquet")

    print("Training model for single run...")
    model = train_xgboost_model(full_feature_df)

    print("Running backtest...")
    equity_curve = run_backtest_and_get_curve(
        model=model,
        data_df=full_feature_df,
        tickers=["AAPL", "GOOG"],
        initial_capital=100000.0,
    )

    print("Generating performance report...")
    report_name = "single_xgboost_report.html"
    output_path = REPORTS_DIR / report_name
    qs.reports.html(
        equity_curve["returns"], output=str(output_path), title="XGBoost Strategy"
    )
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    main()
