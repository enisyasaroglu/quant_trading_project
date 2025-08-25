# scripts/run_backtest.py

import os
import quantstats as qs
import pandas as pd
from qmind_quant.core.event_manager import EventManager
from qmind_quant.data_management.data_handler import HistoricalDataHandler
from qmind_quant.simulation.backtest_engine import BacktestEngine
from qmind_quant.strategies.library.ma_crossover_strategy import (
    MovingAverageCrossoverStrategy,
)
from qmind_quant.portfolio_management.portfolio import Portfolio
from qmind_quant.execution.execution import SimulatedExecutionHandler


def run_single_backtest(
    data_file: str,
    tickers: list[str],
    initial_capital: float,
    max_drawdown_pct: float,
    short_window: int,
    long_window: int,
) -> float:
    """
    Runs a single backtest with a given set of parameters and returns the Sharpe Ratio.
    """
    event_manager = EventManager()
    data_handler = HistoricalDataHandler(file_path=data_file, tickers=tickers)
    strategy = MovingAverageCrossoverStrategy(
        tickers, event_manager, short_window, long_window
    )
    portfolio = Portfolio(
        event_manager, data_handler, initial_capital, max_drawdown_pct
    )
    execution_handler = SimulatedExecutionHandler(event_manager, data_handler)

    engine = BacktestEngine(
        event_manager, data_handler, strategy, portfolio, execution_handler
    )

    engine.run_backtest()

    equity_curve = portfolio.get_equity_curve()

    # Calculate Sharpe Ratio, return 0 if there are no trades
    if equity_curve["returns"].std() == 0:
        return 0.0

    sharpe = qs.stats.sharpe(equity_curve["returns"])
    return sharpe if pd.notna(sharpe) else 0.0


# If you still want to run a single backtest directly, you can add a main block like this:
if __name__ == "__main__":
    project_root = os.path.join(os.path.dirname(__file__), "..")
    data_file_path = os.path.join(
        project_root, "data/processed/us_equities_daily.parquet"
    )

    # Run a single instance with default parameters
    final_sharpe = run_single_backtest(
        data_file=data_file_path,
        tickers=["AAPL", "GOOG"],
        initial_capital=100000.0,
        max_drawdown_pct=0.15,
        short_window=10,
        long_window=30,
    )
    print(f"\nSingle run complete. Final Sharpe Ratio: {final_sharpe:.2f}")
