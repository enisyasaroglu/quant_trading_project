# scripts/run_drl_backtest.py

import os
import quantstats as qs
from stable_baselines3 import PPO

from qmind_quant.core.event_manager import EventManager
from qmind_quant.data_management.data_handler import HistoricalDataHandler
from qmind_quant.simulation.backtest_engine import BacktestEngine
from qmind_quant.strategies.library.rl_strategy import RLStrategy
from qmind_quant.portfolio_management.portfolio import Portfolio
from qmind_quant.execution.execution import SimulatedExecutionHandler
from qmind_quant.config.paths import DATA_DIR, MODELS_DIR, REPORTS_DIR  # Use DATA_DIR


def main():
    """
    Main function to configure and run a backtest for the DRL strategy.
    """
    # --- THIS IS THE FIX ---
    # The strategy now needs the raw data, not the pre-calculated feature data.
    data_file = DATA_DIR / "processed" / "us_equities_daily.parquet"
    model_file = MODELS_DIR / "drl_ppo_v1.zip"

    tickers = ["AAPL"]
    initial_capital = 100000.0

    event_manager = EventManager()
    data_handler = HistoricalDataHandler(file_path=str(data_file), tickers=tickers)
    agent = PPO.load(model_file)
    strategy = RLStrategy(tickers, event_manager, agent)
    portfolio = Portfolio(event_manager, data_handler, initial_capital)
    execution_handler = SimulatedExecutionHandler(event_manager, data_handler)

    engine = BacktestEngine(
        event_manager, data_handler, strategy, portfolio, execution_handler
    )
    engine.run_backtest()

    print("Generating performance report...")
    equity_curve = portfolio.get_equity_curve()

    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_name = "drl_ppo_strategy_report.html"
    output_path = REPORTS_DIR / report_name

    qs.reports.html(
        equity_curve["returns"], output=str(output_path), title="DRL PPO Strategy"
    )
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    main()
