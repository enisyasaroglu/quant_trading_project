# qmind_quant/optimization/portfolio_optimizer.py

import pandas as pd
import quantstats as qs
from qmind_quant.ml_models.model_trainer import train_xgboost_model

# Important: We import the function from the script, not the other way around
from scripts.run_backtest import run_backtest_and_get_curve
from qmind_quant.config.paths import FEATURES_DATA_DIR


class PortfolioOptimizer:
    """
    This class is responsible for evaluating the performance of a combined
    portfolio of multiple strategies given a specific capital allocation.
    """

    def __init__(self, tickers: list[str], initial_capital: float):
        self.tickers = tickers
        self.initial_capital = initial_capital

        # Load the data once
        full_feature_df = pd.read_parquet(FEATURES_DATA_DIR / "ml_feature_data.parquet")

        # --- Run backtests for each strategy ONCE to get their returns ---
        print("--- Pre-calculating individual strategy returns... ---")

        # 1. XGBoost Strategy
        print("Running backtest for XGBoost Strategy...")
        xgb_model = train_xgboost_model(full_feature_df)
        xgb_equity_curve = run_backtest_and_get_curve(
            model=xgb_model,
            data_df=full_feature_df.copy(),
            tickers=self.tickers,
            initial_capital=self.initial_capital,
        )
        self.xgb_returns = xgb_equity_curve["returns"]

        # 2. Add other strategies here in the future (e.g., MA Crossover)
        # For now, we'll compare the XGBoost strategy to a simple Buy-and-Hold
        print("Calculating returns for Buy-and-Hold Strategy...")
        # For simplicity, we use SPY as the benchmark for Buy-and-Hold
        spy_df = pd.read_parquet(FEATURES_DATA_DIR / "ml_feature_data.parquet")
        spy_df = spy_df[spy_df["ticker"] == "SPY"].set_index("date")
        self.bnh_returns = spy_df["close"].pct_change().fillna(0.0)

        print("--- Pre-calculation complete. Optimizer is ready. ---")

    def evaluate(self, weights: list[float]) -> float:
        """
        The 'scorecard' function. Takes a list of weights and returns the
        portfolio's Sharpe Ratio.

        Args:
            weights (list[float]): A list of weights, e.g., [w_xgb, w_bnh].
                                   Must sum to 1.
        """
        # Combine the individual strategy returns using the given weights
        # We need to align the indexes to handle any date mismatches
        combined_returns = pd.DataFrame(
            {"xgb": self.xgb_returns, "bnh": self.bnh_returns}
        ).dropna()

        # The core portfolio math: weighted sum of returns
        portfolio_returns = (combined_returns * weights).sum(axis=1)

        # Calculate the Sharpe Ratio for the combined portfolio
        # We return a negative Sharpe because the PSO algorithm minimizes by default.
        # So, minimizing a negative Sharpe is the same as maximizing a positive one.
        sharpe = qs.stats.sharpe(portfolio_returns)
        return -sharpe if pd.notna(sharpe) else 0


# --- A small test block to ensure it works ---
if __name__ == "__main__":
    optimizer = PortfolioOptimizer(tickers=["AAPL", "GOOG"], initial_capital=100000.0)

    # Test a 50/50 allocation
    test_weights = [0.5, 0.5]
    score = optimizer.evaluate(test_weights)

    print(f"\nTest run with weights {test_weights}")
    print(f"Resulting (negative) Sharpe Ratio: {score:.4f}")
