# scripts/run_optimization.py

import os
import optuna
from scripts.run_backtest import run_single_backtest

# --- Configuration ---
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_FILE = os.path.join(PROJECT_ROOT, "data/processed/us_equities_daily.parquet")
TICKERS = ["AAPL", "GOOG"]
INITIAL_CAPITAL = 100000.0
MAX_DRAWDOWN_PCT = 0.20  # Loosen drawdown for optimization runs
N_TRIALS = 50  # The number of different parameter combinations to test


def objective(trial: optuna.Trial) -> float:
    """
    The function for Optuna to optimize. It suggests parameters and runs a backtest.
    """
    # Define the search space for our parameters
    short_window = trial.suggest_int("short_window", 5, 50)
    long_window = trial.suggest_int("long_window", 20, 200)

    # Constraint: Ensure short_window is smaller than long_window
    if short_window >= long_window:
        # Prune this trial (tell Optuna this is not a valid combination)
        raise optuna.exceptions.TrialPruned()

    print(f"\n--- Starting Trial #{trial.number} ---")
    print(f"Parameters: short_window={short_window}, long_window={long_window}")

    # Run the backtest with the suggested parameters
    sharpe_ratio = run_single_backtest(
        data_file=DATA_FILE,
        tickers=TICKERS,
        initial_capital=INITIAL_CAPITAL,
        max_drawdown_pct=MAX_DRAWDOWN_PCT,
        short_window=short_window,
        long_window=long_window,
    )

    print(f"Trial #{trial.number} Finished. Sharpe Ratio: {sharpe_ratio:.2f}")

    return sharpe_ratio


def main():
    # We want to maximize the Sharpe Ratio, so the direction is 'maximize'
    study = optuna.create_study(direction="maximize")

    # Start the optimization process
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n--- Optimization Finished ---")
    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value (Sharpe Ratio): {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
