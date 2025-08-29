# scripts/run_walk_forward.py

import os
import pandas as pd
import numpy as np
import quantstats as qs
from qmind_quant.data_management.feature_engineer import FeatureEngineer
from qmind_quant.ml_models.model_trainer import train_xgboost_model
from scripts.run_backtest import run_backtest_for_model
from qmind_quant.config.paths import DATA_DIR, REPORTS_DIR


def run_walk_forward_validation(
    full_data_df: pd.DataFrame,
    train_period_days: int,
    test_period_days: int,
    tickers: list[str],
    initial_capital: float,
):
    """
    Performs a full walk-forward validation of the ML strategy.
    """
    print("--- Starting Walk-Forward Validation ---")

    all_dates = pd.to_datetime(sorted(full_data_df["date"].unique()))

    # --- THIS IS THE FIX ---
    # We create the walk-forward windows based on indices, not dates, which is more robust.
    train_window_size = train_period_days
    test_window_size = test_period_days
    step_size = test_window_size  # Slide the window by the test period size

    all_equity_curves = []
    engineer = FeatureEngineer()

    fold_number = 1
    start_index = 0

    while start_index + train_window_size + test_window_size < len(all_dates):
        print(f"\n--- Processing Fold #{fold_number} ---")

        # 1. Define Time Windows using indices
        train_start_index = start_index
        train_end_index = train_start_index + train_window_size
        test_start_index = train_end_index
        test_end_index = test_start_index + test_window_size

        train_start_date = all_dates[train_start_index]
        train_end_date = all_dates[train_end_index]
        test_start_date = all_dates[test_start_index]
        test_end_date = all_dates[test_end_index]

        print(
            f"  Window: Train {train_start_date.date()} - {train_end_date.date()} | Test {test_start_date.date()} - {test_end_date.date()}"
        )

        # 2. Split Data
        train_df = full_data_df[
            (full_data_df["date"] >= train_start_date)
            & (full_data_df["date"] <= train_end_date)
        ]
        test_df = full_data_df[
            (full_data_df["date"] >= test_start_date)
            & (full_data_df["date"] <= test_end_date)
        ]

        print(f"  Train DF shape: {train_df.shape}, Test DF shape: {test_df.shape}")

        if train_df.empty or test_df.empty:
            print("  Skipping fold: Empty data slice.")
            start_index += step_size
            fold_number += 1
            continue

        # 3. Engineer Features and Train Model
        train_feature_df = engineer.create_features(train_df)
        print(f"  Train Feature DF shape after engineering: {train_feature_df.shape}")

        if (
            len(train_feature_df) < 50
        ):  # Check if there's enough data after feature engineering
            print("  Skipping fold: Not enough data to create features for training.")
            start_index += step_size
            fold_number += 1
            continue

        model = train_xgboost_model(train_feature_df)

        # 4. Run Backtest on Out-of-Sample Data
        print(f"  Backtesting on {len(test_df)} out-of-sample bars...")
        out_of_sample_equity = run_backtest_for_model(
            model=model,
            data_df=test_df,
            tickers=tickers,
            initial_capital=initial_capital,
        )
        all_equity_curves.append(out_of_sample_equity)
        print(f"  Fold #{fold_number} complete. Equity curve generated.")

        # 5. Slide Window Forward
        start_index += step_size
        fold_number += 1

    # 6. Stitch Results
    if not all_equity_curves:
        print("\n--- ERROR: No equity curves were generated. ---")
        print(
            "This may be due to the date range being too short for the specified train/test periods."
        )
        return

    print("\n--- Walk-Forward Validation Complete ---")
    full_equity_curve = pd.concat(all_equity_curves)
    full_equity_curve["returns"] = (
        full_equity_curve["total_value"].pct_change().fillna(0.0)
    )

    # 7. Generate Final Report
    print("Generating final performance report...")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_name = "walk_forward_xgboost_report.html"
    output_path = REPORTS_DIR / report_name
    qs.reports.html(
        full_equity_curve["returns"],
        output=str(output_path),
        title="Walk-Forward XGBoost Strategy",
    )
    print(f"Report saved to {output_path}")


def main():
    """Main function to configure and run the walk-forward validation."""
    full_data_file = DATA_DIR / "processed" / "us_equities_daily.parquet"
    full_df = pd.read_parquet(full_data_file)

    # We use number of trading days now, not calendar days
    run_walk_forward_validation(
        full_data_df=full_df,
        train_period_days=252 * 2,  # Approx 2 years of trading days
        test_period_days=126,  # Approx 6 months of trading days
        tickers=["AAPL", "GOOG"],
        initial_capital=100000.0,
    )


if __name__ == "__main__":
    main()
