# scripts/run_ingestion.py

import os
from qmind_quant.data_management.data_collector import DataCollector
from qmind_quant.config.paths import PROCESSED_DATA_DIR


def main():
    """Downloads a longer history of data and saves it to the processed data directory."""

    # --- Configuration ---
    tickers_to_fetch = ["SPY", "AAPL", "MSFT", "GOOG"]
    # Updated start date to provide a much larger dataset
    start_date = "2018-01-01"
    end_date = "2025-08-28"  # Use a recent date
    output_file = PROCESSED_DATA_DIR / "us_equities_daily.parquet"

    # --- Execution ---
    collector = DataCollector()
    daily_data = collector.fetch_daily_data(
        tickers=tickers_to_fetch, start_date=start_date, end_date=end_date
    )

    if not daily_data.empty:
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        collector.save_to_parquet(daily_data, str(output_file))
    else:
        print("No data was fetched. The output file was not created.")


if __name__ == "__main__":
    main()
