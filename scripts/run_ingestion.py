# scripts/run_ingestion.py

import os
from qmind_quant.data_management.data_collector import DataCollector

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    """
    Main function to run the data ingestion process.
    """
    # --- Configuration ---
    tickers_to_fetch = ["SPY", "AAPL", "MSFT", "GOOG"]
    start_date = "2024-01-01"
    end_date = "2024-08-22"  # Using a recent date for end_date
    output_file = os.path.join(project_root, "data/processed/us_equities_daily.parquet")

    # --- Execution ---
    collector = DataCollector()

    # Fetch the data
    daily_data = collector.fetch_daily_data(
        tickers=tickers_to_fetch, start_date=start_date, end_date=end_date
    )

    # Save the data
    if not daily_data.empty:
        collector.save_to_parquet(daily_data, output_file)
    else:
        print("No data was fetched. The output file was not created.")


if __name__ == "__main__":
    main()
