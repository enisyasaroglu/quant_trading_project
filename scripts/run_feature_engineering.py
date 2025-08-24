# scripts/run_feature_engineering.py

import os
import pandas as pd
from qmind_quant.data_management.feature_engineer import FeatureEngineer


def main():
    """
    Main function to run the feature engineering process.
    """
    # --- Configuration ---
    project_root = os.path.join(os.path.dirname(__file__), "..")
    input_file = os.path.join(project_root, "data/processed/us_equities_daily.parquet")
    output_file = os.path.join(project_root, "data/features/ml_feature_data.parquet")

    # --- Execution ---
    print(f"Loading data from {input_file}...")
    ohlcv_data = pd.read_parquet(input_file)

    engineer = FeatureEngineer()

    print("Starting feature engineering...")
    feature_data = engineer.create_features(ohlcv_data)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Saving feature-rich data to {output_file}...")
    feature_data.to_parquet(output_file, index=False)

    print("Feature engineering complete.")
    print("\n--- Data Head ---")
    print(feature_data.head())
    print("\n--- Data Info ---")
    feature_data.info()


if __name__ == "__main__":
    main()
