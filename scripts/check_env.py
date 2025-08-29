# scripts/check_env.py

import pandas as pd
from stable_baselines3.common.env_checker import check_env
from qmind_quant.ml_models.environments.trading_env import TradingEnv
from qmind_quant.config.paths import FEATURES_DATA_DIR


def main():
    """
    Loads data and uses the Stable-Baselines3 checker to verify that the
    custom TradingEnv is correctly implemented and follows the Gymnasium API.
    """
    print("Loading data for environment check...")
    feature_file = FEATURES_DATA_DIR / "ml_feature_data.parquet"
    df = pd.read_parquet(feature_file)

    # An environment is typically trained on a single asset at a time.
    # We'll use AAPL data to perform the check.
    aapl_df = df[df["ticker"] == "AAPL"].reset_index(drop=True)

    print("Creating and checking the environment...")
    env = TradingEnv(df=aapl_df)

    # This function will raise an informative error if the environment is not valid.
    check_env(env)

    print(
        "\n✅ Environment check passed! Your custom environment is ready for training."
    )


if __name__ == "__main__":
    main()
