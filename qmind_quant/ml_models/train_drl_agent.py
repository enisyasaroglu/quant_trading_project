# qmind_quant/ml_models/train_drl_agent.py

import os
import argparse
import pandas as pd
from stable_baselines3 import PPO

from qmind_quant.ml_models.environments.trading_env import TradingEnv

# Import paths for local execution
from qmind_quant.config.paths import FEATURES_DATA_DIR, MODELS_DIR


def main():
    """
    This script is designed to be run as an AWS SageMaker training job,
    but also supports local execution for testing and debugging.
    """
    parser = argparse.ArgumentParser()

    # --- SageMaker specific arguments (will be None locally) ---
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))

    # --- Model hyperparameters ---
    parser.add_argument(
        "--total_timesteps", type=int, default=20000
    )  # Reduced for quick local testing
    parser.add_argument("--initial_capital", type=int, default=100000)
    parser.add_argument("--ticker", type=str, default="AAPL")

    args, _ = parser.parse_known_args()

    # --- THIS IS THE FIX ---
    # Detect if running in SageMaker or locally
    if args.train:
        # SageMaker environment: Paths are provided by arguments
        print("--- Running in SageMaker environment ---")
        training_data_path = os.path.join(args.train, "ml_feature_data.parquet")
        model_save_dir = args.model_dir
    else:
        # Local environment: Use paths from our config file
        print("--- Running in local environment ---")
        training_data_path = FEATURES_DATA_DIR / "ml_feature_data.parquet"
        model_save_dir = MODELS_DIR

    # --- Load Data ---
    print(f"Loading training data from: {training_data_path}")
    df = pd.read_parquet(training_data_path)

    ticker_df = df[df["ticker"] == args.ticker].reset_index(drop=True)
    print(f"Training on {len(ticker_df)} samples for ticker {args.ticker}.")

    # --- Create the Environment ---
    env = TradingEnv(df=ticker_df, initial_capital=args.initial_capital)

    # --- Initialize and Train the Model ---
    model = PPO("MlpPolicy", env, verbose=1)

    print(f"\n--- Starting DRL Agent Training for {args.total_timesteps} timesteps ---")
    model.learn(total_timesteps=args.total_timesteps)
    print("--- Training Complete ---")

    # --- Save the Model ---
    # Ensure the save directory exists
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, "drl_ppo_v1.zip")
    model.save(model_save_path)
    print(f"\nDRL agent saved to: {model_save_path}")


if __name__ == "__main__":
    main()
