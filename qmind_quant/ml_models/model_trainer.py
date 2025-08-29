# qmind_quant/ml_models/model_trainer.py

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier


def train_xgboost_model(feature_df: pd.DataFrame):
    """
    Trains an XGBoost classifier on the provided feature DataFrame.

    Args:
        feature_df (pd.DataFrame): The DataFrame containing features and the 'target' column.

    Returns:
        An instance of the trained XGBoost model.
    """
    # Define the feature set to be used for training
    features = [
        "ema_12",
        "ema_26",
        "macd",
        "adx_14",
        "rsi_14",
        "stoch_k_14",
        "bb_width",
        "atr_14",
        "obv",
        "vwap",
    ]

    X_train = feature_df[features]
    y_train = feature_df["target"]

    # Initialize and train the XGBoost classifier
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )

    print(f"  Training model on {len(X_train)} samples...")
    model.fit(X_train, y_train)

    return model
