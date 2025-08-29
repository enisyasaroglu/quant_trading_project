# qmind_quant/ml_models/environments/trading_env.py

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from collections import deque


class TradingEnv(gym.Env):
    """
    A custom stock trading environment with a Sharpe Ratio-based reward function.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, initial_capital=100000, lookback_window=30):
        super(TradingEnv, self).__init__()

        self.df = df.copy()
        self.initial_capital = initial_capital
        self.lookback_window = lookback_window

        self.action_space = spaces.Discrete(3)  # 0:Hold, 1:Buy, 2:Sell

        num_features = len(self.df.columns) - 2
        observation_shape = (self.lookback_window, num_features + 2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=observation_shape, dtype=np.float32
        )

        # New: A deque to store the history of returns for Sharpe calculation
        self.returns_history = deque(maxlen=self.lookback_window)

        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.cash = self.initial_capital
        self.current_position = 0
        self.total_value = self.initial_capital
        self.current_step = self.lookback_window
        self.returns_history.clear()  # Clear history for the new episode

        return self._get_observation(), {}

    def _get_observation(self):
        frame = self.df.iloc[
            self.current_step - self.lookback_window : self.current_step
        ]
        features = frame.drop(columns=["date", "ticker"]).values
        position_value = self.current_position * frame["close"].iloc[-1]
        portfolio_info = np.array(
            [[self.cash, position_value] for _ in range(self.lookback_window)]
        )
        obs = np.concatenate([portfolio_info, features], axis=1)
        return obs.astype(np.float32)

    def step(self, action):
        current_price = self.df["close"].iloc[self.current_step]

        if action == 1:  # Buy all-in
            shares_to_buy = self.cash / current_price
            self.current_position += shares_to_buy
            self.cash = 0
        elif action == 2:  # Sell all
            self.cash += self.current_position * current_price
            self.current_position = 0

        new_total_value = self.cash + self.current_position * current_price

        # --- THIS IS THE NEW REWARD LOGIC ---
        # Calculate the percentage return for this step
        step_return = (
            (new_total_value / self.total_value) - 1 if self.total_value != 0 else 0
        )
        self.returns_history.append(step_return)

        # Calculate Sharpe Ratio as the reward once we have enough history
        if len(self.returns_history) == self.lookback_window:
            returns_std = np.std(self.returns_history)
            if returns_std > 1e-6:  # Avoid division by zero
                # Annualized Sharpe Ratio (assuming daily steps)
                sharpe_ratio = (
                    np.mean(self.returns_history) / returns_std * np.sqrt(252)
                )
                reward = sharpe_ratio
            else:
                reward = 0.0  # No risk, no reward
        else:
            reward = 0.0  # No reward until we have enough data for a stable calculation

        self.total_value = new_total_value
        self.current_step += 1
        terminated = self.total_value <= 0 or self.current_step >= len(self.df) - 1
        observation = self._get_observation()

        return observation, reward, terminated, False, {}

    def render(self, mode="human"):
        print(
            f"Step: {self.current_step}, Total Value: {self.total_value:.2f}, Position: {self.current_position:.2f}, Cash: {self.cash:.2f}"
        )
