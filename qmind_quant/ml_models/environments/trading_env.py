# qmind_quant/ml_models/environments/trading_env.py

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class TradingEnv(gym.Env):
    """
    A custom stock trading environment for reinforcement learning, compliant with the Gymnasium API.
    This environment allows a DRL agent to interact with historical market data.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, initial_capital=100000, lookback_window=30):
        """
        Initializes the trading environment.

        Args:
            df (pd.DataFrame): A DataFrame containing feature-rich historical data for a single stock.
            initial_capital (int): The starting cash balance.
            lookback_window (int): The number of past timesteps to include in each observation.
        """
        super(TradingEnv, self).__init__()

        self.df = df.copy()
        self.initial_capital = initial_capital
        self.lookback_window = lookback_window

        # Define action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)

        # Define observation space (the state)
        # It consists of a window of past data, plus the agent's current cash and position value.
        # We add 2 to the number of features for these two portfolio metrics.
        num_features = len(self.df.columns) - 2  # Excluding 'date' and 'ticker'
        observation_shape = (self.lookback_window, num_features + 2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=observation_shape, dtype=np.float32
        )

        # Initialize the state of the environment
        self.reset()

    def reset(self, seed=None):
        """
        Resets the environment to its initial state for a new episode.
        """
        super().reset(seed=seed)

        self.cash = self.initial_capital
        self.current_position = 0  # Number of shares held
        self.total_value = self.initial_capital
        self.current_step = self.lookback_window

        return self._get_observation(), {}

    def _get_observation(self):
        """
        Constructs the observation array for the current timestep.
        """
        # Get the slice of data for the current lookback window
        frame = self.df.iloc[
            self.current_step - self.lookback_window : self.current_step
        ]

        # Extract market features (e.g., returns, volatility, RSI)
        features = frame.drop(columns=["date", "ticker"]).values

        # Get current portfolio state
        position_value = self.current_position * frame["close"].iloc[-1]

        # Create an array representing the portfolio state for each step in the window
        portfolio_info = np.array(
            [[self.cash, position_value] for _ in range(self.lookback_window)]
        )

        # Combine portfolio info and market features into the final observation
        obs = np.concatenate([portfolio_info, features], axis=1)
        return obs.astype(np.float32)

    def step(self, action):
        """
        Executes one time step within the environment.
        """
        current_price = self.df["close"].iloc[self.current_step]

        # Execute the chosen action
        if action == 1:  # Buy (all-in for simplicity)
            shares_to_buy = self.cash / current_price
            self.current_position += shares_to_buy
            self.cash = 0
        elif action == 2:  # Sell (all shares)
            self.cash += self.current_position * current_price
            self.current_position = 0

        # Calculate the reward for the action
        new_total_value = self.cash + self.current_position * current_price
        reward = new_total_value - self.total_value
        self.total_value = new_total_value

        # Move to the next time step
        self.current_step += 1

        # Determine if the episode is finished (terminated)
        terminated = self.total_value <= 0 or self.current_step >= len(self.df) - 1

        observation = self._get_observation()

        # Return the standard 5-tuple for a gymnasium step
        return observation, reward, terminated, False, {}

    def render(self, mode="human"):
        """Renders the environment's current state to the console."""
        print(
            f"Step: {self.current_step}, Total Value: {self.total_value:.2f}, Position: {self.current_position:.2f}, Cash: {self.cash:.2f}"
        )
