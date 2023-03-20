import numpy as np
import pandas as pd
import gym
from gym import spaces

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, df, window_size, render_mode=None):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.window_size = window_size
        self.render_mode = render_mode

        # Action space: Buy, Sell, Hold (for each stock)
        # For simplicity, let's assume one stock for now
        self.action_space = spaces.Discrete(3) # 0: Hold, 1: Buy, 2: Sell

        # Observation space: Stock prices, current portfolio value, etc.
        # For simplicity, let's use a window of past prices
        self.observation_space = spaces.Box(low=0, high=1, shape=(window_size, 1), dtype=np.float32)

        self.current_step = window_size
        self.holdings = 0
        self.balance = 10000 # Starting balance
        self.net_worth = self.balance
        self.max_net_worth = self.balance

    def _get_observation(self):
        obs = self.df["Close"].iloc[self.current_step - self.window_size : self.current_step].values
        return obs / self.balance # Normalize observation

    def _get_info(self):
        return {"net_worth": self.net_worth, "holdings": self.holdings, "balance": self.balance}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.holdings = 0
        self.balance = 10000
        self.net_worth = self.balance
        self.max_net_worth = self.balance
        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self, action):
        self.current_step += 1
        if self.current_step > len(self.df) - 1:
            self.current_step = len(self.df) - 1 # End of data

        current_price = self.df["Close"].iloc[self.current_step]

        # Apply action
        if action == 1: # Buy
            if self.balance > current_price:
                self.holdings += 1
                self.balance -= current_price
        elif action == 2: # Sell
            if self.holdings > 0:
                self.holdings -= 1
                self.balance += current_price

        self.net_worth = self.balance + self.holdings * current_price

        # Calculate reward
        reward = self.net_worth - self.max_net_worth
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

        # Check if episode is done
        done = self.net_worth <= self.balance / 2 or self.current_step == len(self.df) - 1

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, done, False, info

    def render(self):
        if self.render_mode == "human":
            print(f"Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, Holdings: {self.holdings}, Balance: {self.balance:.2f}")

    def close(self):
        pass

if __name__ == "__main__":
    # Create dummy data for demonstration
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", periods=100, freq="D")
    prices = np.random.rand(100) * 100 + 50 # Prices between 50 and 150
    dummy_df = pd.DataFrame({"Close": prices}, index=dates)

    env = StockTradingEnv(df=dummy_df, window_size=10, render_mode="human")
    obs, info = env.reset()

    for _ in range(50):
        action = env.action_space.sample() # Take a random action
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        if done:
            print("Episode finished.")
            break
    env.close()
