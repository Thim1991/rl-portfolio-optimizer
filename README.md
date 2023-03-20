# Reinforcement Learning Portfolio Optimizer

This project implements a reinforcement learning agent to optimize stock trading strategies and manage a financial portfolio. The agent learns to make buy, sell, or hold decisions based on historical market data.

## Features

- **Customizable Environment**: OpenAI Gym-compatible environment for stock trading.
- **RL Agent Integration**: Designed to integrate with various reinforcement learning algorithms (e.g., Q-learning, DQN, PPO).
- **Performance Tracking**: Monitors portfolio net worth, balance, and holdings.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Environment Setup Example

```python
import pandas as pd
import numpy as np
from portfolio_optimizer import StockTradingEnv

# Create dummy data for demonstration
np.random.seed(42)
dates = pd.date_range(start="2022-01-01", periods=100, freq="D")
prices = np.random.rand(100) * 100 + 50 # Prices between 50 and 150
dummy_df = pd.DataFrame({"Close": prices}, index=dates)

env = StockTradingEnv(df=dummy_df, window_size=10, render_mode="human")
obs, info = env.reset()

for _ in range(10):
    action = env.action_space.sample() # Take a random action
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done:
        print("Episode finished.")
        break
env.close()
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License.
