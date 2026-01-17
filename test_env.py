from trading_env import TradingEnvironment
import numpy as np

prices = np.array([100, 101, 102, 99, 98, 105])
env = TradingEnvironment(prices)

state = env.reset()
done = False

while not done:
    action = np.random.choice([0, 1, 2])
    next_state, reward, done = env.step(action)
    print(f"Price: {next_state}, Reward: {reward}")

