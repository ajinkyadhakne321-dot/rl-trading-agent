import numpy as np
from trading_env import TradingEnvironment

prices = np.array([
    100, 101, 102, 101, 100, 99, 98, 100, 103,
    105, 104, 106, 108, 107, 109, 110
])

env = TradingEnvironment(prices, window_size=3)

state = env.reset()
done = False

portfolio_values = []

while not done:
    action = np.random.choice([0, 1, 2])
    state, reward, done = env.step(action)

    price = prices[env.current_step]
    value = env.cash + env.stock * price
    portfolio_values.append(value)

final_value = portfolio_values[-1]

print("Final Portfolio Value:", final_value)
print("Total Return:", final_value - env.initial_cash)
