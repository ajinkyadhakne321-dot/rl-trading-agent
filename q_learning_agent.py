import numpy as np
from trading_env import TradingEnvironment

prices = np.array([
    100, 101, 102, 101, 100, 99, 98, 100, 103,
    105, 104, 106, 108, 107, 109, 110
])

env = TradingEnvironment(prices, window_size=3)

state_size = len(env.reset())
action_size = 3

bins = 10
state_bins = [np.linspace(-1, 1, bins) for _ in range(state_size)]

def discretize(state):
    return tuple(
        np.digitize(state[i], state_bins[i]) for i in range(state_size)
    )

q_table = {}

alpha = 0.1
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
episodes = 500

for episode in range(episodes):
    state = discretize(env.reset())
    done = False

    while not done:
        if np.random.rand() < epsilon:
            action = np.random.choice(action_size)
        else:
            action = np.argmax(q_table.get(state, np.zeros(action_size)))

        next_state_raw, reward, done = env.step(action)
        next_state = discretize(next_state_raw)

        q_values = q_table.get(state, np.zeros(action_size))
        next_q_values = q_table.get(next_state, np.zeros(action_size))

        q_values[action] = q_values[action] + alpha * (
            reward + gamma * np.max(next_q_values) - q_values[action]
        )

        q_table[state] = q_values
        state = next_state

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

final_value = env.cash + env.stock * prices[env.current_step]
print("Final Portfolio Value:", final_value)
print("Total Return:", final_value - env.initial_cash)
