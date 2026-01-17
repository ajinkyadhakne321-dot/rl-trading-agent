import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from trading_env import TradingEnvironment

prices = np.array([
    100, 101, 102, 101, 100, 99, 98, 100, 103,
    105, 104, 106, 108, 107, 109, 110
])

env = TradingEnvironment(prices, window_size=3)

state_size = len(env.reset())
action_size = 3

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.net(x)

model = DQN(state_size, action_size)
target_model = DQN(state_size, action_size)
target_model.load_state_dict(model.state_dict())

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

memory = deque(maxlen=10000)

gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
episodes = 300
target_update = 10

def choose_action(state):
    if np.random.rand() < epsilon:
        return np.random.choice(action_size)
    state = torch.FloatTensor(state).unsqueeze(0)
    return torch.argmax(model(state)).item()

def replay():
    if len(memory) < batch_size:
        return

    batch = np.random.choice(len(memory), batch_size, replace=False)
    states, actions, rewards, next_states, dones = zip(*[memory[i] for i in batch])

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards).unsqueeze(1)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones).unsqueeze(1)

    q_values = model(states).gather(1, actions)
    next_q_values = target_model(next_states).max(1)[0].unsqueeze(1)

    target = rewards + gamma * next_q_values * (1 - dones)

    loss = loss_fn(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

for episode in range(episodes):
    state = env.reset()
    done = False

    while not done:
        action = choose_action(state)
        next_state, reward, done = env.step(action)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        replay()

    if episode % target_update == 0:
        target_model.load_state_dict(model.state_dict())

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

final_value = env.cash + env.stock * prices[env.current_step]
print("Final Portfolio Value:", final_value)
print("Total Return:", final_value - env.initial_cash)
