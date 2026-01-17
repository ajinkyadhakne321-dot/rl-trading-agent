

# import numpy as np
# import pandas as pd

# class TradingEnvironment:
#     def __init__(self, prices):
#         self.prices = prices
#         self.current_step = 0
#         self.position = 0
#         self.cash = 10000
#         self.stock = 0

#     def reset(self):
#         self.current_step = 0
#         self.position = 0
#         self.cash = 10000
#         self.stock = 0
#         return self.prices[self.current_step]


#     def step(self, action):
#         price = self.prices[self.current_step]

#         if action == 1 and self.position == 0:
#             self.stock = self.cash / price
#             self.cash = 0
#             self.position = 1

#         elif action == 2 and self.position == 1:
#             self.cash = self.stock * price
#             self.stock = 0
#             self.position = 0

#         self.current_step += 1
#         done = self.current_step >= len(self.prices) - 1

#         total_value = self.cash + self.stock * price
#         reward = total_value - 10000

#         return self.prices[self.current_step], reward, done


# Add state representation for RL

# import numpy as np

# class TradingEnvironment:
#     def __init__(self, prices, window_size=5):
#         self.prices = prices
#         self.window_size = window_size
#         self.initial_cash = 10000
#         self.reset()

#     def reset(self):
#         self.current_step = self.window_size
#         self.position = 0
#         self.cash = self.initial_cash
#         self.stock = 0
#         return self._get_state()

#     def _get_state(self):
#         window_prices = self.prices[
#             self.current_step - self.window_size : self.current_step
#         ]

#         returns = np.diff(window_prices) / window_prices[:-1]
#         moving_avg = np.mean(window_prices)
#         volatility = np.std(returns)

#         state = np.concatenate([
#             returns,
#             [moving_avg],
#             [volatility],
#             [self.position]
#         ])

#         return state

#     def step(self, action):
#         price = self.prices[self.current_step]

#         if action == 1 and self.position == 0:
#             self.stock = self.cash / price
#             self.cash = 0
#             self.position = 1

#         elif action == 2 and self.position == 1:
#             self.cash = self.stock * price
#             self.stock = 0
#             self.position = 0

#         self.current_step += 1
#         done = self.current_step >= len(self.prices) - 1

#         total_value = self.cash + self.stock * price
#         reward = total_value - self.initial_cash

#         next_state = self._get_state()

#         return next_state, reward, done


# Add transaction cost + risk-aware reward

import numpy as np

class TradingEnvironment:
    def __init__(self, prices, window_size=5, transaction_cost=0.001, risk_penalty=0.1):
        self.prices = prices
        self.window_size = window_size
        self.transaction_cost = transaction_cost
        self.risk_penalty = risk_penalty
        self.initial_cash = 10000
        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.position = 0
        self.cash = self.initial_cash
        self.stock = 0
        self.max_value = self.initial_cash
        return self._get_state()

    # def _get_state(self):
    #     window_prices = self.prices[
    #         self.current_step - self.window_size : self.current_step
    #     ]

    #     returns = np.diff(window_prices) / window_prices[:-1]
    #     moving_avg = np.mean(window_prices)
    #     volatility = np.std(returns)

    #     state = np.concatenate([
    #         returns,
    #         [moving_avg],
    #         [volatility],
    #         [self.position]
    #     ])

    
        return state

    def step(self, action):
        price = self.prices[self.current_step]
        prev_value = self.cash + self.stock * price

        if action == 1 and self.position == 0:
            self.stock = (self.cash * (1 - self.transaction_cost)) / price
            self.cash = 0
            self.position = 1

        elif action == 2 and self.position == 1:
            self.cash = self.stock * price * (1 - self.transaction_cost)
            self.stock = 0
            self.position = 0

        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1

        total_value = self.cash + self.stock * price
        self.max_value = max(self.max_value, total_value)

        drawdown = (self.max_value - total_value) / self.max_value
        reward = (total_value - prev_value) - self.risk_penalty * drawdown

        next_state = self._get_state()

        return next_state, reward, done


