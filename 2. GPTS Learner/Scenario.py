import numpy as np

class Scenario:

    def __init__(self, daily_budgets, sigma, fun):

        self.fun = fun
        self.daily_budgets = daily_budgets

        self.mean = self.fun(self.daily_budgets)

        self.sigma = sigma

    def play_round(self, daily_budget_index):
        mean = self.fun(self.daily_budgets[daily_budget_index])
        return max(0, np.random.normal(mean, self.sigma))