import numpy as np


class ThreePhasesScenario:

    def __init__(self, daily_budgets, sigma, phase_1, phase_2, phase_3, phase_duration):
        self.daily_budgets = daily_budgets
        
        self.sigma = sigma
        
        self.phase_duration = phase_duration
        self.phase = 1
        self.t = 0

        self.phase_1 = phase_1
        self.phase_2 = phase_2
        self.phase_3 = phase_3

    def fun(self, x):
        if self.phase == 1:
            return self.phase_1(x)
        elif self.phase == 2:
            return self.phase_2(x)
        elif self.phase == 3:
            return self.phase_3(x)

    def play_round(self, daily_budget_index):
        
        self.t += 1
        if self.t > self.phase_duration * self.phase:
            self.phase += 1
            
        mean = self.fun(self.daily_budgets[daily_budget_index])
        return max(0, np.random.normal(mean, self.sigma))
