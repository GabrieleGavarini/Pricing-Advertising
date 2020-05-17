import numpy as np


class ThreePhasesScenario:

    def __init__(self, daily_budgets, sigma, phase_duration):
        self.daily_budgets = daily_budgets
        
        self.sigma = sigma
        
        self.phase_duration = phase_duration
        self.phase = 1
        self.t = 0

    def phase1(self, x):
        return 100 * (1 - np.exp(-5.0*x**2))

    def phase2(self, x):
        return 100 * (1 - np.exp(-6.0*x))

    def phase3(self, x):
        return 80 * (1 - np.exp(-5.0*x))

    def fun(self, x):
        if self.phase == 1:
            return self.phase1(x)
        elif self.phase == 2:
            return self.phase2(x)
        elif self.phase == 3:
            return self.phase3(x)

    def round(self, daily_budget):
        
        self.t += 1
        if self.t > self.phase_duration * self.phase:
            self.phase += 1
            
        mean = self.fun(daily_budget)
        return np.random.normal(mean, self.sigma)
