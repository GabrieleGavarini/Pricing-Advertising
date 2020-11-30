import pandas as pd
import numpy as np


class ThreePhasesScenario:

    def __init__(self, daily_budgets, campaign, phase_duration, var=5, file_name="../Scenarios/User_pricing.csv"):

        self.phase_duration = phase_duration
        self.phase = 0
        self.t = 0

        self.daily_budgets = daily_budgets

        self.file_name = file_name
        self.df = pd.read_csv(self.file_name)

        count_df = self.df.groupby(['Ad_campaign', 'Phase', 'Budget']).count()

        self.y = count_df.loc[campaign]
        self.phase_values = self.y.loc[self.phase].price.values

        self.var = var

    def play_round(self, daily_budget_index):
        try:
            return np.random.normal(self.phase_values[daily_budget_index], self.var)
        except IndexError:
            return 0

    def advance_time(self):
        self.t += 1
        if self.t > self.phase_duration * (self.phase + 1):
            self.advance_phase()

    def advance_phase(self):
        self.phase += 1
        self.phase = min(2, self.phase)

        self.phase_values = self.y.loc[self.phase].price.values
