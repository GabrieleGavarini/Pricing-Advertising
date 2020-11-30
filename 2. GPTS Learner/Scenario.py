import pandas as pd
import numpy as np


class Scenario:

    def __init__(self, daily_budgets, campaign, var=5, file_name="../Scenarios/User_pricing.csv"):

        self.daily_budgets = daily_budgets

        self.file_name = file_name
        self.df = pd.read_csv(self.file_name)

        count_df = self.df.groupby(['Ad_campaign', 'Budget']).count()
        self.y = count_df.loc[campaign].price.values

        self.y = np.append(0, self.y)
        self.y = np.append(0, self.y)
        self.y = np.append(0, self.y)

        self.var = var

    def play_round(self, daily_budget_index):
        return np.random.normal(self.y[daily_budget_index], self.var)
