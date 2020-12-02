import pandas as pd
import numpy as np

from matplotlib import pyplot as plt


class PricingScenario:

    def __init__(self, file_name="../Scenarios/User_pricing.csv"):
        self.file_name = file_name

        self.df = pd.read_csv(self.file_name)

        self.arms = self.df.price.unique()
        self.n_arms = self.arms.size

    def get_optimal_arm(self):
        converted = self.df[(self.df.Converted == 1)].groupby("price").sum().Converted
        total = self.df.sum().Converted

        return self.arms[np.argmax(converted/total)]

    def round(self, pulled_arm):
        return [
                   pulled_arm,
                   self.df[(self.df.price == pulled_arm) & (self.df.Converted == 1)].count()[0],
                   self.df[(self.df.price == pulled_arm)].count()[0]
        ]
