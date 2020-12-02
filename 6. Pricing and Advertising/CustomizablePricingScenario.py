from PricingScenario import *


class CustomizablePricingScenario(PricingScenario):

    def __init__(self, sex=None, under_30=None, file_name="../Scenarios/User_pricing.csv"):
        self.file_name = file_name

        self.df = pd.read_csv(self.file_name)

        if sex is not None:
            self.df = self.df[self.df.Sex == sex]

        if under_30 is not None:
            self.df = self.df[self.df.Under_30 == under_30]

        self.arms = self.df.price.unique()
        self.n_arms = self.arms.size