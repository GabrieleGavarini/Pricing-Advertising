from TS_Learner import *

class Context:

    def __init__(self, n_arms, dict=None):
        self.TS = TS_Learner(n_arms=n_arms)

        if dict is None:
            self.sex = None
            self.age = None
        else:
            if "Sex" in dict.keys():
                self.sex = dict["Sex"]
            else:
                self.sex = None

            if "Under_30" in dict.keys():
                self.age = dict["Under_30"]
            else:
                self.age = None