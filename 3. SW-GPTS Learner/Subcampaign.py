from ThreePhasesScenario import *
from SW_GPTS_Learner import *


class Subcampaign:

    def __init__(self, daily_budgets, sigma, phase_duration, window_length):
        self.scenario = ThreePhasesScenario(daily_budgets, sigma, phase_duration)
        self.learner = SW_GPTS_Learner(len(daily_budgets), daily_budgets, window_length)
        self.daily_budgets = daily_budgets

        self.pulled_arms_index = np.array([], dtype=int)
        self.collected_rewards = np.array([])

    def sample(self):
        """
        Sample the number of clicks for every possible daily budget drawing from the learner gaussian process.
        :return:
        (numpy.array) An array containing the sampled number of clicks for every daily budget
        """
        return self.learner.sample_values()

    def play_round(self, optimal_arm_index):
        """
        Play the optimal arm and sample the result from the true function
        :parameter:
        (int) optimal_arm_index: the index of the optimal arm to play
        """
        reward = self.scenario.round(self.daily_budgets[optimal_arm_index])

        self.pulled_arms_index = np.append(self.pulled_arms_index, optimal_arm_index)
        self.collected_rewards = np.append(self.collected_rewards, reward)

    def update(self):
        """
        Update the learner Gaussian Process based on the results of the last played arm
        """
        self.learner.update(self.pulled_arms_index[-1], self.collected_rewards[-1])

# TESTING PURPOSE ONLY
    def plot(self, ax):
        self.learner.plot(self.scenario.fun, ax)
