from SW_GPTS_Learner import *


class Subcampaign:

    def __init__(self, daily_budgets, window_length):
        self.learner = SW_GPTS_Learner(len(daily_budgets), daily_budgets, window_length=window_length)
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

    def play_round(self, optimal_arm_index, reward):
        """
        Play the optimal arm and sample the result from the true function
        :param
        (int) optimal_arm_index: the index of the optimal arm to play
        """

    def update(self, optimal_arm_index, reward):
        """
        Update the parameters of the model and the gaussian process based on an external sampling
        of the real function
        :param optimal_arm_index: (int) the index of the arm that has been pulled
        :param reward: (float) the reward received by pulling the arm
        :return:
        """
        self.pulled_arms_index = np.append(self.pulled_arms_index, optimal_arm_index)
        self.collected_rewards = np.append(self.collected_rewards, reward)

        self.learner.update(optimal_arm_index, reward)

# TESTING PURPOSE ONLY
    def plot(self, ax, fun):
        self.learner.plot(fun, ax)
