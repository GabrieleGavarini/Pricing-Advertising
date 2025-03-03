import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RBF, ConstantKernel as C


class SW_GPTS_Learner:

    def __init__(self, arms, sigma=5, window_length=0):
        self.arms = arms
        self.n_arms = len(arms)
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms) * sigma

        self.pulled_arms = np.array([])
        self.collected_rewards = np.array([])

        self.window_length = window_length

        alpha = 1.5

        theta = 1
        l = 1

        kernel = C(theta, (1e-5, 1e5)) * RBF(l, (1e-10, 1e10))

        self.gp = GaussianProcessRegressor(kernel=kernel,
                                           alpha=alpha ** 2,
                                           normalize_y=True,
                                           n_restarts_optimizer=10)

    def sample_values(self):
        """
        Sample the value of all the arms from the learner Gaussian Process
        :return:
        (numpy.array) An array containing the sampled values
        """
        sampled_values = np.random.normal(self.means, self.sigmas)
        return sampled_values

    def update_observations(self, arm_index, reward):
        """
        Update the parameters based on an external sampling on the real function

        Parameters:
        arm_index (int): the index of the arm that has been pulled
        reward (float): the reward received by pulling the arm
        """
        self.pulled_arms = np.append(self.pulled_arms, arm_index)
        self.collected_rewards = np.append(self.collected_rewards, reward)

        if len(self.pulled_arms) > self.window_length:
            self.pulled_arms = self.pulled_arms[-self.window_length:]
            self.collected_rewards = self.collected_rewards[-self.window_length:]

    def update_model(self):
        """
        Update the gaussian process based on the recent observations
        """
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards

        if len(y) == 1:
            return

        self.gp.fit(x, y)
        self.means = self.gp.predict(np.array(list(np.ndindex(len(self.arms)))))
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update(self, arm_index, reward):
        """
        Update the parameters of the model and the gaussian process based on an external sampling
        of the real function

        Parameters:
            arm_index (int): the index of the arm that has been pulled
            reward (float): the reward received by pulling the arm
        """
        self.update_observations(arm_index, reward)
        self.update_model()

    def pull_arm(self):
        """
        Choose which arm to pull based on the result of a sampling of the gaussian process

        Returns:
        (int): the index of the arm that maximizes the observation
        """
        sampled_values = np.random.normal(self.means, self.sigmas)
        return np.argmax(sampled_values)

    # FOR TESTING PURPOSES
    def plot(self, ax, function):
        ax.plot(self.arms, function(self.arms))

        ax.plot(self.pulled_arms, self.collected_rewards, 'ro')

        ax.plot(self.arms, self.means)

    def predict(self, point):
        return self.gp.predict([[point]])
