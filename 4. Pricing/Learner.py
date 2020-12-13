import numpy as np


class Learner:
	
	def __init__(self, n_arms):
		self.n_arms = n_arms
		self.t = 0
		self.rewards_per_arm = [[] for _ in range(n_arms)]
		self.collected_rewards = np.array([])
		
	def update_observations(self, pulled_arm, reward):
		"""
		Updates the beta distribution for the pulled_arm using the value of the reward.
		:param pulled_arm: The index of the arm that has been pulled
		:param reward: The reward gained by playing that arm
		"""
		self.rewards_per_arm[pulled_arm].append(reward)
		self.collected_rewards = np.append(self.collected_rewards, reward)
