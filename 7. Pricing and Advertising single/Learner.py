import numpy as np


class Learner:
	
	def __init__(self, n_arms):
		self.n_arms = n_arms
		self.t = 0
		self.rewards_per_arm = [[] for _ in range(n_arms)]
		self.collected_rewards = np.array([])
		
	def update_observations(self, pulled_arm, reward):
		self.rewards_per_arm[pulled_arm].append(reward)
		self.collected_rewards = np.append(self.collected_rewards, reward)
