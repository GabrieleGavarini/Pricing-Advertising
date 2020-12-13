from Learner import *
import numpy as np


class TS_Learner(Learner):
	
	def __init__(self, n_arms):
		super().__init__(n_arms)
		self.beta_parameters = np.ones((n_arms, 2))
		
	def pull_arm(self):
		"""
		Get the index of the arm that maximizes the beta distribution
		:return: the index of the arm that maximizes the beta distribution
		"""
		rand = np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1])

		index = np.argmax(rand)

		return index
		
	def update(self, pulled_arm, reward):
		"""
		Updates the beta distribution for the pulled arm using the value of the reward
		:param pulled_arm: The index of the arm that has been pulled
		:param reward: The reward obtained by playing the pulled_arm
		"""
		self.t += 1
		self.update_observations(pulled_arm, reward)
		self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward
		self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1]+1.0 - reward
