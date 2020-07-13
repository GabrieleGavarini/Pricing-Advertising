from Context import *

import numpy as np

class ContextLearner:

    def __init__(self, n_arms):
        self.n_arms = n_arms

        self.context_list = []
        self.context_list.append(Context(self.n_arms))

    def split(self, partial_observations, attr_list, attribute_dict={}):

        if len(attr_list) == 0:
            return

        partial_observations_grouped = partial_observations.groupby("Price").sum()
        mu = np.max(partial_observations_grouped["Converted"] / partial_observations_grouped["Total"])

        info = []

        for attr in attr_list:
            attr_index = attr_list.index(attr)

            obs_attr_0 = partial_observations[partial_observations[attr_list[attr_index]] == 0].groupby("Price").sum()
            mu_attr_0 = np.max(obs_attr_0["Converted"] / obs_attr_0["Total"])

            obs_attr_1 = partial_observations[partial_observations[attr_list[attr_index]] == 1].groupby("Price").sum()
            mu_attr_1 = np.max(obs_attr_1["Converted"] / obs_attr_1["Total"])

            p = partial_observations[partial_observations[attr_list[attr_index]] == 1].size / partial_observations.size

            attr_info = mu_attr_0 * (1 - p) + mu_attr_1 * p
            info.append(attr_info)

        candidate_info = np.max(info)
        candidate_attr = attr_list[np.argmax(info)]

        if candidate_info > mu:

            attr_dict_0 = attribute_dict.copy()
            attr_dict_0[candidate_attr] = 0
            attr_dict_1 = attribute_dict.copy()
            attr_dict_1[candidate_attr] = 1

            del self.context_list[-1]
            self.context_list.append(Context(n_arms=self.n_arms, dict=attr_dict_0))
            self.context_list.append(Context(n_arms=self.n_arms, dict=attr_dict_1))

            attr_list.remove(candidate_attr)
            self.split(partial_observations[partial_observations[candidate_attr] == 0],
                       attr_list=attr_list,
                       attribute_dict=attr_dict_0)
            self.split(partial_observations[partial_observations[candidate_attr] == 1],
                       attr_list=attr_list,
                       attribute_dict=attr_dict_1)

        return self.context_list