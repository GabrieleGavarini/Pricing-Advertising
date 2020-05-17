from ThreePhasesScenario import *
from SW_GPTS_Learner import *

time_horizon = 60
phase_duration = time_horizon / 3

subcampaigns = np.empty(3)
for sub_index in len(subcampaigns):
    subcampaigns[sub_index] = Subcampaign()

optimizer = Optimizer()

for t in range(0, time_horizon):

    sampled_values = []

    for s in subcampaigns:
        sampled_values = sampled_values.append(s.sample())

    optimal_arms = optimizer.optimize(sampled_values)

    for sub_index in range(0, len(subcampaigns)):
        s = subcampaigns[sub_index]
        s.play_round()
        s.update()