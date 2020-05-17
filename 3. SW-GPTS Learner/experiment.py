from Subcampaign import *
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

daily_budgets = np.linspace(0.0, 1.0, 20)
sigma = 10

time_horizon = 60
phase_duration = time_horizon / 3

window_length = 10

subcampaigns = []
for sub_index in range(0, 3):
    subcampaigns.append(Subcampaign(daily_budgets=daily_budgets,
                                    sigma=sigma,
                                    phase_duration=phase_duration,
                                    window_length=window_length))

# optimizer = Optimizer()

######FOR TESTING ONLY#######
plt.rcParams["figure.figsize"] = [16, 9]
fig, ax = plt.subplots(3, 1)
#####################

for t in range(0, time_horizon):

    sampled_values = np.array([])

    for s in subcampaigns:
        sampled_values = np.append(sampled_values, s.sample())

    # optimal_arms = optimizer.optimize(sampled_values)


    ####################FOR TESTING ONLY######################
    optimal_arms = [1, 1, 1]

    for arm_index in range(0, len(optimal_arms)):
        optimal_arms[arm_index] = np.random.randint(0, 20)
    ##########################################################


    for sub_index in range(0, len(subcampaigns)):
        s = subcampaigns[sub_index]
        s.play_round(optimal_arms[sub_index])
        s.update()

    #FOR TESTING ONLY
    plt.close()
    fig, ax = plt.subplots(3, 1)
    subcampaigns[0].plot(ax[0])
    subcampaigns[1].plot(ax[1])
    subcampaigns[2].plot(ax[2])
    plt.pause(1)
    ####
