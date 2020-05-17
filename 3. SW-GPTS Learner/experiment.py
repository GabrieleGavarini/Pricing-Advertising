from Subcampaign import *
from Optimizer import *
import numpy as np
from matplotlib import pyplot as plt


def phase_1(x):
    return 100 * (1 - np.exp(-5.0 * x ** 2))


def phase_2(x):
    return 800 * (1 - np.exp(-6.0*x))


def phase_3(x):
    return 80 * (1 - np.exp(-5.0*x))


budget_cap = 19
number_of_budgets = budget_cap + 1
daily_budgets = np.linspace(0, budget_cap, number_of_budgets).astype(int)
normalized_daily_budgets = daily_budgets/budget_cap
sigma = 10

time_horizon = 60
phase_duration = time_horizon / 3

window_length = 10

subcampaigns = []

subcampaigns.append(Subcampaign(daily_budgets=normalized_daily_budgets,
                                sigma=sigma,
                                phase_1=phase_1,
                                phase_2=phase_2,
                                phase_3=phase_3,
                                phase_duration=phase_duration,
                                window_length=window_length))

subcampaigns.append(Subcampaign(daily_budgets=normalized_daily_budgets,
                                sigma=sigma,
                                phase_1=phase_2,
                                phase_2=phase_1,
                                phase_3=phase_2,
                                phase_duration=phase_duration,
                                window_length=window_length))

subcampaigns.append(Subcampaign(daily_budgets=normalized_daily_budgets,
                                sigma=sigma,
                                phase_1=phase_3,
                                phase_2=phase_3,
                                phase_3=phase_1,
                                phase_duration=phase_duration,
                                window_length=window_length))

optimizer = Optimizer(daily_budgets, number_of_budgets)

######FOR TESTING ONLY#######
plt.rcParams["figure.figsize"] = [16, 9]
fig, ax = plt.subplots(3, 1)
#####################

for t in range(0, time_horizon):

    sampled_values = []

    for s in subcampaigns:
        sampled_values.append(s.sample())

    optimal_arms = optimizer.optimize(sampled_values)

    optimal_result = 0

    for sub_index in range(0, len(subcampaigns)):
        s = subcampaigns[sub_index]
        optimal_result += s.play_round(optimal_arms[sub_index])
        s.update()

    print('Optimal result:', optimal_result)

    #FOR TESTING ONLY
    plt.close()
    fig, ax = plt.subplots(3, 1)
    subcampaigns[0].plot(ax[0])
    subcampaigns[1].plot(ax[1])
    subcampaigns[2].plot(ax[2])
    plt.pause(1)
    ####
