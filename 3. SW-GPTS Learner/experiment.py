from SW_GPTS_Learner import *
from ThreePhasesScenario import *
from Optimizer import *
import numpy as np
from matplotlib import pyplot as plt


def phase_1(x): return 100 * (1 - np.exp(-5.0 * x ** 2))


def phase_2(x): return 800 * (1 - np.exp(-6.0*x))


def phase_3(x): return 80 * (1 - np.exp(-5.0*x))


budget_cap = 19
number_of_budgets = (budget_cap + 1)
daily_budgets = np.linspace(0, budget_cap, number_of_budgets).astype(int)
normalized_daily_budgets = daily_budgets/budget_cap
sigma = 0

time_horizon = 1000
phase_duration = time_horizon / 3

window_length = 100

scenario = [ThreePhasesScenario(daily_budgets=normalized_daily_budgets,
                                sigma=sigma,
                                phase_1=phase_1,
                                phase_2=phase_2,
                                phase_3=phase_3,
                                phase_duration=phase_duration),
            ThreePhasesScenario(daily_budgets=normalized_daily_budgets,
                                sigma=sigma,
                                phase_1=phase_2,
                                phase_2=phase_3,
                                phase_3=phase_1,
                                phase_duration=phase_duration),
            ThreePhasesScenario(daily_budgets=normalized_daily_budgets,
                                sigma=sigma,
                                phase_1=phase_3,
                                phase_2=phase_1,
                                phase_3=phase_2,
                                phase_duration=phase_duration)]

subcampaigns = [SW_GPTS_Learner(arms=normalized_daily_budgets,
                                sigma=sigma,
                                window_length=window_length),
                SW_GPTS_Learner(arms=normalized_daily_budgets,
                                sigma=sigma,
                                window_length=window_length),
                SW_GPTS_Learner(arms=normalized_daily_budgets,
                                sigma=sigma,
                                window_length=window_length)]

optimizer = Optimizer(daily_budgets, number_of_budgets)

######FOR TESTING ONLY#######
plt.rcParams["figure.figsize"] = [16, 9]
fig, ax = plt.subplots(3, 1)
#####################

for t in range(0, time_horizon):

    sampled_values = []

    # Sample the Gaussian Process for each subcampaign
    for s in subcampaigns:
        sampled_values.append(s.sample_values())

    # Find the best combination of arms to play based on the sampling of the Gaussian Processes
    optimal_arms = optimizer.optimize(sampled_values)

    optimal_result = 0

    for sub_index in range(0, len(subcampaigns)):
        # Sample the true function to get a reward for the played arm
        reward = scenario[sub_index].play_round(optimal_arms[sub_index])
        optimal_result += reward

        # Update the model based on the result of the played arm
        subcampaigns[sub_index].update(optimal_arms[sub_index], reward)

    print('Optimal solution:', optimal_arms, "with a result of: ", round(optimal_result, 2))

    #FOR TESTING ONLY
    plt.close()
    fig, ax = plt.subplots(3, 1)
    subcampaigns[0].plot(ax[0], scenario[0].fun)
    subcampaigns[1].plot(ax[1], scenario[1].fun)
    subcampaigns[2].plot(ax[2], scenario[2].fun)
    plt.pause(1)
    ####
