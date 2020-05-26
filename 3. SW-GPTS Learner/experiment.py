from SW_GPTS_Learner import *
from ThreePhasesScenario import *
from Optimizer import *

from tqdm import tqdm
import math
import numpy as np
from matplotlib import pyplot as plt


def phase_1(x): return 100 * (1 - np.exp(-5.0 * x ** 2))


def phase_2(x): return 800 * (1 - np.exp(-6.0*x))


def phase_3(x): return 80 * (1 - np.exp(-5.0*x))


def phase_4(x): return 90 * (1 - np.exp(-7.0*x))


def phase_5(x): return 500 * (1 - np.exp(-5.0*x))


number_of_experiments = 10

budget_cap = 10000
number_of_budgets = 100
daily_budgets = np.linspace(0, budget_cap, number_of_budgets).astype(int)
normalized_daily_budgets = daily_budgets/budget_cap
sigma = 5

time_horizon = 200
phase_duration = int(time_horizon / 3)

# See slide 67 1-09
window_coefficient = 1
window_length = int(window_coefficient * math.sqrt(time_horizon))

regret = []
for e in range(0, number_of_experiments):
    print('Starting experiment', e)

    scenarios = [ThreePhasesScenario(daily_budgets=normalized_daily_budgets,
                                     sigma=sigma,
                                     phase_1=phase_1,
                                     phase_2=phase_2,
                                     phase_3=phase_5,
                                     phase_duration=phase_duration),
                 ThreePhasesScenario(daily_budgets=normalized_daily_budgets,
                                     sigma=sigma,
                                     phase_1=phase_2,
                                     phase_2=phase_3,
                                     phase_3=phase_4,
                                     phase_duration=phase_duration),
                 ThreePhasesScenario(daily_budgets=normalized_daily_budgets,
                                     sigma=sigma,
                                     phase_1=phase_5,
                                     phase_2=phase_4,
                                     phase_3=phase_3,
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
    ideal_optimizer = Optimizer(daily_budgets, number_of_budgets)

    # The optimal and the ideal result obtained by playing the best possible combination of arms known
    optimal_rewards_per_round = np.zeros(time_horizon)
    ideal_rewards_per_round = np.zeros(time_horizon)

    for t in tqdm(range(0, time_horizon)):

        # Sample the Gaussian Process for each subcampaign
        sampled_values = []
        for subcampaign in subcampaigns:
            sampled_values.append(subcampaign.sample_values())

        # Compute the real value of the tru function
        real_values = []
        for scenario in scenarios:
            real_values.append(scenario.mean)

        # FIND OPTIMAL ARMS
        # Find the best combination of arms to play at time t based on the sampling of the Gaussian Processes
        optimal_arms = optimizer.optimize(sampled_values)
        # The optimal result obtainable at time t by playing the optimal arms
        optimal_result = 0

        # FIND IDEAL ARMS (to compute regret)
        # Given the real value of the true function, find the ideal combination of arms
        ideal_arms = ideal_optimizer.optimize(real_values)
        # The ideal result obtainable at time t
        ideal_result = 0

        for sub_index in range(0, len(subcampaigns)):
            # Sample the true function to get a reward for the played arm
            reward = scenarios[sub_index].play_round(optimal_arms[sub_index])
            optimal_result += reward

            # Update the model based on the result of the played arm
            subcampaigns[sub_index].update(optimal_arms[sub_index], reward)

            # Compute the ideal optimal result
            ideal_result += scenarios[sub_index].mean[ideal_arms[sub_index]]

            # Advance the time of the scenario and eventually trigger a new phase
            scenarios[sub_index].advance_time()

        optimal_rewards_per_round[t] = optimal_result
        ideal_rewards_per_round[t] = ideal_result
        # print("[", t, "]Optimal solution:", optimal_arms, " (", ideal_arms, ")", "with a result of: ", round(optimal_result, 2), " (", round(ideal_result, 2), ").")

    regret.append(ideal_rewards_per_round - optimal_rewards_per_round)

# Plotting the regret
plt.rcParams["figure.figsize"] = [16, 9]
fig, ax = plt.subplots(1, 1)
ax.axvline(phase_duration*1, color='r')
ax.axvline(phase_duration*2, color='r')
ax.axvline(phase_duration*3, color='r')

ax.plot(np.cumsum(np.mean(regret, axis=0)))
ax.set_xlim([0, time_horizon])
plt.show()
