from SW_GPTS_Learner import *
from ThreePhasesScenario import *
from Optimizer import *

from tqdm import tqdm
import math
import numpy as np
from matplotlib import pyplot as plt

number_of_experiments = 10
window_coefficients = [0, 1, 2]

# The budget cap and the possible daily budget that can be set
budget_cap = 27000
daily_budgets = np.linspace(5000, budget_cap, endpoint=True, num=45).astype(int)

# The number of possible budgets that can be allocated to each subcampaign
number_of_budgets = 11

time_horizon = 300
phase_duration = int(time_horizon / 3)

regret = []

fig, ax = plt.subplots(1, 3, figsize=(36, 9))

for i in range(0, 3):

    # See slide 67 1-09
    window_coefficient = window_coefficients[i]
    window_length = int(window_coefficient * math.sqrt(time_horizon))

    for e in range(0, number_of_experiments):

        print('\n')
        print('Starting experiment', e + 1)

        scenarios = [ThreePhasesScenario(daily_budgets=daily_budgets,
                                         campaign=0,
                                         phase_duration=phase_duration),
                     ThreePhasesScenario(daily_budgets=daily_budgets,
                                         campaign=1,
                                         phase_duration=phase_duration),
                     ThreePhasesScenario(daily_budgets=daily_budgets,
                                         campaign=2,
                                         phase_duration=phase_duration)]

        subcampaigns = [SW_GPTS_Learner(arms=daily_budgets[:number_of_budgets],
                                        window_length=window_length),
                        SW_GPTS_Learner(arms=daily_budgets[:number_of_budgets],
                                        window_length=window_length),
                        SW_GPTS_Learner(arms=daily_budgets[:number_of_budgets],
                                        window_length=window_length)]

        optimizer = Optimizer(daily_budgets, len(daily_budgets))
        ideal_optimizer = Optimizer(daily_budgets, len(daily_budgets))

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
                real_values.append(scenario.phase_values[:number_of_budgets])

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
                ideal_result += scenarios[sub_index].phase_values[ideal_arms[sub_index]]

                # Advance the time of the scenario and eventually trigger a new phase
                scenarios[sub_index].advance_time()

            optimal_rewards_per_round[t] = optimal_result
            ideal_rewards_per_round[t] = ideal_result
            # print("[", t, "]Optimal solution:", optimal_arms, " (", ideal_arms, ")",
            #       "with a result of: ", round(optimal_result, 2), " (", round(ideal_result, 2), ").")

        regret.append(ideal_rewards_per_round - optimal_rewards_per_round)

    # PLOTTING THE REGRET

    ax[i].set_title("Regret per day", fontsize=17, ha='center')

    ax[i].grid(axis='x', alpha=0.3, linestyle='--')

    ax[i].set_ylim(bottom=0)
    ax[i].set_xlim(left=0)

    ax[i].set_xlabel("Day", fontsize=14)
    ax[i].set_ylabel("Regret", fontsize=14)

    ax[i].tick_params(length=0)
    ax[i].set_yticks(np.linspace(0, 30000, 11))
    ax[i].set_yticklabels(np.linspace(0, 30000, 11).astype(np.int64), fontsize=12, alpha=0.7)

    ax[i].spines['right'].set_alpha(0)
    ax[i].spines['left'].set_alpha(0.3)
    ax[i].spines['top'].set_alpha(0)
    ax[i].spines['bottom'].set_alpha(0.3)

    ax[i].axvline(phase_duration*1, color='lightcoral')
    ax[i].axvline(phase_duration*2, color='lightcoral')

    ax[i].plot(np.cumsum(np.mean(regret, axis=0)), linewidth='2', label='c=' + str(window_coefficient))
    ax[i].set_xlim([0, time_horizon])

    ax[i].legend(frameon=False)

fig.savefig('chapter3_regret.png')

fig.show()
