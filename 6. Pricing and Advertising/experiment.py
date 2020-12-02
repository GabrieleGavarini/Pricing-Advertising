from SW_GPTS_Learner import *
from Scenario import *
from Optimizer import *

from CustomizablePricingScenario import *
from TS_Learner import *

from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt


def compute_pricing_reward(extended_reward):
    if extended_reward[2] == 0:
        return 0
    return extended_reward[1] / extended_reward[2]


def compute_ideal(_advertising_scenarios, _pricing_scenarios):
    # Compute the real number of clicks
    real_numbers = []
    for scenario in _advertising_scenarios:
        real_numbers.append(scenario.y[:number_of_budgets])

    # Compute the ideal value of clicks
    real_values = []
    pricing_arms = []
    for scenario in _pricing_scenarios:
        pricing_arm = scenario.get_optimal_arm()
        pricing_arms.append(pricing_arm)

        real_value = scenario.round(pricing_arm)
        real_values.append(compute_pricing_reward(real_value))

    # FIND IDEAL ARMS (to compute regret)
    # Given the real value of the true function, find the ideal combination of arms
    ideal_campaign_values = [a * b for a, b in zip(real_numbers, real_values)]
    arms = ideal_optimizer.optimize(ideal_campaign_values)

    # Compute the ideal optimal result
    result = 0
    for index, scenario in enumerate(_advertising_scenarios):
        result += (scenario.y[arms[index]] * real_values[index])

    # print("\n",
    #       "Ideal pricing solution: ", pricing_arms, "\n",
    #       "Ideal advertising solution: ", arms, "\n",
    #       "Ideal result: ", round(result, 2)
    #       )

    return arms, result


number_of_experiments = 1

# The budget cap and the possible daily budget that can be set
budget_cap = 27000
daily_budgets = np.linspace(5000, budget_cap, endpoint=True, num=45).astype(int)

# The number of possible budgets that can be allocated to each subcampaign
number_of_budgets = 11

time_horizon = 300

regret = []
for e in range(0, number_of_experiments):

    print('\n')
    print('Starting experiment', e + 1)

    advertising_scenarios = [
        Scenario(daily_budgets=daily_budgets,
                 campaign=0,
                 var=0),
        Scenario(daily_budgets=daily_budgets,
                 campaign=1,
                 var=0),
        Scenario(daily_budgets=daily_budgets,
                 campaign=2,
                 var=0)
        ]

    advertising_learners = [
        SW_GPTS_Learner(arms=daily_budgets[:number_of_budgets]),
        SW_GPTS_Learner(arms=daily_budgets[:number_of_budgets]),
        SW_GPTS_Learner(arms=daily_budgets[:number_of_budgets])
    ]

    pricing_scenarios = [
        CustomizablePricingScenario(sex=1, under_30=1),
        CustomizablePricingScenario(sex=1, under_30=0),
        CustomizablePricingScenario(sex=0)
    ]

    pricing_learners = [
        TS_Learner(n_arms=pricing_scenarios[0].n_arms),
        TS_Learner(n_arms=pricing_scenarios[1].n_arms),
        TS_Learner(n_arms=pricing_scenarios[2].n_arms)
    ]

    optimizer = Optimizer(daily_budgets, len(daily_budgets))
    ideal_optimizer = Optimizer(daily_budgets, len(daily_budgets))

    # The optimal and the ideal result obtained by playing the best possible combination of arms known
    optimal_rewards_per_round = np.zeros(time_horizon)

    # The ideal arms to pull
    ideal_arms, ideal_result = compute_ideal(advertising_scenarios, pricing_scenarios)

    for t in tqdm(range(0, time_horizon)):

        # Sample the Gaussian Process for each subcampaign
        number_clicks = []
        for subcampaign in advertising_learners:
            number_clicks.append(subcampaign.sample_values())

        # Obtain the value per click and the arms to pull by sampling the learners
        value_clicks = []
        pricing_arms_indices = []
        pricing_arms = []
        for pricing_learner in pricing_learners:
            arm, value = pricing_learner.pull_arm()
            value_clicks.append(value)
            pricing_arms_indices.append(arm)
            pricing_arms.append(pricing_scenarios[0].arms[arm])

        # Find the best budget allocation based on the estimation of the number and value of clicks
        campaign_values = [a*b for a, b in zip(number_clicks, value_clicks)]
        optimal_advertising_arms = optimizer.optimize(campaign_values)
        # The optimal result obtainable at time t by allocating the optimal budget
        # The optimal result obtainable at time t by allocating the optimal budget
        optimal_result = 0

        for sub_index in range(0, len(advertising_learners)):
            # Sample the true function to get a reward for the played arm
            advertising_reward = advertising_scenarios[sub_index].play_round(optimal_advertising_arms[sub_index])
            pricing_reward = compute_pricing_reward(pricing_scenarios[sub_index].round(pricing_arms[sub_index]))

            optimal_result += advertising_reward * pricing_reward

            # Update the advertising model
            advertising_learners[sub_index].update(optimal_advertising_arms[sub_index], advertising_reward)

            # Update the pricing model
            pricing_learners[sub_index].update((pricing_arms_indices[sub_index]), pricing_reward)

        optimal_rewards_per_round[t] = optimal_result
        # print("\n",
        #       "[", t, "]Optimal advertising solution:", optimal_advertising_arms, "\n",
        #       "\t Optimal pricing solution:", pricing_arms, "\n",
        #       "\t with a result of: ", round(optimal_result, 2), "\n")

    regret.append(ideal_result - optimal_rewards_per_round)

# PLOTTING THE REGRET
y = np.cumsum(np.mean(regret, axis=0))
max_y = np.max(y)

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot()

ax.set_title("Regret per day", fontsize=17, ha='center')

ax.grid(axis='x', alpha=0.3, linestyle='--')

ax.set_ylim(bottom=0)
ax.set_xlim(left=0)

ax.set_xlabel("Day", fontsize=14)
ax.set_ylabel("Regret", fontsize=14)

ax.tick_params(length=0)
ax.set_yticks(np.linspace(0, max_y * 1.5, 11))
ax.set_yticklabels(np.linspace(0, max_y * 1.5, 11).astype(np.int64), fontsize=12, alpha=0.7)

ax.spines['right'].set_alpha(0)
ax.spines['left'].set_alpha(0.3)
ax.spines['top'].set_alpha(0)
ax.spines['bottom'].set_alpha(0.3)

ax.plot(y, linewidth='2')
ax.set_xlim([0, time_horizon])

plt.savefig('chapter6_regret.png')

plt.show()
