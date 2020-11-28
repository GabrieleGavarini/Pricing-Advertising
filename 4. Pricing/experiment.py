from PricingScenario import *
from ContextLearner import *

from matplotlib import pyplot as plt
from tqdm import tqdm


def compute_reward(extended_reward):
    return extended_reward[1] / extended_reward[2]


scenario = PricingScenario()

arms = scenario.arms
n_arms = scenario.n_arms

TS = TS_Learner(n_arms=n_arms)

# -- IDEAL SCENARIO -- #
arm = scenario.get_optimal_arm()
ideal_reward = scenario.round(arm)
# -------------------- #

# -- OPTIMAL SCENARIO -- #
max_week = 5
iteration_per_day = 24*6

regret = []

for day in tqdm(range(0, max_week * 7 * iteration_per_day)):

    # Pull an arm
    pulled_arm = TS.pull_arm()

    # Play a round with the selected arm
    reward = scenario.round(pulled_arm=arms[pulled_arm])

    # Update the distribution
    TS.update(pulled_arm, compute_reward(reward))

    # Compute the iteration regret
    regret += [ideal_reward[1] - reward[1]]


# PLOTTING THE REGRET
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot()

ax.set_title("Regret per week", fontsize=17, ha='center')

ax.grid(axis='x', alpha=0.3, linestyle='--')

ax.set_ylim(bottom=0)
ax.set_xlim(left=0)

ax.set_xlabel("Week", fontsize=14)
ax.set_ylabel("Missed conversion", fontsize=14)

ax.tick_params(length=0)
ax.set_xticks(np.linspace(iteration_per_day * 7, iteration_per_day * 7 * max_week, max_week))
ax.set_xticklabels(np.linspace(1, max_week, max_week).astype(np.int64), fontsize=12, alpha=0.7)
ax.set_yticks(np.linspace(30000, 300000, 10))
ax.set_yticklabels(np.linspace(30, 300, 10).astype(np.int64), fontsize=12, alpha=0.7)

ax.spines['right'].set_alpha(0)
ax.spines['left'].set_alpha(0.3)
ax.spines['top'].set_alpha(0)
ax.spines['bottom'].set_alpha(0.3)

ax.plot(np.cumsum(regret), linewidth='2')

plt.savefig('chapter4_regret.png')
