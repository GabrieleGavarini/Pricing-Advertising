from PricingScenario import *
from ContextLearner import *

from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd

def compute_reward(extended_reward):
    n = 0
    d = 0
    for r in extended_reward:
        n += r[3]
        d += r[4]
    return n/d

scenario = PricingScenario()
scenario.plot_scenario_distribution()

arms = scenario.arms
n_arms = scenario.n_arms

context_list = [
    Context(n_arms, {"Under_30": 0, "Sex": 0}),
    Context(n_arms, {"Under_30": 0, "Sex": 1}),
    Context(n_arms, {"Under_30": 1})
]

# -- IDEAL SCENARIO -- #
all_context = [
    Context(n_arms, {}),
    Context(n_arms, {"Sex": 0}),
    Context(n_arms, {"Sex": 1}),
    Context(n_arms, {"Under_30": 0}),
    Context(n_arms, {"Under_30": 1}),
    Context(n_arms, {"Sex": 0, "Under_30": 0}),
    Context(n_arms, {"Sex": 0, "Under_30": 1}),
    Context(n_arms, {"Sex": 1, "Under_30": 0}),
    Context(n_arms, {"Sex": 1, "Under_30": 1}),
]
ideal_context_reward = []
for context in all_context:
    arm = scenario.get_optimal_arm(sex=context.sex, age= context.age)
    ideal_context_reward += scenario.round(arm, sex=context.sex, age= context.age)

ideal_context_df = pd.DataFrame(ideal_context_reward, columns=["Price", "Sex", "Under_30", "Converted", "Total"])
ideal_context_learner = ContextLearner(n_arms)
ideal_context_list = ideal_context_learner.split(ideal_context_df, ["Sex", "Under_30"])

daily_ideal_reward = 0
for ideal_context in ideal_context_list:
    ideal_arm = scenario.get_optimal_arm(sex=ideal_context.sex, age=ideal_context.age)
    extended_reward = scenario.round(ideal_arm, sex=ideal_context.sex, age=ideal_context.age)
    daily_ideal_reward += sum(list(zip(*extended_reward))[3])
# -------------------- #

# -- OPTIMAL SCENARIO -- #
max_week = 5
iteration_per_day = 24*6

regret = []
observation_rows = []

for week in tqdm(range(0, max_week)):

    # TO create a dataframe with all the observations

    for day in range(0, 7 * iteration_per_day):

        total_converted = 0

        for context in context_list:
            sex = context.sex
            age = context.age

            # Pull an arm
            pulled_arm = context.TS.pull_arm()

            # Play a round with the selected arm
            extended_reward = scenario.round(pulled_arm=arms[pulled_arm], sex=context.sex, age=context.age)
            # Update the database of observations
            observation_rows += extended_reward
            # Compute how many users you've actually converted
            total_converted += sum(list(zip(*extended_reward))[3])

            # Update the distribution
            reward = compute_reward(extended_reward)
            context.TS.update(pulled_arm, reward)

        # Compute the iteration regret (for all the contexts)
        regret += [daily_ideal_reward - total_converted]

    # Create the dataframe with all the observations
    observations = pd.DataFrame(observation_rows, columns=["Price", "Sex", "Under_30", "Converted", "Total"])
    context_learner = ContextLearner(n_arms)
    context_list = context_learner.split(observations, ["Sex", "Under_30"])

# PLOTTING THE REGRET
fig = plt.figure(figsize=(12,9))
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

plt.savefig('chapter5_regret.png')