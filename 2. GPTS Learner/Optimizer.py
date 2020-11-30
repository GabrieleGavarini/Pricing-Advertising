import numpy as np


class Optimizer:

    def __init__(self, daily_budgets, number_of_budgets):
        self.daily_budgets = daily_budgets
        self.number_of_budgets = number_of_budgets

    def optimize(self, sampled_values):
        """
        Knapsack-like algorithm to find the best possible assignment of daily budgets to subcampaign based on the
        reward each daily budgets.
        :param sampled_values: (numpy.array) A numpy array containing the sampled value for each subcampaign
        :return: (numpy.array) An array containing the indexes of the best arm to be pulled
        """
        incremental_table = np.zeros(shape=(len(sampled_values) + 1,
                                            self.number_of_budgets))

        # Initialization of the assignment table to 0
        assignment_table = np.zeros(shape=(len(sampled_values)+1,
                                           self.number_of_budgets,
                                           len(sampled_values)+1))

        for subcampaign_index in range(0, len(sampled_values)):
            # The array of the sampled values for the current subcampaign
            sub_sampled_values = sampled_values[subcampaign_index]

            # incremental_table.append(np.zeros(self.number_of_budgets))

            # for every budget cap explore the possible combination of previous optimal and new sampled
            # values to select the optimal play
            for budget_index in range(0, self.number_of_budgets):

                if budget_index == 0:
                    new_optimal_maximum = sub_sampled_values[0]
                    new_optimal_index = 0
                else:
                    new_optimal_maximum = max(sub_sampled_values[:min(budget_index, len(sub_sampled_values) - 1) + 1])
                    new_optimal_index = np.argmax(sub_sampled_values[:min(budget_index, len(sub_sampled_values) - 1)])

                incremental_table[subcampaign_index + 1][budget_index] = new_optimal_maximum
                assignment_table[subcampaign_index + 1][budget_index][subcampaign_index + 1] = new_optimal_index

                for previous_optimal_index in range(0, budget_index + 1):

                    # The optimal value of the previous assignment, assuming all the budget is assigned to it
                    previous_optimal_value = incremental_table[subcampaign_index][previous_optimal_index]

                    # How much value is left to allocate
                    value_to_allocate = self.daily_budgets[budget_index] - self.daily_budgets[previous_optimal_index]

                    # Find the index of the largest budget that can be allocated given a portion of the budget cap
                    # assigned to the previous optimal coalition
                    sampled_optimal_index = 0
                    sampled_optimal_value = 0
                    for i in range(0, len(sub_sampled_values)):
                        if sub_sampled_values[i] > sampled_optimal_value:
                            sampled_optimal_value = sub_sampled_values[i]
                            sampled_optimal_index = i
                        if self.daily_budgets[i] >= value_to_allocate:
                            break

                    # If a better maximum is found
                    if sampled_optimal_value + previous_optimal_value > new_optimal_maximum:
                        # Update the temporary optimal maximum
                        new_optimal_maximum = sampled_optimal_value + previous_optimal_value

                        # Update the selected budget by coping the assignment for the optimal solution of the previous
                        # optimal solution and modifying the selection for the current campaign
                        assignment_table[subcampaign_index + 1][budget_index] = assignment_table[subcampaign_index][previous_optimal_index + 1]
                        assignment_table[subcampaign_index + 1][budget_index][subcampaign_index + 1] = sampled_optimal_index

                incremental_table[subcampaign_index + 1][budget_index] = new_optimal_maximum

        # print(incremental_table)
        # print(assignment_table[-1])

        return assignment_table[-1][-1][1:].astype(int)
