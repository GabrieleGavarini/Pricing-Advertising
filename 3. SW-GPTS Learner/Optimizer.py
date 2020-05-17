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

                new_optimal_maximum = 0

                for previous_optimal_index in range(0, budget_index + 1):

                    # The optimal value of the previous assignment, assuming all the budget is assigned to it
                    previous_optimal_value = incremental_table[subcampaign_index][previous_optimal_index]

                    # How much value is left to allocate
                    value_to_allocate = self.daily_budgets[budget_index] - self.daily_budgets[previous_optimal_index]

                    # Find the index of the new element which budget is equal to the amount of budget that still needs
                    # to be allocated
                    # sampled_optimal_index = self.daily_budgets.index(value_to_allocate)
                    sampled_optimal_index = 0
                    if len(np.where(self.daily_budgets == value_to_allocate)[0]) > 0:
                        sampled_optimal_index = np.where(self.daily_budgets == value_to_allocate)[0][0]
                    else:
                        print('Error allocating value: ', value_to_allocate)
                    sampled_optimal_value = sub_sampled_values[sampled_optimal_index]

                    if sampled_optimal_value + previous_optimal_value > new_optimal_maximum:
                        new_optimal_maximum = sampled_optimal_value + previous_optimal_value
                        # Update the selected budget by coping the assignment for the optimal solution of the previous
                        # optimal solution and modifying the selection for the current campaign
                        assignment_table[subcampaign_index + 1][budget_index] = assignment_table[subcampaign_index][previous_optimal_index][:]
                        assignment_table[subcampaign_index + 1][budget_index][subcampaign_index + 1] = sampled_optimal_index

                incremental_table[subcampaign_index + 1][budget_index] = new_optimal_maximum

        index_of_optimum = np.unravel_index(incremental_table.argmax(), incremental_table.shape)
        return assignment_table[index_of_optimum][1:].astype(int)
