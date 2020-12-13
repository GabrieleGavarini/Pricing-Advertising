from PricingScenario import *

from matplotlib import pyplot as plt


class ExtendedPricingScenario(PricingScenario):

    def __init__(self, file_name="../Scenarios/User_pricing.csv"):
        super().__init__(file_name)

    def get_optimal_arm(self, sex=None, age=None):
        """
        Get the arm that maximizes the conversion rate.
        :param sex: The sex of the users we are interested in
        :param age: The age of the users we are interested in
        :return: the arm that maximizes the conversion rate.
        """
        if sex is None and age is None:
            converted = self.df[(self.df.Converted == 1)].groupby("price").sum().Converted
            total = self.df.sum().Converted
        elif sex is None and age is not None:
            converted = self.df[(self.df.Under_30 == age) & (self.df.Converted == 1)].groupby("price").sum().Converted
            total = self.df[(self.df.Under_30 == age)].sum().Converted
        elif age is None and sex is not None:
            converted = self.df[(self.df.Sex == sex) & (self.df.Converted == 1)].groupby("price").sum().Converted
            total = self.df[(self.df.Sex == sex)].sum().Converted
        else:
            converted = self.df[(self.df.Sex == sex) & (self.df.Under_30 == age) & (self.df.Converted == 1)].groupby("price").sum().Converted
            total = self.df[(self.df.Sex == sex) & (self.df.Under_30 == age)].sum().Converted

        print(converted/total)
        return self.arms[np.argmax(converted/total)]

    def round(self, pulled_arm, sex=None, age=None):
        """
        Play the pulled_arm and obtain a reward.
        :param pulled_arm: The index of the arm to be played
        :param sex: The sex of the users we are interested in
        :param age: The age of the users we are interested in
        :return: An array containing, for every possible context:
            - The index of the pulled arm
            - The sex
            - The age
            - The number of users converted
            - The total number of users
        """

        if sex is None and age is None:
            return [
                [pulled_arm, 0, 0,
                 self.df[(self.df.price == pulled_arm) & (self.df.Sex == 0) & (self.df.Under_30 == 0) & (self.df.Converted == 1)].count()[0],
                 self.df[(self.df.price == pulled_arm) & (self.df.Sex == 0) & (self.df.Under_30 == 0)].count()[0]],
                [pulled_arm, 0, 1,
                 self.df[(self.df.price == pulled_arm) & (self.df.Sex == 0) & (self.df.Under_30 == 1) & (self.df.Converted == 1)].count()[0],
                 self.df[(self.df.price == pulled_arm) & (self.df.Sex == 0) & (self.df.Under_30 == 1)].count()[0]],
                [pulled_arm, 1, 0,
                 self.df[(self.df.price == pulled_arm) & (self.df.Sex == 1) & (self.df.Under_30 == 0) & (self.df.Converted == 1)].count()[0],
                 self.df[((self.df.price == pulled_arm) & self.df.Sex == 1) & (self.df.Under_30 == 0)].count()[0]],
                [pulled_arm, 1, 1,
                 self.df[(self.df.price == pulled_arm) & (self.df.Sex == 1) & (self.df.Under_30 == 1) & (self.df.Converted == 1)].count()[0],
                 self.df[(self.df.price == pulled_arm) & (self.df.Sex == 1) & (self.df.Under_30 == 1)].count()[0]]
            ]
        elif sex is None and age is not None:
            return[
                [pulled_arm, 0, age,
                 self.df[(self.df.price == pulled_arm) & (self.df.Sex == 0) & (self.df.Under_30 == age) & (self.df.Converted == 1)].count()[0],
                 self.df[(self.df.price == pulled_arm) & (self.df.Sex == 0) & (self.df.Under_30 == age)].count()[0]],
                [pulled_arm, 1, age,
                 self.df[(self.df.price == pulled_arm) & (self.df.Sex == 1) & (self.df.Under_30 == age) & (self.df.Converted == 1)].count()[0],
                 self.df[(self.df.price == pulled_arm) & (self.df.Sex == 1) & (self.df.Under_30 == age)].count()[0]]
            ]
        elif age is None and sex is not None:
            return[
                [pulled_arm, sex, 0,
                 self.df[(self.df.price == pulled_arm) & (self.df.Sex == sex) & (self.df.Under_30 == 0) & (self.df.Converted == 1)].count()[0],
                 self.df[(self.df.price == pulled_arm) & (self.df.Sex == sex) & (self.df.Under_30 == 0)].count()[0]],
                [pulled_arm, sex, 1,
                 self.df[(self.df.price == pulled_arm) & (self.df.Sex == sex) & (self.df.Under_30 == 1) & (self.df.Converted == 1)].count()[0],
                 self.df[(self.df.price == pulled_arm) & (self.df.Sex == sex) & (self.df.Under_30 == 1)].count()[0]]
            ]
        else:
            return[
                [pulled_arm, sex, age,
                 self.df[(self.df.price == pulled_arm) & (self.df.Sex == sex) & (self.df.Under_30 == age) & (self.df.Converted == 1)].count()[0],
                 self.df[(self.df.price == pulled_arm) & (self.df.Sex == sex) & (self.df.Under_30 == age)].count()[0]]
            ]

    def plot_scenario_distribution(self):
        """
        Utility function used to plot the modeled distribution
        """
        x = self.arms

        y = self.df.groupby('price').mean().Converted[x]
        y_sex_0 = self.df[self.df.Sex == 0].groupby('price').mean().Converted[x]
        y_sex_1 = self.df[self.df.Sex == 1].groupby('price').mean().Converted[x]
        y_age_0 = self.df[self.df.Under_30 == 0].groupby('price').mean().Converted[x]
        y_age_1 = self.df[self.df.Under_30 == 1].groupby('price').mean().Converted[x]

        fig, ax_list = plt.subplots(2,1, figsize=(12, 9))

        for ax in ax_list:
            ax.grid(alpha=0.3, linestyle='--')

            ax.set_ylim(bottom=0, top=0.6)
            ax.set_xlim(left=50, right=104)

            ax.set_xlabel("Price", fontsize=14)
            ax.set_ylabel("Conversion Rate", fontsize=14)

            ax.set_xticks(self.arms)
            ax.set_xticklabels(self.arms.astype(np.int64), fontsize=12, alpha=0.7)
            ax.set_yticks(np.linspace(0, 0.7, 8))
            ax.set_yticklabels([str((i * 100).astype(np.int64)) + "%" for i in np.linspace(0, 0.7, 8)], fontsize=12, alpha=0.7)

            ax.spines['right'].set_alpha(0)
            ax.spines['left'].set_alpha(0.3)
            ax.spines['top'].set_alpha(0)
            ax.spines['bottom'].set_alpha(0.3)

        ax_list[0].plot(x, y, label='Global')
        ax_list[0].plot(x, y_sex_0, label='Male', color='moccasin')
        ax_list[0].plot(x, y_sex_1, label='Female', color='darkorange')

        ax_list[1].plot(x, y, label='Global')
        ax_list[1].plot(x, y_age_0, label='Under 30', color='red')
        ax_list[1].plot(x, y_age_1, label='Over 30', color='darkred')

        ax_list[0].legend()
        ax_list[1].legend()

        fig.suptitle("Conversion Rate", fontsize=22)

        fig.show()

        plt.savefig('chapter5_pricing.png')