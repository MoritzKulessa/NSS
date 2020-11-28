import numpy as np
from scipy import stats

'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''


class Benchmark:

    def __init__(self, syndrome_counter, distribution="nb", min_parameter=1):
        """
        Initializing
        :param syndrome_counter: the module producing syndrome counts
        :param distribution: the distribution used for the benchmark ("gaussian"/"poisson"/"nb"/"fisher")
        :param min_parameter: the minimum parameter used for the distribution
        """
        self.syndrome_counter = syndrome_counter
        self.distribution = distribution
        self.min_parameter = min_parameter

    '''
    ***********************************************************************************************************
    ***********************************************************************************************************
    ***********************************************************************************************************
    '''

    def evaluate(self, time_slot):
        """
        Computes the score for the given time slot.
        :param time_slot: the time slot for which the score is computed
        :return: the score for the given time slot
        """

        # obtain previous and current syndrome counts
        first_time_slot = self.syndrome_counter.data_stream.get_info()["first_time_slot"]
        syndrome_counts_df = self.syndrome_counter.get_syndrome_df(first_time_slot, time_slot)
        assert (syndrome_counts_df.iloc[-1]["time_slot"] == time_slot)  # assume that the syndrome counts df is sorted
        syndrome_counts_df = syndrome_counts_df.drop("time_slot", axis=1)

        # Load the total number of cases for the previous time slots and the current time slot for the fisher benchmark
        if self.distribution == "fisher":
            n_cases_before = sum([len(cases) for (cases, _) in
                                  self.syndrome_counter.data_stream.get_history(first_time_slot, time_slot - 1)])
            n_cases_time_slot = len(self.syndrome_counter.data_stream.get_cases(time_slot))

        # Compute for every observed syndrome of the current time slot a p-value
        p_vals = []
        for i in range(syndrome_counts_df.values.shape[1]):

            # obtain previous counts
            counts = syndrome_counts_df.values[:, i]
            train_counts = counts[:-1]
            check_val = counts[-1]

            if check_val == 0:
                continue

            if self.distribution == "gaussian":
                avg_train = np.mean(train_counts)
                std_train = np.max([np.std(train_counts), self.min_parameter])
                p_val = 1 - stats.norm.cdf(check_val, loc=avg_train, scale=std_train)

            elif self.distribution == "poisson":
                avg_train = np.mean(train_counts)
                if avg_train <= self.min_parameter:
                    avg_train = self.min_parameter
                p_val = 1 - stats.poisson.cdf(check_val, avg_train)

            elif self.distribution == "nb":
                mu = np.max([np.mean(train_counts), self.min_parameter])
                sig = np.std(train_counts)
                m = (sig ** 2 - mu)
                if m <= 0.0000001:
                    m = 0.0000001
                r = (mu ** 2) / m
                p = r / (r + mu)
                p_val = 1 - stats.nbinom.cdf(k=check_val, n=r, p=p)

            elif self.distribution == "fisher":
                n1 = n_cases_before
                ns1 = np.sum(train_counts)
                n2 = n_cases_time_slot
                ns2 = check_val

                set_a = np.array([ns1, n1 - ns1])
                set_b = np.array([ns2, n2 - ns2])
                contingency_table = np.array([set_a, set_b]).T
                _, p_val = stats.fisher_exact(contingency_table, alternative="less")

            else:
                raise Exception("Distribution not available: " + str(self.distribution))

            p_vals.append(p_val)

        # Choose the minimal p-value and transform it to a score
        score = 1 - np.min(p_vals)
        return score
