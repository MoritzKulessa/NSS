import pandas as pd
import numpy as np
from scipy import stats
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from sklearn.preprocessing import OneHotEncoder

'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''


class DMSS(object):

    def __init__(self, data_stream, min_support_set=5, min_support_rule=5, ref_windows=[-1]):
        """
        Initialize
        :param data_stream: the module to access data
        :param min_support_set: minimum support for the item set miner
        :param min_support_rule: minimum support for the association rule
        :param ref_windows: reference time slots for creating the reference set. They are relative to the current evaluated time slot. (e.g. [-1, -2] = take the cases of the previous two time slots for the reference set)
        """
        self.data_stream = data_stream

        assert (min_support_set >= 0)
        self.min_support_set = min_support_set

        assert (min_support_rule >= 0)
        self.min_support_rule = min_support_rule

        assert (all([ref_window < 0 for ref_window in ref_windows]))
        self.ref_windows = np.array(ref_windows)

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

        # obtain reference set and set of cases for the current day
        reference_df = pd.concat([self.data_stream.get_cases(time_slot + j) for j in self.ref_windows])
        check_df = self.data_stream.get_cases(time_slot)

        # encode data for frequent itemset algorithm
        enc = OneHotEncoder(handle_unknown='ignore')
        df = pd.concat([reference_df, check_df])
        enc.fit(df)

        # Transform reference_df
        reference_data = enc.transform(reference_df.values).toarray()
        ohe_reference_df = pd.DataFrame(reference_data, columns=enc.get_feature_names(reference_df.columns))

        # Transform check_df
        check_data = enc.transform(check_df.values).toarray()
        ohe_check_df = pd.DataFrame(check_data, columns=enc.get_feature_names(check_df.columns))

        # Generate item sets
        freq_sets = fpgrowth(ohe_check_df, min_support=self.min_support_set / len(ohe_check_df), use_colnames=True)
        # print("Sets: " + str(len(freq_sets)))

        # if no item set have been found return score 0
        if len(freq_sets) == 0: return 0

        # Generate rules using the item sets
        rules = association_rules(freq_sets, metric="support", min_threshold=self.min_support_rule / len(ohe_check_df))
        # print("Rules: " + str(len(rules)))

        # Add empty body rules
        empty_body_rules = []
        for _, row in freq_sets.iterrows():
            empty_body_rules.append(
                [frozenset([]), row["itemsets"]] + [1., row["support"], row["support"], row["support"], np.nan, np.nan,
                                                    np.nan])
        empty_body_rules_df = pd.DataFrame(empty_body_rules,
                                           columns=['antecedents', 'consequents', 'antecedent support',
                                                    'consequent support', 'support', 'confidence', 'lift', 'leverage',
                                                    'conviction'])
        rules = pd.concat([rules, empty_body_rules_df])

        # if no rules have been found return score 0
        if len(rules) == 0: return 0

        p_vals = []
        for _, row in rules.iterrows():

            # Compute antecedents support on reference_df
            bols_left = np.full(len(ohe_reference_df), fill_value=True)
            for cond in row["antecedents"]:
                bols_left &= ohe_reference_df[cond] == 1

            # Continue if the support of the body of the rule is not meet
            if bols_left.sum() < self.min_support_rule: continue

            # Compute consequents support on oh_reference_df
            bols_right = np.full(len(ohe_reference_df), fill_value=True)
            for cond in row["consequents"]:
                bols_right &= ohe_reference_df[cond] == 1

            # Compute confidences
            sup_body = bols_left.sum()
            sup_complete_rule = (bols_left & bols_right).sum()
            reference_confidence = sup_complete_rule / sup_body
            reference_n = sup_body

            check_confidence = row["confidence"]
            check_n = int(round((row["antecedent support"]) * len(ohe_check_df)))

            # Perform hypothesis test on confidences
            set_a = np.array([reference_confidence * reference_n, (1 - reference_confidence) * reference_n])
            set_b = np.array([check_confidence * check_n, (1 - check_confidence) * check_n])
            contigency_table = np.array([set_a, set_b]).T
            _, p_value = stats.fisher_exact(contigency_table, alternative="less")
            p_vals.append(p_value)

        # Choose the minimal p-value and transform it to a score
        score = 1 - np.min(p_vals)
        return score
