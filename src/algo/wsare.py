import os
import numpy as np
import pandas as pd
import pyAgrum as gum
from scipy import stats

from util import io


class WSARE(object):

    def __init__(self, syndrome_counter, version="2.0", randomization=10):
        """
        Initialize
        :param syndrome_counter: the module producing syndrome counts
        :param version: the algorithm used to generate the reference set ("2.0" for WSARE 2.0, "2.5" for WSARE 2.5, and "3.0" for WSARE 3.0)
        :param randomization: the number of iterations for the randomization test, if randomization is None then the minimal p-value is reported 
        """
        self.data_stream = syndrome_counter.data_stream
        self.syndrome_counter = syndrome_counter
        self.version = version
        self.randomization = randomization

    '''
    ***********************************************************************************************************
    ***********************************************************************************************************
    ***********************************************************************************************************
    '''

    def _obtain_reference_set_WSARE2(self, time_slot):
        """
        Returns the reference set according to the WSARE 2.0 algorithm.
        :param time_slot: the time slot for which the reference sets needs to be generated
        :return: a dataframe containing cases
        """

        # merging the cases of the prior time slots 35, 42 49, and 56
        return pd.concat([self.data_stream.get_cases(time_slot - 56),
                          self.data_stream.get_cases(time_slot - 49),
                          self.data_stream.get_cases(time_slot - 42),
                          self.data_stream.get_cases(time_slot - 35)])

    def _obtain_reference_set_WSARE25(self, time_slot):
        """
        Returns the reference set according to the WSARE 2.5 algorithm. If no cases with the same environmental setting can be found return an empty dataframe.
        :param time_slot: the time slot for which the reference sets needs to be generated
        :return: a dataframe containing cases
        """

        # merging the cases of the prior time slots which match the same environmental setting as the current time slot
        first_time_slot = self.data_stream.get_info()["first_time_slot"]
        history = self.data_stream.get_history(first_time_slot, time_slot - 1)
        cur_env = self.data_stream.get_env(time_slot)

        ref_dfs = []
        for (cases_df, env_df) in history:
            if all(cur_env.iloc[0] == env_df.iloc[0]):
                ref_dfs.append(cases_df)

        if len(ref_dfs) > 0:
            return pd.concat(ref_dfs)
        else:
            return pd.DataFrame()

    def _obtain_reference_set_WSARE3(self, time_slot):
        """
        Returns the reference set according to the WSARE 3.0 algorithm.
        :param time_slot: the time slot for which the reference sets needs to be generated
        :return: a dataframe containing cases
        """

        # obtain cases of the current time slot
        cur_cases_df = self.data_stream.get_cases(time_slot)

        # learn a Bayesian network from which the reference set is sampled
        first_time_slot = self.data_stream.get_info()["first_time_slot"]
        history = self.data_stream.get_history(first_time_slot, time_slot - 1)
        cur_env = self.data_stream.get_env(time_slot)

        # create dataframe for training which contains all cases (including the environmental setting for each case)
        train_dfs = []
        for (cases_df, env_df) in history:
            for col_name, val in env_df.iloc[0].items():
                cases_df[col_name] = [val] * len(cases_df)
            train_dfs.append(cases_df)
        train_df = pd.concat(train_dfs)

        # write temporarily to disk, because pyAgrum learns from disk
        if not os.path.exists(io.get_project_directory() + "_tmp/"):
            os.mkdir(io.get_project_directory() + "_tmp/")
        tmp_path = io.get_project_directory() + "_tmp/WSARE3_" + self.data_stream.get_ident() + "_" + str(
            self.randomization) + ".csv"
        train_df.to_csv(tmp_path, index=False)

        # prepare the bayesian network
        learner = gum.BNLearner(tmp_path)
        learner.useLocalSearchWithTabuList()

        # do not allow that environmental variables have parents
        for env_col in cur_env.columns:
            for response_col in cur_cases_df.columns:
                learner.addForbiddenArc(response_col, env_col)

        # learn they bayesian network
        bn = learner.learnBN()

        '''
        Unfortunately pyAgrum does not allow to sample with evidence, therefore we have adapted the structure of the bayesian entwork with 
        respect to the environmental setting of the current evaluated time slot and then used the "generateCSV" method to obtain the samples.
        '''
        for env_col in cur_env.columns:

            # update stats of the bayesian node with respect to the given environmental setting of the current time slot
            # The probability for all values other then "env_val" is set to 0
            s = str(bn.cpt(env_col))
            new_probs = []
            for node in s.split("/"):
                splitted = node.split("::")
                prob = float(splitted[1])
                entry = splitted[0].strip()[1:-1]

                # only allow a probability for the entry in the bayesian node, if all the condition for the entry match the environmental setting
                new_prob = prob
                if "|" in splitted[0]:
                    for cond in entry.split("|"):
                        cond_attr, cond_val = cond.split(":")

                        if (cond_attr in cur_env.iloc[0]) and (cur_env.iloc[0][cond_attr] != cond_val):
                            new_prob = 0
                            break
                else:
                    cond_attr, cond_val = entry.split(":")
                    if (cond_attr in cur_env.iloc[0]) and (cur_env.iloc[0][cond_attr] != cond_val):
                        new_prob = 0

                new_probs.append(new_prob)

            # Avoid that the node obtains for all values a probability of 0
            if np.sum(new_probs) == 0:
                new_probs = [1] * len(new_probs)

            new_probs = list(np.array(new_probs) / np.sum(new_probs))
            bn.cpt(env_col).fillWith(new_probs)

        # generate the samples
        gum.generateCSV(bn, tmp_path, 10000, False, with_labels=True)

        # read samples
        samples = pd.read_csv(tmp_path)

        # only obtain the response attributes
        return samples[list(cur_cases_df.columns)]

    '''
    ***********************************************************************************************************
    ***********************************************************************************************************
    ***********************************************************************************************************
    '''

    def _compute_p_values(self, cases_A, cases_B):
        """
        Compute the p-values with the fisher test for the the cases_A given the cases of cases_B as reference .
        :param cases_A: the cases which need to be checked
        :param cases_B: the cases which serve as reference
        """
        syndrome_counts_A = self.syndrome_counter.count_syndromes(cases_A)
        syndrome_counts_B = self.syndrome_counter.count_syndromes(cases_B)

        p_vals = []
        for syndrome, count in syndrome_counts_A.items():

            if syndrome not in syndrome_counts_B:
                ref_count = 0
            else:
                ref_count = syndrome_counts_B[syndrome]

            set_a = np.array([ref_count, len(cases_B) - ref_count])
            set_b = np.array([count, len(cases_A) - count])
            contigency_table = np.array([set_a, set_b]).T
            _, p_val = stats.fisher_exact(contigency_table, alternative="less")
            p_vals.append(p_val)

        return p_vals

    def _compute_score(self, time_slot, reference_df):
        """
        Compute the score for the cases of the current time slot given the cases of the reference set. If configured a randomization test is performed which might be slow.
        :param time_slot: the time slot which is evaluated
        :param reference_df: a dataframe which contains the cases of the reference set
        :return: the score for the given time slot
        """

        # obtain cases of the current time slot
        cur_cases_df = self.data_stream.get_cases(time_slot)

        # compute the p-values for the observed cases cur_cases_df compared to the reference_df
        p_vals = self._compute_p_values(cur_cases_df, reference_df)

        # perform randomization test, if necessary
        if self.randomization is not None:

            # dataframe which contains all cases
            cases_df = pd.concat([cur_cases_df, reference_df])

            # perform randomizations
            randomization_min_p_vals = []
            for j in range(self.randomization):
                if j % 100 == 0: print("\t\t" + str(j))

                cases_shuffeled_df = cases_df.reindex(np.random.permutation(cases_df.index))
                randomization_reference_df = cases_shuffeled_df[:-len(cur_cases_df)]
                randomization_cases_df = cases_shuffeled_df[-len(cur_cases_df):]

                randomization_p_vals = self._compute_p_values(randomization_cases_df, randomization_reference_df)
                randomization_min_p_vals.append(np.min(randomization_p_vals))

            p_val = np.sum(randomization_min_p_vals <= np.min(p_vals)) / self.randomization
            score = 1. - p_val

        else:
            # Just report the score of the smallest p-value (do not perofrm the randomization test)
            score = 1 - np.min(p_vals)

        return score

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

        # obtain reference set
        if self.version == "2.0":

            reference_df = self._obtain_reference_set_WSARE2(time_slot)

        elif self.version == "2.5":

            reference_df = self._obtain_reference_set_WSARE25(time_slot)

        elif self.version == "3.0":

            reference_df = self._obtain_reference_set_WSARE3(time_slot)

        else:
            raise Exception("Unknown version for WSARE: " + str(self.version))

        # if the reference set contains less than 5 cases return a score of 0
        if len(reference_df) < 5:
            return 0

        # compute the score
        score = self._compute_score(time_slot, reference_df)

        return score
