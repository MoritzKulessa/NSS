import ast
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from simple_spn import spn_handler
from spn.algorithms import Inference
from spn.algorithms import Condition
from spn.algorithms.Statistics import get_structure_stats
from spn.structure.Base import Product, Sum, get_nodes_by_type
from spn.structure.leaves.parametric.Parametric import Gaussian, Categorical

'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''


def _min_p_val(node, children, dtype=np.float64, **kwargs):
    llchildren = np.concatenate(children, axis=1)
    assert llchildren.dtype == dtype
    return np.min(llchildren, axis=1).reshape(-1, 1)


'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''


def _prod_multiply_p_val(node, children, dtype=np.float64, **kwargs):
    matrix = np.concatenate(children, axis=1)
    new_output = []
    for row in matrix:
        non_nan_vals = row[~np.isnan(row)]
        if len(non_nan_vals) >= 1:
            new_output.append(np.prod(non_nan_vals))
        else:
            new_output.append(np.nan)
    new_output = np.array(new_output).reshape(-1, 1)
    assert new_output.dtype == dtype
    return new_output


def _prod_fisher_p_val(node, children, dtype=np.float64, **kwargs):
    matrix = np.concatenate(children, axis=1)
    new_output = []
    for row in matrix:
        non_nan_vals = row[~np.isnan(row)]
        if len(non_nan_vals) == 1:
            new_output.append(non_nan_vals[0])
        elif len(non_nan_vals) > 1:
            non_nan_vals[non_nan_vals == 0] = 0.00000000000000000000000000000000000001
            new_output.append(stats.combine_pvalues(non_nan_vals, method="fisher")[1])
        else:
            new_output.append(np.nan)
    new_output = np.array(new_output).reshape(-1, 1)
    assert new_output.dtype == dtype
    return new_output


def _prod_stouffer_p_val(node, children, dtype=np.float64, **kwargs):
    matrix = np.concatenate(children, axis=1)
    new_output = []
    for row in matrix:
        non_nan_vals = row[~np.isnan(row)]
        if len(non_nan_vals) == 1:
            new_output.append(non_nan_vals[0])
        elif len(non_nan_vals) > 1:
            non_nan_vals[non_nan_vals == 0] = 0.00000000000000000000000000000000000001
            new_output.append(stats.combine_pvalues(non_nan_vals, method="stouffer")[1])
        else:
            new_output.append(np.nan)
    new_output = np.array(new_output).reshape(-1, 1)
    assert new_output.dtype == dtype
    return new_output


'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''


def _sum_weighted_average_p_val(node, children, dtype=np.float64, **kwargs):
    matrix = np.concatenate(children, axis=1)
    new_output = []
    for row in matrix:
        assert (sum(np.isnan(row)) == 0 or sum(np.isnan(row)) == len(row))
        non_nan_vals = row[~np.isnan(row)]
        if len(non_nan_vals) > 0:
            weights = np.array(node.weights, dtype=dtype)
            new_output.append(sum(weights * non_nan_vals))
        else:
            new_output.append(np.nan)
    new_output = np.array(new_output).reshape(-1, 1)
    assert new_output.dtype == dtype
    return new_output

def _sum_weighted_geometric_p_val(node, children, dtype=np.float64, **kwargs):
    matrix = np.concatenate(children, axis=1)
    new_output = []
    for row in matrix:
        assert (sum(np.isnan(row)) == 0 or sum(np.isnan(row)) == len(row))
        non_nan_vals = row[~np.isnan(row)]
        if len(non_nan_vals) > 0:
            non_nan_vals[non_nan_vals == 0] = 0.00000000000000000000000000000000000001
            weights = np.array(node.weights, dtype=dtype)
            new_output.append(np.exp(np.sum(weights * np.log(non_nan_vals)) / np.sum(weights)))
        else:
            new_output.append(np.nan)
    new_output = np.array(new_output).reshape(-1, 1)
    assert new_output.dtype == dtype
    return new_output

def _sum_weighted_harmonic_p_val(node, children, dtype=np.float64, **kwargs):
    matrix = np.concatenate(children, axis=1)
    new_output = []
    for row in matrix:
        assert (sum(np.isnan(row)) == 0 or sum(np.isnan(row)) == len(row))
        non_nan_vals = row[~np.isnan(row)]
        if len(non_nan_vals) > 0:
            non_nan_vals[non_nan_vals == 0] = 0.00000000000000000000000000000000000001
            weights = np.array(node.weights, dtype=dtype)
            new_output.append(np.sum(weights) / np.sum(weights/non_nan_vals))
        else:
            new_output.append(np.nan)
    new_output = np.array(new_output).reshape(-1, 1)
    assert new_output.dtype == dtype
    return new_output

def _sum_weighted_stouffer_p_val(node, children, dtype=np.float64, **kwargs):
    matrix = np.concatenate(children, axis=1)
    new_output = []
    for row in matrix:
        assert (sum(np.isnan(row)) == 0 or sum(np.isnan(row)) == len(row))
        non_nan_vals = row[~np.isnan(row)]
        if len(non_nan_vals) > 1:
            non_nan_vals[non_nan_vals == 0] = 0.00000000000000000000000000000000000001
            new_output.append(stats.combine_pvalues(non_nan_vals, method="stouffer", weights=node.weights)[1])
        else:
            new_output.append(np.nan)
    new_output = np.array(new_output).reshape(-1, 1)
    assert new_output.dtype == dtype
    return new_output


def _sum_weighted_average_corrected_p_val(node, children, dtype=np.float64, **kwargs):
    matrix = np.concatenate(children, axis=1)
    new_output = []
    for row in matrix:
        assert (sum(np.isnan(row)) == 0 or sum(np.isnan(row)) == len(row))
        non_nan_vals = row[~np.isnan(row)]
        if len(non_nan_vals) > 0:
            weights = np.array(node.weights, dtype=dtype)
            new_p_val = sum(weights * non_nan_vals)
            corrected_p_val = np.min([new_p_val * 2.0, 1.0])
            new_output.append(corrected_p_val)
        else:
            new_output.append(np.nan)
    new_output = np.array(new_output).reshape(-1, 1)
    assert new_output.dtype == dtype
    return new_output


def _sum_weighted_average_corrected2_p_val(node, children, dtype=np.float64, **kwargs):
    matrix = np.concatenate(children, axis=1)
    new_output = []
    for row in matrix:
        assert (sum(np.isnan(row)) == 0 or sum(np.isnan(row)) == len(row))
        non_nan_vals = row[~np.isnan(row)]
        if len(non_nan_vals) > 0:
            weights = np.array(node.weights, dtype=dtype)
            index_max = np.argmax(weights)
            if weights[index_max] <= 0.5:
                new_p_val = sum(weights * non_nan_vals)
                corrected_p_val = np.min([new_p_val * 2.0, 1.0])
                new_output.append(corrected_p_val)
            else:
                other_weights = np.array([w for i, w in enumerate(weights) if i != index_max])
                other_vals = np.array([val for i, val in enumerate(non_nan_vals) if i != index_max])
                new_p_val = non_nan_vals[index_max] + sum(other_weights * other_vals) / weights[index_max]
                corrected_p_val = np.min([new_p_val * 2.0, 1.0])
                new_output.append(corrected_p_val)
        else:
            new_output.append(np.nan)
    new_output = np.array(new_output).reshape(-1, 1)
    assert new_output.dtype == dtype
    return new_output


def _sum_weighted_geometric_corrected_p_val(node, children, dtype=np.float64, **kwargs):
    matrix = np.concatenate(children, axis=1)
    new_output = []
    for row in matrix:
        assert (sum(np.isnan(row)) == 0 or sum(np.isnan(row)) == len(row))
        non_nan_vals = row[~np.isnan(row)]
        if len(non_nan_vals) > 0:
            non_nan_vals[non_nan_vals == 0] = 0.00000000000000000000000000000000000001
            weights = np.array(node.weights, dtype=dtype)
            new_p_val = np.exp(np.sum(weights * np.log(non_nan_vals)) / np.sum(weights))
            corrected_p_val = np.min([new_p_val * np.exp(1) * np.log(matrix.shape[1]), 1.0])
            new_output.append(corrected_p_val)
        else:
            new_output.append(np.nan)
    new_output = np.array(new_output).reshape(-1, 1)
    assert new_output.dtype == dtype
    return new_output


def _sum_weighted_harmonic_corrected_p_val(node, children, dtype=np.float64, **kwargs):
    matrix = np.concatenate(children, axis=1)
    new_output = []
    for row in matrix:
        assert (sum(np.isnan(row)) == 0 or sum(np.isnan(row)) == len(row))
        non_nan_vals = row[~np.isnan(row)]
        if len(non_nan_vals) > 0:
            non_nan_vals[non_nan_vals == 0] = 0.00000000000000000000000000000000000001
            weights = np.array(node.weights, dtype=dtype)
            new_p_val = np.sum(weights) / np.sum(weights/non_nan_vals)
            corrected_p_val = np.min([new_p_val * np.exp(1), 1.0])
            new_output.append(corrected_p_val)
        else:
            new_output.append(np.nan)
    new_output = np.array(new_output).reshape(-1, 1)
    assert new_output.dtype == dtype
    return new_output



'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''


def _gaussian_p_val(node, data, min_parameter=0, dtype=np.float64, **kwargs):
    assert len(node.scope) == 1, node.scope
    data = data[:, node.scope[0]]
    nans = np.isnan(data)
    results = np.full(data.shape[0], fill_value=np.nan, dtype=dtype)

    p_vals = 1 - stats.norm.cdf(data[~nans], loc=node.mean, scale=np.max([node.stdev, min_parameter]))

    if np.sum(~nans) > 0: results[~nans] = p_vals
    return results.reshape(-1, 1)


def _poisson_p_val(node, data, min_parameter=0, dtype=np.float64, **kwargs):
    assert len(node.scope) == 1, node.scope
    data = data[:, node.scope[0]]
    nans = np.isnan(data)
    results = np.full(data.shape[0], fill_value=np.nan, dtype=dtype)

    p_vals = 1. - stats.poisson.cdf(data[~nans], np.max([node.mean, min_parameter]))

    if np.sum(~nans) > 0: results[~nans] = p_vals
    return results.reshape(-1, 1)


def _nb_p_val(node, data, min_parameter=0, dtype=np.float64, **kwargs):
    assert len(node.scope) == 1, node.scope
    data = data[:, node.scope[0]]
    nans = np.isnan(data)
    results = np.full(data.shape[0], fill_value=np.nan, dtype=dtype)

    mu = np.max([node.mean, min_parameter])
    sig = node.stdev
    m = (sig ** 2 - mu)
    if m <= 0: m = 0.0000001
    r = (mu ** 2) / m
    p = r / (r + mu)

    p_vals = 1. - stats.nbinom.cdf(data[~nans], n=r, p=p)

    if np.sum(~nans) > 0: results[~nans] = p_vals
    return results.reshape(-1, 1)


'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''


class DensitySPN(object):

    def __init__(self, syndrome_counter, rdc=0.3, mis=0.1, rows="kmeans", cols="rdc", use_cache=True):
        self.syndrome_counter = syndrome_counter
        self.data_stream = syndrome_counter.data_stream

        self.rdc = rdc
        self.mis = mis
        self.rows = rows
        self.cols = cols
        self.use_cache = use_cache
        self.spn, self.f_info = self._create_spn(self.data_stream.get_info()["start_test_part"] - 1)

    def _create_spn(self, till_time_slot):

        data_ident = self.data_stream.get_ident() + "$" + self.syndrome_counter.get_hash()
        spn_ident = data_ident + "$till=" + str(till_time_slot)

        if spn_handler.exist_spn(spn_ident, self.rdc, self.mis, self.cols, self.rows) and self.use_cache:
            spn, f_info, const_time = spn_handler.load_spn(spn_ident, self.rdc, self.mis, self.cols, self.rows)
        else:
            first_time_slot = self.syndrome_counter.data_stream.get_info()["first_time_slot"]
            last_time_slot = self.syndrome_counter.data_stream.get_info()["last_time_slot"]

            syndrome_counts_df = self.syndrome_counter.get_syndrome_df(first_time_slot, last_time_slot)
            assert ((np.diff(syndrome_counts_df["time_slot"]) > 0).all())  # assert time slots are sorted
            bool_array = (syndrome_counts_df["time_slot"] == till_time_slot)
            assert (sum(bool_array) == 1)  # only one entry for time slot
            last_index = list(bool_array).index(True)
            syndrome_counts_df = syndrome_counts_df.drop("time_slot", axis=1)

            spn_handler.create_spn_data(syndrome_counts_df, data_ident, ["numeric"] * len(syndrome_counts_df.columns))
            spn_data_df, f_info, parametric_types = spn_handler.load_spn_data(data_ident)
            spn_data_df = spn_data_df[:last_index]
            assert (len(spn_data_df) == last_index)
            spn, const_time = spn_handler.learn_parametric_spn(spn_data_df.values, parametric_types, self.rdc, self.mis,
                                                               self.cols, self.rows)
            spn_handler.save_spn(spn, const_time, spn_ident, self.rdc, self.mis, self.cols, self.rows, f_info)

        return spn, f_info

    def evaluate(self, time_slot):
        """
        Computes the score for the given time slot.
        :param time_slot: the time slot for which the score is computed
        :return: the score for the given time slot
        """

        # obtain previous and current syndrome counts
        check_df = self.syndrome_counter.get_counts(time_slot)

        # Compute density of the observation
        data = check_df.values
        density = Inference.likelihood(self.spn, data, dtype=np.float64).reshape(len(data))[0]

        # negative density represents the score
        score = -density
        return score


'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''


class SyndromicSPN(object):

    def __init__(self, syndrome_counter, distribution, min_parameter=1.0, evidence="single",
                 product_combine="multiply", sum_combine="weighted_average",
                 rdc=0.3, mis=0.1, rows="kmeans", cols="rdc", cluster_univariate=False, use_cache=True):
        '''
        :param syndrome_counter: module to count the occurrences of syndromes
        :param distribution: the distribution used in the leaves ("gaussian"/"poisson"/"nb")
        :param min_parameter: the minimum parameter used for mean or std before fitting the distribution in a leave
        :param evidence: evaluation strategy for the query set ("single" is Q1, "single_double" is Q2)
        :param product_combine: the method to combine p-values in the product ndoe ("multiply"/"fisher"/"stouffer")
        :param sum_combine: the method to combine p-values in the product ndoe ("weighted_average"/"weighted_harmonic"/"weighted_geometric")
        :param rdc: the rdc parameter for SPN construction
        :param mis: the mis parameter for SPN construction
        :param rows: the method used create sum nodes for SPN construction
        :param cols: the method used create product nodes for SPN construction
        :param cluster_univariate: allow to split univariate attributes for SPN construction
        :param use_cache: if true and exists, load pre-computed SPN from file-system
        '''

        self.syndrome_counter = syndrome_counter
        self.data_stream = syndrome_counter.data_stream
        self.distribution = distribution
        self.min_parameter = min_parameter
        self.evidence = evidence
        self.product_combine = product_combine
        self.sum_combine = sum_combine

        self.rdc = rdc
        self.mis = mis
        self.rows = rows
        self.cols = cols
        self.cluster_univariate = cluster_univariate
        self.use_cache = use_cache
        self.spn, self.f_info = self._create_spn(self.data_stream.get_info()["start_test_part"] - 1)

        # choose double evidences
        self.eval_combos = self._eval_combos()

        # choose combination for product node
        if self.product_combine == "multiply":
            product_likelihood = _prod_multiply_p_val
        elif self.product_combine == "fisher":
            product_likelihood = _prod_fisher_p_val
        elif self.product_combine == "stouffer":
            product_likelihood = _prod_stouffer_p_val
        else:
            raise Exception("Unknown product-combine : " + str(self.product_combine))

        # choose combination for sum node
        if self.sum_combine == "weighted_average":
            sum_likelihood = _sum_weighted_average_p_val
        elif self.sum_combine == "weighted_geometric":
            sum_likelihood = _sum_weighted_geometric_p_val
        elif self.sum_combine == "weighted_harmonic":
            sum_likelihood = _sum_weighted_harmonic_p_val
        elif self.sum_combine == "weighted_stouffer":
            sum_likelihood = _sum_weighted_stouffer_p_val
        elif self.sum_combine == "weighted_average_corrected":
            sum_likelihood = _sum_weighted_average_corrected_p_val
        elif self.sum_combine == "weighted_average_corrected2":
            sum_likelihood = _sum_weighted_average_corrected2_p_val
        elif self.sum_combine == "weighted_geometric_corrected":
            sum_likelihood = _sum_weighted_geometric_corrected_p_val
        elif self.sum_combine == "weighted_harmonic_corrected":
            sum_likelihood = _sum_weighted_harmonic_corrected_p_val
        else:
            raise Exception("Unknown sum-combine : " + str(self.sum_combine))

        # select node likelihood with respect to the used distribution
        if self.distribution == "gaussian":
            self.node_likelihood = {Sum: sum_likelihood, Product: product_likelihood, Gaussian: _gaussian_p_val}
        elif self.distribution == "poisson":
            self.node_likelihood = {Sum: sum_likelihood, Product: product_likelihood, Gaussian: _poisson_p_val}
        elif self.distribution == "nb":
            self.node_likelihood = {Sum: sum_likelihood, Product: product_likelihood, Gaussian: _nb_p_val}
        else:
            raise Exception("Unknown compute p-values: " + str(self.distribution))

    def _create_spn(self, till_time_slot):

        data_ident = self.data_stream.get_ident() + "$" + self.syndrome_counter.get_hash()
        spn_ident = data_ident + "$till=" + str(till_time_slot)

        if spn_handler.exist_spn(spn_ident, self.rdc, self.mis, self.cols, self.rows, self.cluster_univariate) and self.use_cache:
            spn, f_info, const_time = spn_handler.load_spn(spn_ident, self.rdc, self.mis, self.cols, self.rows, self.cluster_univariate)
        else:
            # get info
            first_time_slot = self.syndrome_counter.data_stream.get_info()["first_time_slot"]

            if not (spn_handler.exist_spn_data(data_ident) and self.use_cache):
                # get all syndrome counts
                last_time_slot = self.syndrome_counter.data_stream.get_info()["last_time_slot"]
                syndrome_df = self.syndrome_counter.get_syndrome_df(first_time_slot, last_time_slot)
                assert (len(syndrome_df) == last_time_slot - first_time_slot + 1)

                # get environmental settings
                all_env_df = self.data_stream.get_all_envs()
                df = pd.merge(syndrome_df, all_env_df, how="inner", on=["time_slot"])
                feature_types = ["numeric"] * (len(syndrome_df.columns) - 1) + ["discrete"] * (len(all_env_df.columns) - 1)

                # sort data
                df = df.sort_values("time_slot")

                # create spn data
                df = df.drop("time_slot", axis=1)
                spn_handler.create_spn_data(df, data_ident, feature_types)

            # load spn data
            spn_data_df, f_info, parametric_types = spn_handler.load_spn_data(data_ident)
            spn_data_df = spn_data_df[:till_time_slot - first_time_slot + 1]

            # create spn
            spn, const_time = spn_handler.learn_parametric_spn(spn_data_df.values, parametric_types, self.rdc, self.mis, self.cols, self.rows, self.cluster_univariate)

            # regularize categorical nodes (avoid 0 probability due to usage of log in conditioning)
            nodes = get_nodes_by_type(spn, Categorical)
            for node in nodes:
                for i in range(len(node.p)):
                    node.p[i] = (node.p[i] + 0.001) / (1.0 + len(node.p) * 0.001)

            # save spn
            spn_handler.save_spn(spn, const_time, spn_ident, self.rdc, self.mis, self.cols, self.rows, self.cluster_univariate, f_info)

        print(get_structure_stats(spn))
        return spn, f_info

    def _eval_combos(self):

        if self.syndrome_counter.combos != 1:
            print("No double evidences, since syndromes do not only contain single conditions")
            return []

        f_dict = {}
        for i, f_name in enumerate(self.f_info.get_all_feature_names()):
            if self.f_info.is_numeric(f_name):
                syndrome = ast.literal_eval(f_name)
                col = syndrome[0][0]
                if col not in f_dict:
                    f_dict[col] = []
                f_dict[col].append(i)

        eval_combos = []
        for s1, s2 in itertools.combinations([f_name for f_name, _ in f_dict.items()], r=2):
            for val1 in f_dict[s1]:
                for val2 in f_dict[s2]:
                    eval_combos.append([val1, val2])

        return eval_combos


    def evaluate(self, time_slot):
        """
        Computes the score for the given time slot.
        :param time_slot: the time slot for which the score is computed
        :return: the score for the given time slot
        """

        # obtain current syndrome counts
        check_df = self.syndrome_counter.get_counts(time_slot)
        assert (len(check_df) == 1)
        check_values = list(check_df.values[0])

        # obtain current env nad transform according to spn data
        env_df = self.data_stream.get_env(time_slot)
        env_values = []
        if not env_df.empty:
            assert (len(env_df) == 1)
            env_values = [self.f_info.get_inverse_value_mapping(col)[list(env_df[col])[0]] for col in env_df.columns]

        # condition SPN based on the environmental setting
        if len(env_values) > 0:
            env_evidence = np.array([np.nan] * len(check_values) + env_values)
            spn = Condition.condition(self.spn, env_evidence.reshape(1, -1))
        else:
            spn = self.spn

        # Check syndromes with one attribute
        single_evidences = []
        if "single" in self.evidence:
            for i, val in enumerate(check_values):
                single_evidence = np.full(len(check_values), fill_value=np.nan)
                single_evidence[i] = val
                single_evidences.append(single_evidence)

        # check syndromes with two attributes
        double_evidences = []
        if "double" in self.evidence:
            for eval_combo in self.eval_combos:
                double_evidence = np.full(len(check_values), fill_value=np.nan)
                for index in eval_combo:
                    double_evidence[index] = check_values[index]
                double_evidences.append(double_evidence)

        evidences = np.array(single_evidences + double_evidences)

        if "condition" in self.evidence:
            ''' ONLY FOR DEVELOPMENT '''
            if "condition1" in self.evidence:
                x = np.array(check_values).reshape(1, -1)
                all_result = Inference.likelihood(spn, x, dtype=np.float64, min_parameter=self.min_parameter).T[0]

                check_values = np.array(check_values)

                divide_evidences = []
                for evidence in evidences:
                    e = np.array([np.nan] * len(evidence))
                    e[np.isnan(evidence)] = check_values[np.isnan(evidence)]
                    divide_evidences.append(e)
                divide_evidences = np.array(divide_evidences)
                probs = Inference.likelihood(spn, divide_evidences, dtype=np.float64, min_parameter=self.min_parameter).T[0]

                probs[probs == 0] = 0.000000000000000000000000000000000000000000000000001

                p_values = (all_result[0]/probs)

            elif "condition2" in self.evidence:
                x = np.array(check_values).reshape(1, -1)
                all_result = Inference.likelihood(spn, x, dtype=np.float64, node_likelihood=self.node_likelihood, min_parameter=self.min_parameter).T[0]

                check_values = np.array(check_values)

                divide_evidences = []
                for evidence in evidences:
                    e = np.array([np.nan] * len(evidence))
                    e[np.isnan(evidence)] = check_values[np.isnan(evidence)]
                    divide_evidences.append(e)
                divide_evidences = np.array(divide_evidences)
                probs = Inference.likelihood(spn, divide_evidences, dtype=np.float64, node_likelihood=self.node_likelihood, min_parameter=self.min_parameter).T[0]

                probs[probs == 0] = 0.000000000000000000000000000000000000000000000000001
                p_values = all_result[0] / probs
            elif "condition3" in self.evidence:

                check_values = np.array(check_values)

                p_values = []
                for evidence in evidences:
                    e = np.array([np.nan] * len(evidence))
                    e[np.isnan(evidence)] = check_values[np.isnan(evidence)]
                    x_spn = Condition.condition(spn, e.reshape(1, -1))
                    p_value = Inference.likelihood(x_spn, evidence.reshape(1, -1), dtype=np.float64, node_likelihood=self.node_likelihood, min_parameter=self.min_parameter).T[0]
                    p_values.append(p_value[0])

        else:
            p_values = Inference.likelihood(spn, evidences, dtype=np.float64, node_likelihood=self.node_likelihood, min_parameter=self.min_parameter).T[0]

        min_p_val = np.min(p_values)
        score = 1 - min_p_val

        return score
