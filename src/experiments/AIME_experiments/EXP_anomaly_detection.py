import os
import sys

file_dir = os.path.dirname(os.path.realpath(__file__)) + "/../../"
sys.path.append(file_dir)

import itertools
from util import io
from evaluation import evaluation_engine
from data.data_stream import WSARE_DATA, ED_DATA_RANDOM_ONE_STREAM3
from data.syndrome_counting import BasicSyndromeCounting
from algo.anomaly_detection import SyndromeCountAnomalyDetector
import logging

logger = logging.getLogger(__name__)


def get_combinations(parameters):
    param_names = [parameter[0] for parameter in parameters]
    param_values = [parameter[1] for parameter in parameters]

    parameter_combinations = []
    for combo in itertools.product(*param_values):

        parameter_combination = {}
        for i, param_name in enumerate(param_names):
            parameter_combination[param_name] = combo[i]
        parameter_combinations.append(parameter_combination)
    return parameter_combinations


if __name__ == '__main__':

    # show logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # to compute each data stream independently on the cluster
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
        data_stream_ids = [int(sys.argv[2])]
        environmental_valriables = bool(int(sys.argv[3]))

        if len(sys.argv) > 4:
            select = int(sys.argv[4])
        else:
            select = None

    else:
        data_stream_ids = range(0, 100)
        select = None
        dataset_name = "ED_DATA_RANDOM_ONE_STREAM2"
        environmental_valriables = True

    algo_settings = []

    # the settings for the algorithms
    for s in [0, 1, 2]:
        if s == 0:
            ################
            # Auto Encoder #
            ################
            from pyod.models.auto_encoder import AutoEncoder

            parameters = [
                ["hidden_neurons", [[32, 16, 16, 32], [16, 8, 8, 16], [8, 4, 4, 8]]],
                ["epochs", [100, 500]],
                ["dropout_rate", [0.1, 0.2]],
                ["l2_regularizer", [0.1, 0.01, 0.005, 0.001]],
                ["random_state", [1]],
                ["output_activation", ['sigmoid', 'relu']],
                ["preprocessing", [True, False]]
            ]
            param_combos = get_combinations(parameters)
            algo_settings += [
                [SyndromeCountAnomalyDetector,
                 {"incremental": False, "anomaly_detector_class": AutoEncoder, "anomaly_detector_parameters": param_combo, "incremental": False}
                 ] for param_combo in param_combos]

        elif s == 1:
            #################
            # One-Class SVM #
            #################
            from pyod.models.ocsvm import OCSVM

            parameters = [
                ["kernel", ["linear"]],
                ["nu", [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]],
                ["coef0", [0, 1]],
            ]
            param_combos = get_combinations(parameters)
            algo_settings += [
                [SyndromeCountAnomalyDetector,
                 {"anomaly_detector_class": OCSVM, "anomaly_detector_parameters": param_combo, "incremental": False}
                 ] for param_combo in param_combos]
            algo_settings += [
                [SyndromeCountAnomalyDetector,
                 {"normalize_data": True, "anomaly_detector_class": OCSVM, "anomaly_detector_parameters": param_combo, "incremental": False}
                 ] for param_combo in param_combos]

            parameters = [
                ["kernel", ["rbf"]],
                ["nu", [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]],
                ["degree", [2]],
                ["gamma", ['auto', 'scale']],
                ["coef0", [0]],
            ]

            param_combos = get_combinations(parameters)
            algo_settings += [
                [SyndromeCountAnomalyDetector,
                 {"anomaly_detector_class": OCSVM, "anomaly_detector_parameters": param_combo, "incremental": False}
                 ] for param_combo in param_combos]
            algo_settings += [
                [SyndromeCountAnomalyDetector,
                 {"normalize_data": True, "anomaly_detector_class": OCSVM, "anomaly_detector_parameters": param_combo, "incremental": False}
                 ] for param_combo in param_combos]


        elif s == 2:
            ####################
            # Gaussian Mixture #
            ####################
            from sklearn.mixture import GaussianMixture

            parameters = [
                ["n_components", [1, 3, 5]],
                ["tol", [1e-2, 1e-3, 1e-4]],
                ["reg_covar", [1e-5, 1e-6, 1e-7]],
                ["max_iter", [100, 200]],
                ["random_state", [1]],
            ]
            param_combos = get_combinations(parameters)
            algo_settings += [
                [SyndromeCountAnomalyDetector,
                 {"anomaly_detector_class": GaussianMixture, "anomaly_detector_parameters": param_combo, "incremental": False}
                 ] for param_combo in param_combos]

    if select is not None:
        algo_settings = [algo_settings[select]]

    # the settings for the syndrome counters
    syndrome_counter_settings = [
        [BasicSyndromeCounting, {"combos": 2}],
        [BasicSyndromeCounting, {"combos": 1}]
    ]

    # select data streams
    if dataset_name == "WSARE":
        data_stream_class = WSARE_DATA
        multiple_data_stream_params = [{"data_stream_id": data_stream_id,
                                        "environmental_variables": environmental_valriables}
                                       for data_stream_id in data_stream_ids]
    elif dataset_name == "ED_DATA_RANDOM_ONE_STREAM3":
        data_stream_class = ED_DATA_RANDOM_ONE_STREAM3
        multiple_data_stream_params = [{"environmental_variables": environmental_valriables, "outbreak_sizes": 40, "n_syndromes": 100}]
    else:
        raise Exception("Unknown dataset name: " + str(dataset_name))


    # perform evaluation
    df = evaluation_engine.evaluate_algorithms(
        data_stream_class=data_stream_class,
        multiple_data_stream_params=multiple_data_stream_params,
        algo_settings=algo_settings,
        syndrome_counter_settings=syndrome_counter_settings,
        recompute=False
    )

    io.print_pretty_table(df[["algo_params", "syndrome_counter_params", "macro AMOC 0.05"]])
