import os
import sys

file_dir = os.path.dirname(os.path.realpath(__file__)) + "/../../"
sys.path.append(file_dir)

from util import io
from evaluation import evaluation_engine
from data.data_stream import WSARE_DATA, ED_DATA_RANDOM_ONE_STREAM3
from data.syndrome_counting import BasicSyndromeCounting
from algo.benchmarks import Benchmark
import logging

logger = logging.getLogger(__name__)

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
        dataset_name = "ED_DATA_RANDOM_ONE_STREAM"  # ED_DATA_RANDOM
        environmental_valriables = True

    # the settings for the algorithms
    algo_settings = [
        [Benchmark, {"distribution": "gaussian", "min_parameter": 1, "incremental": False}],
        [Benchmark, {"distribution": "poisson", "min_parameter": 1, "incremental": False}],
        [Benchmark, {"distribution": "nb", "min_parameter": 1, "incremental": False}],
    ]

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
