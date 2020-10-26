import os
import sys

file_dir = os.path.dirname(os.path.realpath(__file__)) + "/../"
sys.path.append(file_dir)

import numpy as np
from util import io
from evaluation import evaluation_engine
from data.data_stream import WSARE_DATA
from data.syndrome_counting import BasicSyndromeCounting
from algo.benchmarks import Benchmark
import logging

logger = logging.getLogger(__name__)

if __name__ == '__main__':

    # show logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # to compute each data stream independently on the cluster
    if len(sys.argv) > 1:
        data_stream_ids = [int(sys.argv[1])]
    else:
        data_stream_ids = range(0, 100)

    # the settings for the algorithms
    algo_settings = []
    for min_parameter in [round(x, 2) for x in np.arange(0.2, 2.2, 0.2)]:
        algo_settings.append([Benchmark, {"distribution": "gaussian", "min_parameter": min_parameter}])
        algo_settings.append([Benchmark, {"distribution": "poisson", "min_parameter": min_parameter}])
        algo_settings.append([Benchmark, {"distribution": "nb", "min_parameter": min_parameter}])

    # the settings for the syndrome counters
    syndrome_counter_settings = [
        [BasicSyndromeCounting, {}]
    ]

    # perform evaluation
    df = evaluation_engine.evaluate_algorithms(
        data_stream_class=WSARE_DATA,
        multiple_data_stream_params=[{"data_stream_id": data_stream_id} for data_stream_id in data_stream_ids],
        algo_settings=algo_settings,
        syndrome_counter_settings=syndrome_counter_settings,
        recompute=False
    )

    # print results
    io.print_pretty_table(df[["algo_params", "macro-averaged parital AMOC 5%"]])