import os
import sys

file_dir = os.path.dirname(os.path.realpath(__file__)) + "/../../"
sys.path.append(file_dir)

import numpy as np
from util import io
from evaluation import evaluation_engine
from data.data_stream import WSARE_DATA
from algo.dmss import DMSS
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

    ref_windows = [np.arange(-30, 0), np.arange(-90, 0), np.arange(-180, 0)]
    min_supports = [3, 5, 7, 9, 11, 13, 15]

    # the settings for the algorithms
    algo_settings = []
    for ref_window in ref_windows:
        for min_support in min_supports:
            algo_settings.append(
                [DMSS, {"min_support_set": min_support, "min_support_rule": min_support, "ref_windows": ref_window}])

    # the settings for the syndrome counters
    syndrome_counter_settings = [
        [None, {}]
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
