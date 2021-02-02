import os
import sys

file_dir = os.path.dirname(os.path.realpath(__file__)) + "/../../"
sys.path.append(file_dir)
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from util import io
from evaluation import evaluation_engine
from data.data_stream import ED_DATA_RANDOM
from data.syndrome_counting import BasicSyndromeCounting
from algo.wsare import WSARE
import logging

logger = logging.getLogger(__name__)

if __name__ == '__main__':

    # show logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # to compute each data stream independently on the cluster
    if len(sys.argv) > 1:
        data_stream_ids = [int(sys.argv[1])]
        if len(sys.argv) > 4:
            data_stream_ids = range(0, 100)
    else:
        data_stream_ids = range(0, 5)

    k = 1.0

    # the settings for the algorithms
    algo_settings = [
        [WSARE, {"version": "2.0", "randomization": None}],
        [WSARE, {"version": "2.5", "randomization": None}],
        [WSARE, {"version": "3.0", "randomization": None}],
    ]

    # the settings for the syndrome counters
    syndrome_counter_settings = [
        [BasicSyndromeCounting, {"combos": 2}],
        [BasicSyndromeCounting, {"combos": 1}]
    ]

    # perform evaluation
    df = evaluation_engine.evaluate_algorithms(
        data_stream_class=ED_DATA_RANDOM,
        multiple_data_stream_params=[{"data_stream_id": data_stream_id, "k": k} for data_stream_id in data_stream_ids],
        algo_settings=algo_settings,
        syndrome_counter_settings=syndrome_counter_settings,
        recompute=False
    )

    # print results
    io.print_pretty_table(
        df[["algo_class", "algo_params", "syndrome_counter_params", "macro-averaged parital AMOC 5%"]])
