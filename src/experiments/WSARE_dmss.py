import os
import sys

file_dir = os.path.dirname(os.path.realpath(__file__)) + "/../"
sys.path.append(file_dir)

import hashlib
import numpy as np
from data.data_stream import WSARE_DATA
from algo.dmss import DMSS
from evaluation import measures
from util import io
import logging

logger = logging.getLogger(__name__)


def get_hash(string, numHashChars=10):
    return hashlib.md5(string.encode()).hexdigest()[-numHashChars:]


if __name__ == '__main__':

    # to compute each data stream independently on the cluster
    if len(sys.argv) > 1:
        data_stream_ids = [int(sys.argv[1])]
    else:
        data_stream_ids = range(0, 100)

    # show logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # if false, already computed results may be loaded from the file system
    recompute = False
    module_name = os.path.basename(__file__)[:-3]

    min_supports = [3, 5, 7]
    ref_windows_confs = [np.arange(-30, 0), np.arange(-90, 0), np.arange(-180, 0)]

    for min_support in min_supports:
        for ref_windows in ref_windows_confs:

            # Compute the results for all data streams
            measure_results = []
            for data_stream_id in data_stream_ids:
                logger.info("Evaluate data stream: " + str(data_stream_id))

                # load data
                data_stream = WSARE_DATA(data_stream_id)
                data_info = data_stream.get_info()

                # configure algorithm
                algo = DMSS(data_stream, min_support_set=min_support, min_support_rule=min_support,
                            ref_windows=ref_windows)

                # path to store computed results
                result_ident = module_name + "_min_support=" + str(min_support) + "_ref_windows=" + get_hash(
                    str(ref_windows)) + "_" + data_stream.get_ident()
                cache_path = io.get_project_directory() + "_results/" + result_ident + ".pkl"

                if os.path.exists(cache_path) and (not recompute):
                    logger.info("\tLoad results from file...")

                    scores = io.load(result_ident, loc="_results/")

                else:
                    logger.info("\tCompute scores...")

                    scores = []
                    for time_slot in range(data_info["start_test_part"], data_info["last_time_slot"] + 1):
                        if time_slot % 100 == 0: logger.info("\t\tEvaluate time slot: " + str(time_slot))
                        scores.append(algo.evaluate(time_slot))

                    # Save results
                    io.save(scores, result_ident, loc="_results")

                # Algin the outbreaks with the scores (e.g. scores only start from the test part of the data stream
                outbreaks = []
                for [start_outbreak, end_outbreak] in data_info["outbreaks"]:
                    outbreaks.append(
                        [start_outbreak - data_info["start_test_part"], end_outbreak - data_info["start_test_part"]])

                # Compute area under partial AMOC-curve (FAR <= 5%)
                roc_values = measures.compute_roc_values(scores, outbreaks)
                amoc_auc5 = measures.compute_area_under_curve(roc_values, x_measure="FAR*0.05",
                                                              y_measure="detectionDelay")
                measure_results.append(amoc_auc5)
                print(amoc_auc5)

            print()
            print("Macro-averaged area under partial AMOC-curve (FAR <= 5%): " + str(np.mean(measure_results)))
