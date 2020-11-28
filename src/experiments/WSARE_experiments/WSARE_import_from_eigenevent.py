import os
import sys

file_dir = os.path.dirname(os.path.realpath(__file__)) + "/../../"
sys.path.append(file_dir)

import scipy.io
import numpy as np
from evaluation import measures
from util import io
import logging

logger = logging.getLogger(__name__)

if __name__ == '__main__':

    # Load eigenevent results
    imported_results = scipy.io.loadmat(io.get_project_directory() + '_data/from_eigenevent_repository/eigenevent.mat')

    # Load wsare results
    wsare_results = scipy.io.loadmat(io.get_project_directory() + '_data/from_eigenevent_repository/wsare_results.mat')

    imported_results["wsare2"] = wsare_results["wsare2"]
    imported_results["wsare25"] = wsare_results["wsare25"]
    imported_results["wsare3"] = wsare_results["wsare3"]

    methods = ["eigenevent", "wsare2", "wsare25", "wsare3"]
    for method in methods:
        measure_results = []
        for data_stream_id in range(0, 100):
            # read extract information about the outbreak
            path = io.get_project_directory() + "_data/wsare/"
            f = open(path + "release_daycode.txt", "r")
            lines = f.readlines()
            outbreak_time_slot = int(lines[data_stream_id].split(" ")[1]) - 73779
            outbreak_duration = 14
            f.close()
            outbreaks = [[outbreak_time_slot - 365, np.min(
                [outbreak_time_slot + outbreak_duration - 1, 729]) - 365]]  # end of outbreak is inclusive

            # select the correct p-values
            p_vals = list(imported_results[method][0][data_stream_id][0])
            scores = 1 - np.array(p_vals)

            # Compute area under partial AMOC-curve (FAR <= 5%)
            roc_values = measures.compute_roc_values(scores, outbreaks)
            amoc_auc5 = measures.compute_area_under_curve(roc_values, x_measure="FAR*0.3", y_measure="detectionDelay")
            measure_results.append(amoc_auc5)

        print(method)
        print("Macro-averaged area under partial AMOC-curve (FAR <= 5%): " + str(np.mean(measure_results)))
