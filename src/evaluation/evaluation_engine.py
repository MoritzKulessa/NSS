import os
import hashlib
import itertools
import pandas as pd
from evaluation import measures
from util import io
import logging

logger = logging.getLogger(__name__)

'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''


def get_hash(string, num_hash_chars=32):
    return hashlib.md5(string.encode()).hexdigest()[-num_hash_chars:]


'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''


def evaluate_data_streams(data_stream_class, data_stream_params={}, algo_class=None, algo_params={},
                          syndrome_counter_class=None, syndrome_counter_params={}, recompute=False):
    # generate identifier and identifier hash
    identifier = str(data_stream_class.__name__) + "_" + str(data_stream_params) + "$" + \
                 str(algo_class.__name__) + "_" + str(algo_params)
    if syndrome_counter_class is not None:
        identifier += "$" + str(syndrome_counter_class.__name__) + "_" + str(syndrome_counter_params)
    identifier_hash = get_hash(identifier)

    logger.info("Evaluate " + str(identifier))

    # perhaps load results from file system
    result_path = io.get_project_directory() + "_results/" + identifier_hash + ".pkl"
    if os.path.exists(result_path) and (not recompute):
        logger.info("\tLoad results from file...")

        result_dict = io.load(identifier_hash, loc="_results/")

    else:
        logger.info("\tCompute scores...")

        # initialize classes
        data_stream = data_stream_class(**data_stream_params)
        data_info = data_stream.get_info()
        if syndrome_counter_class is not None:
            syndrome_counter = syndrome_counter_class(data_stream, **syndrome_counter_params)
            algo = algo_class(syndrome_counter, **algo_params)
        else:
            algo = algo_class(data_stream, **algo_params)

        # compute scores
        scores = []
        for time_slot in range(data_info["start_test_part"], data_info["last_time_slot"] + 1):
            if time_slot % 1 == 0:
                logger.info("\t\tEvaluate time slot: " + str(time_slot))
            scores.append(algo.evaluate(time_slot))

        # algin the outbreaks with the scores (e.g. scores only start from the test part of the data stream
        outbreaks = []
        for [start_outbreak, end_outbreak] in data_info["outbreaks"]:
            outbreaks.append(
                [start_outbreak - data_info["start_test_part"], end_outbreak - data_info["start_test_part"]])

        result_dict = {"scores": scores, "outbreaks": outbreaks}

        # save results
        io.save(result_dict, identifier_hash, loc="_results")

    return result_dict


'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''


def evaluate_multiple_data_streams(data_stream_class, multiple_data_stream_params=[{}], algo_class=None, algo_params={},
                                   syndrome_counter_class=None, syndrome_counter_params={}, recompute=False):
    # Evaluate data streams
    result_dicts = []
    for data_stream_params in multiple_data_stream_params:
        result_dict = evaluate_data_streams(data_stream_class=data_stream_class, data_stream_params=data_stream_params,
                                            algo_class=algo_class, algo_params=algo_params,
                                            syndrome_counter_class=syndrome_counter_class,
                                            syndrome_counter_params=syndrome_counter_params,
                                            recompute=recompute)
        result_dicts.append(result_dict)

    # compute values for the measures
    measure_result = measures.evaluate_results(result_dicts)
    return measure_result


'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''


def evaluate_algorithms(data_stream_class, multiple_data_stream_params=[{}], algo_settings=[],
                        syndrome_counter_settings=[], recompute=False):
    algo_results = []
    all_combos = itertools.product(*[syndrome_counter_settings, algo_settings])
    for [syndrome_counter_class, syndrome_counter_params], [algo_class, algo_params] in all_combos:
        measure_result = evaluate_multiple_data_streams(data_stream_class, multiple_data_stream_params,
                                                        algo_class, algo_params,
                                                        syndrome_counter_class, syndrome_counter_params,
                                                        recompute=recompute)

        algo_result = {
            "data_stream_class": data_stream_class,
            "multiple_data_stream_params": str(multiple_data_stream_params),
            "algo_class": algo_class,
            "algo_params": algo_params,
            "syndrome_counter_class": syndrome_counter_class,
            "syndrome_counter_params": syndrome_counter_params
        }
        algo_result.update(measure_result)
        algo_results.append(algo_result)

    # create dataframe
    df = pd.DataFrame(algo_results)
    return df
