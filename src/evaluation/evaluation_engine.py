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


def get_identifier(data_stream_class, data_stream_params={}, algo_class=None, algo_params={},
                   syndrome_counter_class=None, syndrome_counter_params={}):
    """
    Returns the identifier (string representation) for the specific setting.

    :param data_stream_class: the data stream class
    :param data_stream_params: the data stream parameters
    :param algo_class: the algorithm class
    :param algo_params: the algorithm parameters
    :param syndrome_counter_class: the syndrome counter class
    :param syndrome_counter_params: the syndrome counter parameters
    :return: the string identifier for the specific setting
    """
    # generate identifier and identifier hash
    identifier = str(data_stream_class.__name__) + "_" + str(data_stream_params) + "$" + \
                 str(algo_class.__name__) + "_" + str(algo_params)
    if syndrome_counter_class is not None:
        identifier += "$" + str(syndrome_counter_class.__name__) + "_" + str(syndrome_counter_params)
    return identifier


'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''


def evaluate_data_stream(data_stream_class, data_stream_params={}, algo_class=None, algo_params={},
                         syndrome_counter_class=None, syndrome_counter_params={}, recompute=False):
    """
    Evaluate the specified data stream with the specified algorithm. The results will be saved to the "_results" folder.
    If results exist they will be loaded from the file system.

    :param data_stream_class: the data stream class
    :param data_stream_params: the data stream parameters
    :param algo_class: the algorithm class
    :param algo_params: the algorithm parameters
    :param syndrome_counter_class: the syndrome counter class
    :param syndrome_counter_params: the syndrome counter parameters
    :param recompute: if True, pre-computed results will not be loaded from the file system
    :return: a dictionary containing the results of the experiment
    ("scores" --> the computed scores of the algorithm and "outbreaks" --> the indexes of the outbreaks)
    """
    # generate identifier and identifier hash
    identifier = get_identifier(data_stream_class, data_stream_params, algo_class, algo_params,
                                syndrome_counter_class, syndrome_counter_params)
    identifier_hash = get_hash(identifier)

    logger.info("Evaluate " + identifier)

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
            if time_slot % 100 == 0:
                logger.info("\t\tEvaluate time slot: " + str(time_slot))
            scores.append(algo.evaluate(time_slot))

        # speed up for experiments on real data (e.g. do not need to re-evaluate other time slots than the outbreak day)
        if data_stream_class.__name__ == "ED_DATA" and data_stream_params["scenario"] != 0:

            start_test_part = 365

            new_data_stream_params = data_stream_params.copy()
            new_data_stream_params["scenario"] = 0
            new_data_stream_params["outbreak_size"] = 0
            new_data_stream_params["data_stream_id"] = 0
            new_identifier = get_identifier(data_stream_class, new_data_stream_params, algo_class, algo_params,
                                            syndrome_counter_class, syndrome_counter_params)
            new_identifier_hash = get_hash(new_identifier)
            loaded_result_dict = io.load(new_identifier_hash, loc="_results/")
            loaded_scores = loaded_result_dict["scores"]

            outbreak_time_slots = range(data_info["start_test_part"], data_info["last_time_slot"] + 1)
            assert (len(outbreak_time_slots) == 1)
            outbreak_time_slot = outbreak_time_slots[0]

            loaded_scores[outbreak_time_slot - start_test_part] = scores[0]
            scores = loaded_scores

            outbreaks = []
            for [start_outbreak, end_outbreak] in data_info["outbreaks"]:
                outbreaks.append([start_outbreak - start_test_part, end_outbreak - start_test_part])

        else:
            # algin the outbreaks with the scores (e.g. scores only start from the test part of the data stream)
            outbreaks = []
            for [start_outbreak, end_outbreak] in data_info["outbreaks"]:
                outbreaks.append(
                    [start_outbreak - data_info["start_test_part"], end_outbreak - data_info["start_test_part"]])

        # create result dictionary
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
    """
    Evaluate the specified data streams with the specified algorithm. The results will be saved to the "_results"
    folder. If results exist they will be loaded from the file system.

    :param data_stream_class: the data stream class
    :param multiple_data_stream_params: a list of data stream parameters
    :param algo_class: the algorithm class
    :param algo_params: the algorithm parameters
    :param syndrome_counter_class: the syndrome counter class
    :param syndrome_counter_params: the syndrome counter parameters
    :param recompute: if True, pre-computed results will not be loaded from the file system
    :return: a dictionary containing the results for the evaluation measures of the evaluated algorithm computed
    over all data streams.
    """
    # Evaluate data streams
    result_dicts = []
    for data_stream_params in multiple_data_stream_params:
        result_dict = evaluate_data_stream(data_stream_class=data_stream_class, data_stream_params=data_stream_params,
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
    """
    Evaluate all algorithms on the specified data streams. The results will be saved to the "_results" folder. If
    results exist they will be loaded from the file system.

    :param data_stream_class: the data stream class
    :param multiple_data_stream_params: a list of data stream parameters
    :param algo_settings: a list of algorithm configurations. Each configuration is a pair containing the algorithm
    class and algorithm parameters
    :param syndrome_counter_settings: a list of syndrome counter configurations. Each configuration is a pair containing
    the syndrome counter class and syndrome counter parameters
    :param recompute: if True, pre-computed results will not be loaded from the file system
    :return: a dictionary containing the results for the evaluation measures of the evaluated algorithm computed
    over all data streams.
    """
    algo_results = []
    all_combos = itertools.product(*[syndrome_counter_settings, algo_settings])
    for i, [[syndrome_counter_class, syndrome_counter_params], [algo_class, algo_params]] in enumerate(all_combos):
        logger.info("Evaluate combo: " + str(i + 1) + "/" + str(len(algo_settings) * len(syndrome_counter_settings)))
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
