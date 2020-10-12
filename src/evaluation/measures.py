import numpy as np
from sklearn.metrics import roc_curve

'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''


def _get_ground_truth(outbreaks, size):
    """
    Generates binary labels for the time slots indicating if an outbreak is taking place
    :param outbreaks: an array of tuples, each tuple represents one outbreak, the first value represents the time slot when the outbreak started started and the second value represents the time slot when the outbreak ended (inclusive) 
    :param size: the total number of time slots contained in the boolean array
    :return: Returns an array of booleans, the index represents the time slot and the boolean represents whether an outbreak is going on
    """
    gt = np.full(size, fill_value=False)
    for [start_outbreak, end_outbreak] in outbreaks:
        for i in range(start_outbreak, end_outbreak + 1):
            gt[i] = True
    return gt


'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''


def compute_roc_values(scores, outbreaks):
    """
    Computes the receiver operating characteristics for the given scores and outbreaks. 
    In addition, the detection-delay and the detection-rate is computed for every threshold.
    :param scores: the scores used for the evaluation
    :param outbreaks: an array of tuples, each tuple represents one outbreak, the first value represents the time slot when the outbreak started started and the second value represents the time slot when the outbreak ended (inclusive) 
    :return: a dictionary containing receiver operating characteristics including detection-delay and detection-rate
    """

    ground_truth = _get_ground_truth(outbreaks, len(scores))
    fpr, tpr, thresholds = roc_curve(ground_truth, scores, pos_label=True, drop_intermediate=False)

    n_outbreaks = len(outbreaks)
    if n_outbreaks == 0: return fpr, tpr, np.array([np.nan] * len(tpr)), thresholds

    # Transform epi_scores
    min_epi_scores = []
    for [start_outbreak, end_outbreak] in outbreaks:
        scroes_for_epi = scores[start_outbreak:end_outbreak + 1]
        min_epi_score = np.max(scroes_for_epi)
        min_epi_scores.append([min_epi_score, scroes_for_epi])

    sort_scores = sorted(min_epi_scores, reverse=True, key=lambda x: x[0])

    dr = []
    det_delays = []
    cur_epi_index = 0
    for thresh in thresholds:

        # compute detection rate for the given threshold
        while (cur_epi_index < len(sort_scores) and sort_scores[cur_epi_index][0] >= thresh):
            cur_epi_index += 1
        dr.append(cur_epi_index / n_outbreaks)

        # compute detection delay for the given threshold
        tmp_det_delays = []
        for score, scores_for_epi in sort_scores:
            if score < thresh: break

            alarms = scores_for_epi >= thresh
            tmp_det_delays.append(np.argmax(alarms))

        tmp_det_delays += [len(scores_for_epi) for score, scores_for_epi in sort_scores[cur_epi_index:]]
        assert (len(outbreaks) == len(tmp_det_delays))

        det_delay = np.nanmean(tmp_det_delays)
        det_delays.append(det_delay)

    return {"thresholds": thresholds, "FAR": fpr, "detectionDelay": det_delays, "detectionRate": dr, "TPR": tpr}


def compute_area_under_curve(roc_values, x_measure, y_measure):
    """
    Computes the area under curve for the given measures. If a partial area is computed, the area will be normalized.
    :param roc_values: the receiver operating characteristics 
    :param x_measure: the measure placed on the x-axis, add "*x" to compute only a spatial area (e.g. FAR*0.05)
    :param y_measure: the measure placed on the x-axis
    :return: the area under curve for thegiven measures
    """
    splitted = x_measure.split("*")
    if len(splitted) > 1:
        x_limit = float(splitted[1])
    else:
        x_limit = 1

    x_vals = roc_values[splitted[0]]
    y_vals = roc_values[y_measure]

    # Filter according to x_limit
    if x_limit < 1:
        index_limit = np.searchsorted(x_vals, [x_limit], side="left")[0]

        prev_x_val = x_vals[index_limit - 1]
        next_x_val = x_vals[index_limit]
        prev_y_val = y_vals[index_limit - 1]
        next_y_val = y_vals[index_limit]
        intermediate_y_val = ((x_limit - prev_x_val) / (next_x_val - prev_x_val)) * (
                    next_y_val - prev_y_val) + prev_y_val

        limit_x_vals = list(x_vals[:index_limit]) + [x_limit]
        limit_y_vals = list(y_vals[:index_limit]) + [intermediate_y_val]

        auc = np.trapz(limit_y_vals, limit_x_vals)

    else:
        auc = np.trapz(y_vals, x_vals)

    # normalize the area
    auc = auc / x_limit

    return auc


'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''


def example():
    # Array containing the information abou the outbreaks outbreaks
    outbreaks = [
        # Outbreak 1
        [5, 8],  # Start and end day of the outbreak (end date is inclusive)
        # Outbreak 2
        [9, 11],
        # Outbreak3
        # ...
    ]

    # p-values for each day one
    p_values = [0.4, 0.5, 0.3, 0.005, 0.2, 0.02, 0.01, 0.1, 0.9, 0.00001, 0.8, 0.9, 0.8]

    scores = 1 - np.array(p_values)

    roc_values = compute_roc_values(scores, outbreaks)

    import matplotlib.pyplot as plt

    plt.plot(roc_values["FAR"], roc_values["TPR"])
    plt.title("ROC")
    plt.show()

    plt.plot(roc_values["FAR"], roc_values["detectionRate"])
    plt.title("dROC")
    plt.show()

    plt.plot(roc_values["FAR"], roc_values["detectionDelay"])
    plt.title("AMOC")
    plt.show()

    AUC = compute_area_under_curve(roc_values, x_measure="FAR", y_measure="TPR")
    print("AUC: " + str(AUC))
    AUC5 = compute_area_under_curve(roc_values, x_measure="FAR*0.05", y_measure="TPR")
    print("Partial AUC with FAR<=5%: " + str(AUC5))

    dAUC = compute_area_under_curve(roc_values, x_measure="FAR", y_measure="detectionRate")
    print("dAUC: " + str(dAUC))
    dAUC5 = compute_area_under_curve(roc_values, x_measure="FAR*0.05", y_measure="detectionRate")
    print("Partial dAUC with FAR<=5%: " + str(dAUC5))

    amoc_auc = compute_area_under_curve(roc_values, x_measure="FAR", y_measure="detectionDelay")
    print("AMOC AUC: " + str(amoc_auc))
    amoc_auc5 = compute_area_under_curve(roc_values, x_measure="FAR*0.05", y_measure="detectionDelay")
    print("Partial AMOC AUC with FAR<=5%: " + str(amoc_auc5))
