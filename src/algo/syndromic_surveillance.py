import os
os.environ["R_USER"] = "R_USER"
import numpy as np
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage

'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''


def _load_rmodule(moduleName):
    rpath = os.path.dirname(os.path.realpath(__file__)) + "/../../_rModules/"

    with open(rpath + moduleName, "r") as rfile:
        code = ''.join(rfile.readlines())
        rmodule = SignatureTranslatedAnonymousPackage(code, "rf")
    numpy2ri.activate()

    return rmodule


'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''


class EARS(object):

    def __init__(self, syndrome_counter, method="C1", window_size=7, min_parameter=1):
        """
        Initialize
        :param syndrome_counter: the module producing syndrome counts
        :param method: the algorithm of the EARS-family
        :param window_size: the window size used for the syndromic surveillance method
        :param min_parameter: the minimum standard deviation used for fitting the Gaussian distribution
        """
        self.syndrome_counter = syndrome_counter
        self.method = method
        self.window_size = window_size
        self.min_parameter = min_parameter
        self.rmodule = _load_rmodule("ears_score.R")

    def evaluate(self, time_slot):
        """
        Computes the score for the given time slot.
        :param time_slot: the time slot for which the score is computed
        :return: the score for the given time slot
        """

        if self.method == "C1":
            w = self.window_size
        elif self.method == "C2":
            w = self.window_size + 2
        elif self.method == "C3":
            w = self.window_size + 5
        else:
            raise Exception("Unknown method: " + str(self.method))

        # obtain the syndrome counts
        syndrome_df = self.syndrome_counter.get_syndrome_df(time_slot - w, time_slot)

        # compute the scores for all syndromes
        scores = []
        for syndrome_name in syndrome_df.columns:
            if syndrome_name == "time_slot": continue

            # obtain counts for syndrome
            counts = np.array(syndrome_df[syndrome_name])

            # call R-module
            result = self.rmodule.getEarsScore(counts, freq=365, method=self.method, baseline=self.window_size,
                                               alpha=0.05, minSigma=self.min_parameter)

            # extract the result which is already a score
            score = np.asarray(result[2]).T[0][-1]
            scores.append(score)

        # Report the most significant observation
        return np.max(scores)


'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''


class Bayes(object):

    def __init__(self, syndrome_counter, window_size=7):
        """
        Initialize
        :param syndrome_counter: the module producing syndrome counts
        :param window_size: the window size used for the syndromic surveillance method
        """
        self.syndrome_counter = syndrome_counter
        self.window_size = window_size
        self.rmodule = _load_rmodule("bayes_score.R")

    def evaluate(self, time_slot):
        """
        Computes the score for the given time slot.
        :param time_slot: the time slot for which the score is computed
        :return: the score for the given time slot
        """
        # obtain the syndrome counts
        syndrome_df = self.syndrome_counter.get_syndrome_df(time_slot - self.window_size, time_slot)

        # compute the scores for all syndromes
        scores = []
        for syndrome_name in syndrome_df.columns:
            if syndrome_name == "time_slot": continue

            # obtain counts for syndrome
            counts = np.array(syndrome_df[syndrome_name])

            # call R-module
            result = self.rmodule.getBayesScore(counts, freq=365, b=0, w=self.window_size, actY=True, alpha=0.05)

            # extract score
            results = np.asarray(result[2]).T[0]  # already returns the score
            assert (len(results) == 1)
            scores.append(results[0])

        # Report the most significant observation
        return np.max(scores)


'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''


class RKI(object):

    def __init__(self, syndrome_counter, window_size=7):
        """
        Initialize
        :param syndrome_counter: the module producing syndrome counts
        :param window_size: the window size used for the syndromic surveillance method
        """
        self.syndrome_counter = syndrome_counter
        self.window_size = window_size
        self.rmodule = _load_rmodule("rki_score.R")

    def evaluate(self, time_slot):
        """
        Computes the score for the given time slot.
        :param time_slot: the time slot for which the score is computed
        :return: the score for the given time slot
        """
        # obtain the syndrome counts
        syndrome_df = self.syndrome_counter.get_syndrome_df(time_slot - self.window_size, time_slot)

        # compute the scores for all syndromes
        scores = []
        for syndrome_name in syndrome_df.columns:
            if syndrome_name == "time_slot": continue

            # obtain counts for syndrome
            counts = np.array(syndrome_df[syndrome_name])

            # call R-module
            result = self.rmodule.getRKIScore(counts, freq=365, b=0, w=self.window_size, actY=True)

            # extract score
            results = np.asarray(result[2]).T[0]  # already returns the score
            assert (len(results) == 1)
            scores.append(results[0])

        # Report the most significant observation
        return np.max(scores)
