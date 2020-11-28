class SyndromeCountAnomalyDetector(object):

    def __init__(self, syndrome_counter, anomaly_detector_class, anomaly_detector_parameters, incremental=True,
                 normalize_data=False):
        """
        Initialize
        :param syndrome_counter: the module to access the syndrome counts
        :param anomaly_detector_class: the sklearn or pyod algorithm class of the anomaly detector
        :param anomaly_detector_parameters: the parameters for anomaly detector
        :param normalize_data: if true the data is normalized beforehand
        """
        self.syndrome_counter = syndrome_counter
        self.anomaly_detector_class = anomaly_detector_class
        self.anomaly_detector_parameters = anomaly_detector_parameters
        self.normalize_data = normalize_data
        self.incremental = incremental

        # Ensure that all syndrome counts have been computed
        data_info = self.syndrome_counter.data_stream.get_info()
        self.syndrome_counter.get_syndrome_df(data_info['first_time_slot'], data_info['last_time_slot'])

        # extract the library of the class
        if "sklearn" in str(self.anomaly_detector_class):
            self.detector_library = "sklearn"
        elif "pyod" in str(self.anomaly_detector_class):
            self.detector_library = "pyod"
        else:
            raise Exception("Unknown anomaly detection algorithm: " + str(self.anomaly_detector_class))

        if not self.incremental:
            till_timeslot = self.syndrome_counter.data_stream.get_info()["start_test_part"] - 1
            self.model = self._get_model(till_timeslot)

    def _normalize_data(self, df):
        for column in self.normalize_dict:
            if self.normalize_dict[column]["max"] > self.normalize_dict[column]["min"]:
                df[column] = (df[column] - self.normalize_dict[column]["min"]) / (
                        self.normalize_dict[column]["max"] - self.normalize_dict[column]["min"])

    def _get_model(self, till_timeslot):
        # obtain previous and current syndrome counts
        first_time_slot = self.syndrome_counter.data_stream.get_info()["first_time_slot"]
        syndrome_counts_df = self.syndrome_counter.get_syndrome_df(first_time_slot, till_timeslot)
        assert (syndrome_counts_df.iloc[-1][
                    "time_slot"] == till_timeslot)  # assume that the syndrome counts df is sorted
        syndrome_counts_df = syndrome_counts_df.drop("time_slot", axis=1)

        if self.normalize_data:
            self.normalize_dict = {}
            for column in syndrome_counts_df.columns:
                self.normalize_dict[column] = {}
                self.normalize_dict[column]["min"] = syndrome_counts_df[column].min()
                self.normalize_dict[column]["max"] = syndrome_counts_df[column].max()
            self._normalize_data(syndrome_counts_df)

        # fit anomaly detector
        train_data = syndrome_counts_df.values
        anomaly_detector = self.anomaly_detector_class(**self.anomaly_detector_parameters)

        if self.detector_library == "sklearn":
            anomaly_detector = anomaly_detector.fit(train_data)
        elif self.detector_library == "pyod":
            anomaly_detector.fit(train_data)
        else:
            raise Exception("Unknown anomaly detection algorithm: " + str(self.anomaly_detector_class))

        return anomaly_detector

    def evaluate(self, time_slot):
        """
        Computes the score for the given time slot.
        :param time_slot: the time slot for which the score is computed
        :return: the score for the given time slot
        """

        if self.incremental:
            anomaly_detector = self._get_model(time_slot - 1)
        else:
            anomaly_detector = self.model

        # obtain test data
        count_df = self.syndrome_counter.get_counts(time_slot)
        if self.normalize_data:
            self._normalize_data(count_df)
        test_data = count_df.iloc[-1].values.reshape(1, -1)

        if self.detector_library == "sklearn":
            res = anomaly_detector.score_samples(test_data)
            anomaly_score = - res[0]
        elif self.detector_library == "pyod":
            anomaly_score = anomaly_detector.decision_function(test_data)[0]
        else:
            raise Exception("Unknown anomaly detection algorithm: " + str(self.anomaly_detector_class))

        return anomaly_score
