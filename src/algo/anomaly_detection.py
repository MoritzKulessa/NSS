import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class SyndromeCountAnomalyDetector(object):

    def __init__(self, syndrome_counter, anomaly_detector_class, anomaly_detector_parameters, incremental=True, normalize_data=False):
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

        self.dataset = self._generate_dataset()

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

    def _generate_dataset(self):
        data_info = self.syndrome_counter.data_stream.get_info()
        syndrome_df = self.syndrome_counter.get_syndrome_df(data_info['first_time_slot'], data_info['last_time_slot'])
        syndrome_tss = list(syndrome_df["time_slot"])
        syndrome_df = syndrome_df.drop("time_slot", axis=1)
        env_df = self.syndrome_counter.data_stream.get_all_envs()
        env_tss = list(env_df["time_slot"])
        env_df = env_df.drop("time_slot", axis=1)

        for i in range(len(syndrome_tss)):
            assert (env_tss[i] == syndrome_tss[i])

        if len(env_df.columns) == 0:
            # no environmental variables
            dataset = syndrome_df
        else:
            # with environmental variables --> one-hot encoding
            enc = OneHotEncoder(handle_unknown='ignore')
            enc.fit(env_df)
            ohe_env = enc.transform(env_df.values).toarray()

            new_data = [list(row) + list(ohe_env[i]) for i, row in enumerate(syndrome_df.values)]
            dataset = pd.DataFrame(new_data, columns=list(syndrome_df.columns) + [str(i) for i in range(ohe_env.shape[1])])

        return dataset

    def _normalize_data(self, df):
        all_vals = []
        for column in df.columns:
            if self.normalize_dict[column]["max"] > self.normalize_dict[column]["min"]:
                vals = list((df[column] - self.normalize_dict[column]["min"]) / (self.normalize_dict[column]["max"] - self.normalize_dict[column]["min"]))
                all_vals.append(vals)
            else:
                vals = list(df[column])
                all_vals.append(vals)
        return pd.DataFrame(np.array(all_vals).T, columns=list(df.columns))

    def _get_model(self, till_timeslot):

        train_df = pd.DataFrame(self.dataset[:(till_timeslot + 1)])

        if self.normalize_data:
            self.normalize_dict = {}
            for column in train_df.columns:
                self.normalize_dict[column] = {}
                self.normalize_dict[column]["min"] = train_df[column].min()
                self.normalize_dict[column]["max"] = train_df[column].max()
            train_df = self._normalize_data(train_df)

        # fit anomaly detector
        train_data = train_df.values
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

        count_df = self.dataset[time_slot:time_slot + 1]
        if self.normalize_data:
            count_df = self._normalize_data(count_df)
        test_data = count_df.iloc[-1].values.reshape(1, -1)

        if self.detector_library == "sklearn":
            res = anomaly_detector.score_samples(test_data)
            anomaly_score = - res[0]
        elif self.detector_library == "pyod":
            anomaly_score = anomaly_detector.decision_function(test_data)[0]
        else:
            raise Exception("Unknown anomaly detection algorithm: " + str(self.anomaly_detector_class))

        return anomaly_score
