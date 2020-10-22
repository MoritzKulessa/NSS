import numpy as np
import pandas as pd
from util import io


class WSARE_DATA:

    def __init__(self, data_stream_id):
        """
        Initializing
        :param data_stream_id: the id of the WSARE data stream (0-99)
        """
        self.data_stream_id = data_stream_id
        self._load_data()

    '''
    ***********************************************************************************************************
    ***********************************************************************************************************
    ***********************************************************************************************************
    '''

    def _load_data(self):
        """
        Loads the data of stream from the file system.
        """
        path = io.get_project_directory() + "_data/wsare/"

        # read data
        cur_path = path + "patientstatus" + str(self.data_stream_id) + ".ds"
        df = pd.read_csv(cur_path)

        # prepare cases dataframe and environmental dataframe
        min_daynum = np.min(df["daynum"])
        df["time_slot"] = df["daynum"] - min_daynum
        self.cases_df = df[["time_slot", "XY", "age", "gender", "action", "reported_symptom", "drug"]]
        environmentals = df[["time_slot", "flu", "day_of_week", "weather", "season"]]
        self.environmental_df = environmentals.drop_duplicates()

        #prepare history
        self.history = []
        for time_slot in range(np.min(df["time_slot"]), np.max(df["time_slot"])):
            ts_cases_df = self.cases_df[self.cases_df["time_slot"] == time_slot]
            ts_cases_df = ts_cases_df.drop(['time_slot'], axis=1)
            ts_environmental_df = self.environmental_df[self.environmental_df["time_slot"] == time_slot]
            ts_environmental_df = ts_environmental_df.drop(['time_slot'], axis=1)
            self.history.append([ts_cases_df, ts_environmental_df])

        # read extract information about the outbreak
        f = open(path + "release_daycode.txt", "r")
        lines = f.readlines()
        outbreak_time_slot = int(lines[self.data_stream_id].split(" ")[1]) - min_daynum
        outbreak_duration = 14
        f.close()
        self.outbreaks = [[outbreak_time_slot, np.min(
            [outbreak_time_slot + outbreak_duration - 1, np.max(df["time_slot"])])]]  # end of outbreak is inclusive

    '''
    ***********************************************************************************************************
    ***********************************************************************************************************
    ***********************************************************************************************************
    '''

    def get_cases(self, time_slot):
        """
        Returns the cases for a specific time slot
        :param time_slot: the time slot for which the cases should be returned
        :return: a dataframe containing the cases for the specified time slot 
        """
        return self.history[time_slot][0]

    def get_all_cases(self):
        """
        Returns all cases.
        :return: a dataframe containing all cases including a column "time_slot" specifying which case belongs to which time slot
        """
        return self.cases_df

    def get_env(self, time_slot):
        """
        Returns the environmental setting for a specific time slot
        :param time_slot: the time slot for which the environmental setting should be returned
        :return: a dataframe containing the environmental setting for the specified time slot 
        """
        return self.history[time_slot][1]

    def get_all_envs(self):
        """
        Returns all environmental settings for all available time slots.
        :return: a dataframe containing all environmental settings including a column "time_slot" specifying which environmental setting belongs to which time slot
        """
        return self.environmental_df

    def get_history(self, time_slot_start, time_slot_end):
        """
        Returns the histroy for a specific range of time slots
        :param time_slot_start: the first time slot of the history
        :param time_slot_end: the last time slot of the history (inclusive)
        :return: a list containing the information for the specified range of time slots [[cases_{time_slot_start}, env_{time_slot_start}], ..., [cases_{time_slot_end}, env_{time_slot_end}]] 
        """
        return self.history[time_slot_start:time_slot_end+1]

    def get_info(self):
        """
        Returns additional information about the data stream (e.g. range of time slots, outbreak information, ...)
        The entries "first_time_slot", "last_time_slot", "start_test_part" and "outbreaks" are necessary for the evaluation.
        """
        return {
            "first_time_slot": 0,
            "last_time_slot": 729,
            "start_test_part": 365,
            "outbreaks": self.outbreaks
        }

    '''
    ***********************************************************************************************************
    ***********************************************************************************************************
    ***********************************************************************************************************
    '''

    def get_params(self):
        return [["data_stream_id", self.data_stream_id]]

    def get_ident(self):
        return "WSARE_DATA$" + "$".join([key + "=" + str(val) for key, val in self.get_params()])
