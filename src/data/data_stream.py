import os
import itertools
import numpy as np
import pandas as pd
from util import io


'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''

class WSARE_DATA:

    def __init__(self, data_stream_id, environmental_variables=True):
        """
        Initializing
        :param data_stream_id: the id of the WSARE data stream (0-99)
        """
        self.data_stream_id = data_stream_id
        self.environmental_variables = environmental_variables
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

        # prepare history
        self.history = []
        for time_slot in range(np.min(df["time_slot"]), np.max(df["time_slot"]) + 1):
            ts_cases_df = self.cases_df[self.cases_df["time_slot"] == time_slot]
            ts_cases_df = ts_cases_df.drop(['time_slot'], axis=1)
            ts_environmental_df = self.environmental_df[self.environmental_df["time_slot"] == time_slot]
            ts_environmental_df = ts_environmental_df.drop(['time_slot'], axis=1)

            if self.environmental_variables:
                self.history.append([ts_cases_df, ts_environmental_df])
            else:
                self.history.append([ts_cases_df, pd.DataFrame()])

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
        return self.history[time_slot_start:time_slot_end + 1]

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
        return [["data_stream_id", self.data_stream_id], ["environmental_variables", self.environmental_variables]]

    def get_ident(self):
        return "WSARE_DATA$" + "$".join([key + "=" + str(val) for key, val in self.get_params()])


'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''


class ED_DATA_RANDOM:

    def __init__(self, data_stream_id, k=1.0, environmental_variables=True, random_seed=0):
        """
        Initializing
        """
        self.data_stream_id = data_stream_id
        self.k = k
        self.environmental_variables = environmental_variables
        self.random_seed = random_seed
        self._load_data()

    '''
    ***********************************************************************************************************
    ***********************************************************************************************************
    ***********************************************************************************************************
    '''

    def _load_data(self):
        """
        Loads the data of stream from the file system and adds the synthetic outbreak.
        """
        # load data stream from cache if available
        cache_location = "_cache/streams/"
        if io.exist(self.get_ident(), loc=cache_location):
            [self.cases_df, self.environmental_df, self.history, self.start_test_part, self.last_time_slot,
             self.outbreaks, self.syndrome, self.outbreak_size] = io.load(self.get_ident(), loc=cache_location)
        else:

            # load basic data
            load_path = io.get_project_directory() + "_data/ed_data/ed1_v2.csv"
            df = pd.read_csv(load_path)
            min_timeslot = np.min(df["time_slot"])
            df["time_slot"] = df["time_slot"] - min_timeslot
            assert (np.max(df["time_slot"]) == 730)

            # extract cases and environmental settings
            self.cases_df = df[["time_slot", "gender", "age", "mts", "fever", "pulse",
                                "respiration", "oxygensaturation", "bloodpressure"]]
            environmentals = df[["time_slot", "week_day", "season"]]
            self.environmental_df = environmentals.drop_duplicates()

            # initialize random number generator
            rand_gen = np.random.RandomState(seed=self.random_seed + self.data_stream_id * 10000)

            # compute syndromes
            copy_cases_df = self.cases_df.drop(columns=["time_slot"])

            # Replace normal values with nan, these will not be considered for defining outbreaks
            copy_cases_df["mts"] = copy_cases_df["mts"].replace(["Sonstiges"], np.nan)
            copy_cases_df["fever"] = copy_cases_df["fever"].replace(["normal"], np.nan)
            copy_cases_df["pulse"] = copy_cases_df["pulse"].replace(["normal"], np.nan)
            copy_cases_df["respiration"] = copy_cases_df["respiration"].replace(["10-20 (normal)"], np.nan)
            copy_cases_df["oxygensaturation"] = copy_cases_df["oxygensaturation"].replace([">=95 (normal)"], np.nan)
            copy_cases_df["bloodpressure"] = copy_cases_df["bloodpressure"].replace(["normal"], np.nan)

            eval_cols = list(copy_cases_df.columns)
            syndromes = []
            combos = 2
            for combo in range(1, combos + 1):
                for eval_combo in itertools.combinations(eval_cols, r=combo):
                    for val_combo, group_df in copy_cases_df.groupby(list(eval_combo)):

                        if len(group_df) >= 1:
                            if combo == 1:
                                val_combo = [val_combo]
                            else:
                                val_combo = list(val_combo)
                            s = [[eval_combo[i], val_combo[i]] for i in range(len(eval_combo))]
                            syndromes.append([s, len(group_df) / 731])

            # select syndrome
            if self.data_stream_id < 20:
                syndromes = list(filter(lambda x: x[1] <= 1, syndromes))
                self.syndrome = syndromes[rand_gen.randint(len(syndromes))][0]
            else:
                syndromes = list(filter(lambda x: x[1] > 1, syndromes))
                self.syndrome = syndromes[rand_gen.randint(len(syndromes))][0]

            # set evaluation details and generate random outbreak
            self.start_test_part = 366
            self.last_time_slot = 730
            day_counts = [len(group_df) for _, group_df in df.groupby("time_slot")]
            daily_std = np.std(day_counts)
            self.outbreak_size = rand_gen.poisson(lam=daily_std * self.k)
            outbreak_time_slot = rand_gen.randint(low=self.start_test_part, high=self.last_time_slot + 1)
            self.outbreaks = [[outbreak_time_slot, outbreak_time_slot]]

            # generate outbreak cases
            sample_df = self.cases_df
            for [cond_col, cond_val] in self.syndrome:
                sample_df = sample_df[sample_df[cond_col] == cond_val]
            sample_df = sample_df.sample(self.outbreak_size, random_state=rand_gen, replace=True)
            sample_df["time_slot"] = [outbreak_time_slot] * len(sample_df)

            # add cases to cases dataframe
            self.cases_df = pd.concat([self.cases_df, sample_df])
            self.cases_df = self.cases_df.sort_values(["time_slot"])

            # prepare history
            self.history = []
            for time_slot in range(np.min(df["time_slot"]), np.max(df["time_slot"]) + 1):
                ts_cases_df = self.cases_df[self.cases_df["time_slot"] == time_slot]
                ts_cases_df = ts_cases_df.drop(['time_slot'], axis=1)
                ts_environmental_df = self.environmental_df[self.environmental_df["time_slot"] == time_slot]
                ts_environmental_df = ts_environmental_df.drop(['time_slot'], axis=1)

                if self.environmental_variables:
                    self.history.append([ts_cases_df, ts_environmental_df])
                else:
                    self.history.append([ts_cases_df, pd.DataFrame()])

            if not os.path.isdir(io.get_project_directory() + cache_location):
                os.makedirs(io.get_project_directory() + cache_location)
            data = [self.cases_df, self.environmental_df, self.history, self.start_test_part, self.last_time_slot,
                    self.outbreaks, self.syndrome, self.outbreak_size]
            io.save(data, self.get_ident(), loc=cache_location)

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
        return self.history[time_slot_start:time_slot_end + 1]

    def get_info(self):
        """
        Returns additional information about the data stream (e.g. range of time slots, outbreak information, ...)
        The entries "first_time_slot", "last_time_slot", "start_test_part" and "outbreaks" are necessary for the evaluation.
        """
        return {
            "first_time_slot": 0,
            "last_time_slot": self.last_time_slot,
            "start_test_part": self.start_test_part,
            "outbreaks": self.outbreaks
        }

    '''
    ***********************************************************************************************************
    ***********************************************************************************************************
    ***********************************************************************************************************
    '''

    def get_params(self):
        return [["data_stream_id", self.data_stream_id], ["k", self.k],
                ["environmental_variables", self.environmental_variables], ["random_seed", self.random_seed]]

    def get_ident(self):
        return "ED_DATA_RANDOM$" + "$".join([key + "=" + str(val) for key, val in self.get_params()])
