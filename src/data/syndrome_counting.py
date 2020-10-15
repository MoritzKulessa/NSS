import os
import hashlib
import itertools
import numpy as np
import pandas as pd
from util import io

import logging

logger = logging.getLogger(__name__)

'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''


class BasicSyndromeCounting:

    def __init__(self, data_stream, combos=2, min_count=1, remove_cols=[], blacklist={}, clear_cache=False):
        """
        Initializing
        :param data_stream: the module to access data
        :param combos: maximum length of the syndromes (NOT IMPORTANT FOR US)
        :param min_count: the minimum number of occurrences on the current day in order to be counted (NOT IMPORTANT FOR US)
        :param remove_cols: the columns which should not be used for forming syndromes
        :param blacklist: a dictionary, defining which values should not be used for forming syndromes. key=column/attribute value=list of values of that attribute
        :param clear_cache: removes pre-computed results
        """
        self.data_stream = data_stream
        self.combos = combos
        self.min_count = min_count
        self.remove_cols = remove_cols
        self.blacklist = blacklist
        self.directory = io.get_project_directory() + "_cache/basic_syndrome_counting/" + data_stream.get_ident() + "/"
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)

        self.cache_path = self.directory + self.get_hash() + ".pkl"
        if clear_cache and os.path.exists(self.cache_path):
            os.remove(self.cache_path)

        if os.path.exists(self.cache_path):
            self.syndrome_df = pd.read_pickle(self.cache_path)
        else:
            self.syndrome_df = pd.DataFrame({"time_slot": []})

    '''
    ***********************************************************************************************************
    ***********************************************************************************************************
    ***********************************************************************************************************
    '''

    def update_cache(self, time_slot_or_set_of_time_slots):
        """
        Updates the syndrome count dataframe with the specified range of time slots (last time slot is inclusive). 
        The updated version is saved to the file system.
        :param time_slot_or_set_of_time_slots: a time slot or a set of time slots
        """

        time_slot_set = set(time_slot_or_set_of_time_slots)

        # Generate syndrome counts
        time_slots = []
        syndrome_counts = []
        for i in time_slot_set:
            time_slots.append(i)
            syndrome_counts.append(self.count_syndromes(self.data_stream.get_cases(i)))
            self.syndrome_df = self.syndrome_df[self.syndrome_df["time_slot"] != i]

        # create dataframe for syndrome counts of the new time slots
        new_syndromes_df = pd.DataFrame(syndrome_counts)
        new_syndromes_df["time_slot"] = time_slots

        # add new syndrome counts
        self.syndrome_df = pd.concat([self.syndrome_df, new_syndromes_df])
        self.syndrome_df = self.syndrome_df.fillna(0)
        self.syndrome_df = self.syndrome_df.sort_values("time_slot")
        self.syndrome_df.to_pickle(self.cache_path)

    '''
    ***********************************************************************************************************
    ***********************************************************************************************************
    ***********************************************************************************************************
    '''

    def get_counts(self, time_slot):
        """
        Returns the syndrome counts for a specific time slot.
        :param time_slot: the time slot for which the syndrome counts should be returned
        :return: a dataframe with one row containing the syndrome counts (columns = syndromes, row = the counts for the respective syndrome)
        """
        if time_slot not in set(self.syndrome_df["time_slot"]):
            self.update_cache(time_slot)
        selected_syndrome_df = self.syndrome_df[self.syndrome_df["time_slot"] == time_slot]
        assert (len(selected_syndrome_df) == 1)
        return selected_syndrome_df.drop(['time_slot'], axis=1)

    def get_syndrome_df(self, time_slot_start, time_slot_end):
        """
        Returns the syndrome counts for the specified range of time slots.
        :param time_slot_start: the first time slot 
        :param time_slot_end: the last time slot (inclusive)
        :return: a dataframe containing the syndrome counts (columns = time slot + syndromes, rows = the syndrome counts for the respective time slot)
        """
        time_slots = set(range(time_slot_start, time_slot_end + 1))
        available_time_slots = set(self.syndrome_df["time_slot"])
        missing_time_slots = time_slots.difference(available_time_slots)
        if len(missing_time_slots) > 0:
            self.update_cache(missing_time_slots)

        selected_syndrome_df = self.syndrome_df[np.logical_and(self.syndrome_df["time_slot"] >= time_slot_start,
                                                               self.syndrome_df["time_slot"] <= time_slot_end)]

        assert(len(selected_syndrome_df) == time_slot_end - time_slot_start + 1)
        return selected_syndrome_df

    '''
    ***********************************************************************************************************
    ***********************************************************************************************************
    ***********************************************************************************************************
    '''

    def count_syndromes(self, df):
        """
        Compute the counts for the syndromes for the given the cases as a pandas dataframe. 
        Columns which have been specified by *remove_cols* will be disregarded.
        Values for specific columns specified by the *blacklist* will be diregarded as well.
        :param df: the dataframe containing the cases
        :return: the syndrome counts as a dictionary for the specified day. The key is the list of symptoms as a string and the value is the count.
        """
        eval_cols = list(df.columns)
        for remove_col in self.remove_cols:
            eval_cols.remove(remove_col)

        # Replace the values of the blacklist with nan, these will not be considered for defining syndromes
        for col, vals in self.blacklist.items():
            df[col] = df[col].replace(vals, np.nan)

        syndromes = {}
        for combo in range(1, self.combos + 1):
            for eval_combo in itertools.combinations(eval_cols, r=combo):
                for val_combo, group_df in df.groupby(list(eval_combo)):

                    if len(group_df) >= self.min_count:
                        if combo == 1:
                            val_combo = [val_combo]
                        else:
                            val_combo = list(val_combo)
                        s = [[eval_combo[i], val_combo[i]] for i in range(len(eval_combo))]
                        syndromes[str(s)] = len(group_df)

        return syndromes

    '''
    ***********************************************************************************************************
    ***********************************************************************************************************
    ***********************************************************************************************************
    '''

    def get_params(self):
        return [["combos", self.combos], ["min_count", self.min_count], ["remove_cols", self.remove_cols],
                ["blacklist", self.blacklist]]

    def get_ident(self):
        return "BasicSyndromeCounting$" + "$".join([key + "=" + str(val) for key, val in self.get_params()])

    def get_hash(self, numHashChars=32):
        return hashlib.md5(self.get_ident().encode()).hexdigest()[-numHashChars:]
