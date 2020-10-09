'''
Created on 09.10.2020

@author: Moritz
'''

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression as LR



'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''



class ControlChart(object):


    def __init__(self, data_stream):
        '''
        Initialize
        :param data_stream: the module to access data
        '''
        self.data_stream = data_stream
    
    
    def evaluate(self, time_slot):
        '''
        Computes the score for the given time slot.
        :param time_slot: the time slot for which the score is computed
        :return: the score for the given time slot
        '''
        #obtain the history (prior seven days) and extract the counts for each time slot
        first_time_slot = self.data_stream.get_info()["first_time_slot"]
        history = self.data_stream.get_history(first_time_slot, time_slot-1)
        count_history = [len(cases_df) for cases_df, _ in history]

        #obtain number of cases for the given time slot
        n_cases = len(self.data_stream.get_cases(time_slot))
        
        #compute p-value
        avg_train = np.mean(count_history)
        std_train = np.std(count_history)
        p_value = 1 - stats.norm.cdf(n_cases, loc=avg_train, scale=std_train)
        
        #Compute score
        score = 1- p_value
        return score
        
        

'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''
   
   
        
class MovingAverage(object):


    def __init__(self, data_stream):
        '''
        Initialize
        :param data_stream: the module to access data
        '''
        self.data_stream = data_stream
    
    
    
    def evaluate(self, time_slot):
        '''
        Computes the score for the given time slot.
        :param time_slot: the time slot for which the score is computed
        :return: the score for the given time slot
        '''
        #obtain the history (prior seven days) and extract the counts for each time slot
        history = self.data_stream.get_history(time_slot-7, time_slot-1)
        count_history = [len(cases_df) for cases_df, _ in history]
        
        #obtain number of cases for the given time slot
        n_cases = len(self.data_stream.get_cases(time_slot))
        
        #compute p-value
        avg_train = np.mean(count_history)
        std_train = np.std(count_history)
        p_value = 1 - stats.norm.cdf(n_cases, loc=avg_train, scale=std_train)
        
        #Compute score
        score = 1- p_value
        return score
    
    
   
'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''    
   
   
    
class LinearRegression(object):


    def __init__(self, data_stream):
        '''
        Initialize
        :param data_stream: the module to access data
        '''
        self.data_stream = data_stream
    
    
    
    def evaluate(self, time_slot):
        '''
        Computes the score for the given time slot.
        :param time_slot: the time slot for which the score is computed
        :return: the score for the given time slot
        '''
        #obtain the history of case counts and environmental settings
        first_time_slot = self.data_stream.get_info()["first_time_slot"]
        history = self.data_stream.get_history(first_time_slot, time_slot-1)
        count_history = np.array([len(cases_df) for cases_df, _ in history])
        envs_df = pd.concat([env_df for _, env_df in history])
        
        #generate one hot encoding for environementals
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(envs_df)
        ohe_envs = enc.transform(envs_df).toarray()
        
        #learn the model
        model = LR()
        model.fit(ohe_envs, count_history)
        
        #predict train counts to compute the standard error
        preds = model.predict(ohe_envs)
        standard_error = np.sqrt(np.sum(((count_history-preds) * (count_history-preds)))/(len(preds)-2))
        
        #obtain environmental setting for current time slot and use it to predict the cases
        cur_env_df = self.data_stream.get_env(time_slot)
        ohe_cur_env = enc.transform(cur_env_df).toarray()
        cur_pred = model.predict(ohe_cur_env)
        
        #obtain current number of cases
        n_cases = len(self.data_stream.get_cases(time_slot))
        
        #compute p-value
        p_value = 1 - stats.norm.cdf(n_cases, loc=cur_pred, scale=standard_error)
        
        #Compute score
        score = 1- p_value
        return score
        
        
        