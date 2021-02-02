'''
Created on 31.10.2019

@author: Moritz
'''

import os 
import pandas as pd
import numpy as np

from simple_spn import spn_handler
from simple_spn import functions as fn
from simple_spn import io_helper
from spn.experiments.AQP.Ranges import NominalRange, NumericRange



def get_titanic(col_names=None):
    path = "data/titanic/train.csv"

    df = pd.read_csv(path)
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)
    
    if col_names is not None:
        df = df[col_names]
    
    #Fill missing values    
    df["Age"].fillna(int(df["Age"].mean()), inplace=True)
    df["Embarked"].fillna("S", inplace=True)

    return fn.transform_dataset(df)



if __name__ == '__main__':
    
    
    '''
    SPN_experiments construction titanic
    '''
    
    dataset_name = "titanic"
    
    
    #parameters for the construction
    rdc_threshold = 0.3
    min_instances_slice = 0.01
    
    
    if not spn_handler.exist_spn(dataset_name, rdc_threshold, min_instances_slice):
        print("Creating SPN_experiments ...")
        
        #get z_data
        df, value_dict, parametric_types = get_titanic()
        
        #print z_data (top 5 rows)
        io_helper.print_pretty_table(df.head(5))
        
        #print value-dict
        print(value_dict)
        
        #Creates the SPN_experiments and saves to a file
        spn, const_time = spn_handler.learn_parametric_spn(df.values, parametric_types, rdc_threshold, min_instances_slice)
        spn_handler.save_spn(spn, const_time, dataset_name, rdc_threshold, min_instances_slice, value_dict=value_dict)
        
    
    #Load SPN_experiments
    spn, value_dict, _ = spn_handler.load_spn(dataset_name, rdc_threshold, min_instances_slice)
    
    #Print some statistics
    fn.print_statistics(spn)
    
    
    
    
    #Example value dict generation
    
    
    
    path = "data/titanic/train.csv"
    df = pd.read_csv(path)
    
    #print z_data (top 5 rows)
    io_helper.print_pretty_table(df.head(5))
    
    df = df[["Survived", "Sex", "Age", "Fare", "Pclass"]]
    df, val_dict, param_types = fn.transform_dataset(df)
    
    #print z_data after transformation (top 5 rows)
    io_helper.print_pretty_table(df.head(5))
    
    ''''
    SPN_experiments functions
    '''
    
    
    #Load synthetic example SPN_experiments (very simple SPN_experiments)
    from simple_spn.example import example_spns
    spn = example_spns.get_gender_spn()
    
    #plot spn
    #fn.plot_spn(spn, "sample_spn.pdf")
    
    
    #generate samples
    samples = fn.sampling(spn, n_samples=10, random_seed=1)
    print(samples)
    
    samples = fn.sampling_rang(spn, rang=[None, None, None, None], n_samples=10, random_seed=1)
    print(samples)
    
    samples = fn.sampling_rang(spn, rang=[None, None, NumericRange([[10,11], [29,30]])], n_samples=10, random_seed=1)
    print(samples)
    
    samples = fn.sampling_rang(spn, rang=[NominalRange([0]), None, NumericRange([[14,15], [29,30]])], n_samples=10, random_seed=1)
    print(samples)
    
    
    
    #Test probabilities
    rang = [None, None, None]
    prob = fn.prob(spn, rang)
    print(prob)
    
    rang = [NominalRange([0]), NominalRange([1]), NumericRange([[20]])]
    prob = fn.prob(spn, rang)
    print(prob)
    
    ranges = np.array([[None, None, NumericRange([[0,20]])],
                       [NominalRange([0]), None, None],
                       [None, NominalRange([1]), None]])
    probs = fn.probs(spn, ranges)
    print(probs)
    
    inst = [0, np.nan, np.nan]
    prob = fn.prob_spflow(spn, inst)
    print(prob)
    
    
    data = np.array([[0, np.nan, np.nan], [0, 1, np.nan]])
    probs = fn.probs_spflow(spn, data)
    print(probs)
    
    
    
    
    #Marguinalize SPN_experiments
    spn1 = fn.marg(spn, [2])
    #fn.plot_spn(spn1, "marg1.pdf")
    
    spn2 = fn.marg(spn, [0])
    #fn.plot_spn(spn2, "marg2.pdf")
    
    spn3 = fn.marg(spn, [1])
    #fn.plot_spn(spn3, "marg3.pdf")
    
    spn4 = fn.marg(spn, [1,2])
    #fn.plot_spn(spn4, "marg4.pdf")
    
    rang = [None, NominalRange([1]), None]
    prob, spn5 = fn.marg_rang(spn, rang)
    #fn.plot_spn(spn5, "marg5.pdf")
    
    rang = [None, NominalRange([1]), NumericRange([[10,12]])]
    prob, spn6 = fn.marg_rang(spn, rang)
    #fn.plot_spn(spn6, "marg6.pdf")
    
    rang = [NominalRange([0]), NominalRange([1]), None]
    prob = fn.prob(spn, rang)
    print(prob)
    prob = fn.prob(spn6, rang)
    print(prob)
    
    
    
    
    #Expectation
    rang = [None, None, None]
    expect = fn.expect(spn, feature_id=2, rang=rang)
    print(expect)
    
    rang = [NominalRange([0]), None, None]
    expect = fn.expect(spn, feature_id=2, rang=rang)
    print(expect)
    
    rang = [NominalRange([1]), None, None]
    expect = fn.expect(spn, feature_id=2, rang=rang)
    print(expect)
    
    rang = [None, NominalRange([0]), None]
    expect = fn.expect(spn, feature_id=2, rang=rang)
    print(expect)
    
    feature_scope = {2}
    data = np.array([[np.nan, np.nan, np.nan]])
    expect = fn.expects_spnflow(spn, feature_scope, data)
    print(expect)
    
    feature_scope = {2}
    data = np.array([np.nan, np.nan, np.nan])
    expect = fn.expect_spnflow(spn, feature_scope, data)
    print(expect)
    
    
    
    
    #Sub-population
    sub_pops = fn.get_sub_populations(spn)
    print(sub_pops)
    
    
    #Value_dict
    f_info = fn.generate_adhoc_feature_info(spn)
    print(val_dict)
    
    #overall_population
    overall_pop = fn.get_overall_population(spn)
    
    
    

    
    spn, f_info, _ = spn_handler.load_spn(dataset_name, rdc_threshold, min_instances_slice)
    
    
    #Classify
    ranges = np.array([[None, NominalRange([1]), None, None, None, None, None, None],
                       [None, NominalRange([0]), None, None, None, None, None, None], 
                       [None, NominalRange([0]), None, None, None, None, None, None]])
    res = fn.classifies(spn, target_id=0, ranges=ranges, f_info=f_info)
    print(res)
    
    res = fn.classify(spn, target_id=0)
    print(res)
    
    
    
    
    #MPE (most probable explanation) (similar to classify)
    #---> Warnung wird evtl geschmissen: C:\Users\Moritz\Anaconda3\lib\site-packages\numpy\core\_methods.py:32: RuntimeWarning: overflow encountered in reduce  return umr_sum(a, axis, dtype, out, keepdims)
    df,_,_ =  get_titanic()
    input_data = df.values
    res = fn.mpe_spflow(spn, 0, input_data)
    print(res[:10])
    
    
