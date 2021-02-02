'''
Created on 08.07.2019

@author: Moritz
'''

import numpy as np
import pandas as pd


def generate_gender_age_data(num_instances, rand_seed):    
    '''
    Correlations:
    P(gender=m) = 50%
    P(gender=f) = 50%
    P(student=yes|gender=m) = 30%
    P(student=yes|gender=f) = 80%
    P(age) = N(mu=20, sigma=3) ...
    '''
    
    np.random.seed(rand_seed)

    data = [] 
    for _ in range(num_instances):
        inst = {}
        if np.random.random() < 0.5:
            
            inst["gender"] = "male"
            if np.random.random() < 0.3:
                inst["student"] = "yes"
            else:
                inst["student"] = "no"

        else:
            inst["gender"] = "female"
            if np.random.random() < 0.8:
                inst["student"] = "yes"
            else:
                inst["student"] = "no" 
            
        inst["age"] = int(np.random.normal(20, 5))
        
        data.append(inst)
    
    df = pd.DataFrame(data)
    return fn.transform_dataset(df[["gender", "student", "age"]], feature_types=["discrete", "discrete", "numeric"])



if __name__ == '__main__':
    
    
    from simple_spn import functions as fn
    from simple_spn import spn_handler
    from simple_spn import io_helper
    

    dataset_name = "genderAge"
    
    #parameters for the construction
    rdc_threshold = 0.3
    min_instances_slice = 0.01
    

    #get z_data
    df, value_dict, parametric_types = generate_gender_age_data(10000, 1)
    
    
    #print z_data (top 5 rows)
    io_helper.print_pretty_table(df.head(100))
    
    #print value-dict
    print(value_dict)
    
    #Creates the SPN_experiments and saves to a file
    spn, const_time = spn_handler.learn_parametric_spn(df.values, parametric_types, rdc_threshold, min_instances_slice, colSplit="None")
    
    #Print some statistics
    fn.print_statistics(spn)
    
    #fn.plot_spn(spn)



