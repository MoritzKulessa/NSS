'''
Created on 05.10.2020

@author: Moritz
'''

import os, sys
file_dir = os.path.dirname(os.path.realpath(__file__)) + "/../"
sys.path.append(file_dir)


import numpy as np

from data.data_stream import WSARE_DATA
from data.syndrome_counting import BasicSyndromeCounting
from algo.benchmarks import Benchmark

from evaluation import measures

from util import io

import logging
logger = logging.getLogger(__name__)



if __name__ == '__main__':
    
    #to compute each data stream independently on the cluster
    if len(sys.argv) > 1:
        data_stream_ids = [int(sys.argv[1])]
    else:
        data_stream_ids = range(0, 100)
        
    #show logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  
    
    #if false, already computed results may be loaded from the file system
    recompute = False
    module_name = os.path.basename(__file__)[:-3]
    
    #Compute the results for all data streams
    measure_results = []
    for data_stream_id in data_stream_ids:
        logger.info("Evaluate data stream: " + str(data_stream_id))
        
        #load data
        data_stream = WSARE_DATA(data_stream_id)
        data_info = data_stream.get_info()
        
        #configure algorithm
        syndrome_counter = BasicSyndromeCounting(data_stream)
        algo = Benchmark(syndrome_counter, distribution="nb", min_parameter=1)
        
        #path to store computed results
        result_ident = module_name + "_" + data_stream.get_ident()
        cache_path = io.get_project_directory() + "_results/" + result_ident + ".pkl"
        
        
        if os.path.exists(cache_path) and (not recompute):
            logger.info("\tLoad results from file..." )
            
            scores = io.load(result_ident, loc="_results/")
        
        else:
            logger.info("\tCompute scores...")
           
            scores = []
            for time_slot in range(data_info["start_test_part"], data_info["last_time_slot"]+1):
                if time_slot % 100 == 0 : logger.info("\t\tEvaluate time slot: " + str(time_slot))
                scores.append(algo.evaluate(time_slot))

            #Save results
            io.save(scores, result_ident, loc="_results")
        
        #Algin the outbreaks with the scores (e.g. scores only start from the test part of the data stream 
        outbreaks = []
        for [start_outbreak, end_outbreak] in data_info["outbreaks"]:
            outbreaks.append([start_outbreak - data_info["start_test_part"], end_outbreak - data_info["start_test_part"]])
        
        #Compute area under partial AMOC-curve (FAR <= 5%)
        roc_values = measures.compute_roc_values(scores, outbreaks)
        amoc_auc5 = measures.compute_area_under_curve(roc_values, x_measure="FAR*0.05", y_measure="detectionDelay")
        measure_results.append(amoc_auc5)
    
    print()
    print("Macro-averaged area under partial AMOC-curve (FAR <= 5%): " + str(np.mean(measure_results)))
    
            
        
        
        
        
    
    
    
    
    