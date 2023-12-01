"""

This will be a script to create the images from the pickled results.
Has to be called in the directory where the 'results' directory lies.
"""

from pathlib import Path
import pandas as pd
import numpy as np

from mylib import class_distributions

def create_images(save=True):

    print("Creating images")

    results_folder = Path('results/')

    for training_method in ['continued_training', 'add_trees']:
        for largest_or_smallest_class in ['largest class', 'smallest class']:
            for sort_type in ['closest', 'furthest']:

                # get results of full models
                batch_results = helper_funcs.unpack_batch_results(training_method, largest_or_smallest_class)
            
                experiment_results = helper_funcs.unpack_results(training_method, 
                                                                 data_selection_method, 
                                                                 sort_type, 
                                                                 largest_or_smallest_class)
                helper_funcs.plot_results(training_method, 
                                          experiment_results, 
                                          batch_results, 
                                          data_selection_method, 
                                          sort_type, 
                                          largest_or_smallest_class, 
                                          save)
        
        # and once for random method
        experiment_results = helper_funcs.unpack_results(training_method, 'random', 'closest', largest_or_smallest_class)
        helper_funcs.plot_results(training_method, experiment_results, batch_results, 'random', 'closest', largest_or_smallest_class, save)


