"""

This will be a script to create the images from the pickled results.
Has to be called in the directory where the 'results' directory lies.
"""

import pandas as pd
import numpy as np

from mylib import class_distributions

def create_images(save=True):

    print("Creating images")

    for training_method in ['continued_training', 'add_trees']:
        print(f'Training method: {training_method}')
        for largest_or_smallest_class in ['largest class', 'smallest class']:
            
            batch_results = helper_funcs.unpack_batch_results(training_method, largest_or_smallest_class)
            
            for data_selection_method in ['split_criterion', 'dist_to_mean', 'nearest_neighbors', 'entropy']:
                for sort_type in ['closest', 'furthest']:
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


