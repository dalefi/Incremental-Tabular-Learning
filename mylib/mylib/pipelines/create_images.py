"""

This will be a script to create the images from the pickled results.

"""

import pandas as pd
import numpy as np

from mylib import class_distributions

def create_images(filepath):

    # read and prepare data
    data = pd.read_csv(filepath)

    # need feature matrix X and labels labels for xgboost
    labels = data["Class"]
    X = data.drop(["Class"],axis=1,inplace=False)

    label_proportions = class_distributions.label_proportions(labels)
    largest_class_label = max(label_proportions.items(), key=operator.itemgetter(1))[0]
    smallest_class_label = min(label_proportions.items(), key=operator.itemgetter(1))[0]

    

    for training_method in ['continued_training', 'add_trees']:
        print(f'Training method: {training_method}')
        for new_class_idx in [largest_class_label, smallest_class_label]:
            if new_class_idx == largest_class_label:
                largest_or_smallest_class = 'largest class'
            elif new_class_idx == smallest_class_label:
                largest_or_smallest_class = 'smallest class'
    
            print(largest_or_smallest_class)
            
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
                                              save=True)
            
            # and once for random method
            experiment_results = helper_funcs.unpack_results(training_method, 'random', 'closest', largest_or_smallest_class)
            helper_funcs.plot_results(training_method, experiment_results, batch_results, 'random', 'closest', largest_or_smallest_class, save=True)