"""
This script implements the entire experiment pipeline for an arbitrary dataset.
It will do the experiments for both the largest and the smallest class (by number of samples)
in the dataset. That means all combinations of training methods and data selection methods.
It is to be called from the command line. In the directory where it is called it creates the
subdirectory results, where the results of the experiments are saved.

I subsampled the datasets to around 10,000 samples, because especially the add_trees training method
scales badly with the number of samples.

I set the parameters in the following way:
    num_models is at 20, but this number can be arbitrarily high, it doesn't change anything just takes longer
    (and ideally delivers more accurate results)

    num_round is at 10, this can be varied to increase model performance. Of course, the higher the number, the longer
    the experiments will take and also I think there will be a higher chance of overfitting?

    max_depth is the most tricky parameter ... for the Beans dataset I set it to 3 which was enough for models
    with over 90% accuracy. For the ForestCover dataset I set it to 5, which yielded around 70% accuracy, but
    with higher values I could have gotten better performing models. I always did some preliminary experiments
    to figure out a suitable value for reasonable accuracy and then ran with that ...


An exemplary call from the command line looks like this:
python3 experiments.py path/to/file/data.csv 20 10 3


Parameters:
    filepath (string)            : filepath to a file that can be read by pandas.read_csv()
                                   needs a column ["Class"] with labels in [0, 1, ...., num_labels]
    num_models (int)             : number of models trained for one proportion of old data, after all are trained the mean
                                   result is returned
    num_round (int)              : the number of rounds with which the original, small model is trained
    max_depth (int)              : the max_depth parameter which is used everywhere for XGBoost
"""


import pandas as pd
import operator
import sys

from mylib import class_distributions

from mylib.pipelines import full_models
from mylib.pipelines import updating_pipeline


# read in the command line arguments
filepath = sys.argv[1]
num_models = int(sys.argv[2])
num_round = int(sys.argv[3])
max_depth = int(sys.argv[4])


data = pd.read_csv(filepath)
labels = data["Class"]

# calculate the class proportions and largest and smallest classes
label_proportions = class_distributions.label_proportions(labels)
largest_class_label = max(label_proportions.items(), key=operator.itemgetter(1))[0]
smallest_class_label = min(label_proportions.items(), key=operator.itemgetter(1))[0]


# use all available training methods
for training_method in ['continued_training', 'add_trees']:
    # do the experiments for both largest and smallest class
    for new_class_idx in [largest_class_label, smallest_class_label]:

        # training the comparison models
        full_models.full_models(filepath,
                                training_method,
                                new_class_idx,
                                num_models,
                                num_round,
                                max_depth)
        
        # the update process - the actual experiments
        for data_selection_method in ['split_criterion', 'dist_to_mean', 'nearest_neighbors', 'entropy']:
            for sort_type in ['closest', 'furthest']:
                updating_pipeline.updating_pipeline(filepath,
                                                    training_method,
                                                    new_class_idx,
                                                    data_selection_method,
                                                    sort_type,
                                                    num_models,
                                                    num_round,
                                                    max_depth)
        
        # for random selection there is no differentitation between closest and furthest
        updating_pipeline.updating_pipeline(filepath,
                                            training_method,
                                            new_class_idx,
                                            'random',
                                            'closest',
                                            num_models,
                                            num_round,
                                            max_depth)
    
    
    
    
