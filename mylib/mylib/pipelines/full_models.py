"""
Trains the full models for comparison.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn as skl

import operator
import pickle

from mylib import class_distributions
from mylib import data_selection
from mylib import helper_funcs

from mylib.my_xgb import BinaryDecisionTree as BDT
from mylib.my_xgb import add_class



def full_models(filepath,
                training_method,
                new_class_idx,
                num_models,
                num_round=10,
                max_depth=3):
    """
    Parameters:
        filepath (string)            : filepath to a file that can be read by pandas.read_csv()
                                       needs a column ["Class"] with labels in [0, 1, ...., num_labels]
        training_method (string)     : the training method which the full models are to be compared to
                                       currently available options:
                                           "continued_training"
                                           "add_trees"
        new_class_idx (int)          : index of the new class
        num_models (int)             : number of models trained for one proportion of old data
        num_round (int)              : the number of rounds with which the original, small model is trained
        max_depth (int)              : the max_depth parameter which is used everywhere for XGBoost
    """
    
    # set the random seed for numpy
    np.random.seed(42)

    print(f"Training full models in preparation to add class {new_class_idx} using the {training_method} training method")

    # create the results directory
    Path('results').mkdir(parents=True, exist_ok=True)


    # read and prepare data
    data = pd.read_csv(filepath)

    # need feature matrix X and labels labels for xgboost
    labels = data["Class"]
    X = data.drop(["Class"],axis=1,inplace=False)

    largest_or_smallest_class = helper_funcs.largest_or_smallest_class(labels, new_class_idx)
    



    
    # prepare smaller dataset with only first num_labels classes of beans
    old_classes = np.delete(labels.unique(), new_class_idx)
    new_class = new_class_idx

    # compute number of old labels used
    num_labels = len(old_classes)

    # relabel for XGBoost
    labels, relabel_dict = helper_funcs.relabel(labels, old_classes, new_class)
    
    # the "original" training data
    data_small = X[labels < num_labels]
    labels_small = labels[labels < num_labels]
    
    # to check full model on all of the old data
    dsmall = xgb.DMatrix(data_small, label=labels_small)

    # attempt to retrain with new data
    new_class_data = X[labels == num_labels]
    new_class_labels = labels[labels == num_labels]
    
    # only to check performance on the newly added data
    dnew_class = xgb.DMatrix(new_class_data, label=new_class_labels)

    # also train a model with all the data availale for comparison
    data_full = pd.concat([data_small, new_class_data])
    labels_full = pd.concat([labels_small, new_class_labels])

    # to check full model on the full data
    dfull = xgb.DMatrix(data_full, label=labels_full)
    

    # specify parameters for XGBoost
    if training_method == "continued_training":
        # the fair comparison model needs double the number of training rounds than the original model
        num_round_full = 2*num_round
    elif training_method == 'add_trees':
        # the fair comparison model needs the same number of training rounds as the original model
        num_round_full = num_round
    
    eta = .1
    
    # parameters for full model
    param_full = {'max_depth': max_depth,
                  'eta': eta,
                  'objective': 'multi:softprob',
                  "num_class": num_labels+1}
    param_full['nthread'] = 4
    param_full['eval_metric'] = 'mlogloss'
    
            
    old_acc = np.zeros(num_models)
    new_acc = np.zeros(num_models)
    full_acc = np.zeros(num_models)

    # training
    for model_num in range(num_models):
        
        # split data into train- and test-data
        seed = np.random.randint(0,100)
        X_train_full, X_test_full, y_train_full, y_test_full = skl.model_selection.train_test_split(data_full,
                                                                                                    labels_full,
                                                                                                    test_size=.2,
                                                                                                    random_state=seed)

        dtrain_full = xgb.DMatrix(X_train_full, label=y_train_full)
        dtest_full = xgb.DMatrix(X_test_full, label=y_test_full)
        
        evallist_full = [(dtrain_full, 'train'), (dtest_full, 'eval')]


        # training a model with all the training data
        bst_full = xgb.train(param_full,
                             dtrain_full,
                             num_round_full,
                             evals=evallist_full,
                             verbose_eval=False)
        
        old_acc[model_num] = skl.metrics.accuracy_score(np.argmax(bst_full.predict(dsmall), axis=1),
                                                         labels_small)
        new_acc[model_num] = skl.metrics.accuracy_score(np.argmax(bst_full.predict(dnew_class), axis=1),
                                                         new_class_labels)
        full_acc[model_num] = skl.metrics.accuracy_score(np.argmax(bst_full.predict(dtest_full), axis=1),
                                                         y_test_full)
    
    print("Accuracy of full model on old data: ", old_acc.mean())
    print("Accuracy of full model on new data: ", new_acc.mean())
    print("Accuracy of full model on full data: ", full_acc.mean())

    # save batch training results
    batch_training_results = dict()
    batch_training_results['old_data'] = old_acc.mean()
    batch_training_results['new_data'] = new_acc.mean()
    batch_training_results['full_data'] = full_acc.mean()
    
    pickle.dump(batch_training_results, open(f'results/{training_method}_{largest_or_smallest_class}_batch_training_results.pkl','wb'))

    return True