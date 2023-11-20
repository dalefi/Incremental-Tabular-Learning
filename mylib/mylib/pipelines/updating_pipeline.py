"""
Implements the experiment pipeline for the updating process.
Most (hyper-) parameters are set here for simplicity.
The user only gives very few arguments.
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



def updating_pipeline(filepath,
                      training_method,
                      new_class_idx,
                      data_selection_method,
                      sort_type,
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
        data_selection_method (str)  : data selection method
                                           'split_criterion'
                                           'dist_to_mean'
                                           'nearest_neighbors'
                                           'entropy'
                                           'random'
        sort_type (str)              : sort type for the selected data
                                            'closest'
                                            'furthest'
        num_models (int)             : number of models trained for one proportion of old data
        num_round (int)              : the number of rounds with which the original, small model is trained
        max_depth (int)              : the max_depth parameter which is used everywhere for XGBoost
    """
    
    # set the random seed for numpy
    np.random.seed(42)

    print(f"Adding class {new_class_idx} with {training_method}")
    print(f'Used data selection method: {data_selection_method}. Sort type: {sort_type}')

    # read and prepare data
    data = pd.read_csv(filepath)

    # need feature matrix X and labels labels for xgboost
    labels = data["Class"]
    X = data.drop(["Class"],axis=1,inplace=False)

    # need to figure out which class we are going to add for naming scheme purposes
    largest_or_smallest_class = helper_funcs.largest_or_smallest_class(labels, new_class_idx)

    # create folder where results are stored
    Path(f'results/{training_method}_{data_selection_method}_{sort_type}_{largest_or_smallest_class}').mkdir(parents=True, exist_ok=True)

    # give it a name
    results_folder = Path(f'results/{training_method}_{data_selection_method}_{sort_type}_{largest_or_smallest_class}')

    # create model file 
    Path('models').mkdir(parents=True, exist_ok=True)
    model_folder = Path("models/")

    
    # prepare smaller dataset with only first num_labels classes of beans
    old_classes = np.setdiff1d(labels.unique(), new_class_idx)
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

    # the new class data
    new_class_data = X[labels == num_labels]
    new_class_labels = labels[labels == num_labels]
    
    # only to check performance on the newly added data
    dnew_class = xgb.DMatrix(new_class_data, label=new_class_labels)

    # the entire training data
    data_full = pd.concat([data_small, new_class_data])
    labels_full = pd.concat([labels_small, new_class_labels])

    # to check full model on the full data
    dfull = xgb.DMatrix(data_full, label=labels_full)


    # some parameters
    proportion_of_old_data = [i*0.1 for i in range(1,10)]
    # I don't have the time to vary this
    num_round_update=[num_round]
    eta = .1
    
    # parameters for small model
    param_small = {'max_depth': max_depth,
                   'eta': eta,
                   'objective': 'multi:softprob',
                   "num_class": num_labels}
    param_small['nthread'] = 4
    param_small['eval_metric'] = 'mlogloss'
    
    
    # parameters for update model (the same as for full model, but just in case I want to ever change them)
    param_update = {'max_depth': max_depth,
                    'eta': eta,
                    'objective': 'multi:softprob',
                    "num_class": num_labels+1}
    param_update['nthread'] = 4
    param_update['eval_metric'] = 'mlogloss'

    # these dictionaries are filled with the results and later pickled
    
    old_data_mean_results = dict()
    old_data_std_results = dict()
    new_data_mean_results = dict()
    new_data_std_results = dict()
    update_data_mean_results = dict()
    update_data_std_results = dict()
    full_data_mean_results = dict()
    full_data_std_results = dict()

    # the update routine
    for num_round_update_idx, num_round_updt in enumerate(num_round_update):

        # initialize arrays where results are stored
        old_data_mean = np.zeros(len(proportion_of_old_data))
        old_data_std = np.zeros(len(proportion_of_old_data))
        new_data_mean = np.zeros(len(proportion_of_old_data))
        new_data_std = np.zeros(len(proportion_of_old_data))
        update_data_mean = np.zeros(len(proportion_of_old_data))
        update_data_std = np.zeros(len(proportion_of_old_data))
        full_data_mean = np.zeros(len(proportion_of_old_data))
        full_data_std = np.zeros(len(proportion_of_old_data))
    
        for proportion_num, proportion in enumerate(proportion_of_old_data):
            print(f"Current target proportion of old data in use: {proportion}")

            # initialize arrays where temporary results are stored
            old_data_tmp = np.zeros(num_models)
            new_data_tmp = np.zeros(num_models)
            update_data_tmp = np.zeros(num_models)
            full_data_tmp = np.zeros(num_models)
    
            for model_num in range(num_models):

                # training the original model
                
                seed = np.random.randint(0,100)
                # split original data into train- and test-data
                X_train_small, X_test_small, y_train_small, y_test_small = skl.model_selection.train_test_split(data_small, 
                                                                                                                labels_small,
                                                                                                                test_size=.2,
                                                                                                                random_state=seed)
    
                # specify DMatrices
                dtrain_small = xgb.DMatrix(X_train_small, label=y_train_small)
                dtest_small = xgb.DMatrix(X_test_small, label=y_test_small)
                
                evallist_small = [(dtrain_small, 'train'), (dtest_small, 'eval')]
                
                bst_small = xgb.train(param_small,
                                      dtrain_small,
                                      num_round,
                                      evals=evallist_small,
                                      verbose_eval=False)
    
                bst_small.save_model(fname=model_folder / 'small_model.json')
                
                # use the given selection method
                
                if data_selection_method == 'split_criterion':
                    bst_small_df = bst_small.trees_to_dataframe()

                    # compute important features in model
                    important_features = data_selection.important_features_by_class(bst_small_df, num_labels, criterion="gain")
                    
                    # data selection
                    selected_data = pd.DataFrame(dtype='float64')
                    selected_data_labels = pd.Series(dtype='int8')
                    
                    for (label, feature, split_val) in important_features:
                        tmp, tmp_labels = data_selection.get_samples_split_value(data_small,
                                                                                 labels_small,
                                                                                 feature,
                                                                                 split_val,
                                                                                 label,
                                                                                 ratio_return_total=proportion,
                                                                                 sort_type=sort_type)
                        
                        selected_data = pd.concat([selected_data, tmp])
                        selected_data_labels = pd.concat([selected_data_labels, tmp_labels])

                elif data_selection_method == 'dist_to_mean':
                    # get critical data
                    selected_data, selected_data_labels = data_selection.get_samples_euclidean(data_small,
                                                                                               labels_small,
                                                                                               new_class_data,
                                                                                               ratio_return_total = proportion,
                                                                                               normalization="min_max",
                                                                                               sort_type=sort_type)

                elif data_selection_method == "nearest_neighbors":
                    # get critical data
                    selected_data, selected_data_labels = data_selection.get_samples_nearest_neighbors(data_small,
                                                                                                       labels_small,
                                                                                                       new_class_data,
                                                                                                       ratio_return_total = proportion,
                                                                                                       normalization="min_max",
                                                                                                       alpha=0.5,
                                                                                                       remove_duplicates=False,
                                                                                                       sort_type=sort_type)
                elif data_selection_method == "entropy":
                    # get critical data
                    selected_data, selected_data_labels = data_selection.get_samples_entropy(data_small,
                                                                                             labels_small,
                                                                                             bst_small,
                                                                                             ratio_return_total=proportion,
                                                                                             sort_type=sort_type)
                    
                    
                elif data_selection_method == 'random':
                    seed = np.random.randint(0,100)
                    _, selected_data, _, selected_data_labels = skl.model_selection.train_test_split(data_small,
                                                                                                     labels_small,
                                                                                                     test_size=proportion,
                                                                                                     random_state=seed,
                                                                                                     stratify=labels_small)

                else:
                    raise ValueError("This data selection method does not exist")
                    return
                    
                # concatenate selected data with data of new class
                data_update = pd.concat([selected_data, new_class_data])
                labels_update = pd.concat([selected_data_labels, new_class_labels])

                """
                # split the update data into train and test sets
                seed = np.random.randint(0,100)
                X_train_update, X_test_update, y_train_update, y_test_update = skl.model_selection.train_test_split(data_update,
                                                                                                                    labels_update,
                                                                                                                    test_size=.2,
                                                                                                                    random_state=seed)
                
                # create DMatrices
    
                dtrain_update = xgb.DMatrix(X_train_update, label=y_train_update)
                dtest_update = xgb.DMatrix(X_test_update, label=y_test_update)
    
                evallist_update = [(dtrain_update, 'train'), (dtest_update, 'eval')]
                """

                # use all the update data to update the model
                dtrain_update = xgb.DMatrix(data_update, label=labels_update)
                
                # update model
                bst_update = xgb.train(param_update,
                                      dtrain_update,
                                      num_round_updt,
                                      #evals=evallist_update,
                                      verbose_eval=False,
                                      xgb_model=model_folder/"small_model.json")
    
                
                old_data_tmp[model_num] = skl.metrics.accuracy_score(np.argmax(bst_update.predict(dsmall), axis=1),
                                                                       labels_small)
                new_data_tmp[model_num] = skl.metrics.accuracy_score(np.argmax(bst_update.predict(dnew_class), axis=1),
                                                                       new_class_labels)
                update_data_tmp[model_num] = skl.metrics.accuracy_score(np.argmax(bst_update.predict(dtrain_update), axis=1),
                                                                         labels_update)
                full_data_tmp[model_num] = skl.metrics.accuracy_score(np.argmax(bst_update.predict(dfull), axis=1),
                                                                        labels_full)
    
            old_data_mean[proportion_num] = old_data_tmp.mean()
            old_data_std[proportion_num] = old_data_tmp.std()
            new_data_mean[proportion_num] = new_data_tmp.mean()
            new_data_std[proportion_num] = new_data_tmp.std()  
            update_data_mean[proportion_num] = update_data_tmp.mean()
            update_data_std[proportion_num] = update_data_tmp.std()  
            full_data_mean[proportion_num] = full_data_tmp.mean()
            full_data_std[proportion_num] = full_data_tmp.std()

        old_data_mean_results[num_round_updt] = old_data_mean
        old_data_std_results[num_round_updt] = old_data_std
        new_data_mean_results[num_round_updt] = new_data_mean
        new_data_std_results[num_round_updt] = new_data_std
        update_data_mean_results[num_round_updt] = update_data_mean
        update_data_std_results[num_round_updt] = update_data_std
        full_data_mean_results[num_round_updt] = full_data_mean
        full_data_std_results[num_round_updt] = full_data_std

    pickle.dump(old_data_mean_results, open(results_folder / f'{training_method}_{data_selection_method}_{sort_type}_{largest_or_smallest_class}_old_data_mean_results.pkl','wb'))
    pickle.dump(old_data_std_results, open(results_folder / f'{training_method}_{data_selection_method}_{sort_type}_{largest_or_smallest_class}_old_data_std_results.pkl','wb'))
    pickle.dump(new_data_mean_results, open(results_folder / f'{training_method}_{data_selection_method}_{sort_type}_{largest_or_smallest_class}_new_data_mean_results.pkl','wb'))
    pickle.dump(new_data_std_results, open(results_folder / f'{training_method}_{data_selection_method}_{sort_type}_{largest_or_smallest_class}_new_data_std_results.pkl','wb'))
    pickle.dump(update_data_mean_results, open(results_folder / f'{training_method}_{data_selection_method}_{sort_type}_{largest_or_smallest_class}_update_data_mean_results.pkl','wb'))
    pickle.dump(update_data_std_results, open(results_folder / f'{training_method}_{data_selection_method}_{sort_type}_{largest_or_smallest_class}_update_data_std_results.pkl','wb'))
    pickle.dump(full_data_mean_results, open(results_folder / f'{training_method}_{data_selection_method}_{sort_type}_{largest_or_smallest_class}_full_data_mean_results.pkl','wb'))
    pickle.dump(full_data_std_results, open(results_folder / f'{training_method}_{data_selection_method}_{sort_type}_{largest_or_smallest_class}_full_data_std_results.pkl','wb'))
    
    # remove the saved model file, so that it isn't somehow loaded mistakenly later
    Path.unlink(model_folder / 'small_model.json')
    
    return True