"""
Functions to implement a pipeline to add a class to an XGBoost model by adding class-trees.
"""

import numpy as np
import xgboost as xgb
from pathlib import Path

import json

from mylib.my_xgb import loss_functions
from mylib.my_xgb import BinaryDecisionTree as BDT
from mylib.my_xgb import merge_tree_model_file as merge



def add_class(model_file=None,
              data=None,
              labels=None,
              num_tree_grps=None,
              num_orig_classes=None,
              num_iterations=None,
              params=None,
              weights=None):
    """
    Implements the pipeline to add a class to an XGBoost model. 
    
    Parameters:
        model_file (.json)           : .json-dump of XGBoost model, that a class is supposed to be added to
        data (pd.DataFrame)          : training data
        labels (pd.Series)           : labels for the training data
        num_tree_grps (int)          : number of tree groups in original model (equals num_round)
        num_orig_classes (int)       : number of classes the original model had
        num_iterations (int)         : number of iterations that new trees should be trained
        params (dict)                : parameters of XGBoost model (need only be passed to build_tree function)
        weights (np.ndarray)         : weights for gradient and hessian
            
    Returns: 
        Modified model.
    """

    # want to update the model step-wise
    current_model = model_file
    
    if num_iterations is None:
        num_iterations = num_tree_grps
        
    for iteration in range(max(num_tree_grps, num_iterations)):
        #print("Adding tree number",  iteration+1)
        # note classes are 0-based
        
        new_tree = build_tree(current_model,
                              data,
                              labels,
                              iteration,
                              target_class=num_orig_classes,
                              num_orig_classes=num_orig_classes,
                              weights=weights,
                              params=params)
        
        current_model = merge._merge_tree_model(current_model,
                                                  new_tree,
                                                  target_class=num_orig_classes,
                                                  num_orig_classes = num_orig_classes,
                                                  iteration=iteration)
        
        current_model = merge._adjust_parameters(current_model,
                                                   iteration=iteration)
        
        with open('current_model.json', 'w', encoding='utf-8') as f:
            json.dump(current_model, f, separators=(',', ':'))
            f.close()
            
        current_model = 'current_model.json'
        
        
    bst_mod = xgb.Booster()
    bst_mod.load_model('current_model.json')

    # remove the current_model file
    Path.unlink(Path('current_model.json'))
    
    return bst_mod


def build_tree(model_file=None,
               data=None,
               labels=None,
               iteration=None,
               target_class=None,
               num_orig_classes=None,
               weights=None,
               params=None):
    """
    Builds a single decision tree in a gradient boosted fashion like XGBoost. The new tree
    is for a new class.
    """
    
    # load the model
    current_model = xgb.Booster()
    current_model.load_model(model_file)
    
    num_samples, num_features = data.shape
    
    # get previous prediction
    if iteration != 0:
        prev_prediction = _get_current_prediction(current_model, data, labels, iteration)
        
    else:
        # need to artificially create the previous prediction
        prev_prediction = _create_initial_prediction(num_orig_classes, num_samples)
    
    new_tree = BDT.BinaryDecisionTree(root=None,
                                      class_label=target_class,
                                      num_features=num_features,
                                      params=params)

    new_tree.fit(data, labels, prev_prediction, weights)
    return new_tree
    
    
def _get_current_prediction(model, data, labels, iteration):
    """
    Returns current prediction at iteration step.
    """
    
    """
    prediction = model.predict(xgb.DMatrix(data, label=labels),
                               output_margin=True,
                               iteration_range=(0, iteration))
    """

    prediction = model.predict(xgb.DMatrix(data, label=labels),
                               output_margin=True)
        
    return prediction


def _create_initial_prediction(num_orig_classes, num_samples):
    """
    Creates the initial prediction.
    """
    
    prediction = (1/(num_orig_classes+1)) * np.ones((num_orig_classes+1)*num_samples).reshape(num_samples,
                                                                                              num_orig_classes+1)
    
    return prediction
