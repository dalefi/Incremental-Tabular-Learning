import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn as skl


def label_proportions(series):
    """
    Returns series with label proportions.
    """
    
    return series.value_counts()/len(series)

def max_label_proportion_difference(labels1, labels2):
    label_proportions1 = label_proportions(labels1)
    label_proportions2 = label_proportions(labels2)
    
    label_prop_diff = abs(label_proportions1-label_proportions2)
    
    return label_prop_diff.max()

def mean_label_proportion_difference(labels1, labels2):
    label_proportions1 = label_proportions(labels1)
    label_proportions2 = label_proportions(labels2)
    
    label_prop_diff = abs(label_proportions1-label_proportions2)
    
    return label_prop_diff.mean()