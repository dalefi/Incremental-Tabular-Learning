"""
Library for data selection algorithms.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn as skl

from sklearn.neighbors import KDTree

from mylib import class_distributions
from mylib import helper_funcs


def weighted_euclidean_distance(x,y,weights):
    """
    Computes weighted euclidean distance between x and y.
    
    Parameters:
        x       (numpy.array): first data point
        y       (numpy.array): second data point
        weights (numpy.array): weights
    """
    
    
    if x.shape[1]!=y.shape[1]:
        print("Data dimension doesn't match.")
        return
    if len(weights)!=x.shape[1]:
        print("Weights don't have the same dimension as data.")
        return
        
    dist = np.sqrt((weights*np.square(x-y)).sum(axis=1))
    
    return dist



def create_weights_from_fscores(data, fscores):
    """
    In XGBoost the features that don't appear as splitting values also don't appear in the fscores.
    I need them as weights, so I assign them a weight of 0.
    Also I normalize the fscores, so they sum up to 1.
    
    Parameters:
        data (Pandas DataFrame): data to get the feature names
        fscores (dict)         : fscores as returned by XGBoost
        
    """
    
    fscores_with_zeroes = {}

    for feature in (data.columns):
        if feature not in fscores.keys():
            fscores_with_zeroes[feature] = 0
        else:
            fscores_with_zeroes[feature] = fscores[feature]
    
    weights = np.fromiter(fscores_with_zeroes.values(), dtype=float)
    weights_normalized = weights/weights.sum()
    
    return weights_normalized



def important_features_by_class(tree_df, num_labels, criterion="gain"):
    """
    Returns a list of tuples with the most important features by the specified criterion for each class
    and their respective mean split-values.
    
    Parameters:
        tree_df (Pandas Dataframe): XGBoost-model converted into dataframe
        num_labels (int)          : number of classes in XGBoost-model
        criterion (str)           : criterion by which importance is measured
                                    Admissible criteria are:
                                    "gain" for total gain.
                                    "avg_gain" for average gain.
                                    "num_splits" for the number of times the feature is used as splitting criterion.
                                    "cover" for cover.
                                    "avg_cover" for average cover.
    """
    
    important_features = []
    
    if criterion == "gain":
        for i in range(num_labels):
            tmp=tree_df[tree_df["Tree"]%6==i]
            
            gain = tmp.groupby(["Feature"]).sum(numeric_only=True)["Gain"]
            feature = gain.idxmax()
            split_val = tmp.groupby(["Feature"]).mean(numeric_only=True)["Split"][feature]

            important_features.append((i, feature, split_val))
    
    if criterion == "avg_gain":
        for i in range(num_labels):
            tmp=tree_df[tree_df["Tree"]%6==i]
            
            gain=tmp.groupby(["Feature"]).sum(numeric_only=True)["Gain"]
            num_splits=tmp.groupby(["Feature"]).count()["Tree"]
            feature=(gain/num_splits).idxmax()
            split_val=tmp.groupby(["Feature"]).mean(numeric_only=True)["Split"][feature]

            important_features.append((i, feature, split_val))

    if criterion == "num_splits":
        for i in range(num_labels):
            tmp=tree_df[tree_df["Tree"]%6==i]
            
            num_splits=tmp.groupby(["Feature"]).count()["Tree"]
            feature = num_splits.idxmax()
            split_val=tmp.groupby(["Feature"]).mean(numeric_only=True)["Split"][feature]
            
            important_features.append((i, feature, split))
            
    if criterion == "cover":
        for i in range(num_labels):
            tmp=tree_df[tree_df["Tree"]%6==i]
            
            gain = tmp.groupby(["Feature"]).sum(numeric_only=True)["Cover"]
            feature = gain.idxmax()
            split = tmp.groupby(["Feature"]).mean(numeric_only=True)["Split"][feature]

            important_features.append((i, feature, split_val))
    
    if criterion == "avg_cover":
        for i in range(num_labels):
            tmp=tree_df[tree_df["Tree"]%6==i]
            
            gain=tmp.groupby(["Feature"]).sum(numeric_only=True)["Cover"]
            num_splits=tmp.groupby(["Feature"]).count()["Tree"]
            feature=(gain/num_splits).idxmax()
            split_val=tmp.groupby(["Feature"]).mean(numeric_only=True)["Split"][feature]

            important_features.append((i, feature, split_val))
    
    return important_features



def get_samples_split_value(dataframe,
                            labels,
                            feature_name,
                            split_value,
                            label,
                            ratio_return_total,
                            sort_type="closest"):
    """
    Returns Dataframe with ratio_return_total percent of data_update.
    Returned data are close to (or far away from) split_value for feature_name.
    
    Parameters:
        dataframe (Pandas Dataframe): XGBoost-model converted into dataframe
        labels    (Pandas Series)   : labels of dataset
        feature_name (str)          : name of the feature to select by
        split_value (float)         : the (mean) split value that selected samples should be close to in the chosen feature
        label (int)                 : the label from which the data is to be selected
                                      if None then all the data is selected from
        ratio_return_total (float)  : ratio of number of returned samples to number of total samples
        sort_type (str)             : choose if closest or furthest samples from split_value should be selected
                                      Admissible sort_types are:
                                      "closest" : closest values
                                      "furthest": furthest values
    """

    num_labels = len(labels.unique())
    upper_limit = int(len(dataframe)*ratio_return_total)
    
    if sort_type=="closest":
        if label is not None:
            # compute proportion of data which belongs to given label
            label_proportion = class_distributions.label_proportions(labels)[label]
            # compute the number of labels that are supposed to be selected
            upper_limit_for_label = int(len(dataframe)*ratio_return_total*label_proportion)

            # select labels
            dataframe = dataframe[labels==label]
            idx = abs((dataframe[feature_name]-split_value)).sort_values()[:upper_limit_for_label].index
        else:
            idx = abs((dataframe[feature_name]-split_value)).sort_values()[:upper_limit].index
            
    elif sort_type=="furthest":
        if label is not None:
            # compute proportion of data which belongs to given label
            label_proportion = class_distributions.label_proportions(labels)[label]
            # compute the number of labels that are supposed to be selected
            upper_limit_for_label = int(len(dataframe)*ratio_return_total*label_proportion)

            # select labels
            dataframe = dataframe[labels==label]
            idx = abs((dataframe[feature_name]-split_value)).sort_values()[-upper_limit_for_label:].index
        else:
            idx = abs((dataframe[feature_name]-split_value)).sort_values()[-upper_limit:].index
    else:
        print("Not a supported sort_type")
        return
        
    return dataframe.loc[idx], labels.loc[idx]


def get_samples_euclidean(data,
                          labels,
                          data_update,
                          ratio_return_total,
                          normalization="min_max",
                          sort_type = "closest",
                          weights = None):
    """
    Returns Dataframe with ratio_return_total percent of data_update.
    Returned data are closest to (or furthest away from) the mean of data_update in euclidean norm.
    With or without normalization.
    
    Parameters:
        data (Pandas Dataframe)       : old feature data
        labels (Pandas Series)        : labels of old feature data
        data_update (Pandas Dataframe): feature data of new class
        ratio_return_total (float)    : ratio of number of returned samples to number of total samples
        normalization (bool)          : which kind of normalization should be applied
                                        Admissible normalization values are:
                                        "min_max": min-max normalization
                                        "mean"   : mean-normalization
                                        None     : No normalization
        sort_type (str)               : choose if closest or furthest samples from split_value should be selected
                                        Admissible sort_types are:
                                        "closest" : closest values
                                        "furthest": furthest values
        weights (numpy.array)         : weights for weighted euclidean distance, should add up to 1
                                        None for unweighted euclidean distance
    """
    num_labels = len(labels.unique())
    upper_limit = int(len(data)*ratio_return_total)

    # initialize returned data
    selected_data = pd.DataFrame(dtype='float64')
    selected_data_labels = pd.Series(dtype='int8')
    
    # normalize
    data_normal, data_update_normal = helper_funcs.normalize_data(normalization, data, data_update)

    # calculate mean of update_data
    data_update_normal_mean = data_update_normal.mean(axis=0)
    
    if weights is not None:
        # check if weights sum up to 1
        if weights.sum() - 1 > 10e-6:
            print("Weights don't sum up to 1.")
            return
        
        distances = pd.Series(weighted_euclidean_distance(data_normal,
                                                          data_update_normal_mean.to_numpy().reshape(1,-1),
                                                          weights = weights),
                                                          index=data.index).sort_values()

    else:
        # calculate distances to mean
        distances = pd.Series(weighted_euclidean_distance(data_normal,
                                                          data_update_normal_mean.to_numpy().reshape(1,-1),
                                                          weights = np.ones(data.shape[1])),
                                                          index=data.index).sort_values()
    
    
    if sort_type == "closest":
        for label in labels.unique():
            # compute proportion of data which belongs to given label
            label_proportion = class_distributions.label_proportions(labels)[label]
            # compute the number of labels that are supposed to be selected
            upper_limit_for_label = int(len(data)*ratio_return_total*label_proportion)

            # select labels
            idx = distances[labels==label].sort_values()[:upper_limit_for_label].index
            tmp = data.loc[idx]
            tmp_labels = labels.loc[idx]

            selected_data = pd.concat([selected_data, tmp])
            selected_data_labels = pd.concat([selected_data_labels, tmp_labels])
            
    elif sort_type == "furthest":
        for label in labels.unique():
            # compute proportion of data which belongs to given label
            label_proportion = class_distributions.label_proportions(labels)[label]
            # compute the number of labels that are supposed to be selected
            upper_limit_for_label = int(len(data)*ratio_return_total*label_proportion)

            # select labels
            idx = distances[labels==label].sort_values()[-upper_limit_for_label:].index
            tmp = data.loc[idx]
            tmp_labels = labels.loc[idx]

            selected_data = pd.concat([selected_data, tmp])
            selected_data_labels = pd.concat([selected_data_labels, tmp_labels])
    else:
        print("Not a supported sort_type")
        return
    
    return selected_data, selected_data_labels


def get_samples_nearest_neighbors(data, 
                                  labels,
                                  data_update, 
                                  ratio_return_total,
                                  normalization="min_max",
                                  alpha=1,
                                  remove_duplicates=False,
                                  sort_type="closest"):
    
    """
    Returns Dataframe with ratio_return_total percent of data_update.
    For each class in the old data we add the closest points that are not in the class to the returned data (=O).
    Also we add the closest points in the old data to the new data (data_update) to the returned data (=N).
    Alpha controls the composition.
    Returned data should be stratified.
    
    Parameters:
        data (Pandas Dataframe)       : old feature data
        labels (Pandas Series)        : labels of old feature data
        data_update (Pandas Dataframe): feature data of new class
        ratio_return_total (float)    : ratio of number of returned samples to number of total samples
        normalization (bool)          : which kind of normalization should be applied
                                        Admissible normalization values are:
                                        "min_max": min-max normalization
                                        "mean"   : mean-normalization
                                        None     : No normalization
        alpha (float)                 : controls how much of returned data is close to new data and how much is
                                        from the boundaries of old classes.
                                        default alpha=1 means all returned data is close to new data.
        remove_duplicates (bool)      : the data might contain duplicates. by default they are removed.
    """

    # compute distribution of classes in old data
    class_distribution = class_distributions.label_proportions(labels)
    
    upper_limit = int(len(data)*ratio_return_total)

    # normalize
    data_normal, data_update_normal = helper_funcs.normalize_data(normalization, data, data_update)

    # first focus on N
    N = pd.DataFrame(dtype='float64')
    N_labels = pd.Series(dtype='int8')
    # build kd-tree and find the nearest neighbor in data_update to every point in data
    tree = skl.neighbors.KDTree(data_update_normal, leaf_size=50)
    dist, ind = tree.query(data_normal, k=1)

    if sort_type=="closest":
        for label in labels.unique():
            dist_class = dist[labels==label]
        
            idx_sort = np.argsort(dist_class.flatten())
            
            # pick the data with the smallest distance
            label_freq = class_distribution.loc[label]
            upper_limit_for_label = int(len(data)*ratio_return_total*label_freq*alpha)
            
            tmp = data[labels==label].iloc[idx_sort[:upper_limit_for_label]]
            tmp_labels = labels[labels==label].iloc[idx_sort[:upper_limit_for_label]]
    
            N = pd.concat([N, tmp])
            N_labels = pd.concat([N_labels, tmp_labels])

    if sort_type=="furthest":
        for label in labels.unique():
            dist_class = dist[labels==label]
        
            idx_sort = np.argsort(dist_class.flatten())
            
            # pick the data with the smallest distance:
            label_freq = class_distribution.loc[label]
            upper_limit_for_label = int(len(data)*ratio_return_total*label_freq*alpha)
            
            tmp = data[labels==label].iloc[idx_sort[-upper_limit_for_label:]]
            tmp_labels = labels[labels==label].iloc[idx_sort[-upper_limit_for_label:]]
    
            N = pd.concat([N, tmp])
            N_labels = pd.concat([N_labels, tmp_labels])

    # now focus on O
    
    O = pd.DataFrame()
    O_labels = pd.DataFrame(dtype='int64')

    if sort_type=="closest":
        for label in labels.unique():
            # get all the data corresponding to the label
            class_data = data_normal.loc[labels[labels==label].index]
            class_data_labels = labels.loc[labels[labels==label].index]
            rest_data = data_normal.loc[labels[labels!=label].index]
            
            tree = KDTree(rest_data, leaf_size=50)
            dist, ind = tree.query(class_data, k=1)
            
            idx_sort = np.argsort(dist.flatten())
            
            # get frequency of current label
            label_freq = class_distribution.loc[label]
            limit = int(len(data)*ratio_return_total*(1-alpha)*label_freq)
            
            O = pd.concat([O, class_data.iloc[idx_sort[:limit]]])
            O_labels = pd.concat([O_labels, class_data_labels.iloc[idx_sort[:limit]]])

    if sort_type=="furthest":
        for label in labels.unique():
            # get all the data corresponding to the label
            class_data = data_normal.loc[labels[labels==label].index]
            class_data_labels = labels.loc[labels[labels==label].index]
            rest_data = data_normal.loc[labels[labels!=label].index]
            
            tree = skl.neighbors.KDTree(rest_data, leaf_size=50)
            dist, ind = tree.query(class_data, k=1)
            
            idx_sort = np.argsort(dist.flatten())
            
            # get frequency of current label
            label_freq = class_distribution.loc[label]
            limit = int(len(data)*ratio_return_total*(1-alpha)*label_freq)
            
            O = pd.concat([O, class_data.iloc[idx_sort[-limit:]]])
            O_labels = pd.concat([O_labels, class_data_labels.iloc[idx_sort[-limit:]]])

    return_data = pd.concat([N,O])
    return_data_labels = pd.concat([N_labels, O_labels])
    
    # remove duplicates
    if remove_duplicates:
        return_data = return_data[~return_data.index.duplicated(keep='first')]
        return_data_labels = return_data_labels[~return_data_labels.index.duplicated(keep='first')]
        
    return return_data, return_data_labels.astype(int)


def entropy(distribution):
    """
    Calculates entropy of a probability distribution.

    Parameters:
        distribution (list or np.ndarray): probability distribution

    Returns:
        entropy (float)
    """

    if isinstance(distribution, list):
        prob_distrib = np.array(distribution)
    elif isinstance(distribution, np.ndarray):
        prob_distrib = distribution
    else:
        print("Distribution has to be list or numpy array")

    entropy = -np.sum(prob_distrib*np.log2(prob_distrib), axis=1)

    return entropy
    

    

def get_samples_entropy(data, labels, model, ratio_return_total, sort_type="closest"):
    """
    Chooses data based on entropy of softmaxed predictions of model.
    The higher the entropy, the more unsure the model is about a prediction.
    """
        
    limit = int(ratio_return_total*len(data))

    # define DMatrix
    ddata = xgb.DMatrix(data, label=labels)
    
    predictions = model.predict(ddata)
    entropy_scores = entropy(predictions)

    # initialize returned data
    selected_data = pd.DataFrame(dtype='float64')
    selected_data_labels = pd.Series(dtype='int8')
    
    # sort by entropy
    entropy_sort_idx = np.argsort(entropy_scores)
    data_sorted = data.iloc[entropy_sort_idx]
    labels_sorted = labels.iloc[entropy_sort_idx]

    # note that here we want to select for the LARGEST entropy, hence the inverted sorting compared to previous methods
    if sort_type == "closest":
        for label in labels.unique():
            # compute proportion of data which belongs to given label
            label_proportion = class_distributions.label_proportions(labels)[label]
            # compute the number of labels that are supposed to be selected
            upper_limit_for_label = int(len(data)*ratio_return_total*label_proportion)
            
            tmp = data_sorted.loc[labels==label]
            tmp = tmp[-upper_limit_for_label:]
            tmp_labels = labels_sorted[labels==label]
            tmp_labels = tmp_labels[-upper_limit_for_label:]

            selected_data = pd.concat([selected_data, tmp])
            selected_data_labels = pd.concat([selected_data_labels, tmp_labels])
        
    elif sort_type == "furthest":
        for label in labels.unique():
            # compute proportion of data which belongs to given label
            label_proportion = class_distributions.label_proportions(labels)[label]
            # compute the number of labels that are supposed to be selected
            upper_limit_for_label = int(len(data)*ratio_return_total*label_proportion)
            
            tmp = data_sorted.loc[labels==label]
            tmp = tmp[:upper_limit_for_label]
            tmp_labels = labels_sorted.loc[labels==label][:upper_limit_for_label]

            selected_data = pd.concat([selected_data, tmp])
            selected_data_labels = pd.concat([selected_data_labels, tmp_labels])
    
    return selected_data, selected_data_labels






