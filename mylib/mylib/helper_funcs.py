import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

import operator

from mylib import class_distributions

import matplotlib.pyplot as plt
# use Latex in plots
plt.rcParams['text.usetex'] = True
import seaborn as sns

from pathlib import Path



def create_numbered_categories(data, column):
    """
    If a categorical column contains strings its not possible to use all the data_selection methods.
    Thus we turn them into numbers.
    """

    labels_dict = {key:value for (value,key) in enumerate(data[column].unique())}
    data[column] = data[column].map(labels_dict)

    return data
    

def subsample(data, target_num_of_samples, stratify=True):
    """
    Subsamples a dataset.

    Parameters:
        data (pd.DataFrame)        : The data. Needs a column ['Class']
        target_num_of_samples (int): The target number of samples after subsampling.
        stratify (bool)            : If to stratify after ['Class'] while subsampling.
    """

    labels = data['Class']
    X = data.drop(["Class"],axis=1,inplace=False)

    target_proportion = 1-(target_num_of_samples/len(X))

    X_subsample, _, y_subsample, _ = train_test_split(X,
                                                    labels,
                                                    test_size=target_proportion,
                                                    stratify=labels)


    return pd.concat([X_subsample, y_subsample], axis=1)
    


def normalize_data(norm_method, data_small, new_class_data):
    """
    Normalizes the data TOGETHER.

    Parameters:
        norm_method (str)            : The normalization method.
                                           Available:
                                           "min_max"
                                           "mean"
        data_small (pd.DataFrame)     : The original training data.
        new_class_data (pd.DataFrame) : The class to be added. 
    """

    # first need to concatenate the data, so that they are normalized together, otherwise
    # normalization doesn't make sense
    
    full_data = pd.concat([data_small, new_class_data])
    
    if norm_method == 'min_max':
        full_data_normal = (full_data-full_data.min())/np.maximum((full_data.max()-full_data.min()), 10e-6)

    elif norm_method == 'mean':
        full_data_normal = (full_data-full_data.mean())/full_data.std()

    else:
        raise ValueError(f"Normalization method {norm_method} not available!")

    data_small_normal = full_data_normal[:len(data_small)]
    new_class_data_normal = full_data_normal[len(data_small):]

    return data_small_normal, new_class_data_normal


def relabel(labels, old_classes, new_class):
    """
    Relabels the labels in order to be able to learn any old_classes - new_class combination.
    XGBoost requires the labels to be in [0, num_class).
    
    Parameters:
        labels (Pandas Series): the labels.
        old_classes (list)    : the labels of the old classes in training.
        new_class (int)       : the label of the new class in training.
    
    Returns:
        relabeled (Pandas Series): relabeled labels.
    """
    
    relabel_dict = {}
    
    # relabel the old classes
    for i, old_class in enumerate(old_classes):
        relabel_dict[old_class] = i
    
    # relabel the new class
    relabel_dict[new_class] = len(old_classes)
    
    tmp = len(old_classes) + 1
    # relabel the rest
    for label in labels.unique():
        if label not in relabel_dict.keys():
            relabel_dict[label] = tmp
            tmp += 1
            
    return labels.map(relabel_dict), relabel_dict


def largest_or_smallest_class(labels, new_class_idx):
    """
    Returns 'largest class' if largest class is trained, 'smallest class' if smallest class is trained
    and else 'intermediary class'.
    """

    # we need to determine if it's the smallest or largest class that we are adding or neither
    label_proportions = class_distributions.label_proportions(labels)

    largest_class_label = max(label_proportions.items(), key=operator.itemgetter(1))[0]
    smallest_class_label = min(label_proportions.items(), key=operator.itemgetter(1))[0]

    if new_class_idx == largest_class_label:
        largest_or_smallest_class = 'largest class'
    elif new_class_idx == smallest_class_label:
        largest_or_smallest_class = 'smallest class'
    else:
        largest_or_smallest_class = 'intermediary class'

    return largest_or_smallest_class


def create_folder_path(training_method, data_selection_method, sort_type, largest_or_smallest_class):

    """
    Creates the path to the results folder.
    training_method = {continued_training, add_trees}
    """

    folder_path = Path(f'results/{training_method}_{data_selection_method}_{sort_type}_{largest_or_smallest_class}')
    
    return folder_path


def create_file_path(training_method, data_selection_method, sort_type, largest_or_smallest_class, which_data, mean_or_std):

    """
    Creates the path to the results file.
    which_data:
        old_data, new_data, update_data, full_data
    mean_or_std:
        mean, std
    """
    
    results_folder = create_folder_path(training_method, data_selection_method, sort_type, largest_or_smallest_class)

    file_path = Path(f'{training_method}_{data_selection_method}_{sort_type}_{largest_or_smallest_class}_{which_data}_{mean_or_std}_results.pkl')
    
    return file_path


def unpack_results(training_method, data_selection_method, sort_type, largest_or_smallest_class):
    
    folder_path = create_folder_path(training_method, data_selection_method, sort_type, largest_or_smallest_class)

    results = {}
    for which_data in ['old_data', 'new_data', 'update_data', 'full_data']:
        for mean_or_std in ['mean', 'std']:
            file_path = create_file_path(training_method, data_selection_method, sort_type, largest_or_smallest_class, which_data, mean_or_std)
            results_dict = pickle.load(open(folder_path / file_path, 'rb'))
            results[f'{which_data}_{mean_or_std}'] = results_dict

    return results

def unpack_batch_results(training_method, largest_or_smallest_class):

    results = pickle.load(open(f'results/{training_method}_{largest_or_smallest_class}_batch_training_results.pkl', 'rb'))

    return results

def plot_results(training_method,
                 results,
                 batch_results,
                 data_selection_method,
                 sort_type,
                 largest_or_smallest_class,
                 save = False):

    """
    Creates the plots from the results.
    The argument largest_or_smallest_class should get passed a string, either 'largest class' or 'smallest class'.
    This argument is inserted directly in the plot title.
    """

    # create folder where the images are stored
    Path('images').mkdir(parents=True, exist_ok=True)
   
    # give it a name
    images_folder = Path('images')
    
    # set the seaborn standard theme for plotting
    sns.set_theme()

    # i'll just fix it
    proportion_of_old_data = [i*0.1 for i in range(1,10)]

    # unpack the results that are stored in a dict further
    old_data_mean = results['old_data_mean']
    old_data_std = results['old_data_std']
    new_data_mean = results['new_data_mean']
    new_data_std = results['new_data_std']
    update_data_mean = results['update_data_mean']
    update_data_std = results['update_data_std']
    full_data_mean = results['full_data_mean']
    full_data_std = results['full_data_std']

    old_acc_batch = batch_results['old_data']
    new_acc_batch = batch_results['new_data']
    full_acc_batch = batch_results['full_data']


    # in case I did the experiments with varying update rounds there will be multiple keys in each of the dicts
    # the keys indicate the number of performed update rounds
    
    for num_round_updt in old_data_mean.keys():
        # plot performances
        fig = plt.figure(figsize=(14,6))
        ax = plt.gca()
        #ax.set_xlim([0, 1])
        #ax.set_ylim([0.8, 1.05])
    
        #ax.set_xlim([0, 1])
        #ax.set_ylim([0, 1])

        plot_title = f"{training_method}, {data_selection_method}, {sort_type}, {largest_or_smallest_class}"
            
        plt.title(plot_title)
    
        plt.errorbar(proportion_of_old_data,
                     old_data_mean[num_round_updt],
                     yerr=old_data_std[num_round_updt],
                     label="$f^{new}$ on old data",
                     color = 'blue')
    
        plt.errorbar(proportion_of_old_data,
                     new_data_mean[num_round_updt],
                     yerr=new_data_std[num_round_updt],
                     label="$f^{new}$ on new class",
                     color = 'orange')

        """
        plt.errorbar(proportion_of_old_data,
                     update_data_mean[num_round_updt],
                     yerr=update_data_std[num_round_updt],
                     label="update data")
        """
        
        plt.errorbar(proportion_of_old_data,
                     full_data_mean[num_round_updt],
                     yerr=full_data_std[num_round_updt],
                     label="$f^{new}$ on full data",
                     color='red')
        
        plt.axhline(old_acc_batch,
                    color = "blue",
                    linestyle = "--",
                    label = "$f^{full}$ on old data")
        
        plt.axhline(new_acc_batch,
                    color = "orange",
                    linestyle = "--",
                    label = "$f^{full}$ on new class")
        
        plt.axhline(full_acc_batch,
                    color = "red",
                    linestyle = "--",
                    label = "$f^{full}$ on full data")
    
        plt.xlabel("Percentage of old data used in updating")
        plt.xticks(proportion_of_old_data)
        plt.ylabel("Accuracy")
        plt.legend()

        if save:
            filename = plot_title
            savepath = Path(images_folder / f'{filename}')
            plt.savefig(savepath)

        plt.show();