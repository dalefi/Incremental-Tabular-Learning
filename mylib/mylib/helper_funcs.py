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
        relabel_dict (dict)      : the dictionary used for relabelling
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

    file_path = Path(f'{training_method}_{data_selection_method}_{sort_type}_{largest_or_smallest_class}_{which_data}_{mean_or_std}_results.pkl')
    
    return file_path


def unpack_results(training_method, data_selection_method, sort_type, largest_or_smallest_class):
    
    folder_path = create_folder_path(training_method, data_selection_method, sort_type, largest_or_smallest_class)

    results = {}
    for which_data in ['old_data', 'new_data', 'update_data', 'full_data']:
        for mean_or_std in ['mean', 'std']:
            file_path = create_file_path(training_method, 
                                         data_selection_method, 
                                         sort_type, 
                                         largest_or_smallest_class, 
                                         which_data, 
                                         mean_or_std)
            
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
                     label="$f^{new}$ on $\mathcal{D}^{\mathrm{old}}$",
                     color = 'blue')
    
        plt.errorbar(proportion_of_old_data,
                     new_data_mean[num_round_updt],
                     yerr=new_data_std[num_round_updt],
                     label="$f^{new}$ on $\mathcal{D}^{\mathrm{new}}$",
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
                     label="$f^{new}$ on $\mathcal{D}^{\mathrm{full}}$",
                     color='red')
        
        plt.axhline(old_acc_batch,
                    color = "blue",
                    linestyle = "--",
                    label = "$f^{full}$ on $\mathcal{D}^{\mathrm{old}}$")
        
        plt.axhline(new_acc_batch,
                    color = "orange",
                    linestyle = "--",
                    label = "$f^{full}$ on $\mathcal{D}^{\mathrm{new}}$")
        
        plt.axhline(full_acc_batch,
                    color = "red",
                    linestyle = "--",
                    label = "$f^{full}$ on $\mathcal{D}^{\mathrm{full}}$")
    
        plt.xlabel("Size of the exemplar set")
        plt.xticks(proportion_of_old_data)
        plt.ylabel("Accuracy")
        plt.legend()

        if save:
            filename = plot_title
            savepath = Path(images_folder / f'{filename}')
            plt.savefig(savepath)

        plt.show();


def unpack_results_for_subplots(training_method, sort_type, largest_or_smallest_class, which_data):
    """
    I thought of a better visualization so I need a new unpack function.
    Returns a dictionary with the data selection methods as keys and arrays with the results as values.
    The arrays are of shape (2,9), the first row contains the means, the second one the stds.
    The results are specific to a training method, a sort type, the largest resp. smallest class, and the which_data (old, new, full).
    """

    plot_data = {}
    for data_selection_method in ['split_criterion', 'dist_to_mean', 'nearest_neighbors', 'entropy', 'random']:

        # for random we don't have furthest sort_type
        if data_selection_method == 'random':
            sort_type = 'closest'
        
        folder_path = create_folder_path(training_method, data_selection_method, sort_type, largest_or_smallest_class)

        result_array = np.empty((2,9))
        for idx, mean_or_std in enumerate(['mean', 'std']):
            file_path = create_file_path(training_method, 
                         data_selection_method, 
                         sort_type, 
                         largest_or_smallest_class, 
                         which_data, 
                         mean_or_std)
            
            results_dict = pickle.load(open(folder_path / file_path, 'rb'))
            # there will always be only one key, I just don't have the time for more
            for key in results_dict.keys():
                result_array[idx,:] = np.array(results_dict[key])
        plot_data[data_selection_method] = result_array

    return plot_data
    

def create_single_subplot(plot_data, batch_data, ax):
    """
    Parameters:
        plot_data (dict): a dictionary with all data_selection_methods as keys. the values are
                          2-dim arrays with the mean results in the first- and the std results in
                          the second row
        ax (plt.Axes)   : an ax object to plot to
    """

    #plot_title = f"{training_method}, {sort_type}, {largest_or_smallest_class}"
                    
    #plt.title(plot_title)

    proportion_of_old_data = [.1*i for i in range(1,10)]

    for data_selection_method in plot_data.keys():
    
        ax.errorbar(proportion_of_old_data,
                     plot_data[data_selection_method][0],
                     yerr=plot_data[data_selection_method][1],
                     label = data_selection_method)

    ax.axhline(batch_data,
                linestyle = "--",
               label = '$f^{\mathrm{full}}$')
        
    ax.set_xlabel("Size of the exemplar set")
    ax.set_xticks(proportion_of_old_data)
    ax.set_ylabel("Accuracy")
    ax.legend(borderpad=0.2)
    
    return ax

def subplots_for_training_method_and_sort_type(results_folder,
                                               training_method,
                                               sort_type,
                                               y_lims = None,
                                               save = False,
                                               show = True):

    """
    The goal is to create subplots with three columns and two rows.
    In the first row the results for the update with the largest class is plotted,
    in the second row the results for the smallest class.

    In the first column we plot the results on the old data, in the second and third
    the results on new and full data respectively.

    For one dataset we will need 4 such plots to plot everything relevant I think.

    Parameters:
        results_folder (Path)          : The path to where the results are stored.
        training_method (str)          : The training method ('continued_training' or 'add_trees')
        sort_type (str)                : 'closest' or 'furthest'
        y_lims (tuple)                 : (y_lim_low, y_lim_high) are set if not None
        save (bool)                    : if images are to saved after creation
        show (bool)                    : if images need to be shown
    """
    
    # create folder where the images are stored
    Path('images').mkdir(parents=True, exist_ok=True)
    # give it a name
    images_folder = Path('images')
    
    # set the seaborn standard theme for plotting
    sns.set_theme()

    # i'll just fix it
    proportion_of_old_data = [i*0.1 for i in range(1,10)]

    # create a subplots with 2 rows and 3 cols
    #fig, axs = plt.subplots(2, 3, sharey=True, figsize=(15, 10))
    fig, axs = plt.subplots(2, 3, sharey=False, figsize=(15, 10))
    
    plot_title = f'Subplot_{training_method}_{sort_type}'

    for row_idx, largest_or_smallest_class in enumerate(['largest class', 'smallest class']):
        
        for col_idx, which_data in enumerate(['old_data', 'new_data', 'full_data']):

            # load the batch results
            batch_results = unpack_batch_results(training_method, largest_or_smallest_class)
            batch_data = batch_results[f'{which_data}']

            # load the update results
            plot_data = unpack_results_for_subplots(training_method, sort_type, largest_or_smallest_class, which_data)
            create_single_subplot(plot_data, batch_data, axs[row_idx, col_idx])

    # lets try to add some captions to rows and columns
    cols = ['$\mathcal{D}^{\mathrm{old}}$', '$\mathcal{D}^{\mathrm{new}}$', '$\mathcal{D}^{\mathrm{full}}$']
    rows = ['largest class', 'smallest class']
    
    for ax, col in zip(axs[0], cols):
        ax.set_title(col)
    
    for ax, row in zip(axs[:,0], rows):
        ax.set_ylabel(row, rotation=90, size='large')

    # if we want to adapt the ylims
    if y_lims is not None:
        for ax in axs:
            for sub_ax in ax:
                sub_ax.set_ylim(y_lims)
    
    fig.tight_layout()
    
    if save:
        filename = plot_title
        savepath = Path(images_folder / f'{filename}')
        plt.savefig(savepath, dpi=300, bbox_inches='tight')

    if show:
        plt.show();

    return (fig, axs)


def create_all_subplots(results_folder,
                        save = False,
                        show = True):
    """
    Creates all the subplots for a dataset. All ylims should be the same, thats why we need to create the plots twice.
    """
    y_lim_low_min = np.inf
    y_lim_high_max = -np.inf
    
    # create all the plots once
    for training_method in ['continued_training', 'add_trees']:
        for sort_type in ['closest', 'furthest']:
            print(f'{training_method}, {sort_type}')
            
            fig, axs = subplots_for_training_method_and_sort_type(results_folder,
                                                                    training_method,
                                                                    sort_type,
                                                                    y_lims = None,
                                                                    save = False,
                                                                    show = False)
            # get ylims
            for ax in axs:
                for sub_ax in ax:
                    y_lim_low, y_lim_high = sub_ax.get_ylim()

                    if y_lim_low < y_lim_low_min:
                        y_lim_low_min = y_lim_low
        
                    if y_lim_high > y_lim_high_max:
                        y_lim_high_max = y_lim_high

    # clear previous plots
    plt.close('all')
    
    # create them again but set the ylims
    for training_method in ['continued_training', 'add_trees']:
        for sort_type in ['closest', 'furthest']:
            print(f'{training_method}, {sort_type}')
            
            fig, axs = subplots_for_training_method_and_sort_type(results_folder,
                                                                    training_method,
                                                                    sort_type,
                                                                    y_lims = (y_lim_low_min, y_lim_high_max),
                                                                    save = save,
                                                                    show = show)

    return True


    





    