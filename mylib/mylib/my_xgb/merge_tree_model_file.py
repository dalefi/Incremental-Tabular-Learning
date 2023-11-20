import json


def _merge_tree_model(model_file=None, new_tree=None, target_class=None, num_orig_classes=None, iteration=None):
    """
    Merges decision tree for a new class with an existing XGBoost model.
    """
    
    # load file
    with open(model_file, 'r') as file:
        model_file = json.load(file)
        file.close()
        
    model_file_copy = model_file.copy()
    tree_list = model_file_copy['learner']['gradient_booster']['model']['trees']
    tree_list = tree_list.insert(iteration*(num_orig_classes+1) + target_class, new_tree.to_xgboost_dict())
    
    return model_file_copy


def _get_num_classes(model_file):
    """
    Takes .json of an XGBoost-model as input and returns the num_class parameter.
    """
    
    num_class = int(model_file['learner']['learner_model_param']['num_class'])
    
    return num_class


def _get_num_trees(model_file):
    """
    Takes .json of an XGBoost-model as input and returns the num_trees parameter.
    This is the number of individual trees! So it is equal to the number of rounds 
    (or number of tree groups) times the number of classes.
    """
    
    num_trees = int(model_file['learner']['gradient_booster']['model']['gbtree_model_param']['num_trees'])
    
    return num_trees


def _adjust_parameters(model_file, iteration):
    """
    Adjusts all relevant parameters in the model_file.
    """
    
    # adjust num_class
    if iteration == 0:   # only in the first iteration
        model_file = _adjust_num_class(model_file)
        
    # adjust num_trees
    model_file  = _adjust_num_trees(model_file)
    
    # adjust tree_info
    model_file = _adjust_tree_info(model_file, iteration)
    
    # adjust the tree ids
    model_file = _adjust_tree_ids(model_file)
    
    
    return model_file
    
    
def _adjust_num_class(model_file):
    """
    Increases all num_class parameters in a model file by 1.
    """
    
    model_file_copy = model_file.copy()
    
    num_class = int(model_file_copy['learner']['learner_model_param']['num_class'])
    model_file_copy['learner']['learner_model_param']['num_class'] = str(num_class+1)
    model_file_copy['learner']['objective']['softmax_multiclass_param']['num_class'] = str(num_class+1)
    
    return model_file_copy


def _adjust_num_trees(model_file):
    """
    Increases the number of trees by one in each iteration.
    """
    
    model_file_copy = model_file.copy()
    
    num_trees = int(model_file_copy['learner']['gradient_booster']['model']['gbtree_model_param']['num_trees'])
    model_file_copy['learner']['gradient_booster']['model']['gbtree_model_param']['num_trees'] = str(num_trees+1)
    
    return model_file_copy


def _adjust_tree_info(model_file, iteration):
    """
    Modifies the tree info list.
    """
    
    model_file_copy = model_file.copy()
    
    # get number that has to be added
    tree_info_list = model_file_copy['learner']['gradient_booster']['model']['tree_info']
    tree_num = max(tree_info_list)
    
    if iteration==0:  # in the first iteration we add max + 1, afterwards only max
        tree_info_list.insert(tree_num+1, tree_num+1)
        
    else:
        tree_info_list.insert(iteration*(tree_num+1) + tree_num, tree_num)
        
    model_file_copy['learner']['gradient_booster']['model']['tree_info'] = tree_info_list
    
    return model_file_copy


def _adjust_tree_ids(model_file):
    """
    Resets the tree_ids after a new tree was added.
    """
    
    model_file_copy = model_file.copy()
    
    num_trees = _get_num_trees(model_file_copy)
    
    for i in range(int(num_trees)):
        model_file_copy['learner']['gradient_booster']['model']['trees'][i]['id'] = i
    
    return model_file_copy