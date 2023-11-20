import json
import xgboost as xgb



def reset_tree_info(model_file, num_old_classes):
    
    """
    Modifies tree_info in the json.
    Also returns num_deleted_trees for tree_deletion
    """

    modified_file = model_file.copy()
    tree_info = modified_file['learner']['gradient_booster']['model']['tree_info']
    
    #find the point where the new learned trees start
    idx = tree_info.index(num_old_classes)       # num_old_classes is 0-indexed
    idx = idx - num_old_classes
    
    new_tree_info = tree_info[idx:]
    modified_file['learner']['gradient_booster']['model']['tree_info'] = new_tree_info
    
    num_deleted_trees = int(len(tree_info[:idx])/num_old_classes)
    
    return modified_file, num_deleted_trees

def delete_old_trees(model_file, num_old_classes, num_deleted_trees):
    """
    Deletes the old part of the model.
    """
    
    modified_file = model_file.copy()
    
    trees = modified_file['learner']['gradient_booster']['model']['trees']
    
    new_trees = trees[int(num_old_classes*num_deleted_trees):]
    modified_file['learner']['gradient_booster']['model']['trees'] = new_trees
    
    return modified_file
    

def reset_num_trees(model_file):
    """
    Modifies num_trees in the json.
    Only invoke after reset_tree_info.
    """
    
    modified_file = model_file.copy()
    
    tree_info = modified_file['learner']['gradient_booster']['model']['tree_info']
    num_class = int(modified_file['learner']['learner_model_param']['num_class'])
    
    # the number of trees is the number of occurrences of 0 in tree_info times the number of classes/labels
    modified_file['learner']['gradient_booster']['model']['gbtree_model_param']['num_trees'] = str(int(num_class*tree_info.count(0)))
    
    return modified_file
    

def reset_ids(model_file):
    """
    Modifies num_trees in the json.
    Only invoke after both reset_tree_info and reset_num_trees.
    """
    
    modified_file = model_file.copy()
    
    num_trees = int(modified_file['learner']['gradient_booster']['model']['gbtree_model_param']['num_trees'])
    
    for i in range(int(num_trees)):
        modified_file['learner']['gradient_booster']['model']['trees'][i]['id'] = i
    
    return modified_file


def modify(model_file, num_old_classes):
    """
    Implements the modification pipeline.
    """
    
    modified_file, num_deleted_trees = reset_tree_info(model_file, num_old_classes)
    modified_file = delete_old_trees(modified_file, num_old_classes, num_deleted_trees)
    modified_file = reset_num_trees(modified_file)
    modified_file = reset_ids(modified_file)
    
    return modified_file


def create_modified_model(model_file, num_old_classes):
    """
    Takes the json of the updated model and returns an XGBoost model
    that consists only of the "update part".
    """
    
    with open(model_file, 'r') as file:
        updated_model_file = json.load(file)
        modified = modify(updated_model_file, num_old_classes)

        with open('modified.json', 'w', encoding='utf-8') as f:
            json.dump(modified, f, separators=(',', ':'))
            f.close()

        file.close()
        
    bst_mod = xgb.Booster()
    bst_mod.load_model("modified.json")
        
    return bst_mod