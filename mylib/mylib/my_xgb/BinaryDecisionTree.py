"""
Basic Binary Decision Tree Class.
"""

import numpy as np
import pandas as pd

import time
from numba import njit 
from queue import Queue

from mylib.my_xgb import loss_functions


class Node:
    
    def __init__(self, left=None, right=None, parent=None, split_feat=0, split_val=None,
                 leaf_val=None, base_weight=None, default_left=1, gain=None,
                 cover=None, node_id = None):
        
        self.left = left
        self.right = right
        self.parent = parent
        self.split_feat = split_feat
        self.split_val = split_val   # only if Node is inner Node
        self.leaf_val = leaf_val     # only if Node is a leaf
        self.base_weight = base_weight
        self.default_left = default_left
        self.gain = gain
        self.cover = cover
        self.node_id = node_id
        

class BinaryDecisionTree:
    
    def __init__(self, root=None, class_label=None, num_features=None, **params):
        
        # set min_child_weight=0 for simplicity. The default of XGBoost is 1 !
        # for now this means that I don't need to check the stopping criterion induced by min_child_weight
        
        """
        Parameters:
            root (Node)             : root of the tree
            class_label (int)       : index of the class to be learned (0-based)
            num_features (int)      : number of features
            params (dict)           : Parameters
                max_depth (int)         : max_depth as specified for XGBoost
                eta (float)             : learning rate
                min_child_weight (float): minimum Cover that the children in a split need for that split to happen
                                          otherwise next best split is taken [NOT IMPLEMENTED]
                gamma (float)           : minimum gain required for a split to happen
                lamb (float)            : hyperparameter used in calculations
        """
        
        # I don't know whats going on here ...
        params = params['params']
        
        self.root = root
        self.class_label = class_label
        self.num_features = num_features
        
        self.max_depth = params.get('max_depth', 10)
        self.eta = params.get('eta', 1)
        self.min_child_weight = params.get('min_child_weight', 0)
        self.gamma = params.get('gamma', 0)
        self.lam = params.get('lam', 1)
        
    def fit(self, features, labels, previous_prediction, weights=None):
        """
        Fits a tree.
        
        Parameters:
            features (np.ndarray or pd.DataFrame): the data without labels
            labels (np.ndarray or pd.Series)     : the labels
            previous_prediction (np.ndarray)     : previous prediction of the current model
            weights (np.ndarray)                 : weights for gradient and hessian
        """
        
        if isinstance(features, pd.DataFrame):
            features = features.to_numpy()
        if isinstance(labels, pd.DataFrame) or isinstance(labels, pd.Series):
            labels = labels.to_numpy()
        
        assert isinstance(previous_prediction, np.ndarray)
        
        if isinstance(features, np.ndarray) and isinstance(labels, np.ndarray):
            
            # computes gradient statistics
            grad, hess = loss_functions.softprob_obj(features, labels, previous_prediction, weights)

            self.root = self._grow_tree(features, labels, grad, hess)
            
            # create node_ids in a BFS-manner
            self._create_node_ids()
            # create parent ids
            self._create_parent_ids()
              
        else:
            print("features and labels should have datatype np.ndarray or pd.Dataframe.")
            return
    

    def _grow_tree(self, features, labels, grad, hess, depth=0):
        """
        Grows a tree for given features and labels. The depth where the tree's root is at, is passed as an argument.
        """
        
        best_split_feature, best_split_value, max_gain, cover = _calc_best_split(features=features,
                                                                                 labels=labels,
                                                                                 grad=grad,
                                                                                 hess=hess,
                                                                                 class_label=self.class_label,
                                                                                 lam=self.lam)
        base_weight = self._calc_base_weight(grad, hess)
        
        # check stopping criteria
        if self._stopping_criteria(depth, max_gain):
            leaf_value, cover = self._calc_leaf_value(grad, hess)
            base_weight = self._calc_base_weight(grad, hess)
            return Node(split_val=leaf_value,
                        leaf_val=leaf_value,
                        base_weight=base_weight,
                        default_left=0,
                        gain=leaf_value,
                        cover=cover)
        
        left_idx, right_idx = self._split(features, best_split_feature, best_split_value)
        
        left  = self._grow_tree(features[left_idx],  labels[left_idx],  grad[left_idx],  hess[left_idx],  depth+1)
        right = self._grow_tree(features[right_idx], labels[right_idx], grad[right_idx], hess[right_idx], depth+1)
        
        return Node(left=left, right=right, split_feat=best_split_feature, split_val=best_split_value,
                    base_weight=base_weight, default_left=1, gain=max_gain, cover=cover)
    
    
    def _split(self, features, split_feature, split_value):
        """
        Splits features according to split_feature and split_value.
        """
        left_idx  = np.argwhere(features[:, split_feature] <  split_value).flatten()
        right_idx = np.argwhere(features[:, split_feature] >= split_value).flatten()
        
        return left_idx, right_idx
    

    def _calc_leaf_value(self, grad, hess):
        """
        Calculates a leaf value.
        """
        G = grad[:, self.class_label].sum()
        H = hess[:, self.class_label].sum()
        leaf_value = -self.eta*(G/(H+self.lam))
        cover = H
        
        return leaf_value, cover
    
    
    def _calc_base_weight(self, grad, hess):
        """
        Calculates the base weight of a node. This is basically the leaf value
        that this node would have, but without regard for the learning rate.
        """
        
        G = grad[:, self.class_label].sum()
        H = hess[:, self.class_label].sum()
        base_weight = -G/(H+self.lam)
        
        return base_weight
        
        
    def _stopping_criteria(self, depth, max_gain):
        """
        Implements stopping criteria. Returns True if growing should stop.
        """
        if depth >= self.max_depth:
            return True
        if max_gain < self.gamma:
            return True

        return False

    
    def _create_node_ids(self):
        """
        Gives the tree-nodes IDs in BFS-order.
        """
        
        if self.root==None:
            return
        
        Q=Queue()
        Q.put(self.root)
        ID = 0
        
        while(not Q.empty()):
            node=Q.get()
            if node is None:
                continue
                
            node.node_id = ID
            ID = ID+1
        
            Q.put(node.left)
            Q.put(node.right)
        
        
    def _create_parent_ids(self):
        """
        Assigns parent pointers to nodes.
        """
        
        if self.root==None:
            return
        
        Q=Queue()
        Q.put((self.root, None))
        
        while(not Q.empty()):
            node, parent = Q.get()
            if node is None:
                continue
            
            node.parent = parent
            Q.put((node.left, node))
            Q.put((node.right, node))
           
        
    def _calc_num_nodes(self):
        """
        Counts number of nodes in a tree.
        """
        
        count = 0
        
        if self.root==None:
            return count
        
        Q=Queue()
        Q.put(self.root)
        
        while(not Q.empty()):
            node=Q.get()
            if node is None:
                continue
            
            count = count + 1
        
            Q.put(node.left)
            Q.put(node.right)
        
        return count
    
    
    def tree_to_dataframe(self):
        """
        Writes information in tree into a pd.DataFrame.
        """
        
        if self.root==None:
            return
        
        Q=Queue()
        Q.put(self.root)
        ID = 0
        
        rows_list = []
        
        while(not Q.empty()):
            node=Q.get()
            if node is None:
                continue
            
            new_row = {}
            new_row['Node'] = ID
            ID = ID+1
            
            if node.split_feat is None:
                new_row['Feature'] = 'Leaf'
            else:
                new_row['Feature'] = node.split_feat
                
            new_row['Split'] = node.split_val
            new_row['Gain'] = node.gain
            new_row['Base Weight'] = node.base_weight
            new_row['Cover'] = node.cover
            new_row['Default left'] = node.default_left
            new_row['node_id'] = node.node_id
            if node.parent is not None:
                new_row['parent'] = node.parent.node_id
            else:
                new_row['parent'] = 2147483647
            
            rows_list.append(new_row)
            
            Q.put(node.left)
            Q.put(node.right)
            
        tree_df = pd.DataFrame(rows_list)
        
        return tree_df
    
    
    def to_xgboost_dict(self):
        """
        Creates a dictionary as XGBoost does for each tree.
        """
        
        tree_dict = {}
        
        base_weights = []
        categories = []
        categories_nodes = []
        categories_segments = []
        categories_sizes = []
        default_left = []
        tree_id = 0
        left_children = []
        loss_changes = []
        parents = []
        right_children = []
        split_conditions = []
        split_indices = []
        split_type = []
        sum_hessian = []
        tree_param = {}

        
        if self.root==None:
            return
        
        Q=Queue()
        Q.put(self.root)
        
        while(not Q.empty()):
            node=Q.get()
            if node is None:
                continue
                
            # base weights
            base_weights.append(round(node.base_weight,8))
            # default left
            default_left.append(node.default_left)
            
            # left_children
            if node.left is not None:
                left_children.append(node.left.node_id)
            else:
                left_children.append(-1)
                
            # loss_changes
            if node.leaf_val is None:  #if it's not a leaf
                loss_changes.append(round(node.gain, 8))
            else:
                loss_changes.append(0.0)
                
            # parents
            if node.parent is not None:
                parents.append(node.parent.node_id)
            else:
                parents.append(2147483647)
            
            # right_children
            if node.right is not None:
                right_children.append(node.right.node_id)
            else:
                right_children.append(-1)
                
            # split conditions
            split_conditions.append(round(node.split_val, 8))
            # split_indices
            split_indices.append(node.split_feat)
            # split_type
            split_type.append(0)
            # sum_hessian
            sum_hessian.append(round(node.cover, 8))
            
            # tree param
            tree_param['num_deleted'] = '0'   # seems to always be 0
            tree_param['num_feature'] = str(self.num_features)
            tree_param['num_nodes'] = str(self._calc_num_nodes())
            tree_param['size_leaf_vector'] = '0'   # also seems to always be 0
            
            Q.put(node.left)
            Q.put(node.right)
            
        tree_dict['base_weights'] = base_weights
        
        #these are all empty
        tree_dict['categories'] = categories
        tree_dict['categories_nodes'] = categories_nodes
        tree_dict['categories_segments'] = categories_segments
        tree_dict['categories_sizes'] = categories_sizes
        
        tree_dict['default_left'] = default_left
        tree_dict['id'] = tree_id   # i will reset these later
        tree_dict['left_children'] = left_children
        tree_dict['loss_changes'] = loss_changes
        tree_dict['parents'] = parents
        tree_dict['right_children'] = right_children
        tree_dict['split_conditions'] = split_conditions
        tree_dict['split_indices'] = split_indices
        tree_dict['split_type'] = split_type
        tree_dict['sum_hessian'] = sum_hessian
        tree_dict['tree_param'] = tree_param
        
        return tree_dict
    

@njit
def _calc_best_split(features, labels, grad, hess, class_label, lam):
    """
    Calculates best split for given features, labels and loss statistics.
    """

    num_samples, num_features = features.shape

    best_gain = -np.inf
    best_split_feature = None
    best_split_value = None


    # these are always the same
    G_root = grad[:, class_label].sum()
    H_root = hess[:, class_label].sum()

    for split_feature in range(num_features):
        sorted_feature_ids = features[:, split_feature].argsort()
        G_left = 0
        H_left = 0

        for split_index in range(1, num_samples):
            G_left += grad[sorted_feature_ids[split_index-1], class_label]
            H_left += hess[sorted_feature_ids[split_index-1], class_label]

            G_right = G_root - G_left
            H_right = H_root - H_left
            
            current_gain = _calc_split_gain(G_root, H_root, G_left, H_left, G_right, H_right, lam)

            if current_gain > best_gain:
                best_gain = current_gain
                best_split_feature = split_feature
                best_split_value = _calc_split_val(features,
                                                        sorted_feature_ids,
                                                        split_index,
                                                        split_index-1,
                                                        split_feature)
                    
    return best_split_feature, best_split_value, best_gain, H_root
    
    
@njit
def _calc_split_gain(G_root, H_root, G_left, H_left, G_right, H_right, lam):
    """
    For given statistics calculates the gain.
    """

    return _calc_term(G_left, H_left, lam) + _calc_term(G_right, H_right, lam) - _calc_term(G_root, H_root, lam)
    
    
@njit
def _calc_term(g, h, lam):
    """
    Calculates a term.
    """
    return np.square(g) / (h + lam)
    
    
@njit
def _calc_split_val(features, sorted_feature_ids, split_index1, split_index2, split_feature):
    """
    For a given split_index and split_feature calculates the split_value that XGBoost would use
    (always the midpoints between the values).
    """

    split_val1 = features[sorted_feature_ids[split_index1]][split_feature]
    split_val2 = features[sorted_feature_ids[split_index2]][split_feature]

    return (split_val1+split_val2)/2