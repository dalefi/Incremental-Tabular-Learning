import numpy as np
import pandas as pd


def softmax(x):
    return np.exp(x)/sum(np.exp(x))


def xgb_predict_sample(model, sample, num_class, iteration_range = None):
    
    tree_df = model.trees_to_dataframe()
    num_trees = int((tree_df["Tree"].iloc[-1] + 1) / num_class) # one tree consists of num_class trees
    scores = np.zeros(num_class)
    
    if iteration_range is not None:
        start, stop = iteration_range[0], iteration_range[1]
    else:
        start = 0
        stop = num_trees
    
    for i in range(num_class):
        for j in range(start, stop):
            current_tree = tree_df[tree_df["Tree"] == int(i+j*num_class)]
            
            # get splitting feature at root
            ID = current_tree.iloc[0]["ID"]
            current_leaf = current_tree[current_tree["ID"] == ID]
            split_feature = current_leaf["Feature"].iloc[0]
            
            while(split_feature!="Leaf"):
                split_value = sample[split_feature]
                
                if split_value < current_leaf["Split"].iloc[0]:
                    ID = current_tree[current_tree["ID"] == ID]["Yes"].iloc[0]
                else:
                    ID = current_tree[current_tree["ID"] == ID]["No"].iloc[0]
                
                # update current leaf
                current_leaf = current_tree[current_tree["ID"] == ID]
                split_feature = current_leaf["Feature"].iloc[0]
            
            # set scores to output
            scores[i] = scores[i] + current_leaf["Gain"]
    return softmax(scores)


def xgb_predict(model, data, num_class, iteration_range = None):
    if len(data.shape) == 1:
        return np.array([xgb_predict_sample(model, data, num_class, iteration_range)])
    else:
        num_samples = data.shape[0]

        scores = []
        #print(data)
        for index, sample in data.iterrows():
            #print(sample)
            scores.append(xgb_predict_sample(model, sample, num_class, iteration_range))

        return np.array(scores)