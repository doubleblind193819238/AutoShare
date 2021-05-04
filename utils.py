import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist
from scipy import sparse
from tqdm import tqdm

import pkg_resources

ROOT = ""



DIR_PATH = os.path.join(ROOT, "datasets")
ML100K = "ml100k"
ML1M = "ml1m"
NETFLIX = "netflix"
AMAZON_MOVIE = "amazon_movie"
BOOK_CHILDREN = "children_book"
BOOK_POETRY = "poetry_book"


# Directories
BUILTIN_DATASETS = {
    ML100K: [os.path.join(DIR_PATH, "ml100k_train.npz"), os.path.join(DIR_PATH, "ml100k_test.npz"), os.path.join(DIR_PATH, "ml100k_valid.npz")],
    ML1M: [os.path.join(DIR_PATH, "ml1m_train.npz"), os.path.join(DIR_PATH, "ml1m_test.npz"), os.path.join(DIR_PATH, "ml1m_valid.npz")],
    NETFLIX: [os.path.join(DIR_PATH, "netflix_train_v.npz"), os.path.join(DIR_PATH, "netflix_test_v.npz"), os.path.join(DIR_PATH, "netflix_valid_v.npz")],
    AMAZON_MOVIE: [os.path.join(DIR_PATH, "amazon_movie_train_v.npz"), os.path.join(DIR_PATH, "amazon_movie_test_v.npz"), os.path.join(DIR_PATH, "amazon_movie_valid_v.npz")],
    BOOK_POETRY: [os.path.join(DIR_PATH, "poetry_book_train.npz"), os.path.join(DIR_PATH, "poetry_book_test.npz"), os.path.join(DIR_PATH, "poetry_book_valid.npz")],
    BOOK_CHILDREN: [os.path.join(DIR_PATH, "children_book_train.npz"), os.path.join(DIR_PATH, "children_book_test.npz"), os.path.join(DIR_PATH, "children_book_valid.npz")],

}



class DataLoader():
    def __init__(self, targetData, sourceData = None):
        self.targetData = targetData
        self.sourceData = sourceData
        
    def load(self):
        targetTrain, targetTest, targetValid = BUILTIN_DATASETS[self.targetData]
         # laod datasets
        tarTrainData = sparse.load_npz(targetTrain)
        tarTestData = sparse.load_npz(targetTest)
        tarValidData = sparse.load_npz(targetValid)
        
        if(self.sourceData != None):
            sourceTrain, sourceTest, sourceValid = BUILTIN_DATASETS[self.sourceData]
            sorTrainData = sparse.load_npz(sourceTrain)
            sorTestData = sparse.load_npz(sourceTest)
            sorValidData = sparse.load_npz(sourceValid)
            return tarTrainData, tarTestData, tarValidData, sorTrainData, sorTestData, sorValidData
 
        return tarTrainData, tarTestData, tarValidData
        

def getNum_threshold(threshold, simMat):
    poMat = np.flip(simMat.argsort(axis = 1), axis = 1)
    count = 0
    avg_distance = 0
    steps = simMat.shape[0]
    for i in range(steps):
        distance = int(np.where(poMat[i] == i)[0])
        avg_distance += distance
        if(distance <= threshold):
            count+=1
    avg_distance /= steps
    return count, avg_distance

def getTop_N_feature_mat(target, source, top_N):
    distance_mat = np.abs(target - source)
    disArgsort = distance_mat.mean(axis = 0).argsort()
    new_target_hidden = target[:, disArgsort[:top_N]]
    new_source_hidden = source[:, disArgsort[:top_N]]
    sim_mat = 1 - cdist(new_target_hidden, new_source_hidden, metric = "cosine")
    return sim_mat
        
    
def compute_entropy(ratings, use_softmax=False):
    """Compute the shanon entropy of a user's ratings

    Args:
        ratings (pd.Series): user ratings
        use_softmax (bool, Optional): should softmax be used to compute probabilities?

    Example:
        >>> r = pd.Series(list('12223333333444444444455'))
        >>> np.round(compute_entropy(r), 5)
        1.93114

    Todo:
        * Memoization to improve performance.

    Returns:
        float: shanon entropy
    """
    # calculate the user's rating probabilities
    if use_softmax:
        probs = softmax(ratings.value_counts())
    else:
        probs = ratings.value_counts() / len(ratings)

    # calculate the shanon entropy
    entropy = -1 * np.sum(probs * np.log2(probs))
    return entropy


def stratified_train_test_split(df, based_on, split_ratio=0.9, min_ratings=0):
    """
    used for train test split with min number of rating
    """
    test = []
    train = []
    for key, X in tqdm(df.groupby(based_on)):
        # discard users with less than min ratings
        if X.shape[0] < min_ratings:
            continue
        msk = np.random.random(len(X)) < split_ratio

        if len(X[msk]) < 1:
            train.append(X)
            continue

        test.append(X[~msk])
        train.append(X[msk])

    return pd.DataFrame(pd.concat(train)), pd.DataFrame(pd.concat(test))



def plot_grad_flow(grads):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for p in grads:
        ave_grads.append(p.grad.abs().mean())
        max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    
    


def cal_rmse(predict_mat, test_mat):
    mask = (test_mat != 0).astype(np.float)
    predict_mat = predict_mat * mask
    test_mat = test_mat.reshape(-1)
    predict_mat = predict_mat.reshape(-1)
    return np.sqrt(mean_squared_error(test_mat[test_mat != 0], predict_mat[predict_mat != 0]))



def fill_miss_value(source):
    source_copy = source.copy()
    for index, i in enumerate(source_copy):
        row_mean = np.mean(i[i != 0])
        i[i == 0] = row_mean
        source[index, :] = i
    return source