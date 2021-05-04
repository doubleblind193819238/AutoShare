from utils import DataLoader
from utils import compute_entropy
from tqdm import tqdm
from scipy.stats import pearsonr
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
from scipy import sparse
from AutoRecShareLayer_ValidShare_MMD import autoEncoder
from utils import getNum_threshold
from MF import MF

SOURCE = "ml1m"
TARGET = "ml100k"


# load datasets
trainData_T, testData_T, validData_T, trainData_S, testData_S, validData_S = DataLoader(targetData=TARGET, sourceData=SOURCE).load()

# train auto-encoder model
n_items, n_users = trainData_S.shape
params = dict(
    MOVIE_BASED=True,
    modelUTCStr=str(int(time.time())),
    MOVIE_NUM_T=trainData_T.shape[0],
    USER_NUM_T=trainData_T.shape[1],
    MOVIE_NUM_S=trainData_S.shape[0],
    USER_NUM_S=trainData_S.shape[1],
    is_loadModel=False,
    AUTO_BATCH_SIZE=128,
    AUTO_EPOCH=100,
    LATENT_DIM=500,
    LATENT_MID=200,
    SorData=SOURCE,
    TarData=TARGET,
    # hyper-parameters
    LR=0.001,
    LR_DECAY=0.99,
    V_regularWeight=0.05,
    W_regularWeight=0.05,
    GAMMA=10 ^ 3,
    S_weight=1,
    MF_LR=0.001,
    MF_BATCH_SIZE=64,
    MF_EPOCH=130,
    BETA=1000,
    transfer_method="MMD",
    stopTrainEpoch=80,
    is_random_emd=False,
)

auto_encoder_item = autoEncoder(trainData_T, testData_T, validData_T, trainData_S, validData_S, **params)

auto_encoder_item.run()

# evaluate the distance rank
layer2_sim = auto_encoder_item.get_similar_ids_from_sources(10,None, 3)
_, average_distance = getNum_threshold(10, layer2_sim)
print(f"average distance for Layer 3: {average_distance}\n")
# save similarity mat
used_source_num = auto_encoder_item.construct_similar_items_mat(0.93,True)


#pretrain source domain MF
n_items, n_users = trainData_S.shape
mf_params_S = dict(
    n_users = n_users,
    n_items = n_items,
    lr = 0.001,
    batch_size = 256,
    dropout = 0.1,
    n_factors = 10,
    epoch = 100,
    dataset = SOURCE,
)
mf_model_S = MF(trainData_S, testData_S, **mf_params_S)
mf_model_S.run()
# save source domain model
mf_model_S.save_model()

# train target domain MF model with source domain information(beta value = 0.45)
auto_encoder_item.run_MF(trainData_T, testData_T, validData_T, beta_mf = 0.45)
