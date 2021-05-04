import torch as t
import numpy as np
import mmd
from torch import nn


t.cuda.set_device(3)
# device = t.device("cuda" if t.cuda.is_available() else "cpu")
device = t.device("cpu")
t.manual_seed(13)

def mmd_loss(x_src, x_tar):
    return mmd.mix_rbf_mmd2(x_src, x_tar, [GAMMA])

def l2_regularize(array):
    loss = t.sum(array ** 2.0)
    return loss

class MFModel(nn.Module):
    def __init__(self, n_users, n_items, itemSourceEmd, userSourceEmd, itemSimilarMat, userSimilarMat, beta, n_factors = 10, dropout = 0.1, sparse = False, random_emd = False, movie_based = True):
        super(MFModel, self).__init__()
        # target domain model
        self.n_users = n_users
        self.n_items = n_items
        self.beta = beta
        self.movie_based = movie_based
        if(self.movie_based):
            print("*"*10 + "movie based" + "*"*10)
        else:
            print("*"*10 + "user based" + "*"*10)
        
        
        if(self.movie_based):
            if(random_emd):
                source_item_num = itemSourceEmd.weight.size()[0]
                self.SourceEmd = nn.Embedding(source_item_num, n_factors, sparse = sparse).to(device)
            else:
                self.SourceEmd = itemSourceEmd.to(device)
                
            self.simMat = itemSimilarMat
            self.SourceEmd.weight.requires_grad = False
            self.sourceValue = list(self.simMat.keys())
           
        else:
            if(random_emd):
                source_user_num = userSourceEmd.weight.size()[0]
                self.SourceEmd = nn.Embedding(source_user_num, n_factors, sparse = sparse).to(device)
            else:
                self.SourceEmd = userSourceEmd.to(device)
            self.SourceEmd.weight.requires_grad = False
            
            self.simMat = userSimilarMat
            self.sourceValue = list(self.simMat.keys())
            

        

        

        # source value items is list
        
#         
#         self.TensorIds = t.LongTensor(range(1532)).to(device)
  
        
        self.user_biases = nn.Embedding(n_users, 1, sparse=sparse).to(device)
        self.item_biases = nn.Embedding(n_items, 1, sparse=sparse).to(device)
        self.user_embedding = nn.Embedding(n_users, n_factors, sparse = sparse).to(device)
        self.item_embedding = nn.Embedding(n_items, n_factors, sparse = sparse).to(device)
        # add a adopt layer for source embedding
        self.adopt_layer = nn.Linear(n_factors, n_factors).to(device)
#         self.adopt_layer_2 = nn.Linear(n_factors, n_factors).to(device)

        
        t.nn.init.xavier_uniform_(self.user_embedding.weight)
        t.nn.init.xavier_uniform_(self.item_embedding.weight)
        self.user_biases.weight.data.fill_(0.)
        self.item_biases.weight.data.fill_(0.)
        self.dropout_p = dropout
        self.dropout = nn.Dropout(p = self.dropout_p).to(device)
        self.sparse = sparse
        
    def forward(self, users, items):
        users_emd = self.user_embedding(users)
        items_emd = self.item_embedding(items)

        pred = (self.dropout(users_emd) * self.dropout(items_emd)).sum(dim=1, keepdim=True)

        ## Item Reg Loss
        if(self.movie_based):
            intersect_ids = np.intersect1d(items.cpu(), self.sourceValue)
        else:
            intersect_ids = np.intersect1d(users.cpu(), self.sourceValue)

        
        if(len(intersect_ids) > 0):
            source_emd = []
            for i in intersect_ids:
                source_ids = self.simMat[i]
#                 tensorIds = self.TensorIds[sourceItem_ids]
                tensorIds = t.LongTensor([source_ids]).to(device)
                mean_emd = self.SourceEmd(tensorIds).mean(axis = 0)
                source_emd.append(mean_emd)
            source_emd = t.cat(source_emd).reshape(-1, 10)
#             source_emd = t.from_numpy(np.array(source_emd)).to(device)
#             self.source_emd.weight.requires_grad = False
            if(self.movie_based):
                item_intersect_emd = self.item_embedding(t.LongTensor(intersect_ids).to(device))
                reg_loss = self.beta * (l2_regularize(item_intersect_emd - self.adopt_layer(source_emd)))
            else:
                user_intersect_emd = self.user_embedding(t.LongTensor(intersect_ids).to(device))
                reg_loss = self.beta * (l2_regularize(user_intersect_emd - self.adopt_layer(source_emd)))
#             item_intersect_emd = self.item_embedding(self.TensorIds[item_intersect_ids])
#             reg_loss = ALPHA * (l2_regularize(self.user_embedding.weight) + l2_regularize(self.item_embedding.weight))
            
        else:
            reg_loss = 0
        
        
        
        
        
        pred += self.user_biases(users)
        pred += self.item_biases(items)
#         reg_loss = item_reg_loss + user_reg_loss 
            
        return pred.squeeze(), reg_loss
        
#         #####################################################################################
#         # get the item similar source embedding
#         itemTensorSimArgMat = t.LongTensor(itemSimilarArgMat).to(device)
#         item_emd_similar = self.itemSourceEmd(itemTensorSimArgMat)
#         itemWeightTmp = itemSimilarMat.sum(axis = 1)
#         itemWeightSim = itemSimilarMat / itemWeightTmp[:, np.newaxis]
#         itemWeightTensor = t.from_numpy(itemWeightSim).float().to(device)
#         itemEmdTmp = t.einsum("ijk, ij -> ijk",item_emd_similar, itemWeightTensor)
#         mean_item_emd_S = itemEmdTmp.mean(axis = 1)
#         #####################################################################################
#         #####################################################################################
#         # get the user similar source embedding
#         userTensorSimArgMat = t.LongTensor(userSimilarArgMat).to(device)
#         user_emd_similar = self.userSourceEmd(userTensorSimArgMat)
#         userWeightTmp = userSimilarMat.sum(axis = 1)
#         userWeightSim = userSimilarMat / userWeightTmp[:, np.newaxis]
#         userWeightTensor = t.from_numpy(userWeightSim).float().to(device)
#         userEmdTmp = t.einsum("ijk, ij -> ijk",user_emd_similar, userWeightTensor)
#         mean_user_emd_S = userEmdTmp.mean(axis = 1)
#         #####################################################################################
        
        
        
        
#         diffLoss = BETA * mmd_loss(items_emd, mean_item_emd_S)
#         diff_emd = self.dropout(items_emd_T) - self.dropout(mean_item_emd_S)
#         transferLoss = BETA * t.sum(diff_emd * diff_emd)

        
        
    
    def __call__(self, *args):
        return self.forward(*args)
    
    def _predict(self, users, items):
#         users_emd = self.user_embedding(users)
#         items_emd = self.item_embedding(items)
#         preds = self.user_biases(users)
        
        
#         preds += self.item_biases(items)
        
        
        
#         preds += (self.dropout(users_emd) * self.dropout(items_emd)).sum(dim = 1, keepdim = True)
        return self.forward(users, items)
    
    def predict(self, users, items):
        
        pred, _ = self._predict(users, items)
        return pred
        
        
        
        
        
       

        
        