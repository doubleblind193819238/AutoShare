import torch as t
import numpy as np
import pickle
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm

t.manual_seed(13)
device = t.device("cuda" if t.cuda.is_available() else "cpu")
t.cuda.set_device(2)
# device = t.device("cpu")

def l2_regularize(array):
    loss = t.sum(array ** 2.0)
    return loss

class MFModel(nn.Module):
    def __init__(self, n_users, n_items, n_factors = 10, dropout = 0, sparse = False):
        super(MFModel, self).__init__()
        self.n_users = n_users 
        self.n_items = n_items
        self.user_biases = nn.Embedding(n_users, 1, sparse=sparse).to(device)
        self.item_biases = nn.Embedding(n_items, 1, sparse=sparse).to(device)
        self.user_embedding = nn.Embedding(n_users, n_factors, sparse = sparse).to(device)
        self.item_embedding = nn.Embedding(n_items, n_factors, sparse = sparse).to(device)
#         self.fc = nn.Linear(2 * n_factors, 1).to(device)
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
#         tmp_emd = t.cat((users_emd, items_emd), axis = 1)
#         preds = self.fc(tmp_emd)
        preds = (self.dropout(users_emd) * self.dropout(items_emd)).sum(dim=1, keepdim=True)
        
        preds += self.user_biases(users)
        
        
        preds += self.item_biases(items)
        
#         reg_loss = 0.05 * (l2_regularize(self.user_embedding.weight) + l2_regularize(self.item_embedding.weight) + l2_regularize(self.item_biases.weight) + l2_regularize(self.user_biases.weight))
#         preds +=  V_regularWeight * (l2_regularize(self.user_embedding.weight) + l2_regularize(self.item_embedding.weight) + l2_regularize(self.item_biases.weight) + l2_regularize(self.user_biases.weight))
#         reg_loss = 0.01 * (l2_regularize(self.user_embedding.weight) + l2_regularize(self.item_embedding.weight))

        
        return preds.squeeze()
    
    def _predict(self, users, items):
#         users_emd = self.user_embedding(users)
#         items_emd = self.item_embedding(items)
#         preds = self.user_biases(users)
        
        
#         preds += self.item_biases(items)
        
        
        
#         preds += (self.dropout(users_emd) * self.dropout(items_emd)).sum(dim = 1, keepdim = True)
        return self.forward(users, items)
    
    def __call__(self, *args):
        return self.forward(*args)
    
    def predict(self, users, items):
        return self._predict(users, items)
    

class MF():
    def __init__(self, trainData, testData, n_users, n_items, dropout, lr, n_factors, batch_size, epoch, dataset):
        self.trainData = trainData
        self.testData = testData
        self.n_users = n_users
        self.n_items = n_items
        self.dropout = dropout
        self.dataset = dataset
        self.lr = lr
        self.n_factors = n_factors
        self.batch_size = batch_size 
        self.epoch = epoch
        ##### record #####
        self.testRmse_records = []
    
    def getModelName(self):
        model_name = self.dataset + "_latent-factor_" + str(self.n_factors) + \
        "_learning-rate_" + str(self.lr) + \
        "_batch_size_" + str(self.batch_size)
        return model_name

    def prepare(self):
        self.mf_model = MFModel(n_users = self.n_users, n_items = self.n_items, n_factors = self.n_factors, dropout = self.dropout)
        self.loss_mse = nn.MSELoss(reduction = "sum")
        self.op = t.optim.SGD(self.mf_model.parameters(), lr = self.lr, weight_decay = 1e-5)
        
    def RMSE(self, pred, groundTrue):
        return np.sqrt(((pred - groundTrue) ** 2).mean())
    
    def test_loss(self, testMat):
        rows_test, cols_test = testMat.nonzero()
        num_ratings = len(rows_test)
        steps=int(np.ceil(num_ratings/self.batch_size))
        epoch_rmse_loss = 0
        tensorRowTest = t.LongTensor(rows_test).to(device)
        tensorColTest = t.LongTensor(cols_test).to(device)
    
    
        for i in tqdm(range(steps)):
            ed = min((i + 1)*self.batch_size, num_ratings)
            rows_batch_ids = tensorRowTest[i*self.batch_size:ed]
            cols_batch_ids = tensorColTest[i*self.batch_size:ed]
            rows_batch = rows_test[i*self.batch_size:ed]
            cols_batch = cols_test[i*self.batch_size:ed]

            batch_len = len(rows_batch_ids)
            groundTrueRatings = np.array(testMat[rows_batch, cols_batch]).reshape(-1)

            pred_t = self.mf_model.predict(cols_batch_ids, rows_batch_ids)
            epoch_rmse_loss += self.RMSE(pred_t.detach().cpu().numpy(), groundTrueRatings)
        epoch_rmse = epoch_rmse_loss / steps
        return epoch_rmse
    
    
    def run(self):
        self.prepare()
        tensorMat = t.from_numpy(self.trainData.toarray()).float().to(device)
        for e in range(self.epoch):
        # target domain
            rows_T, cols_T = self.trainData.nonzero()

            p = np.random.permutation(len(rows_T))
            num_ratings = len(p)
            rows_T, cols_T = rows_T[p], cols_T[p]
            # put all index on GPU
            tensor_rows = t.LongTensor(rows_T).to(device)
            tensor_cols = t.LongTensor(cols_T).to(device)
            

            steps=int(np.ceil(num_ratings/self.batch_size))
            epoch_rmse_loss = 0

            for i in tqdm(range(steps)):
                ed = min((i + 1)*self.batch_size, num_ratings)
                rows_batch_ids = tensor_rows[i*self.batch_size:ed]
                cols_batch_ids = tensor_cols[i*self.batch_size:ed]
                # numpy array

                batch_len = len(rows_batch_ids)

                groundTrueRatings = tensorMat[rows_batch_ids, cols_batch_ids].view(-1)

        #         print(groundTrueRatings_T.shape)
                pred_t = self.mf_model(cols_batch_ids, rows_batch_ids)


                # loss function
                batch_loss = (self.loss_mse(groundTrueRatings, pred_t)) 

                epoch_rmse_loss += self.RMSE(pred_t.detach().cpu().numpy(), groundTrueRatings.detach().cpu().numpy())
                self.op.zero_grad()
                batch_loss.backward()
                
                self.op.step()
            epoch_rmse = epoch_rmse_loss / steps


            print(f"epoch:{e + 1} / {self.epoch} train rmse: {epoch_rmse}")
            test_rmse = self.test_loss(self.testData)
            print(f"epoch:{e + 1} / {self.epoch} test rmse: {test_rmse}")
            self.testRmse_records.append(test_rmse)
            
    def save_model(self, path = "model/"):
        """save the whole model"""
        name = path + self.dataset + ".pkl"

        t.save(self.mf_model, name)
        print(f"save the whole model as :{name}")
    
    def save_baseline(self, path = "MF_records/"):
        """save baseline result"""
        name = path + self.dataset + ".pkl"
        with open(name, "wb") as handle:
            pickle.dump(self.testRmse_records, handle, protocol = pickle.HIGHEST_PROTOCOL)
            
        print(f"save the baseline results as :{name}")
            
    def plot(self, save = False):
        dir = "plot"
        x = list(range(len(self.testRmse_records)))
        fig, ax = plt.subplots(figsize = (14, 7))
        ax.plot(x, self.testRmse_records, label = self.dataset)
        ax.set_title("RMSE on {}".format(self.dataset), fontsize = 18)
        ax.set_xlabel("Epochs", fontsize = "x-large", fontfamily = "sans-serif")
        ax.set_ylabel("RMSE",fontsize = "x-large", fontfamily = 'sans-serif')
        ax.xaxis.set_tick_params(rotation=45,labelsize=12,colors='black') 
        ax.yaxis.set_tick_params(labelsize = 12)
        ax.legend()
        ax.grid()
        name = self.getModelName() + ".png"
        if(save):
            print(f"save fig to: {name}")
            plt.savefig("plot/" + name)
        plt.show()
        
        
        
    
        
    