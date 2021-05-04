import torch as t
import torch.nn as nn
import pickle
import numpy as np

import os
import mmd

from TransMF import MFModel
from scipy.spatial.distance import cdist
from utils import getNum_threshold
from tqdm import tqdm
from torch.autograd import Function
from matplotlib import pyplot as plt

# t.cuda.set_device(3)
# device = t.device("cuda" if t.cuda.is_available() else "cpu")
device = t.device("cpu")
######### reproducibility ##########
t.backends.cudnn.deterministic = True
t.backends.cudnn.benchmark = False
print('device: ' + str(device))
resultFile = 'result.txt'
t.manual_seed(13)
np.random.seed(13)


class autoEncoder():
    def __init__(self, trainData_T, testData_T, validData_T, trainData_S, testData_S, MOVIE_NUM_T, USER_NUM_T,
                 MOVIE_NUM_S, USER_NUM_S, MOVIE_BASED, AUTO_BATCH_SIZE, AUTO_EPOCH, LATENT_DIM, LATENT_MID, LR,
                 LR_DECAY, V_regularWeight, W_regularWeight, GAMMA, S_weight, modelUTCStr, MF_LR, MF_BATCH_SIZE,
                 MF_EPOCH, BETA, stopTrainEpoch, SorData, TarData, is_random_emd, is_loadModel=False,
                 transfer_method="MMD"):

        if (MOVIE_BASED):
            self.trainData_T = trainData_T
            self.testData_T = testData_T
            self.validData_T = validData_T
            self.trainData_S = trainData_S
            self.testData_S = testData_S

        else:
            self.trainData_T = trainData_T.T
            self.testData_T = testData_T.T
            self.validData_T = validData_T.T
            self.trainData_S = trainData_S.T
            self.testData_S = testData_S.T

        self.MOVIE_NUM_T = MOVIE_NUM_T
        self.USER_NUM_T = USER_NUM_T
        self.MOVIE_NUM_S = MOVIE_NUM_S
        self.USER_NUM_S = USER_NUM_S
        self.MOVIE_BASED = MOVIE_BASED
        self.AUTO_BATCH_SIZE = AUTO_BATCH_SIZE
        self.AUTO_EPOCH = AUTO_EPOCH
        self.LATENT_DIM = LATENT_DIM
        self.LATENT_MID = LATENT_MID
        self.modelUTCStr = modelUTCStr
        self.dataset = TarData + SorData
        self.tar_name = TarData
        self.sor_name = SorData
        self.MF_LR = MF_LR
        self.BETA = BETA

        self.LR = LR
        self.LR_DECAY = LR_DECAY
        self.V_regularWeight = V_regularWeight
        self.W_regularWeight = W_regularWeight
        self.is_loadModel = is_loadModel
        self.S_weight = S_weight
        self.GAMMA = GAMMA

        self.trainMask1 = (self.trainData_T != 0)
        self.testMask1 = (self.testData_T != 0)
        self.validMask1 = (self.validData_T != 0)
        self.trainMask2 = (self.trainData_S != 0)
        self.testMask2 = (self.testData_S != 0)
        self.loss_mse = nn.MSELoss(reduction='sum')  # 没有求mean=(x-y)^2
        self.loss_mae = nn.L1Loss(reduction="sum")
        self.domain_loss = nn.NLLLoss()
        self.transfer_method = transfer_method
        self.stopTrainEpoch = stopTrainEpoch
        self.is_random_emd = is_random_emd

        self.curEpoch = 0
        self.train_losses = []
        self.train_RMSEs = []
        self.test_losses = []
        self.test_RMSEs = []
        self.step_losses = []
        self.average_distance = []
        self.grads = []

        ######################## MF Part ##################
        self.MF_BATCH_SIZE = MF_BATCH_SIZE
        self.MF_EPOCH = MF_EPOCH
        self.MF_test_rmse = []

        ######################## gradient Observation ########################
        self.V_share_grad = []
        self.W_share_grad = []
        self.V1_grad = []
        self.W1_grad = []
        self.V2_grad = []
        self.W2_grad = []

    def mmd_loss(self, x_src, x_tar):
        return mmd.mix_rbf_mmd2(x_src, x_tar, [self.GAMMA])

    def GAN_loss(self, domainS, domainT):
        labelS = t.full_like(domainS, 0)[:, 0]
        labelS = labelS.type(t.LongTensor).to(device)
        labelT = t.full_like(domainT, 1)[:, 0]
        labelT = labelT.type(t.LongTensor).to(device)
        disLossS = self.domain_loss(domainS, labelS)
        disLossT = self.domain_loss(domainT, labelT)
        disLoss = disLossS + disLossT
        return disLoss

    def prePareModel(self):
        if self.MOVIE_BASED:
            D_in1 = self.USER_NUM_T
            D_out1 = self.USER_NUM_T
            D_in2 = self.USER_NUM_S
            D_out2 = self.USER_NUM_S
        else:
            D_in1 = self.MOVIE_NUM_T
            D_out1 = self.MOVIE_NUM_T
            D_in2 = self.MOVIE_NUM_S
            D_out2 = self.MOVIE_NUM_S
        ###########################################   
        # prepare for auto encoder model          #
        self.V1 = t.empty(D_in1, self.LATENT_DIM, dtype=t.float, device=device, requires_grad=True)  # 需要求梯度,需要更新
        # first encoder
        self.W1_2 = t.empty(self.LATENT_DIM, D_out1, dtype=t.float, device=device, requires_grad=True)
        # share layer

        self.V_share = t.empty(self.LATENT_DIM, self.LATENT_MID, dtype=t.float, device=device, requires_grad=True)
        self.W_share = t.empty(self.LATENT_MID, self.LATENT_DIM, dtype=t.float, device=device, requires_grad=True)

        # second encoder
        self.V2 = t.empty(D_in2, self.LATENT_DIM, dtype=t.float, device=device, requires_grad=True)
        self.W2_2 = t.empty(self.LATENT_DIM, D_out2, dtype=t.float, device=device, requires_grad=True)
        # reg
        self.b1_1 = t.zeros(self.LATENT_DIM, dtype=t.float, device=device, requires_grad=True)
        self.b_share_V = t.zeros(self.LATENT_MID, dtype=t.float, device=device, requires_grad=True)
        self.b_share_W = t.zeros(self.LATENT_DIM, dtype=t.float, device=device, requires_grad=True)
        self.b1_3 = t.zeros(D_out1, dtype=t.float, device=device, requires_grad=True)

        self.b2_1 = t.zeros(self.LATENT_DIM, dtype=t.float, device=device, requires_grad=True)
        self.b2_3 = t.zeros(D_out2, dtype=t.float, device=device, requires_grad=True)
        # discriminator 
        self.fc1 = t.empty(self.LATENT_DIM, 100, dtype=t.float, device=device, requires_grad=True)
        self.fc1_bias = t.empty(100, dtype=t.float, device=device, requires_grad=True)
        self.fc2 = t.empty(100, 2, dtype=t.float, device=device, requires_grad=True)
        self.fc2_bias = t.empty(2, dtype=t.float, device=device, requires_grad=True)
        self.drop = nn.Dropout2d(0.25).to(device)
        self.relu = nn.ReLU(inplace=True).to(device)
        self.batchNorm = nn.BatchNorm1d(100).to(device)
        self.logsoftmax = nn.LogSoftmax(dim=1).to(device)

        nn.init.xavier_uniform_(self.V1)
        nn.init.xavier_uniform_(self.W1_2)
        nn.init.xavier_uniform_(self.W_share)
        nn.init.xavier_uniform_(self.V_share)
        nn.init.xavier_uniform_(self.V2)
        nn.init.xavier_uniform_(self.W2_2)

        nn.init.xavier_uniform_(self.fc1)
        nn.init.xavier_uniform_(self.fc2)
        #         nn.init.xavier_uniform_(self.fc1_bias)
        #         nn.init.xavier_uniform_(self.fc2_bias)

        self.optimizer = t.optim.Adam(
            [self.V1, self.V2, self.V_share, self.W_share, self.W1_2, self.W2_2, self.b1_1, self.b_share_V,
             self.b_share_W, self.b1_3, self.b2_1, self.b2_3, self.fc1, self.fc2, self.fc1_bias, self.fc2_bias],
            lr=self.LR)

        if (t.cuda.is_available()):
            # train mat
            tmpTrainMat1 = self.trainData_T.toarray()
            self.tensorTrainMat_t = t.from_numpy(tmpTrainMat1).float().to(device)
            tmpTrainMask1 = self.trainMask1.toarray()
            self.tensorTrainMask_t = t.from_numpy(1 * tmpTrainMask1).float().to(device)
            # test mat
            tmpTestMat1 = self.testData_T.toarray()
            self.tensorTestMat_t = t.from_numpy(tmpTestMat1).float().to(device)
            tmpTestMask1 = self.testMask1.toarray()
            self.tensorTestMask_t = t.from_numpy(1 * tmpTestMask1).float().to(device)
            # valid mat
            tmpValidMat1 = self.validData_T.toarray()
            self.tensorValidMat_t = t.from_numpy(tmpValidMat1).float().to(device)
            tmpValidMask1 = self.validMask1.toarray()
            self.tensorValidMask_t = t.from_numpy(1 * tmpValidMask1).float().to(device)

            tmpTrainMat2 = self.trainData_S.toarray()
            self.tensorTrainMat_s = t.from_numpy(tmpTrainMat2).float().to(device)
            tmpTrainMask2 = self.trainMask2.toarray()
            self.tensorTrainMask_s = t.from_numpy(1 * tmpTrainMask2).float().to(device)
            # test mat
            tmpTestMat2 = self.testData_S.toarray()
            self.tensorTestMat_s = t.from_numpy(tmpTestMat2).float().to(device)
            tmpTestMask2 = self.testMask2.toarray()
            self.tensorTestMask_s = t.from_numpy(1 * tmpTestMask2).float().to(device)

    def domain_classifier(self, x):

        reverse_feature = grad_reverse(x, 1)
        reverse_feature = t.mm(reverse_feature, self.fc1) + self.fc1_bias
        norm_feature = self.batchNorm(reverse_feature)
        relu_feature = self.relu(norm_feature)
        output = t.mm(relu_feature, self.fc2) + self.fc2_bias
        return self.logsoftmax(output)

    def auto_model(self, train1, train2):
        # train1 is target domain matrix, train2 is source domain matrix
        self.encoder1_1 = t.sigmoid(t.mm(train1, self.V1) + self.b1_1)
        self.encoder1_2 = t.sigmoid(t.mm(self.encoder1_1, self.V_share) + self.b_share_V)
        self.decoder1_1 = t.mm(self.encoder1_2, self.W_share) + self.b_share_W
        self.decoder1_2 = t.mm(self.decoder1_1, self.W1_2) + self.b1_3

        self.encoder2_1 = t.sigmoid(t.mm(train2, self.V2) + self.b2_1)
        self.encoder2_2 = t.sigmoid(t.mm(self.encoder2_1, self.V_share) + self.b_share_V)
        self.decoder2_1 = t.mm(self.encoder2_2, self.W_share) + self.b_share_W
        self.decoder2_2 = t.mm(self.decoder2_1, self.W2_2) + self.b2_3

        self.domain1 = self.domain_classifier(self.decoder1_1)
        self.domain2 = self.domain_classifier(self.decoder2_1)

        return self.decoder1_2, self.decoder2_2, self.encoder1_1, self.encoder2_1, self.encoder1_2, self.encoder2_2, self.decoder1_1, self.decoder2_1, self.domain1, self.domain2

    def trainModel(self, trainMat1, trainMask1, trainMat2, trainMask2, op, share, epoch):

        if (self.MOVIE_BASED):
            num1 = trainMat1.shape[0]
            num2 = trainMat2.shape[0]
            shuffledIds1 = np.random.permutation(num1)
            shuffledIds2 = np.random.permutation(num2)
            steps = int(np.ceil(num1 / self.AUTO_BATCH_SIZE))
            batchSize_S = int(np.ceil(num2 / steps))
        else:
            num1 = trainMat1.shape[0]
            num2 = trainMat2.shape[0]
            shuffledIds1 = np.random.permutation(num1)
            shuffledIds2 = np.random.permutation(num2)
            steps = int(np.ceil(num1 / self.AUTO_BATCH_SIZE))
            batchSize_S = int(np.ceil(num2 / steps))

        epoch_loss = 0
        epoch_rmse_loss = 0
        epoch_rmse_num = 0
        hidden_loss_count = 0
        for i in tqdm(range(steps)):
            if (self.MOVIE_BASED):
                ed_t = min((i + 1) * self.AUTO_BATCH_SIZE, num1)
                ed_s = min((i + 1) * batchSize_S, num2)
                batch_Ids_t = shuffledIds1[i * self.AUTO_BATCH_SIZE:ed_t]
                batch_Ids_s = shuffledIds2[i * batchSize_S:ed_s]
                batch_len_t = len(batch_Ids_t)
                batch_len_s = len(batch_Ids_s)
            else:
                ed_t = min((i + 1) * self.AUTO_BATCH_SIZE, num1)
                ed_s = min((i + 1) * batchSize_S, num2)
                batch_Ids_t = shuffledIds1[i * self.AUTO_BATCH_SIZE:ed_t]
                batch_Ids_s = shuffledIds2[i * batchSize_S:ed_s]
                batch_len_t = len(batch_Ids_t)
                batch_len_s = len(batch_Ids_s)

            train_t = self.tensorTrainMat_t[batch_Ids_t]
            mask_t = self.tensorTrainMask_t[batch_Ids_t]

            train_s = self.tensorTrainMat_s[batch_Ids_s]
            mask_s = self.tensorTrainMask_s[batch_Ids_s]

            ### add some noise
            #             noise_t = t.empty(train_t.shape).uniform_(0, 0.1)
            #             input_t = train_t + noise_t.to(device)
            #             noise_s = t.empty(train_s.shape).uniform_(0, 0.1)
            #             input_s = train_s + noise_s.to(device)

            tmpTrain_t = trainMat1[batch_Ids_t].toarray()
            tmpMask_t = trainMask1[batch_Ids_t].toarray()

            y_pred_t, y_pred_s, hiddenT_1, hiddenS_1, hiddenT_2, hiddenS_2, hiddenT_3, hiddenS_3, domainT, domainS = self.auto_model(
                train_t, train_s)

            pred_loss = self.loss_mse(y_pred_t * mask_t, train_t) / batch_len_t + self.S_weight * self.loss_mse(
                y_pred_s * mask_s, train_s) / batch_len_s

            common_batchSize = min(len(batch_Ids_t), len(batch_Ids_s))

            if (epoch < self.stopTrainEpoch):
                if (self.transfer_method == "MMD"):
                    hidden_loss = self.mmd_loss(hiddenT_1[:common_batchSize], hiddenS_1[:common_batchSize]) * self.BETA
                if (self.transfer_method == "GAN"):
                    hidden_loss = self.GAN_loss(domainS, domainT) * self.BETA
            else:
                hidden_loss = 0
                self.V_share.requires_grad = False
                self.W_share.requires_grad = False
                self.b_share_V.requires_grad = False
                self.b_share_W.requires_grad = False
            v1_loss = t.sum(self.V1 * self.V1)
            vShare_loss = t.sum(self.V_share * self.V_share)
            wShare_loss = t.sum(self.W_share * self.W_share)

            v2_loss = self.S_weight * t.sum(self.V2 * self.V2)
            w1_loss = t.sum(self.W1_2 * self.W1_2)
            w2_loss = self.S_weight * t.sum(self.W2_2 * self.W2_2)
            batch_loss = pred_loss + self.V_regularWeight * (v1_loss + vShare_loss + v2_loss) + self.W_regularWeight * (
                        w1_loss + w2_loss + wShare_loss) + hidden_loss
            hidden_loss_count += hidden_loss
            epoch_loss += batch_loss.item()
            epoch_rmse_loss += self.RMSE(y_pred_t.cpu().detach().numpy(), tmpTrain_t,
                                         tmpMask_t)  # //////////////////②?????/////////////
            epoch_rmse_num += t.sum(mask_t).item()
            op.zero_grad()

            batch_loss.backward()
            op.step()

        epoch_rmse = np.sqrt(epoch_rmse_loss / epoch_rmse_num)
        return epoch_loss / steps, epoch_rmse, hidden_loss_count / steps

    def testMFModel(self, testMat_T):
        if (self.MOVIE_BASED):
            item_ids, user_ids = testMat_T.nonzero()
        else:
            user_ids, item_ids = testMat_T.nonzero()
        p = np.random.permutation(len(item_ids))
        num_ratings = len(p)
        steps = int(np.ceil(num_ratings / self.MF_BATCH_SIZE))
        epoch_test_rmse = 0

        for i in range(steps):
            ed = min((i + 1) * self.MF_BATCH_SIZE, num_ratings)

            batch_user_ids = user_ids[i * self.MF_BATCH_SIZE:ed]
            batch_item_ids = item_ids[i * self.MF_BATCH_SIZE:ed]

            tensor_batch_users = t.LongTensor(batch_user_ids).to(device)
            tensor_batch_items = t.LongTensor(batch_item_ids).to(device)
            if (self.MOVIE_BASED):
                groundTrueRatings = np.array(testMat_T[batch_item_ids, batch_user_ids]).reshape(-1)
            else:
                groundTrueRatings = np.array(testMat_T[batch_user_ids, batch_item_ids]).reshape(-1)

            pred = self.mf_model.predict(tensor_batch_users, tensor_batch_items)

            epoch_test_rmse += self.RMSE_MF(pred.cpu().detach().numpy(), groundTrueRatings)
        return epoch_test_rmse / steps

    def trainMFModel(self, trainMat_T, op):

        # target domain
        if (self.MOVIE_BASED):
            item_ids, user_ids = trainMat_T.nonzero()
        else:
            user_ids, item_ids = trainMat_T.nonzero()
        p = np.random.permutation(len(item_ids))
        num_ratings_T = len(p)
        item_ids, user_ids = item_ids[p], user_ids[p]
        tensor_user = t.LongTensor(user_ids).to(device)
        tensor_item = t.LongTensor(item_ids).to(device)
        steps_T = int(np.ceil(num_ratings_T / self.MF_BATCH_SIZE))
        epoch_rmse_loss_T = 0
        epoch_rmse_loss_S = 0
        epoch_loss_T = 0
        epoch_loss_S = 0
        epoch_transferLoss = 0


        for i in tqdm(range(steps_T)):
            ed_T = min((i + 1) * self.MF_BATCH_SIZE, num_ratings_T)
            #             ed_S = min((i + 1) * BATCH_SIZE_S, num_ratings_S)
            batch_user_tensor = tensor_user[i * self.MF_BATCH_SIZE:ed_T]
            batch_item_tensor = tensor_item[i * self.MF_BATCH_SIZE:ed_T]

            batch_user_ids = user_ids[i * self.MF_BATCH_SIZE:ed_T]
            batch_item_ids = item_ids[i * self.MF_BATCH_SIZE:ed_T]

            batch_len_T = self.MF_BATCH_SIZE
            if (self.MOVIE_BASED):
                groundTrueRatings_T = np.array(trainMat_T[batch_item_ids, batch_user_ids]).reshape(-1)
            else:
                groundTrueRatings_T = np.array(trainMat_T[batch_user_ids, batch_item_ids]).reshape(-1)

            groundTensor_T = t.from_numpy(groundTrueRatings_T).float().to(device)

            pred_t, transferLoss = self.mf_model(batch_user_tensor, batch_item_tensor)
            epoch_transferLoss += transferLoss
            # loss function
            pred_loss = self.loss_mse(groundTensor_T, pred_t)
            batch_loss_T = pred_loss + transferLoss

            #             batch_loss_S = (self.loss_mse(groundTensor_S, pred_s))
            epoch_loss_T = batch_loss_T

            #             epoch_loss_S = batch_loss_S
            epoch_rmse_loss_T += self.RMSE_MF(pred_t.cpu().detach().numpy(), groundTrueRatings_T)
            #             epoch_rmse_loss_S += self.RMSE_MF(pred_s.detach().numpy(), groundTrueRatings_S)
            batch_loss = batch_loss_T

            op.zero_grad()
            batch_loss_T.backward()
            op.step()
        #         print(f"transfer Loss: {epoch_transferLoss / steps_T}")
        epoch_rmseT = epoch_rmse_loss_T / steps_T
        #         epoch_rmseS = epoch_rmse_loss_S / steps_T
        #         epoch_lossS = epoch_loss_S
        epoch_lossT = epoch_loss_T
        return epoch_rmseT, epoch_lossT

    def run_MF(self, trainData_T, testData_T, validData_T, beta_mf):
        self.MF_test_rmse = []
        self.MF_valid_rmse = []
        self.beta_mf = beta_mf
        num_items, num_users = self.trainData_T.shape
        # load source model
        source_model = t.load("model/{}.pkl".format(self.sor_name))
        print("source model loaded!")

        #         print(f"source similarity matrix loaded!")
        # load item similar mat
        if (self.MOVIE_BASED):
            with open("model/" + self.dataset + "item_similarMat.pickle", "rb") as handle:
                item_similarmat = pickle.load(handle)
                user_similarmat = None
        else:
            with open("model/" + self.dataset + "user_similarMat.pickle", "rb") as handle:
                user_similarmat = pickle.load(handle)
                item_similarmat = None
        ### normal mf
        self.mf_model = MFModel(num_users, num_items, source_model.item_embedding, source_model.user_embedding,
                                item_similarmat, user_similarmat, beta_mf, random_emd=self.is_random_emd,
                                movie_based=self.MOVIE_BASED)
        self.optimizer_MF = t.optim.SGD(self.mf_model.parameters(), lr=self.MF_LR, weight_decay=1e-5)
        for e in range(self.MF_EPOCH):
            epoch_rmseT, epoch_lossT = self.trainMFModel(trainData_T, self.optimizer_MF)
            print("epoch %d/%d, epoch_rmseT=%.4f, epoch_LossT = %.4f" % (e, self.MF_EPOCH, epoch_rmseT, epoch_lossT))
            epoch_rmse_test = self.testMFModel(testData_T)
            epoch_rmse_valid = self.testMFModel(validData_T)
            print("epoch %d/%d, epoch_valid_rmse = %.4f, epoch_test_rmse=%.4f" % (
            e, self.MF_EPOCH, epoch_rmse_valid, epoch_rmse_test))
            self.MF_test_rmse.append(epoch_rmse_test)
            self.MF_valid_rmse.append(epoch_rmse_valid)

    def run_GANMF(self, trainData_T, testData_T):
        self.MF_test_rmse = []
        num_items, num_users = trainData_T.shape
        # load source model
        source_model = t.load("model/mfGAN.pkl")
        print("source model loaded!")

        with open("model/" + self.dataset + "item_similarMat.pickle", "rb") as handle:
            item_similarmat = pickle.load(handle)
        user_similarmat = None
        self.mf_model = MFModel(num_users, num_items, source_model.item_embedding_s, source_model.user_embedding_s,
                                source_model.item_embedding_t, source_model.user_embedding_t,
                                source_model.user_biases_t, source_model.item_biases_t, item_similarmat,
                                user_similarmat)
        self.optimizer_MF = t.optim.SGD(self.mf_model.parameters(), lr=self.MF_LR, weight_decay=1e-5)
        for e in range(self.MF_EPOCH):
            epoch_rmseT, epoch_lossT = self.trainMFModel(trainData_T, self.optimizer_MF)
            print("epoch %d/%d, epoch_rmseT=%.4f, epoch_LossT = %.4f" % (e, self.MF_EPOCH, epoch_rmseT, epoch_lossT))
            epoch_rmse_test = self.testMFModel(testData_T)
            print("epoch %d/%d, epoch_test_rmse=%.4f" % (e, self.MF_EPOCH, epoch_rmse_test))
            self.MF_test_rmse.append(epoch_rmse_test)

    def RMSE_MF(self, pred, groudTrue):
        return np.sqrt(((pred - groudTrue) ** 2).mean())

    def testModel(self, trainMat_T, testMat_T, testMask_T, trainMat_S, testMat_S, testMask_S):
        if (self.MOVIE_BASED):
            num1 = trainMat_T.shape[0]
            num2 = trainMat_S.shape[0]
            shuffledIds1 = np.random.permutation(num1)
            shuffledIds2 = np.random.permutation(num2)
            steps = int(np.ceil(num1 / self.AUTO_BATCH_SIZE))
        else:
            num1 = trainMat_T.shape[0]
            num2 = trainMat_S.shape[0]
            shuffledIds1 = np.random.permutation(num1)
            shuffledIds2 = np.random.permutation(num2)
            steps = int(np.ceil(num1 / self.AUTO_BATCH_SIZE))
            batchSize_S = int(np.ceil(num2 / steps))

        epoch_loss_t = 0
        epoch_rmse_loss_t = 0
        epoch_loss_s = 0
        epoch_rmse_loss_s = 0
        epoch_rmse_num_s = 0
        epoch_rmse_num_t = 0
        for i in range(steps):
            if (self.MOVIE_BASED):
                ed_t = min((i + 1) * self.AUTO_BATCH_SIZE, num1)
                ed_s = min((i + 1) * self.AUTO_BATCH_SIZE, num2)
                batch_Ids_t = shuffledIds1[i * self.AUTO_BATCH_SIZE:ed_t]
                batch_Ids_s = shuffledIds2[i * self.AUTO_BATCH_SIZE:ed_s]

            else:
                ed_t = min((i + 1) * self.AUTO_BATCH_SIZE, num1)
                ed_s = min((i + 1) * batchSize_S, num2)
                batch_Ids_t = shuffledIds1[i * self.AUTO_BATCH_SIZE:ed_t]
                batch_Ids_s = shuffledIds2[i * batchSize_S:ed_s]

            train_t = self.tensorTrainMat_t[batch_Ids_t]
            test_t = self.tensorTestMat_t[batch_Ids_t]
            mask_t = self.tensorTestMask_t[batch_Ids_t]  # bool转换为int

            train_s = self.tensorTrainMat_s[batch_Ids_s]
            test_s = self.tensorTestMat_s[batch_Ids_s]
            mask_s = self.tensorTestMask_s[batch_Ids_s]

            tmpTest_t = testMat_T[batch_Ids_t].toarray()
            tmptestMask_t = testMask_T[batch_Ids_t].toarray()

            tmpTest_s = testMat_S[batch_Ids_s].toarray()
            tmptestMask_s = testMask_S[batch_Ids_s].toarray()

            y_pred_t, y_pred_s, _, _, _, _, _, _, _, _ = self.auto_model(train_t, train_s)

            epoch_rmse_loss_t += self.RMSE(y_pred_t.cpu().detach().numpy(), tmpTest_t, tmptestMask_t)
            epoch_loss_t += epoch_rmse_loss_t  # /////////////???/////////////
            epoch_rmse_num_t += t.sum(mask_t).item()

            epoch_rmse_loss_s += self.RMSE(y_pred_s.cpu().detach().numpy(), tmpTest_s, tmptestMask_s)
            epoch_loss_s += epoch_rmse_loss_s  # /////////////???/////////////
            epoch_rmse_num_s += t.sum(mask_s).item()
        epoch_rmse_s = np.sqrt(epoch_rmse_loss_s / epoch_rmse_num_s)
        epoch_rmse_t = np.sqrt(epoch_rmse_loss_t / epoch_rmse_num_t)
        return epoch_loss_t, epoch_rmse_t, epoch_loss_s, epoch_rmse_s

    def run(self):
        self.prePareModel()

        self.share = True
        for e in range(self.curEpoch, self.AUTO_EPOCH):
            epoch_loss, epoch_rmse, transferLoss = self.trainModel(self.trainData_T, self.trainMask1, self.trainData_S,
                                                                   self.trainMask2, self.optimizer, self.share, e)

            self.train_losses.append(epoch_loss)
            self.train_RMSEs.append(epoch_rmse)
            epoch_loss_t, epoch_rmse_t, epoch_loss_s, epoch_rmse_s = self.testModel(self.trainData_T, self.testData_T,
                                                                                    self.testMask1, self.trainData_S,
                                                                                    self.testData_S, self.testMask2)
            print(f"transfer loss: {transferLoss}")
            print("epoch %d/%d, epoch_loss=%.2f, epoch_rmse=%.4f" % (e, self.AUTO_EPOCH, epoch_loss, epoch_rmse))
            print(
                "epoch %d/%d, cv_epoch_loss_target =%.2f,cv_epoch_rmse_target=%.4f, cv_epoch_loss_source =%.2f,cv_epoch_rmse_source=%.4f" % (
                e, self.AUTO_EPOCH, epoch_loss_t, epoch_rmse_t, epoch_loss_s, epoch_rmse_s))
            # calculate average distance    

            print("calculate average distance at layer 3 ...........")
            layer2_sim = self.get_similar_ids_from_sources(10, None, 3)
            _, average_distance = getNum_threshold(10, layer2_sim)
            print(f"average distance: {average_distance}\n")
            self.average_distance.append(average_distance)
            print("\n")
            #             self.test_losses.append(cv_epoch_loss)
            #             self.test_RMSEs.append(epoch_rmse_t)

            self.curLr = self.adjust_learning_rate(self.optimizer, e)
            self.curEpoch = e
            # if e%10==0 and e!=0:
            #     self.saveModel()
            #     test_epoch_loss=self.testModel(self.trainMat,self.testMat,self.testMask,1)
            #     log("epoch %d/%d, test_epoch_loss=%.2f"%(e, EPOCH, test_epoch_loss))
            #     self.step_losses.append(test_epoch_loss)
            #     # for i in range(len(self.step_losses)):
            #     #     print("***************************")
            #     #     print("rmse = %.4f"%(self.step_rmse[i]))
            #     #     print("***************************")
        #         _,test_rmse=self.testModel(self.trainMat,self.testMat,self.testMask,1)
        #         self.writeResult(test_rmse)
        print("\n")
        #         print("test_rmse=%.4f"%(test_rmse))
        #         self.recordResult(self.test_RMSEs)
        self.getModelName()

    # get top N feature to renew similarity matrix
    def getTop_N_feature_mat(self, target, source, top_N_features):
        if (top_N_features):
            distance_mat = np.abs(target - source)
            disArgsort = distance_mat.mean(axis=0).argsort()
            new_target_hidden = target[:, disArgsort[:top_N_features]]
            new_source_hidden = source[:, disArgsort[:top_N_features]]
            sim_mat = 1 - cdist(new_target_hidden, new_source_hidden, metric="cosine")
            return sim_mat
        else:
            print(f"None feature selection")
            return 1 - cdist(target, source, metric="cosine")

    # get top-N similar(non-similar) item ids in source domain
    def get_similar_ids_from_sources(self, top_n, top_n_features, layer, save=False):
        _, _, self.hiddenTaget_1, self.hiddenSource_1, self.hiddenMid_Target, self.hiddenMid_Source, self.hiddenTarget_2, self.hiddenSource_2, _, _ = self.auto_model(
            self.tensorTrainMat_t, self.tensorTrainMat_s)
        npTar_1, npTar_mid, npTar_2 = self.hiddenTaget_1.cpu().detach().numpy(), self.hiddenMid_Target.cpu().detach().numpy(), self.hiddenTarget_2.cpu().detach().numpy()
        npSource_1, npSource_mid, npSource_2 = self.hiddenSource_1.cpu().detach().numpy(), self.hiddenMid_Source.cpu().detach().numpy(), self.hiddenSource_2.cpu().detach().numpy()
        if (layer == 1):
            self.similarity = self.getTop_N_feature_mat(npTar_1, npSource_1, top_n_features)
            sort_similarity = np.flip(np.sort(self.similarity, axis=1), axis=1)
            self.topNSim = sort_similarity[:, :top_n]
            self.targetSimilarMatArg = np.flip(self.similarity.argsort(axis=1), axis=1)[:, :top_n]

        elif (layer == 2):
            self.similarity = self.getTop_N_feature_mat(npTar_mid, npSource_mid, top_n_features)
            sort_similarity = np.flip(np.sort(self.similarity, axis=1), axis=1)
            self.topNSim = sort_similarity[:, :top_n]
            self.targetSimilarMatArg = np.flip(self.similarity.argsort(axis=1), axis=1)[:, :top_n]

        elif (layer == 3):
            self.similarity = self.getTop_N_feature_mat(npTar_2, npSource_2, top_n_features)
            sort_similarity = np.flip(np.sort(self.similarity, axis=1), axis=1)
            self.topNSim = sort_similarity[:, :top_n]
            self.targetSimilarMatArg = np.flip(self.similarity.argsort(axis=1), axis=1)[:, :top_n]
        if (save):
            if (self.MOVIE_BASED):
                print(f"save items similarity info!")
                np.save("similarity_matrix/item_similarArg.npy", self.targetSimilarMatArg)
                np.save("similarity_matrix/item_similarity.npy", self.topNSim)
            else:
                print(f"save users similarity info!")
                np.save("similarity_matrix/user_similarArg.npy", self.targetSimilarMatArg)
                np.save("similarity_matrix/user_similarity.npy", self.topNSim)
        return self.similarity

    def save_similar_mat(self):
        similar_mat = self.get_similar_ids_from_sources(None, None, 3)
        if (self.MOVIE_BASED):
            self.simMatName = self.dataset + "item_simMat.pickle"
        else:
            self.simMatName = self.dataset + "user_simMat.pickle"
        with open('similarity_matrix/' + self.simMatName, 'wb') as handle:
            pickle.dump(similar_mat, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def construct_similar_items_mat(self, threshold=0.9999, oneItem=False):
        print(f"construct similart mat between target and source")
        if (self.MOVIE_BASED):
            with open('similarity_matrix/' + self.dataset + "item_simMat.pickle", 'rb') as handle:
                similar_mat = pickle.load(handle)
        else:
            with open('similarity_matrix/' + self.dataset + "user_simMat.pickle", 'rb') as handle:
                similar_mat = pickle.load(handle)

        if (oneItem):
            print("only one item from source domain")
            argmaxIdex = similar_mat.argmax(axis=1)
            self.similarMat = dict(zip(range(len(argmaxIdex)), argmaxIdex))
            source_number = len(self.similarMat.values())
        else:
            self.similarMat = {}
            mask = (similar_mat > threshold)
            source_number = np.sum(mask)
            print(source_number)
            for i in range(mask.shape[0]):
                if (np.sum(mask[i] != 0)):
                    self.similarMat[i] = list(mask[i].nonzero()[0])
        if (self.MOVIE_BASED):
            self.name = self.dataset + "item_similarMat.pickle"
        else:
            self.name = self.dataset + "user_similarMat.pickle"

        with open('model/' + self.name, 'wb') as handle:
            print(f"save similar mat!!: {self.name}")
            pickle.dump(self.similarMat, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return source_number

    def adjust_learning_rate(self, optimizer, epoch):
        LR = self.LR * (self.LR_DECAY ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = LR
        return LR

    def saveModel(self):
        history = dict()
        history['losses'] = self.train_losses
        history['val_losses'] = self.test_losses  # 验证
        ModelName = self.getModelName()

        savePath = r'./' + self.dataset + r'/Model/' + ModelName + r'.pth'
        # t.save({
        #     'epoch': self.curEpoch,
        #     'lr': self.learning_rate,
        #     'decay': self.decay,
        #     'V' : self.V,
        #     'W' : self.W,
        #     'b1': self.b1,
        #     'b2' :self.b2,
        #     'v_weight':self.V_regularWeight,
        #     'w_weight':self.W_regularWeight,
        #     'history': history
        #     }, savePath)
        print("save model : " + ModelName)

        with open('./' + self.dataset + r'/History/' + ModelName + '.his', 'wb') as fs:
            pickle.dump(history, fs)

    def getModelName(self):
        ModelName = "TR" + self.dataset + self.modelUTCStr + \
                    "_S-Weight_" + str(self.S_weight) + \
                    "_CV" + str("test") + \
                    "_V-Weight_" + str(self.V_regularWeight) + \
                    "_W-Weight_" + str(self.W_regularWeight) + \
                    "_BATCH_" + str(self.AUTO_BATCH_SIZE) + \
                    "_LATENT_DIM" + str(self.LATENT_DIM) + \
                    "_LR" + str(self.LR) + \
                    "_LR_DACAY" + str(self.LR_DECAY)
        return ModelName

    def getMFName(self):
        ModelName = "TRMF" + self.dataset + \
                    "_MF_LR_" + str(self.MF_LR) + \
                    "_MF_BATCH_SIZE_" + str(self.MF_BATCH_SIZE)
        return ModelName

    def writeResult(self, result):
        with open(resultFile, mode='a') as f:
            modelName = self.getModelName()
            f.write('\r\n')
            f.write(dataset + '\r\n')
            f.write(modelName + '\r\n')
            f.write(str(result) + '\r\n')

    def RMSE(self, decoder, label, mask):
        add_avg = decoder + 3 - 3 * (decoder != 0)
        return np.sum(np.square((add_avg - label) * mask))

    def recordResult(self, result, name):
        path = "results"
        file = os.path.join(path, name + ".pkl")
        with open(file, "wb") as f:
            pickle.dump(result, f)
            print(f"__save__results as : {file}")

    def saveMF_results(self):
        self.recordResult(self.MF_test_rmse, self.getMFName())

    ###### Plot ########
    def plot_sim_distrib(self, save=False):
        from collections import Counter
        import matplotlib.pyplot as plt
        layer2_sim = self.get_similar_ids_from_sources(None, None, 3)
        tmpNp = np.round(layer2_sim, decimals=2).flatten()
        counter = Counter(tmpNp)
        dictCounter = dict(counter)
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.bar(x=list(dictCounter.keys()), height=list(dictCounter.values()), width=0.008)
        ax.set_title("layer 3 similarity distribution", fontsize=23)
        ax.set_xlabel("cousine similarity", fontsize=13)
        ax.set_ylabel("frequency", fontsize=13)
        if (save):
            name = "plot/" + self.getModelName + "SimDis.png"
            plt.savefig(name)
            print(f"save similarity distribution plot as :{name}")
        plt.show()

    def plot_MF(self, save=False):
        baseline_name = "MF_records/{}.pkl".format(self.tar_name)
        with open(baseline_name, "rb") as handle:
            baseline = pickle.load(handle)
        transfer_results = self.MF_test_rmse
        x = list(range(len(transfer_results)))

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(x, transfer_results, label="transfer_model")
        ax.plot(x, baseline, label="baseline")

        ax.set_title(self.dataset, fontsize=18)
        ax.set_xlabel("Epochs", fontsize="x-large", fontfamily="sans-serif")
        ax.set_ylabel("RMSE", fontsize="x-large", fontfamily='sans-serif')
        ax.xaxis.set_tick_params(rotation=45, labelsize=12, colors='black')
        ax.yaxis.set_tick_params(labelsize=12)
        ax.set_ylim([0.9, 1])
        ax.legend()
        ax.grid()
        if (save):
            name = "plot/" + self.dataset + ".png"
            plt.savefig(name)
            print(f"save plot as : {name}")
        plt.show()


####### gradient reverse model        
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd):
    return GradReverse.apply(x, lambd)

# class Domain_classifier(nn.Module):
#     def __init__(self):
#         super(Domain_classifier,self).__init__()
#         self.fc1 = nn.Linear(500, 100).to(device)
#         self.fc2 = nn.Linear(100, 1).to(device)
#         self.drop = nn.Dropout2d(0.25).to(device)

#     def forward(self, x):
#         x = grad_reverse(x, 0.1)
#         x = F.leaky_relu(self.drop(self.fc1(x)))
#         x = self.fc2(x)
#         return F.sigmoid(x)
