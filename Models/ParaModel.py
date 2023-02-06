# deepset
# predict on the parameters

import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
from tqdm import tqdm
import copy
from copy import  deepcopy



class KKTSoft(object):
    def __init__(self, dataset, base_model):
        self.dataset = dataset
        self.model = base_model

    def KKT_loss_soft(self, X_train, train_lab, indx, C, paras, para_true = None):
        '''

        :param X_train:
        :param train_lab: if not binary, should be one_hot encoded
        :param indx:
        :param C:
        :param paras: class_num*(feature_dim + 1)
        :return:
        '''
        if self.model == 'Logistic Regression' and self.dataset == 'MNIST':
            # pred_W = paras.reshape((-1, 10, 785))[:,:, :-1]
            # pred_b = torch.squeeze(paras.reshape((-1, 10, 785))[:, :, -1])
            pred_W = paras.reshape((-1, 10, 129))[:,:, :-1]
            pred_b = torch.squeeze(paras.reshape((-1, 10, 129))[:, :, -1])
            gradient_W = torch.tensor(np.zeros((X_train.shape[0], 10, X_train.shape[-1]))).cuda()
            gradient_b = torch.tensor(np.zeros((X_train.shape[0], 10))).cuda()
            # train_lab_enc = nn.functional.one_hot(deepcopy(train_lab).to(torch.int64), num_classes = 10)
            for i in range(X_train.shape[0]):
                X_use = X_train[i, :indx[i].type(torch.IntTensor), :]
                y_use = train_lab[i, :indx[i].type(torch.IntTensor), :]
                # start = time.time()
                gradient_W[i,:,:] =pred_W[i] + C*torch.sum( #pred_W[i] +
                    torch.stack([torch.kron(torch.reshape((-1)*(y_use[j]-nn.functional.softmax(torch.matmul(X_use[j],
                                  torch.transpose(torch.squeeze(pred_W[i,:,:]), 0, 1))+pred_b[i],
                                  dim=0)), (-1,1)) , torch.reshape(X_use[j], (1, -1)))
                                  for j in range(X_use.shape[0])], dim=0),dim=0) #torch.tensor(1)-

                gradient_b[i, :] = C * torch.sum(
                    torch.stack([(-1)*(y_use[j]-nn.functional.softmax(
                        torch.matmul(X_use[j], torch.transpose(torch.squeeze(pred_W[i, :, :]), 0, 1)) + pred_b[i],
                        dim=0)) for j in range(X_use.shape[0])], dim=0
                                ) ,dim=0)


        elif self.model == 'Logistic Regression' and (self.dataset == 'IRIS'
                                                      or self.dataset == 'SPAM'
                                                    or self.dataset == 'HIGGS'):
            y_train = copy.deepcopy(train_lab)
            w = paras[:, :-1]
            b = paras[:, -1]

            gradient_W = torch.tensor(np.zeros((X_train.shape[0], X_train.shape[-1]))).cuda()
            gradient_b = torch.tensor(np.zeros((X_train.shape[0], 1))).cuda()

            for i in range(X_train.shape[0]):
                # X_train use selected part
                X_use = X_train[i, :indx[i].type(torch.IntTensor), :]
                y_use = y_train[i, :indx[i].type(torch.IntTensor)]

                gradient_W[i, :] = w[i, :] + C * torch.sum(
                    torch.stack(
                        [(1 / (1 + torch.exp(-1 * (torch.dot(X_use[j, :], w[i, :]) + b[i]))) - y_train[i, j]) * (
                            X_use[j, :]) for j in range(X_use.shape[0])]), dim=0)
                gradient_b[i, :] = C * torch.sum(torch.stack(
                    [(1 / (1 + torch.exp(-1 * (torch.dot(X_use[j, :], w[i, :]) + b[i]))) - y_train[i, j]) for
                     j in range(X_use.shape[0])]), dim=0)


        elif self.model =='SVM':
            if self.dataset == 'IRIS' or self.dataset == 'SPAM' or self.dataset == 'HIGGS':
                y_train = copy.deepcopy(train_lab)
                w = paras[:, :-1]
                b = paras[:, -1].view(-1,1)
                gradient_W = torch.tensor(np.zeros((X_train.shape[0], X_train.shape[-1]))).cuda()
                gradient_b = torch.tensor(np.zeros((X_train.shape[0], 1))).cuda()

                for i in range(X_train.shape[0]):
                    X_use = X_train[i, :indx[i].type(torch.IntTensor), :]
                    y_use = y_train[i, :indx[i].type(torch.IntTensor)]
                    gradient_W[i, :] = w[i, :] - C*torch.sum(
                    torch.stack([y_use[j]* X_use[j, :]*torch.tensor(2.).cuda()*torch.max(torch.tensor(0.).cuda(), torch.tensor(torch.tensor(1.).cuda()-y_use[j]*(torch.dot(X_use[j, :], w[i, :])+b[i])))
                                 if torch.tensor(1.).cuda()-y_use[j]*(torch.dot(X_use[j, :], w[i, :])+b[i])>0
                         else torch.zeros_like(w[i, :]) for j in range(X_use.shape[0])]), dim=0)
                    gradient_b[i, :] = b[i] - C *torch.sum(
                    torch.stack([y_use[j]*torch.tensor(2.).cuda()*torch.max(torch.tensor(0.).cuda(),
                                                                            torch.tensor(torch.tensor(1.).cuda()-y_use[j]*(torch.dot(X_use[j, :], w[i, :])+b[i])))
                         if 1-y_use[j]*(torch.dot(X_use[j, :], w[i, :])+b[i])>0
                         else torch.zeros_like(b[i]) for j in range(X_use.shape[0])]), dim=0)

        loss_b = torch.linalg.norm(gradient_b)
        loss_w = torch.linalg.norm(gradient_W)

        return loss_b + loss_w


    def utility_loss_soft(self, X_train, train_lab, indx, C, paras, true_para):
        # calculate the utility loss of predicted para and ture para (training loss)
        # here y is the one hot encoding
        if self.model == 'Logistic Regression' and self.dataset == 'MNIST':
            # pred_W = paras.reshape((-1, 10, 785))[:,:, :-1]
            # pred_b = torch.squeeze(paras.reshape((-1, 10, 785))[:, :, -1])
            pred_W = paras.reshape((-1, 10, 129))[:,:, :-1]
            pred_b = torch.squeeze(paras.reshape((-1, 10, 129))[:, :, -1])
            loss_pred = torch.tensor(np.zeros(X_train.shape[0])).cuda()
            for i in range(X_train.shape[0]):
                X_use = X_train[i, :indx[i].type(torch.IntTensor), :]
                y_use = train_lab[i, :indx[i].type(torch.IntTensor), :]
                # start = time.time()
                y_pred_prob = nn.functional.softmax(torch.matmul(X_use, torch.transpose(torch.squeeze(pred_W[i, :, :]),0,1))
                                                                 + pred_b[i], dim=1)
                loss_pred[i] = torch.mean(-torch.sum(y_use * torch.log(y_pred_prob))) #+ C * 1 / 2 * torch.norm(torch.squeeze(pred_W[i, :, :])) ** 2

            # train_W = true_para.reshape((-1, 10, 785))[:,:, :-1]
            # train_b = torch.squeeze(true_para.reshape((-1, 10, 785))[:, :, -1])
            train_W = true_para.reshape((-1, 10, 129))[:,:, :-1]
            train_b = torch.squeeze(true_para.reshape((-1, 10, 129))[:, :, -1])
            loss_train = torch.tensor(np.zeros(X_train.shape[0])).cuda()
            for i in range(X_train.shape[0]):
                X_use = X_train[i, :indx[i].type(torch.IntTensor), :]
                y_use = train_lab[i, :indx[i].type(torch.IntTensor), :]
                # start = time.time()
                y_train_prob = nn.functional.softmax(torch.matmul(X_use, torch.transpose(torch.squeeze(train_W[i, :, :]),0,1))
                                                                 + train_b[i], dim=1)
                loss_train[i] = torch.mean(-torch.sum(y_use * torch.log(y_train_prob))) #+  C * 1 / 2 * torch.norm(torch.squeeze(train_W[i, :, :])) ** 2

            loss_utility = torch.linalg.norm(loss_train-loss_pred)
            return loss_utility

        elif self.model == 'Logistic Regression' and (self.dataset == 'IRIS' or
                                                      self.dataset == 'SPAM'or self.dataset == 'HIGGS'):
            # here y is the class {0, 1}
            pred_W = paras[:, :-1]
            pred_b = torch.squeeze(paras[:, -1])
            loss_pred = torch.tensor(np.zeros(X_train.shape[0])).cuda()
            for i in range(X_train.shape[0]):
                X_use = X_train[i, :indx[i].type(torch.IntTensor), :]
                y_use = torch.squeeze(train_lab[i, :indx[i].type(torch.IntTensor)])
                y_pred_prob =  torch.clamp(torch.sigmoid(torch.matmul(X_use, pred_W[i, :]) + pred_b[i]), min=1e-5, max = 0.999999)
                loss_pred[i] = torch.mean(-1 * (torch.mul(y_use, torch.log(y_pred_prob)) + torch.mul(1 - y_use, torch.log(
                    1 - y_pred_prob))))  #+  C * 1 / 2 * torch.norm(torch.squeeze(pred_W[i, :])) ** 2

            train_W = true_para[:, :-1]
            train_b = torch.squeeze(true_para[:, -1])
            loss_train = torch.tensor(np.zeros(X_train.shape[0])).cuda()
            for i in range(X_train.shape[0]):
                X_use = X_train[i, :indx[i].type(torch.IntTensor), :]
                y_use = train_lab[i, :indx[i].type(torch.IntTensor)]
                y_pred_prob = torch.clamp(torch.sigmoid(torch.matmul(X_use, train_W[i, :]) + train_b[i]), min=1e-5, max = 0.999999)
                loss_train[i] = torch.mean(
                    -1 * (torch.multiply(y_use, torch.log(y_pred_prob)) + torch.mul(1 - y_use, torch.log(
                        1 - y_pred_prob))))  #+  C * 1 / 2 * torch.norm(torch.squeeze(train_W[i, :])) ** 2

            loss_utility = torch.linalg.norm(loss_train-loss_pred)
            return loss_utility

        elif self.model == 'SVM' and (self.dataset == 'IRIS' or
                                                      self.dataset == 'SPAM' or self.dataset == 'HIGGS'):

            pred_W = paras[:, :-1]
            pred_b = torch.squeeze(paras[:, -1])
            loss_pred = torch.tensor(np.zeros(X_train.shape[0])).cuda()
            for i in range(X_train.shape[0]):
                X_use = X_train[i, :indx[i].type(torch.IntTensor), :]
                y_use = torch.squeeze(train_lab[i, :indx[i].type(torch.IntTensor)])
                y_hat = torch.matmul(X_use, pred_W[i, :]) + pred_b[i]
                loss_pred[i] = torch.mean(torch.max(torch.tensor(0.).cuda(), torch.tensor(1.).cuda()-y_use*y_hat)**2, dim=0)

            train_W = true_para[:, :-1]
            train_b = torch.squeeze(true_para[:, -1])
            loss_train = torch.tensor(np.zeros(X_train.shape[0])).cuda()
            for i in range(X_train.shape[0]):
                X_use = X_train[i, :indx[i].type(torch.IntTensor), :]
                y_use = torch.squeeze(train_lab[i, :indx[i].type(torch.IntTensor)])
                y_hat = torch.matmul(X_use, train_W[i, :]) + train_b[i]
                loss_train[i] = torch.mean(torch.max(torch.tensor(0.).cuda(), torch.tensor(1.).cuda()-y_use*y_hat)**2, dim=0)
            loss_utility = torch.linalg.norm(loss_train-loss_pred)
            return loss_utility


class DeepSet(nn.Module):
    def __init__(self, in_features, in_lab, out_dim, hyper_config
                 ):
        '''
        in_labels should be one_hot encoded
        out_dim: feature dimension
        '''
        super(DeepSet, self).__init__()

        self.in_features = in_features
        self.in_response = in_lab
        self.out_features = hyper_config['set_features']
        self.hidden_ext = hyper_config['hidden_ext']
        self.hidden_reg = hyper_config['hidden_reg']
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features + in_lab, self.hidden_ext, bias=False),
            nn.ELU(inplace=True),
            nn.Linear(self.hidden_ext, self.hidden_ext, bias=False),
            nn.ELU(inplace=True),
            nn.Linear(self.hidden_ext, self.out_features, bias=False)
        )

        self.regressor = nn.Sequential(
            nn.Linear(self.out_features, self.hidden_reg, bias=False),
            # nn.BatchNorm1d(num_features=self.hidden_reg),
            nn.ELU(inplace=True),
            nn.Linear(self.hidden_reg, self.hidden_reg, bias=False),
            nn.ELU(inplace=True),
            nn.Linear(self.hidden_reg, int(self.hidden_reg / 2), bias=False),
            nn.ELU(inplace=True)
        )

        self.linear = nn.Linear(int(self.hidden_reg / 2), out_dim)
        # self.sigmoid = nn.Sigmoid()

        self.add_module('0', self.feature_extractor)
        self.add_module('1', self.regressor)

    def forward(self, input_x, input_y):
        x = input_x
        y = input_y
        merge = torch.cat((x, y), -1)
        z = self.feature_extractor(merge)
        z = z.sum(dim=1)
        z = self.regressor(z)
        z = self.linear(z)
        # z = self.sigmoid(z)
        return z



class Para_KKT_deepset(object):
    def __init__(self, in_dims, in_lab, out_dim, deep_config, configs, model=None, bias=False):

        if model is None:
            self.model = DeepSet(in_dims, in_lab, out_dim, deep_config)

        else:
            self.model = model

        self.lab_size = in_lab

        self.model = self.model.cuda()

        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss(reduction='mean') #sum
        self.lr = configs['learning_rate']
        self.dataset = configs['dataset']
        self.save_path = configs['model_path']
        self.utility_loss = configs['util_loss']
        self.optim = optim.Adam(self.model.parameters(), self.lr )
        # self.optim = optim.AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01,
        #                          amsgrad=False)
        self.KKT_soft = KKTSoft(self.dataset, configs['base_model'])
        self.time = 0

    def fit(self, train_data, train_lab, train_set, valid_set, C, n_epoch,
            l1_lambda=0.01, l2_lambda=0.01,kkt_lambda = 1, Tol=1e-3, batch_size=32):

        train_data = copy.deepcopy(train_data)
        train_lab = copy.deepcopy(train_lab)
        # kkt_lambda = 1/C * 10
        # kkt_lambda = 1 / C
        if self.lab_size > 2:
            train_lab = torch.nn.functional.one_hot(train_lab, self.lab_size)

        N = train_data.shape[0]  # largest training sample size (for permutation)
        k = train_data.shape[1]  # dimension of the features

        X_feature, y_feature = train_set # y feature is the utility (estimates of parameters)
        X_feature_test, y_feature_test = valid_set  # set for Shapley Calculation of testing

        train_size = len(y_feature)  # number of permutations to be trained

        Train_loss, Valid_loss = [], []
        C = torch.tensor(C).cuda()

        self.model.train()

        for epoch in range(n_epoch):
            # Shuffle training utility samples
            ind = np.arange(train_size, dtype=int)
            np.random.shuffle(ind)
            X_feature = [X_feature[i] for i in ind]  # index for perMaM<mutation index
            # y_feature_1 = [y_feature[i] for i in ind]
            y_feature = y_feature[ind]
            # grad_all = [train_grad[i] for i in ind]

            train_loss = 0
            start_ind = 0

            for j in tqdm(range(train_size // batch_size)):
            # for j in range(train_size // batch_size):
                start = time.time()
                start_ind = j * batch_size
                batch_X, batch_lab, batch_y, batch_grad = [], [], [], []  # here y is the utility
                batch_Idx, batch_Idx2 = [], []

                for i in range(start_ind, min(start_ind + batch_size, train_size)):
                    b = np.zeros((N, k))
                    labs = np.zeros((N, self.lab_size))
                    grads = np.zeros((N, k+1))
                    sel_index = np.zeros(N, dtype=bool)
                    sel_index[np.where(X_feature[i]==1)[0]] = True
                    selected_train_data = train_data[sel_index]
                    selected_train_lab = train_lab[sel_index]
                    b[:selected_train_data.shape[0]] = selected_train_data
                    if len(selected_train_lab.shape)==1:
                        selected_train_lab = selected_train_lab.reshape(-1, 1)

                    labs[:selected_train_data.shape[0], :] = selected_train_lab
                    # grads[:selected_train_data.shape[0]] = selected_train_grad

                    index_len = np.where(sel_index.astype(int)==1)[0].shape[0]
                    # index = np.zeros(N)
                    # index[:index_len] = 1
                    # batch_Idx.append(index)
                    batch_Idx.append(index_len)
                    batch_X.append(b)
                    batch_lab.append(labs)
                    # batch_grad.append(grads)

                    batch_y.append(y_feature[i])

                batch_X = np.stack(batch_X)
                batch_lab = np.stack(batch_lab)
                batch_Idx = np.stack(batch_Idx)
                # batch_grad= np.stack(batch_grad)
                batch_X, batch_lab, batch_y = torch.FloatTensor(batch_X).cuda(), torch.FloatTensor(
                    batch_lab).cuda(), torch.FloatTensor(batch_y).cuda()
                batch_Idx = torch.Tensor(batch_Idx).cuda()
                # batch_Idx2 = torch.Tensor(batch_Idx2).cuda()

                self.optim.zero_grad()
                y_pred = self.model(batch_X, batch_lab)

                # import matplotlib.pyplot as plt
                # plt.hist(y_pred[0].cpu().detach().numpy().reshape(-1,1))
                # plt.show()

                loss = self.l2(y_pred, batch_y)

                # add KKT loss

                loss_KKT = self.KKT_soft.KKT_loss_soft(batch_X, batch_lab, batch_Idx, C, y_pred)
                loss += kkt_lambda*loss_KKT

                # add utility loss
                if self.utility_loss == True:
                    loss_utility = self.KKT_soft.utility_loss_soft(batch_X, batch_lab, batch_Idx, C, y_pred, batch_y)
                    loss += loss_utility

                l1_reg = torch.tensor(0.).cuda()
                l2_reg = torch.tensor(0.).cuda()
                for name, param in self.model.named_parameters():
                    if 'weight' in name:
                        l1_reg += torch.norm(param, 1)
                        l2_reg += torch.norm(param)
                loss += l1_lambda * l1_reg
                loss += l2_lambda * l2_reg
                end1 = time.time()
                loss.backward()
                self.optim.step()
                loss_val = loss.data.cpu().numpy().item()
                train_loss += loss_val

                end2 =  time.time()
                print('The {}th batch with KKT time {:.3f}, total time {:.3f}'.format(j, end2-end1, end1-start))
                # print('The {}th batch with loss {:.3f}, loss_KKT {:.3f}'.format(j, loss/batch_size, loss_KKT/batch_size))

            train_loss /= train_size
            test_loss = self.evaluate(train_data, train_lab, valid_set)

            print('Epoch %s Train Loss %s Test Loss %s' % (epoch, train_loss, test_loss))
            Train_loss.append(train_loss)
            Valid_loss.append(test_loss)

            # save the model trained in each epoch
            # model_savename = self.save_path + '/UtlModel_Del0_KKT_Nepoch{}_Iter{}_L1L2_{}_{}.state_dict'.format(
            model_savename = self.save_path + '/Rnd_Del0_KKT_{}_Utl_{}_Nepoch{}_Iter{}_L1L2_{}_{}.state_dict'.format(kkt_lambda, str(self.utility_loss),
                epoch, n_epoch, self.model.out_features, self.model.hidden_ext)
            sys.stdout.flush()
            torch.save(self.model.state_dict(), model_savename)

            if epoch >= 5 and np.abs(Train_loss[-1] - Train_loss[-2]) / Train_loss[-2] < Tol:
                break

        return Train_loss, Valid_loss


    def evaluate(self, train_data, train_lab, valid_set ):
        '''

        :param train_data:
        :param train_lab: should be one_hot encoding!!
        :param valid_set:
        :return:
        '''
        self.model.eval()
        N, k = train_data.shape
        X_feature_test, y_feature_test = valid_set
        train_lab = copy.deepcopy(train_lab)
        test_size = len(y_feature_test)
        test_loss = 0

        for i in range(test_size):

            b = np.zeros((N, k))
            labs = np.zeros((N, self.lab_size))
            sel_index = np.zeros(N, dtype=bool)
            sel_index[np.where(X_feature_test[i] == 1)[0]] = True
            sel_index = sel_index.flatten()
            selected_train_data = train_data[sel_index]
            selected_train_lab = train_lab[sel_index]
            b[:selected_train_data.shape[0]] = selected_train_data
            if len(selected_train_lab.shape) == 1:
                selected_train_lab = selected_train_lab.reshape(-1, 1)
            labs[:selected_train_data.shape[0],:] = selected_train_lab

            batch_X, batch_lab, batch_y = torch.FloatTensor(b).cuda(), torch.FloatTensor(labs).cuda(), \
                                          torch.FloatTensor(y_feature_test[i:i+1]).cuda()
            batch_X, batch_lab, batch_y = batch_X.reshape((1, N, k)), batch_lab.reshape((1, N, self.lab_size)), batch_y.reshape((1, -1))
            y_pred = self.model(batch_X, batch_lab)

            loss = self.l2(y_pred, batch_y)
            loss_val = loss.data.cpu().numpy().item()
            test_loss += loss_val

        test_loss /= test_size
        return test_loss

    def predict(self, train_data, train_lab, X_feature_test):
        self.model.eval()
        N, k = train_data.shape
        test_size = len(X_feature_test)
        train_lab = copy.deepcopy(train_lab)
        if self.lab_size > 2:
            train_lab = torch.nn.functional.one_hot(train_lab, num_classes=self.lab_size)
        YPred = []

        # start = time.time()
        for i in range(test_size):
            b = np.zeros((N, k))
            resps = np.zeros((N, self.lab_size))
            sel_index = np.zeros(N, dtype=bool)
            sel_index[np.where(X_feature_test[i] == 1)[0]] = True
            sel_index = sel_index.flatten()
            selected_train_data = train_data[sel_index]
            selected_train_lab = train_lab[sel_index]
            if len(selected_train_lab.shape) == 1:
                selected_train_lab = selected_train_lab.reshape(-1, 1)
            b[:selected_train_data.shape[0]] = selected_train_data
            resps[:selected_train_data.shape[0]] = selected_train_lab

            batch_X, batch_lab = torch.FloatTensor(b).cuda(), torch.FloatTensor(
                resps).cuda()
            batch_X, batch_resp = batch_X.reshape((1, N, k)), batch_lab.reshape(
                (1, N, self.lab_size))

            y_pred = self.model(batch_X, batch_resp)
            # end = time.time()
            # self.time = end - start
            YPred.append(y_pred.cpu().detach().numpy())

            # batch_X, batch_lab = torch.FloatTensor(b), torch.FloatTensor(
            #     resps)
            # batch_X, batch_resp = batch_X.reshape((1, N, k)), batch_lab.reshape(
            #     (1, N, self.lab_size))
            #
            # y_pred = self.model(batch_X, batch_resp)
            # # end = time.time()
            # # self.time = end - start
            # YPred.append(y_pred.detach().numpy())
        # end = time.time()
        # self.time = end - start
        return YPred


class Para_deepset(object): # no KKT
    def __init__(self, in_dims, in_lab, out_dim, deep_config, configs, model=None, bias=False):

        if model is None:
            self.model = DeepSet(in_dims, in_lab, out_dim, deep_config)
        else:
            self.model = model

        self.lab_size = in_lab

        self.model = self.model.cuda()

        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss(reduction='mean') #sum
        self.lr = configs['learning_rate']
        self.dataset = configs['dataset']
        self.save_path = configs['model_path']
        self.optim = optim.Adam(self.model.parameters(), self.lr )
        # self.optim = optim.AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01,
        #                          amsgrad=False)
        self.KKT_soft = KKTSoft(self.dataset, configs['base_model'])

    def fit(self, train_data, train_lab, train_set, valid_set, C, n_epoch,
            l1_lambda=0.01, l2_lambda=0.01, Tol=1e-3, batch_size=32):
        train_data = copy.deepcopy(train_data)
        train_lab = copy.deepcopy(train_lab)
        N = train_data.shape[0]  # largest training sample size (for permutation)
        k = train_data.shape[1]  # dimension of the features

        X_feature, y_feature = train_set # y feature is the utility (estimates of parameters)
        X_feature_test, y_feature_test = valid_set  # set for Shapley Calculation of testing

        train_size = len(y_feature)  # number of permutations to be trained

        Train_loss, Valid_loss = [], []
        C = torch.tensor(C).cuda()

        for epoch in range(n_epoch):
            # Shuffle training utility samples
            ind = np.arange(train_size, dtype=int)
            np.random.shuffle(ind)
            X_feature = [X_feature[i] for i in ind]  # index for perMaM<mutation index
            # y_feature_1 = [y_feature[i] for i in ind]
            y_feature = y_feature[ind]
            # grad_all = [train_grad[i] for i in ind]

            train_loss = 0
            start_ind = 0

            for j in tqdm(range(train_size // batch_size)):
            # for j in range(train_size // batch_size):
                start = time.time()
                start_ind = j * batch_size
                batch_X, batch_lab, batch_y, batch_grad = [], [], [], []  # here y is the utility
                batch_Idx, batch_Idx2 = [], []

                for i in range(start_ind, min(start_ind + batch_size, train_size)):
                    b = np.zeros((N, k))
                    labs = np.zeros((N, self.lab_size))
                    grads = np.zeros((N, k+1))
                    sel_index = np.zeros(N, dtype=bool)
                    sel_index[np.where(X_feature[i]==1)[0]] = True
                    selected_train_data = train_data[sel_index]
                    selected_train_lab = train_lab[sel_index]
                    # selected_train_grad = grad_all[i]
                    if len(selected_train_lab.shape)==1:
                        selected_train_lab = selected_train_lab.reshape(-1, 1)
                    b[:selected_train_data.shape[0]] = selected_train_data
                    labs[:selected_train_data.shape[0],:] = selected_train_lab
                    # grads[:selected_train_data.shape[0]] = selected_train_grad

                    index_len = np.where(sel_index.astype(int)==1)[0].shape[0]
                    # index = np.zeros(N)
                    # index[:index_len] = 1
                    # batch_Idx.append(index)
                    batch_Idx.append(index_len)
                    batch_X.append(b)
                    batch_lab.append(labs)
                    # batch_grad.append(grads)

                    batch_y.append(y_feature[i])

                batch_X = np.stack(batch_X)
                batch_lab = np.stack(batch_lab)
                batch_Idx = np.stack(batch_Idx)
                # batch_grad= np.stack(batch_grad)
                batch_X, batch_lab, batch_y = torch.FloatTensor(batch_X).cuda(), torch.FloatTensor(
                    batch_lab).cuda(), torch.FloatTensor(batch_y).cuda()
                batch_Idx = torch.Tensor(batch_Idx).cuda()
                # batch_Idx2 = torch.Tensor(batch_Idx2).cuda()

                self.optim.zero_grad()
                y_pred = self.model(batch_X, batch_lab)

                loss = self.l2(y_pred, batch_y)

                l1_reg = torch.tensor(0.).cuda()
                l2_reg = torch.tensor(0.).cuda()
                for name, param in self.model.named_parameters():
                    if 'weight' in name:
                        l1_reg += torch.norm(param, 1)
                        l2_reg += torch.norm(param)
                loss += l1_lambda * l1_reg
                loss += l2_lambda * l2_reg
                end1 = time.time()
                loss_val = loss.data.cpu().numpy().item()
                loss.backward()
                self.optim.step()
                train_loss += loss_val

                end2 =  time.time()
                print('The {}th batch with KKT time {:.3f}, total time {:.3f}'.format(j, end2-end1, end1-start))

            train_loss /= train_size
            test_loss = self.evaluate(train_data, train_lab, valid_set)

            print('Epoch %s Train Loss %s Test Loss %s' % (epoch, train_loss, test_loss))
            Train_loss.append(train_loss)
            Valid_loss.append(test_loss)

            # save the model trained in each epoch
            model_savename = self.save_path + '/UtlModel_Del0_NOKKT_Nepoch{}_L1L2_{}_{}_lr_{}.state_dict'.format(
                epoch, n_epoch, self.model.out_features, self.model.hidden_ext, self.lr)
            sys.stdout.flush()
            torch.save(self.model.state_dict(), model_savename)

            if epoch >= 10 and np.abs(Train_loss[-1] - Train_loss[-2]) / Train_loss[-2] < Tol:
                break

        return Train_loss, Valid_loss


    def evaluate(self, train_data, train_lab, valid_set ):
        N, k = train_data.shape
        X_feature_test, y_feature_test = valid_set
        test_size = len(y_feature_test)
        test_loss = 0

        for i in range(test_size):

            b = np.zeros((N, k))
            labs = np.zeros((N, self.lab_size))
            sel_index = np.zeros(N, dtype=bool)
            sel_index[np.where(X_feature_test[i] == 1)[0]] = True
            sel_index = sel_index.flatten()
            selected_train_data = train_data[sel_index]
            selected_train_lab = train_lab[sel_index]
            if len(selected_train_lab.shape) == 1:
                selected_train_lab = selected_train_lab.reshape(-1, 1)
            b[:selected_train_data.shape[0]] = selected_train_data
            labs[:selected_train_data.shape[0],:] = selected_train_lab

            batch_X, batch_lab, batch_y = torch.FloatTensor(b).cuda(), torch.FloatTensor(labs).cuda(), \
                                          torch.FloatTensor(y_feature_test[i:i+1]).cuda()
            batch_X, batch_lab, batch_y = batch_X.reshape((1, N, k)), batch_lab.reshape((1, N, self.lab_size)), batch_y.reshape((1, -1))
            y_pred = self.model(batch_X, batch_lab)

            loss = self.l2(y_pred, batch_y)
            loss_val = loss.data.cpu().numpy().item()
            test_loss += loss_val

        test_loss /= test_size
        return test_loss

    def predict(self, train_data, train_lab, X_feature_test):
        train_data = copy.deepcopy(train_data)
        train_lab = copy.deepcopy(train_lab)
        N, k = train_data.shape
        test_size = len(X_feature_test)
        YPred = []

        for i in range(test_size):
            b = np.zeros((N, k))
            resps = np.zeros((N, self.lab_size))
            sel_index = np.zeros(N, dtype=bool)
            sel_index[np.where(X_feature_test[i] == 1)[0]] = True
            sel_index = sel_index.flatten()
            selected_train_data = train_data[sel_index]
            selected_train_lab = train_lab[sel_index]
            if len(selected_train_lab.shape) == 1:
                selected_train_lab = selected_train_lab.reshape(-1, 1)
            b[:selected_train_data.shape[0]] = selected_train_data
            resps[:selected_train_data.shape[0]] = selected_train_lab

            batch_X, batch_lab = torch.FloatTensor(b).cuda(), torch.FloatTensor(
                resps).cuda()
            batch_X, batch_resp = batch_X.reshape((1, N, k)), batch_lab.reshape(
                (1, N, self.lab_size))
            # start = time.time()
            y_pred = self.model(batch_X, batch_resp)
            # end = time.time()
            YPred.append(y_pred.cpu().detach().numpy())
            # self.time = end - start

        return YPred