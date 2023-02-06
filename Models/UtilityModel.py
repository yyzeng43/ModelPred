# predict on the final utility (for Shapley Comparison and model parameter estimation evaluation)
'''
Input: trained model
Output: utility + Shapley
'''

import numpy as np
from tqdm import tqdm
import itertools
import pickle
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import torch.nn as nn
import torch

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


class MINST_Utility(object):
    def __init__(self, X_test, y_test, y_test_enc, config, seed, **kwargs):
        '''
        :param X:
        :param y:
        :param X_test:
        :param y_test:
        :param config:
        :param seed:
        :param kwargs:
        '''

        if seed is not None:
            np.random.seed(seed)
        self.X_test = X_test
        self.y_test = y_test
        self.y_test_enc = y_test_enc
        self.config = config
        self.C = 1/self.config['lambda']

    def _testing_loss(self, log_model):
        y_pred_prob = log_model.predict_proba(self.X_test)
        w = log_model.coef_.reshape(-1,1)
        loss = self.C *(-np.sum(self.y_test_enc * np.log(y_pred_prob))) + 1/2* np.linalg.norm(w)**2
        return np.asarray(loss / self.y_test.shape[0]).flatten()[0]


    def _testing_acc(self, log_model):
        y_pred = log_model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        return acc

    def _testing_acc_inital(self):
        y_pred = np.zeros(self.y_test.shape[0])
        acc = accuracy_score(self.y_test, y_pred)
        return acc

    def _deep_loss(self, paras):
        # pred_W = paras.reshape((10, 785))[:, :-1]
        # pred_b = paras.reshape((10, 785))[ :, -1]
        pred_W = paras.reshape((10, 129))[:, :-1]
        pred_b = paras.reshape((10, 129))[ :, -1]
        y_pred_prob = nn.functional.softmax(np.matmul(self.X_test, pred_W.T) + pred_b,
                             dim=1)
        y_pred = torch.clip(y_pred_prob, 1e-9, 1.)
        loss = self.C *(-torch.sum(torch.tensor(self.y_test_enc) * torch.log(y_pred))) \
               + 1/2* np.linalg.norm(pred_W)**2
        # loss = torch.mean(-torch.sum(torch.tensor(self.y_test_enc) * torch.log(y_pred))+self.C*torch.norm(paras)**2)


        # y_pred_prob = nn.functional.softmax(torch.matmul(self.X_test, torch.transpose(torch.squeeze(pred_W), 0, 1)) + pred_b,
        #                       dim=0)
        # y_pred = torch.clip(y_pred_prob, 1e-9, 1.)
        # loss = torch.mean(-torch.sum(self.y_test_enc * torch.log(y_pred))).cpu().numpy()
        return np.asarray(loss / self.y_test.shape[0]).flatten()[0]

    def _deep_acc(self, paras):
        # pred_W = paras.reshape((10, 785))[:, :-1]
        # pred_b = paras.reshape((10, 785))[ :, -1]
        pred_W = paras.reshape((10, 129))[:, :-1]
        pred_b = paras.reshape((10, 129))[ :, -1]
        y_pred_prob =  nn.functional.softmax(np.matmul(self.X_test, pred_W.T) + pred_b,
                              dim=1)
        y_pred = np.argmax(y_pred_prob, axis = 1)
        acc = accuracy_score(self.y_test, y_pred)
        return acc

    def _model_dist(self, paras, act_model):
        fitted_para = np.concatenate((act_model.coef_, act_model.intercept_.reshape(-1, 1)), axis=1).flatten()
        return np.linalg.norm(paras-fitted_para, ord=2)

    @property
    def test_ini_acc(self):
        return self._testing_acc_inital()


class Minst_Shapley(object):
    def __init__(self, X_train,y_train, y_train_enc, X_test, y_test, y_test_enc, path, config, seed):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        self.X_train = X_train
        self.y_train = y_train
        self.y_train_enc = y_train_enc
        self.X_test = X_test
        self.y_test = y_test
        self.y_test_enc = y_test_enc
        self.PATH = path
        self.config = config
        self.C = 1/self.config['lambda']
        self.feature_dim = self.config['dim']
        self.num_class = config['class_number']

        self.le = preprocessing.LabelEncoder().fit(y_train)

    def _create_model(self):
        model = LogisticRegression(max_iter=10000, C = self.C, solver='lbfgs')
        return model


    def _deep_approx_perm(self, deepmodel, X_train_scaled, scaler_Utl):
        # calculate the utility estimated by approximate sampling (without training data)
        # directly use deepmodel to predict the estimated parameter
        Tol = self.config['DeepUtlTol']
        Tmax = self.config['MaxIter']


        self.DeepApproxPerm = {}
        self.DeepApproxPerm['utility_acc'] = []
        self.DeepApproxPerm['utility_loss'] = []
        self.DeepApproxPerm['utility_loss_true']=[]
        self.DeepApproxPerm['Svalue_loss'] = []
        self.DeepApproxPerm['Svalue_acc'] = []
        self.DeepApproxPerm['perm'] = []  # record the permutation
        self.DeepApproxPerm['one_hot'] = []
        self.DeepApproxPerm['fitted_para'] = []
        self.DeepApproxPerm['pred_para'] = []
        self.DeepApproxPerm['para_dist'] = []

        Utility_model = MINST_Utility(self.X_test, self.y_test, self.y_test_enc, self.config, self.seed)
        n_data = self.y_train.shape[0]  # number of data points
        t = 0
        delta = 100

        for t in tqdm(range(Tmax)):
            np.random.seed(t+1234)
            perm = np.random.permutation(n_data)
            self.DeepApproxPerm['perm'].append(perm)
            self.DeepApproxPerm['utility_loss'].append([-1 * np.log(2)])
            self.DeepApproxPerm['utility_loss_true'].append([-1 * np.log(2)])
            self.DeepApproxPerm['utility_acc'].append([Utility_model.test_ini_acc])
            self.DeepApproxPerm['Svalue_loss'].append(np.zeros(n_data))
            self.DeepApproxPerm['Svalue_acc'].append(np.zeros(n_data))
            self.DeepApproxPerm['one_hot'].append([])
            self.DeepApproxPerm['pred_para'].append([])
            self.DeepApproxPerm['fitted_para'].append([])
            self.DeepApproxPerm['para_dist'].append([])

            for i in tqdm(range(n_data)):
                dataid = perm[i]
                datasid = np.asarray(perm[:i + 1]).flatten()
                one_hot = np.zeros(len(self.y_train))
                one_hot[datasid] = 1

                X_use, y_use, y_use_org = self.X_train[datasid], self.y_train_enc[datasid], self.y_train[datasid]
                if len(np.unique(y_use_org)) < 10:
                    loss =  np.log(2)
                    true_loss = np.log(2)
                    acc = Utility_model.test_ini_acc
                    para_dist = 0
                    self.DeepApproxPerm['pred_para'][-1].append(np.zeros((self.num_class, self.feature_dim + 1)))
                    self.DeepApproxPerm['fitted_para'][-1].append(np.zeros((self.num_class, self.feature_dim + 1)))

                else:
                    estimation = deepmodel.predict(X_train_scaled,
                                                   self.y_train, one_hot.reshape(1, -1))
                    estimation = np.array(estimation).reshape(1, -1)
                    if scaler_Utl != None:
                        estimation = scaler_Utl.inverse_transform(estimation)
                    loss = Utility_model._deep_loss(estimation)
                    acc = Utility_model._deep_acc(estimation)

                    model = self._create_model()
                    y_use_le = self.le.transform(y_use_org)
                    model.fit(X_use, y_use_le)
                    fitted_para = np.concatenate((model.coef_, model.intercept_.reshape(-1, 1)), axis=1).flatten()
                    true_loss = Utility_model._testing_loss(model)
                    para_dist = Utility_model._model_dist(estimation, model)

                    self.DeepApproxPerm['pred_para'][-1].append(estimation.reshape((10, 129)))
                    self.DeepApproxPerm['fitted_para'][-1].append(fitted_para.reshape((10, 129)))

                self.DeepApproxPerm['utility_loss'][-1].append(-1*loss)
                self.DeepApproxPerm['utility_loss_true'][-1].append(-1 * true_loss)
                self.DeepApproxPerm['utility_acc'][-1].append(acc)
                self.DeepApproxPerm['para_dist'][-1].append(para_dist)

                if t > 0:
                    self.DeepApproxPerm['Svalue_loss'][-1][dataid] = t / (t + 1) * self.DeepApproxPerm['Svalue_loss'][-2][
                        dataid] + 1 / (t + 1) * (self.DeepApproxPerm['utility_loss'][-1][-1] -
                                                 self.DeepApproxPerm['utility_loss'][-1][-2])
                    self.DeepApproxPerm['Svalue_acc'][-1][dataid] = t / (t + 1) * self.DeepApproxPerm['Svalue_acc'][-2][
                        dataid] + 1 / (t + 1) * (self.DeepApproxPerm['utility_acc'][-1][-1] -
                                                 self.DeepApproxPerm['utility_acc'][-1][-2])

                else:
                    self.DeepApproxPerm['Svalue_loss'][-1][dataid] = self.DeepApproxPerm['utility_loss'][-1][-1] - \
                                                                 self.DeepApproxPerm['utility_loss'][-1][-2]

                    self.DeepApproxPerm['Svalue_acc'][-1][dataid] = self.DeepApproxPerm['utility_acc'][-1][-1] - \
                                                                 self.DeepApproxPerm['utility_acc'][-1][-2]

            if t > 5:
                delta1 = np.max(np.std(np.asarray(self.DeepApproxPerm['Svalue_acc'])[-5:, :], axis=0))
                delta2 = np.max(np.std(np.asarray(self.DeepApproxPerm['Svalue_loss'])[-5:, :], axis=0))
                delta = np.max((delta1, delta2))
            # print('Permutation {} Delta {}'.format(t+1, delta))

            pickle.dump([self.DeepApproxPerm['utility_loss'], self.DeepApproxPerm['utility_loss_true']],
                    open(self.PATH + '/Perm_{}_Iter_{}_DeepLoss_TrueLoss.data'.format(Tmax, t), 'wb'))
            pickle.dump([self.DeepApproxPerm['pred_para'], self.DeepApproxPerm['fitted_para']],
                    open(self.PATH + '/Perm_{}_Iter_{}_DeepPara_TruePara.data'.format(Tmax, t), 'wb'))
            pickle.dump([self.DeepApproxPerm['para_dist']],
                    open(self.PATH + '/Perm_{}_Iter_{}_DeepTrueParaDist.data'.format(Tmax, t), 'wb'))

            if t > 1000 and delta < Tol:
                print('Reaching tolerance at iteration {:d}'.format(t))
                break


    def _deep_perm(self, deepmodel, X_train_scaled, scaler_Utl):
        # calculate the utility estimated by approximate sampling (without training data)
        # directly use deepmodel to predict the estimated parameter
        Tol = self.config['DeepUtlTol']
        Tmax = self.config['MaxIter']


        self.DeepPerm = {}
        self.DeepPerm['utility_acc'] = []
        self.DeepPerm['utility_loss'] = []
        self.DeepPerm['utility_loss_true']=[]
        self.DeepPerm['Svalue_loss'] = []
        self.DeepPerm['Svalue_acc'] = []
        self.DeepPerm['perm'] = []  # record the permutation
        self.DeepPerm['one_hot'] = []
        self.DeepPerm['fitted_para'] = []
        self.DeepPerm['pred_para'] = []
        self.DeepPerm['para_dist'] = []

        Utility_model = MINST_Utility(self.X_test, self.y_test, self.y_test_enc, self.config, self.seed)
        n_data = self.y_train.shape[0]  # number of data points
        t = 0
        delta = 100

        for t in tqdm(range(Tmax)):
            np.random.seed(t+1234)
            perm = np.random.permutation(n_data)


            for i in tqdm(range(n_data)):
                dataid = perm[i]
                datasid = np.asarray(perm[:i + 1]).flatten()
                one_hot = np.zeros(len(self.y_train))
                one_hot[datasid] = 1

                X_use, y_use, y_use_org = self.X_train[datasid], self.y_train_enc[datasid], self.y_train[datasid]
                if len(np.unique(y_use_org)) == 10:
                    self.DeepPerm['perm'].append(perm)
                    estimation = deepmodel.predict(X_train_scaled,
                                                   self.y_train, one_hot.reshape(1, -1))
                    estimation = np.array(estimation).reshape(1, -1)
                    if scaler_Utl != None:
                        estimation = scaler_Utl.inverse_transform(estimation)
                    loss = Utility_model._deep_loss(estimation)
                    acc = Utility_model._deep_acc(estimation)

                    model = self._create_model()
                    y_use_le = self.le.transform(y_use_org)
                    model.fit(X_use, y_use_le)
                    fitted_para = np.concatenate((model.coef_, model.intercept_.reshape(-1, 1)), axis=1).flatten()
                    true_loss = Utility_model._testing_loss(model)
                    para_dist = Utility_model._model_dist(estimation, model)

                    self.DeepPerm['pred_para'].append(estimation.reshape((10, 129)))
                    self.DeepPerm['fitted_para'].append(fitted_para.reshape((10, 129)))

                    self.DeepPerm['utility_loss'].append(-1*loss)
                    self.DeepPerm['utility_loss_true'].append(-1 * true_loss)
                    self.DeepPerm['utility_acc'].append(acc)
                    self.DeepPerm['para_dist'].append(para_dist)
                    self.DeepPerm['one_hot'].append(one_hot)

        pickle.dump([self.DeepPerm['utility_loss'], self.DeepPerm['utility_loss_true']],
                open(self.PATH + '/Perm_{}_DeepLoss_TrueLoss.data'.format(Tmax), 'wb'))
        pickle.dump([self.DeepPerm['pred_para'], self.DeepPerm['fitted_para']],
                open(self.PATH + '/Perm_{}_DeepPara_TruePara.data'.format(Tmax), 'wb'))
        pickle.dump([self.DeepPerm['para_dist']],
                open(self.PATH + '/Perm_{}_DeepTrueParaDist.data'.format(Tmax), 'wb'))


    def _rand_sampling(self, deepmodel, X_train_scaled, num, size_min, size_max, random_state):
        np.random.seed(random_state)

        self.rand_sets = {}
        self.rand_sets['utility_loss'] = []
        self.rand_sets['utility_loss_true'] = []
        self.rand_sets['one_hot']=[]
        self.rand_sets['pred_para']=[]
        self.rand_sets['fitted_para']=[]
        self.rand_sets['para_dist']=[]

        N = self.X_train.shape[0]
        Utility_model = MINST_Utility(self.X_test, self.y_test, self.y_test_enc, self.config, self.seed)

        for i in tqdm(range(num)):
            n_select = np.random.choice(range(size_min, size_max))
            subset_index = np.random.choice(range(N), n_select, replace=False)
            one_hot = np.zeros(len(self.y_train))
            one_hot[subset_index] = 1
            X_use, y_use, y_use_org = self.X_train[subset_index], self.y_train_enc[subset_index], self.y_train[subset_index]

            if len(np.unique(y_use_org)) == 10:
                estimation = deepmodel.predict(X_train_scaled,
                                               self.y_train, one_hot.reshape(1, -1))
                estimation = np.array(estimation).reshape(1, -1)

                loss = Utility_model._deep_loss(estimation)

                model = self._create_model()
                y_use_le = self.le.transform(y_use_org)
                model.fit(X_use, y_use_le)
                fitted_para = np.concatenate((model.coef_, model.intercept_.reshape(-1, 1)), axis=1).flatten()
                true_loss = Utility_model._testing_loss(model)
                para_dist = Utility_model._model_dist(estimation, model)

                self.rand_sets['pred_para'].append(estimation.reshape((10, 129)))
                self.rand_sets['fitted_para'].append(fitted_para.reshape((10, 129)))

                self.rand_sets['utility_loss'].append(-1 * loss)
                self.rand_sets['utility_loss_true'].append(-1 * true_loss)
                self.rand_sets['para_dist'].append(para_dist)
                self.rand_sets['one_hot'].append(one_hot)

        pickle.dump([self.rand_sets['one_hot'], self.rand_sets['para_dist']],
                    open(self.PATH + '/Rand_{}_OneEncoding_TrueParaDist.data'.format(num), 'wb'))
        pickle.dump([self.rand_sets['pred_para'], self.rand_sets['fitted_para']],
                    open(self.PATH + '/Rand_{}_Deep_TruePara.data'.format(num), 'wb'))
        pickle.dump([self.rand_sets['utility_loss'], self.rand_sets['utility_loss_true']],
                    open(self.PATH + '/Rand_{}_Deep_TrueLoss.data'.format(num), 'wb'))

    def _utl_approx_perm(self, deepmodel, X_train_scaled, num, size_min, size_max, random_state):
        # calculate the utility estimated by approximate sampling (without training data)
        # directly use deepmodel to predict the estimated parameter

        self.UtlApproxPerm = {}
        self.UtlApproxPerm['utility_loss'] = []
        self.UtlApproxPerm['utility_loss_true']=[]
        self.UtlApproxPerm['one_hot'] = []

        Utility_model = MINST_Utility(self.X_test, self.y_test, self.y_test_enc, self.config, self.seed)

        N = self.X_train.shape[0]
        for i in tqdm(range(num)):
            n_select = np.random.choice(range(size_min, size_max))
            subset_index = np.random.choice(range(N), n_select, replace=False)
            one_hot = np.zeros(len(self.y_train))
            one_hot[subset_index] = 1
            X_use, y_use, y_use_org = self.X_train[subset_index], self.y_train[subset_index], self.y_train[subset_index]

            if len(np.unique(y_use_org)) == 10:

                estimation = deepmodel.predict(X_train_scaled,
                                               self.y_train, one_hot.reshape(1, -1))
                estimation = np.squeeze(np.array(estimation))
                loss = estimation

                model = self._create_model()
                y_use_le = self.le.transform(y_use_org)
                model.fit(X_use, y_use_le)
                true_loss = Utility_model._testing_loss(model)

                self.UtlApproxPerm['utility_loss'].append(-1 * loss)
                self.UtlApproxPerm['utility_loss_true'].append(-1 * true_loss)
                self.UtlApproxPerm['one_hot'].append(one_hot)

        pickle.dump([self.UtlApproxPerm['one_hot']],
                    open(self.PATH + '/Utl_Rand_{}_OneEncoding.data'.format(num), 'wb'))
        pickle.dump([self.UtlApproxPerm['utility_loss'], self.UtlApproxPerm['utility_loss_true']],
                    open(self.PATH + '/Utl_Rand_{}_Deep_TrueLoss.data'.format(num), 'wb'))


    def _influ_rand(self, influ_train_point, num, size_min, size_max, random_state):
        np.random.seed(random_state)

        self.influ_rand_sets = {}
        self.influ_rand_sets['utility_loss'] = []
        self.influ_rand_sets['utility_loss_true'] = []
        self.influ_rand_sets['one_hot'] = []

        N = self.X_train.shape[0]
        Utility_model = MINST_Utility(self.X_test, self.y_test, self.y_test_enc, self.config, self.seed)

        model = self._create_model()
        model.fit(self.X_train, self.y_train)
        loss_all = Utility_model._testing_loss(model)

        for i in tqdm(range(num)):
            n_select = np.random.choice(range(size_min, size_max))
            subset_index = np.random.choice(range(N), n_select, replace=False)
            one_hot = np.zeros(len(self.y_train))
            one_hot[subset_index] = 1
            X_use, y_use, y_use_org = self.X_train[subset_index], self.y_train[subset_index], self.y_train[subset_index]

            if len(np.unique(y_use_org)) == 10:
                del_index = np.setdiff1d(np.arange(self.X_train.shape[0]),subset_index)
                loss_sel = influ_train_point[del_index]
                loss = loss_all + np.sum(loss_sel) / self.X_test.shape[0]

                model = self._create_model()
                y_use_le = self.le.transform(y_use_org)
                model.fit(X_use, y_use_le)
                true_loss = Utility_model._testing_loss(model)

                self.influ_rand_sets['utility_loss'].append(-1 * loss)
                self.influ_rand_sets['utility_loss_true'].append(-1 * true_loss)

                self.influ_rand_sets['one_hot'].append(one_hot)

        pickle.dump([self.influ_rand_sets['one_hot']],
                    open(self.PATH + '/Influ_Rand_{}_OneEncoding.data'.format(num), 'wb'))
        pickle.dump([self.influ_rand_sets['utility_loss'], self.influ_rand_sets['utility_loss_true']],
                    open(self.PATH + '/Influ_Rand_{}_Deep_TrueLoss.data'.format(num), 'wb'))


class Iris_Utility(object):
    def __init__(self, X_test, y_test, config, seed, **kwargs):
        '''
        :param X:
        :param y:
        :param X_test:
        :param y_test:
        :param config:
        :param seed:
        :param kwargs:
        '''

        if seed is not None:
            np.random.seed(seed)
        self.X_test = X_test
        self.y_test = y_test
        self.config = config
        self.C = 1/self.config['lambda']

    def _testing_loss(self, log_model):
        y_pred_prob = log_model.predict_proba(self.X_test)[:,1]
        w = log_model.coef_.reshape(-1,1)
        loss = self.C * np.sum(-1*(np.multiply(self.y_test, np.log(y_pred_prob))+
                                   np.multiply(1-self.y_test, np.log(1-y_pred_prob)))) + 1/2* np.linalg.norm(w)**2
        return np.asarray(loss / self.y_test.shape[0]).flatten()[0]


    def _testing_acc(self, log_model):
        y_pred = log_model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        return acc

    def _testing_acc_inital(self):
        y_pred = np.zeros(self.y_test.shape[0])
        acc = accuracy_score(self.y_test, y_pred)
        return acc

    def _deep_loss(self, paras):
        w = np.squeeze(paras)[:-1]
        c = np.squeeze(paras)[-1]
        sigmoid = lambda x: 1 / (1 + np.exp(-1 * x))
        y_pred_prob = sigmoid(np.dot(self.X_test, w.reshape(-1, 1)) + c).flatten()
        loss = self.C*np.sum(-1*(np.multiply(self.y_test, np.log(y_pred_prob))+np.multiply(1-self.y_test, np.log(1-y_pred_prob))))\
               + 1/2* np.linalg.norm(w)**2
        return np.asarray(loss / self.y_test.shape[0]).flatten()[0]

    def _deep_acc(self, paras):
        w = np.squeeze(paras)[:-1]
        c = np.squeeze(paras)[-1]
        sigmoid = lambda x: 1 / (1 + np.exp(-1 * x))
        y_pred = sigmoid(np.dot(self.X_test, w.reshape(-1,1))+ c)
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        acc = accuracy_score(self.y_test, y_pred)
        return acc

    def _model_dist(self, paras, act_model):
        fitted_para = np.concatenate((act_model.coef_.T, act_model.intercept_), axis = None)
        return np.linalg.norm(paras-fitted_para, ord=2)

    @property
    def test_ini_acc(self):
        return self._testing_acc_inital()



class Iris_Shapley(object):
    def __init__(self, X_train, y_train, X_test, y_test, path, config, seed):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.PATH = path
        self.config = config
        self.C = 1/self.config['lambda']
        self.feature_dim = self.config['dim']


    def _create_model(self):
        model = LogisticRegression(max_iter=10000, C = self.C, solver='lbfgs')
        return model

    def _deep_approx_perm(self, deepmodel, X_train_scaled, scaler_Utl):
        # calculate the utility estimated by approximate sampling (without training data)
        # directly use deepmodel to predict the estimated parameter
        Tol = self.config['DeepUtlTol']
        Tmax = self.config['MaxIter']


        self.DeepApproxPerm = {}
        self.DeepApproxPerm['utility_acc'] = []
        self.DeepApproxPerm['utility_loss'] = []
        self.DeepApproxPerm['utility_loss_true']=[]
        self.DeepApproxPerm['Svalue_loss'] = []
        self.DeepApproxPerm['Svalue_acc'] = []
        self.DeepApproxPerm['perm'] = []  # record the permutation
        self.DeepApproxPerm['one_hot'] = []
        self.DeepApproxPerm['pred_para'] = []
        self.DeepApproxPerm['fitted_para'] = []
        self.DeepApproxPerm['para_dist'] = []


        Utility_model = Iris_Utility(self.X_test, self.y_test, self.config, self.seed)
        n_data = self.y_train.shape[0]  # number of data points
        t = 0
        delta = 100

        for t in tqdm(range(Tmax)):
            np.random.seed(t+1234)
            perm = np.random.permutation(n_data)
            self.DeepApproxPerm['perm'].append(perm)
            self.DeepApproxPerm['utility_loss'].append([-1 * np.log(2)])
            self.DeepApproxPerm['utility_loss_true'].append([-1 * np.log(2)])
            self.DeepApproxPerm['utility_acc'].append([Utility_model.test_ini_acc])
            self.DeepApproxPerm['Svalue_loss'].append(np.zeros(n_data))
            self.DeepApproxPerm['Svalue_acc'].append(np.zeros(n_data))
            self.DeepApproxPerm['one_hot'].append([])
            self.DeepApproxPerm['fitted_para'].append([])
            self.DeepApproxPerm['pred_para'].append([])
            self.DeepApproxPerm['para_dist'].append([])


            for i in range(n_data):
                dataid = perm[i]
                datasid = np.asarray(perm[:i + 1]).flatten()
                one_hot = np.ones(len(self.y_train)) * (-1)
                one_hot[datasid] = 1

                X_use, y_use = self.X_train[datasid], self.y_train[datasid]
                if len(np.unique(y_use)) == 1:
                    loss =  np.log(2)
                    true_loss = np.log(2)
                    nrmse = Utility_model.test_ini_acc
                    para_dist = 0
                    self.DeepApproxPerm['pred_para'][-1].append(np.zeros(self.feature_dim + 1))
                    self.DeepApproxPerm['fitted_para'][-1].append(np.zeros(self.feature_dim + 1))

                else:
                    estimation = deepmodel.predict(X_train_scaled,
                                                   self.y_train, one_hot.reshape(1, -1))
                    estimation = np.squeeze(np.array(estimation))
                    if scaler_Utl != None:
                        estimation = scaler_Utl.inverse_transform(estimation)
                    loss = Utility_model._deep_loss(estimation)
                    nrmse = Utility_model._deep_acc(estimation)

                    model = self._create_model()
                    model.fit(X_use, y_use)
                    true_loss = Utility_model._testing_loss(model)
                    fitted_para = np.concatenate((model.coef_.T, model.intercept_), axis=None)
                    para_dist = Utility_model._model_dist(estimation, model)

                    self.DeepApproxPerm['pred_para'][-1].append(estimation)
                    self.DeepApproxPerm['fitted_para'][-1].append(fitted_para)


                self.DeepApproxPerm['utility_loss'][-1].append(-1*loss)
                self.DeepApproxPerm['utility_loss_true'][-1].append(-1 * true_loss)
                self.DeepApproxPerm['utility_acc'][-1].append(nrmse)
                self.DeepApproxPerm['para_dist'][-1].append(para_dist)

                if t > 0:
                    self.DeepApproxPerm['Svalue_loss'][-1][dataid] = t / (t + 1) * self.DeepApproxPerm['Svalue_loss'][-2][
                        dataid] + 1 / (t + 1) * (self.DeepApproxPerm['utility_loss'][-1][-1] -
                                                 self.DeepApproxPerm['utility_loss'][-1][-2])
                    self.DeepApproxPerm['Svalue_acc'][-1][dataid] = t / (t + 1) * self.DeepApproxPerm['Svalue_acc'][-2][
                        dataid] + 1 / (t + 1) * (self.DeepApproxPerm['utility_acc'][-1][-1] -
                                                 self.DeepApproxPerm['utility_acc'][-1][-2])

                else:
                    self.DeepApproxPerm['Svalue_loss'][-1][dataid] = self.DeepApproxPerm['utility_loss'][-1][-1] - \
                                                                 self.DeepApproxPerm['utility_loss'][-1][-2]

                    self.DeepApproxPerm['Svalue_acc'][-1][dataid] = self.DeepApproxPerm['utility_acc'][-1][-1] - \
                                                                 self.DeepApproxPerm['utility_acc'][-1][-2]

            if t > 5:
                delta1 = np.max(np.std(np.asarray(self.DeepApproxPerm['Svalue_acc'])[-5:, :], axis=0))
                delta2 = np.max(np.std(np.asarray(self.DeepApproxPerm['Svalue_loss'])[-5:, :], axis=0))
                delta = np.max((delta1, delta2))
            # print('Permutation {} Delta {}'.format(t+1, delta))

            pickle.dump([self.DeepApproxPerm['utility_loss'], self.DeepApproxPerm['utility_loss_true']],
                    open(self.PATH + '/Perm_{}_Iter_{}_DeepLoss_TrueLoss.data'.format(Tmax, t), 'wb'))
            pickle.dump([self.DeepApproxPerm['pred_para'], self.DeepApproxPerm['fitted_para']],
                    open(self.PATH + '/Perm_{}_Iter_{}_DeepPara_TruePara.data'.format(Tmax, t), 'wb'))
            pickle.dump([self.DeepApproxPerm['para_dist']],
                    open(self.PATH + '/Perm_{}_Iter_{}_DeepTrueParaDist.data'.format(Tmax, t), 'wb'))


            if delta < Tol:
                print('Reaching tolerance at iteration {:d}'.format(t))
                break

    def _deep_perm(self, deepmodel, X_train_scaled):
        # calculate the utility estimated by approximate sampling (without training data)
        # directly use deepmodel to predict the estimated parameter
        Tmax = self.config['MaxIter']


        self.DeepPerm = {}
        self.DeepPerm['utility_loss'] = []
        self.DeepPerm['utility_loss_true']=[]
        self.DeepPerm['perm'] = []  # record the permutation
        self.DeepPerm['one_hot'] = []
        self.DeepPerm['fitted_para'] = []
        self.DeepPerm['pred_para'] = []
        self.DeepPerm['para_dist'] = []

        N = self.X_train.shape[0]
        Utility_model = Iris_Utility(self.X_test, self.y_test, self.config, self.seed)

        for t in tqdm(range(Tmax)):
            np.random.seed(t+1234)
            perm = np.random.permutation(N)

            for i in tqdm(range(N)):
                dataid = perm[i]
                datasid = np.asarray(perm[:i + 1]).flatten()
                one_hot = np.zeros(len(self.y_train))
                one_hot[datasid] = 1

                X_use, y_use, y_use_org = self.X_train[datasid], self.y_train[datasid], self.y_train[datasid]
                if len(np.unique(y_use)) > 1:
                    estimation = deepmodel.predict(X_train_scaled,
                                                   self.y_train, one_hot.reshape(1, -1))
                    estimation = np.squeeze(np.array(estimation))
                    loss = Utility_model._deep_loss(estimation)

                    model = self._create_model()
                    model.fit(X_use, y_use)
                    true_loss = Utility_model._testing_loss(model)
                    fitted_para = np.concatenate((model.coef_.T, model.intercept_), axis=None)
                    para_dist = Utility_model._model_dist(estimation, model)


                    self.DeepPerm['pred_para'].append(estimation)
                    self.DeepPerm['fitted_para'].append(fitted_para)

                    self.DeepPerm['utility_loss'].append(-1*loss)
                    self.DeepPerm['utility_loss_true'].append(-1 * true_loss)
                    self.DeepPerm['para_dist'].append(para_dist)
                    self.DeepPerm['one_hot'].append(one_hot)

        pickle.dump([self.DeepPerm['utility_loss'], self.DeepPerm['utility_loss_true']],
                open(self.PATH + '/Perm_{}_DeepLoss_TrueLoss.data'.format(Tmax), 'wb'))
        pickle.dump([self.DeepPerm['pred_para'], self.DeepPerm['fitted_para']],
                open(self.PATH + '/Perm_{}_DeepPara_TruePara.data'.format(Tmax), 'wb'))
        pickle.dump([self.DeepPerm['para_dist']],
                open(self.PATH + '/Perm_{}_DeepTrueParaDist.data'.format(Tmax), 'wb'))


    def _rand_sampling(self, deepmodel, X_train_scaled, num, size_min, size_max, random_state):
        np.random.seed(random_state)

        self.rand_sets = {}
        self.rand_sets['utility_loss'] = []
        self.rand_sets['utility_loss_true'] = []
        self.rand_sets['one_hot']=[]
        self.rand_sets['pred_para']=[]
        self.rand_sets['fitted_para']=[]
        self.rand_sets['para_dist']=[]

        N = self.X_train.shape[0]
        Utility_model = Iris_Utility(self.X_test, self.y_test, self.config, self.seed)

        for i in tqdm(range(num)):
            n_select = np.random.choice(range(size_min, size_max))
            subset_index = np.random.choice(range(N), n_select, replace=False)
            one_hot = np.zeros(len(self.y_train))
            one_hot[subset_index] = 1
            X_use, y_use, y_use_org = self.X_train[subset_index], self.y_train[subset_index], self.y_train[subset_index]

            if len(np.unique(y_use)) > 1:
                estimation = deepmodel.predict(X_train_scaled,
                                               self.y_train, one_hot.reshape(1, -1))
                estimation = np.squeeze(np.array(estimation))
                loss = Utility_model._deep_loss(estimation)

                model = self._create_model()
                model.fit(X_use, y_use)
                true_loss = Utility_model._testing_loss(model)
                fitted_para = np.concatenate((model.coef_.T, model.intercept_), axis=None)
                para_dist = Utility_model._model_dist(estimation, model)

                self.rand_sets['pred_para'].append(estimation)
                self.rand_sets['fitted_para'].append(fitted_para)
                self.rand_sets['utility_loss'].append(-1 * loss)
                self.rand_sets['utility_loss_true'].append(-1 * true_loss)
                self.rand_sets['para_dist'].append(para_dist)
                self.rand_sets['one_hot'].append(one_hot)

        pickle.dump([self.rand_sets['one_hot'], self.rand_sets['para_dist']],
                    open(self.PATH + '/Rand_{}_OneEncoding_TrueParaDist.data'.format(num), 'wb'))
        pickle.dump([self.rand_sets['pred_para'], self.rand_sets['fitted_para']],
                    open(self.PATH + '/Rand_{}_Deep_TruePara.data'.format(num), 'wb'))
        pickle.dump([self.rand_sets['utility_loss'], self.rand_sets['utility_loss_true']],
                    open(self.PATH + '/Rand_{}_Deep_TrueLoss.data'.format(num), 'wb'))



    def _utl_approx_perm(self, deepmodel, X_train_scaled, num, size_min, size_max, random_state):
        # calculate the utility estimated by approximate sampling (without training data)
        # directly use deepmodel to predict the estimated parameter

        self.UtlApproxPerm = {}
        self.UtlApproxPerm['utility_loss'] = []
        self.UtlApproxPerm['utility_loss_true']=[]
        self.UtlApproxPerm['one_hot'] = []


        Utility_model = Iris_Utility(self.X_test, self.y_test, self.config, self.seed)
        N = self.X_train.shape[0]
        for i in tqdm(range(num)):
            n_select = np.random.choice(range(size_min, size_max))
            subset_index = np.random.choice(range(N), n_select, replace=False)
            one_hot = np.zeros(len(self.y_train))
            one_hot[subset_index] = 1
            X_use, y_use, y_use_org = self.X_train[subset_index], self.y_train[subset_index], self.y_train[subset_index]

            if len(np.unique(y_use)) > 1:
                estimation = deepmodel.predict(X_train_scaled,
                                               self.y_train, one_hot.reshape(1, -1))
                estimation = np.squeeze(np.array(estimation))
                loss = estimation

                model = self._create_model()
                model.fit(X_use, y_use)
                true_loss = Utility_model._testing_loss(model)

                self.UtlApproxPerm['utility_loss'].append(-1 * loss)
                self.UtlApproxPerm['utility_loss_true'].append(-1 * true_loss)
                self.UtlApproxPerm['one_hot'].append(one_hot)

        pickle.dump([self.UtlApproxPerm['one_hot']],
                    open(self.PATH + '/Utl_Rand_{}_OneEncoding.data'.format(num), 'wb'))
        pickle.dump([self.UtlApproxPerm['utility_loss'], self.UtlApproxPerm['utility_loss_true']],
                    open(self.PATH + '/Utl_Rand_{}_Deep_TrueLoss.data'.format(num), 'wb'))


    def _influ_rand(self, influ_train_point, num, size_min, size_max, random_state):
        np.random.seed(random_state)

        self.influ_rand_sets = {}
        self.influ_rand_sets['utility_loss'] = []
        self.influ_rand_sets['utility_loss_true'] = []
        self.influ_rand_sets['one_hot'] = []

        N = self.X_train.shape[0]
        Utility_model = Iris_Utility(self.X_test, self.y_test, self.config, self.seed)

        model = self._create_model()
        model.fit(self.X_train, self.y_train)
        loss_all = Utility_model._testing_loss(model)

        for i in tqdm(range(num)):
            n_select = np.random.choice(range(size_min, size_max))
            subset_index = np.random.choice(range(N), n_select, replace=False)
            one_hot = np.zeros(len(self.y_train))
            one_hot[subset_index] = 1
            X_use, y_use, y_use_org = self.X_train[subset_index], self.y_train[subset_index], self.y_train[subset_index]

            if len(np.unique(y_use)) > 1:
                del_index = np.setdiff1d(np.arange(self.X_train.shape[0]), subset_index)
                loss_sel = influ_train_point[del_index]
                loss = loss_all + np.sum(loss_sel) / self.X_test.shape[0]

                model = self._create_model()
                model.fit(X_use, y_use)
                true_loss = Utility_model._testing_loss(model)

                self.influ_rand_sets['utility_loss'].append(-1 * loss)
                self.influ_rand_sets['utility_loss_true'].append(-1 * true_loss)

                self.influ_rand_sets['one_hot'].append(one_hot)

        pickle.dump([self.influ_rand_sets['one_hot']],
                    open(self.PATH + '/Influ_Rand_{}_OneEncoding.data'.format(num), 'wb'))
        pickle.dump([self.influ_rand_sets['utility_loss'], self.influ_rand_sets['utility_loss_true']],
                    open(self.PATH + '/Influ_Rand_{}_Deep_TrueLoss.data'.format(num), 'wb'))



class SVM_bin_Utility(object):
    def __init__(self, X_test, y_test, config, seed, **kwargs):
        '''
        :param X:
        :param y:
        :param X_test:
        :param y_test:
        :param config:
        :param seed:
        :param kwargs:
        '''

        if seed is not None:
            np.random.seed(seed)
        self.X_test = X_test
        self.y_test = y_test
        self.config = config
        self.C = 1/self.config['lambda']

    def _testing_loss(self, svm_model):
        w = svm_model.coef_.reshape(-1,1)
        b = svm_model.intercept_.reshape(-1,1)
        y_hat= np.dot(self.X_test, w)+b
        loss = 0
        for i in range(len(y_hat)):
            loss += self.C * np.max((0, 1 - self.y_test[i] * y_hat[i]))
        loss += 1 / 2 * np.linalg.norm(w) ** 2 + 1 / 2 * np.linalg.norm(b) ** 2
        return np.asarray(loss / self.y_test.shape[0]).flatten()[0]

    def _testing_acc(self, svm_model):
        y_pred = svm_model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        return acc

    def _testing_acc_inital(self):
        y_pred = np.zeros(self.y_test.shape[0])
        acc = accuracy_score(self.y_test, y_pred)
        return acc

    def _deep_loss(self, paras):
        w = np.squeeze(paras)[:-1].reshape(-1,1)
        b = np.squeeze(paras)[-1].reshape(-1,1)
        y_hat = np.dot(self.X_test, w) + b
        loss = 0
        for i in range(len(y_hat)):
            loss += self.C *np.max([0, 1-self.y_test[i]*y_hat[i]])
        loss += 1 / 2 * np.linalg.norm(w) ** 2 + 1 / 2 * np.linalg.norm(b) ** 2
        return np.asarray(loss / self.y_test.shape[0]).flatten()[0]

    def _deep_acc(self, paras):
        w = np.squeeze(paras)[:-1]
        b = np.squeeze(paras)[-1]
        y_hat = np.dot(self.X_test, w) + b
        y_pred = np.ones(len(self.y_test))
        for i in range(len(self.y_test)):
            if 1 - self.y_test[i] * y_hat[i] > 0:
                y_pred[i] = -1 * self.y_test[i]
            else:
                y_pred[i] = self.y_test[i]
        acc = accuracy_score(self.y_test, y_pred)
        return acc

    def _model_dist(self, paras, act_model):
        fitted_para = np.concatenate((act_model.coef_.T, act_model.intercept_), axis = None)
        return np.linalg.norm(paras-fitted_para, ord=2)

    @property
    def test_ini_acc(self):
        return self._testing_acc_inital()



