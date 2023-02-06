import numpy as np
from tqdm import tqdm
import itertools
import pickle
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import torch.nn as nn
import torch
from Models.UtilityModel import MINST_Utility, Iris_Utility, SVM_bin_Utility
from scipy.stats import pearsonr, spearmanr
import time
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt

#%%
def all_shapley_fig(shapley_true, shapley_kkt, shapley_utl, shapley_inf = None):
    mse_kkt = mean_squared_error(shapley_true, shapley_kkt)
    nrmse_kkt = np.sqrt(mean_squared_error(shapley_true, shapley_kkt)) / (np.max(shapley_true) - np.min(shapley_true))
    print('KKT Shapley testing loss MSE: {}, NRMSE: {}'.format(mse_kkt, nrmse_kkt))
    pearson_corr = pearsonr(shapley_true, shapley_kkt)
    spearmanr_corr_kkt = spearmanr(shapley_true, shapley_kkt)
    print('loss pearson', pearson_corr[0])
    print('loss spearman', spearmanr_corr_kkt[0])

    mse_utl = mean_squared_error(shapley_true, shapley_utl)
    nrmse_utl = np.sqrt(mean_squared_error(shapley_true, shapley_utl)) / (np.max(shapley_true) - np.min(shapley_true))
    print('Influence utility testing loss MSE: {}, NRMSE: {}'.format(mse_utl, nrmse_utl))
    pearson_corr = pearsonr(shapley_true, shapley_utl)
    spearmanr_corr_utl = spearmanr(shapley_true, shapley_utl)
    print('loss pearson', pearson_corr[0])
    print('loss spearman', spearmanr_corr_utl[0])

    if shapley_inf != None:
        mse_inf = mean_squared_error(shapley_true, shapley_inf)
        nrmse_inf = np.sqrt(mean_squared_error(shapley_true, shapley_inf)) / (np.max(shapley_true) - np.min(shapley_true))
        print('Deltagrad utility testing loss MSE: {}, NRMSE: {}'.format(mse_inf, nrmse_inf))
        pearson_corr = pearsonr(shapley_true, shapley_inf)
        spearmanr_corr = spearmanr(shapley_true, shapley_inf)
        print('loss pearson', pearson_corr[0])
        print('loss spearman', spearmanr_corr[0])


    fig, ax = plt.subplots(figsize=(10, 7))#figsize=(20, 20)
    font = {'family':'serif','serif':['Times'], 'weight': 'normal', 'size': 20}
    plt.rc('font', **font)
    # plt.scatter(shapley_true, loss_inf, color='green', edgecolors='blue',label = 'Influence function',edgecolor='k',
    #             marker='o', s=[subset_size[i]*5  for i in range(len(subset_size))], alpha=0.8)
    plt.scatter(shapley_true, shapley_utl, color='blue',label = 'UtilityPred', edgecolor='k',
                marker='o', s = 200, alpha=0.6)
    plt.scatter(shapley_true, shapley_kkt, color='orange',label = 'DeepSet', edgecolor='k',
                marker='o',s = 200, alpha=0.6)
    plt.plot(shapley_true, shapley_true, 'r', linewidth=1)
    ax.set_ylabel("Shapley Value (predicted)", fontsize = 20)
    ax.set_xlabel("Shapley Value (true)", fontsize = 20)
    plt.legend(loc='upper left', prop = {'size': 15})
    plt.yticks(fontsize = 20)
    plt.xticks(fontsize=20)
    plt.legend()
    # plt.show()

    return fig, nrmse_kkt, spearmanr_corr_kkt[0], nrmse_utl, spearmanr_corr_utl[0]


#%%
class Minst_Shapley(object):
    def __init__(self, X_train,y_train, y_train_enc, X_test, y_test, y_test_enc, config):
        self.config = config
        self.seed = self.config['random_state']
        self.X_train = X_train
        self.y_train = y_train
        self.y_train_enc = y_train_enc
        self.X_test = X_test
        self.y_test = y_test
        self.y_test_enc = y_test_enc
        self.PATH = self.config['save_path']
        self.C = 1/self.config['lambda']
        self.feature_dim = self.config['dim']
        self.num_class = config['class_number']

        self.le = preprocessing.LabelEncoder().fit(y_train)

    def _create_model(self):
        model = LogisticRegression(max_iter=10000, C=self.C, solver='lbfgs', random_state=self.config['random_state'])
        return model


    def _deep_approx_perm(self, deepmodel):
        # calculate the utility estimated by approximate sampling (without training data)
        # directly use deepmodel to predict the estimated parameter
        Tol = self.config['Tolerance']
        Tmax = self.config['MaxIter']


        self.DeepApproxPerm = {}
        self.DeepApproxPerm['utility_loss'] = []
        self.DeepApproxPerm['utility_loss_true']=[]
        self.DeepApproxPerm['Svalue_loss'] = []
        self.DeepApproxPerm['perm'] = []  # record the permutation
        all_one_hot = []

        Utility_model = MINST_Utility(self.X_test, self.y_test, self.y_test_enc, self.config, self.seed)
        n_data = self.y_train.shape[0]  # number of data points
        t = 0
        delta = 100
        # start_time = time.time()
        for t in tqdm(range(Tmax)):
            np.random.seed(t+1234)
            perm = np.random.permutation(n_data)
            self.DeepApproxPerm['perm'].append(perm)
            all_one_hot.append([])

            for i in tqdm(range(n_data)):
                dataid = perm[i]
                datasid = np.asarray(perm[:i + 1]).flatten()
                one_hot = np.zeros(len(self.y_train))
                one_hot[datasid] = 1
                all_one_hot[-1].append(one_hot)

        X_test_feature = np.array(all_one_hot).reshape(-1, self.X_train.shape[0])

        start_time = time.time()
        estimation_all = deepmodel.predict(self.X_train,self.y_train,
                                                   X_test_feature)
        end_time = time.time()

        estimation_all = np.array(estimation_all).reshape((Tmax, self.X_train.shape[0], -1))

        for t in tqdm(range(Tmax)):
            perm =self.DeepApproxPerm['perm'][t]
            self.DeepApproxPerm['utility_loss'].append([-1 * np.log(2)])
            self.DeepApproxPerm['Svalue_loss'].append(np.zeros(n_data))

            for i in tqdm(range(n_data)):
                dataid = perm[i]
                datasid = np.asarray(perm[:i + 1]).flatten()
                X_use, y_use, y_use_org = self.X_train[datasid], \
                                          self.y_train_enc[datasid], self.y_train[datasid]
                if len(np.unique(y_use_org)) < 10:
                    loss =  np.log(2)
                else:
                    estimation = estimation_all[t,i]
                    estimation = np.array(estimation).reshape(1, -1)
                    loss = Utility_model._deep_loss(estimation)

                self.DeepApproxPerm['utility_loss'][-1].append(-1*loss)

                if t > 0:
                    self.DeepApproxPerm['Svalue_loss'][-1][dataid] = t / (t + 1) * self.DeepApproxPerm['Svalue_loss'][-2][
                        dataid] + 1 / (t + 1) * (self.DeepApproxPerm['utility_loss'][-1][-1] -
                                                 self.DeepApproxPerm['utility_loss'][-1][-2])

                else:
                    self.DeepApproxPerm['Svalue_loss'][-1][dataid] = self.DeepApproxPerm['utility_loss'][-1][-1] - \
                                                                 self.DeepApproxPerm['utility_loss'][-1][-2]
            # if t > 5:
            #     delta = np.max(np.std(np.asarray(self.DeepApproxPerm['Svalue_loss'])[-5:, :], axis=0))

            # print('Permutation {} Delta {}'.format(t+1, delta))

            pickle.dump([self.DeepApproxPerm['utility_loss']],
                    open(self.PATH + '/Perm_{}_Iter_{}_DeepKKT_Loss_Iter_{}.data'.format(Tmax, t, Tmax), 'wb'))

            # if t > 500 and delta < Tol:
            #     print('Reaching tolerance at iteration {:d}'.format(t))
            #     break

        # end_time = time.time()
        print('DeepKKT model time:', end_time-start_time)
        pickle.dump([self.DeepApproxPerm['Svalue_loss'], [(end_time-start_time)/(Tmax*n_data)]],
                    open(self.PATH + '/DeepKKT_Shapley_Time_Iter_{}.data'.format(Tmax), 'wb'))




    def _approx_perm(self):
        # calculate the utility estimated by approximate sampling (without training data)
        # directly use deepmodel to predict the estimated parameter
        Tol = self.config['Tolerance']
        Tmax = self.config['MaxIter']

        self.ApproxPerm = {}
        self.ApproxPerm['utility_loss'] = []
        self.ApproxPerm['Svalue_loss'] = []
        self.ApproxPerm['perm'] = []  # record the permutation
        self.ApproxPerm['time'] = []  # record the time

        Utility_model = MINST_Utility(self.X_test, self.y_test, self.y_test_enc, self.config, self.seed)
        n_data = self.y_train.shape[0]  # number of data points
        t = 0
        delta = 100

        # start_time = time.time()
        for t in tqdm(range(Tmax)):
            np.random.seed(t + 1234)
            perm = np.random.permutation(n_data)
            self.ApproxPerm['perm'].append(perm)
            self.ApproxPerm['utility_loss'].append([-1 * np.log(2)])
            self.ApproxPerm['Svalue_loss'].append(np.zeros(n_data))


            for i in tqdm(range(n_data)):
                dataid = perm[i]
                datasid = np.asarray(perm[:i + 1]).flatten()
                one_hot = np.ones(len(self.y_train)) * (-1)
                one_hot[datasid] = 1

                X_use, y_use, y_use_org = self.X_train[datasid], self.y_train_enc[datasid], self.y_train[datasid]
                if len(np.unique(y_use_org)) < 10:

                    true_loss = np.log(2)
                    # self.ApproxPerm['fitted_para'][-1].append(np.zeros(self.feature_dim + 1))

                else:
                    start_time = time.time()
                    model = self._create_model()
                    y_use_le = self.le.transform(y_use_org)
                    model.fit(X_use, y_use_le)
                    end_time = time.time()
                    true_loss = Utility_model._testing_loss(model)
                    # fitted_para = np.concatenate((model.coef_, model.intercept_.reshape(-1, 1)), axis=1).flatten()
                    # self.ApproxPerm['fitted_para'][-1].append(fitted_para.reshape((10, 129)))
                    self.ApproxPerm['time'].append(end_time - start_time)
                self.ApproxPerm['utility_loss'][-1].append(-1 * true_loss)

                if t > 0:
                    self.ApproxPerm['Svalue_loss'][-1][dataid] = \
                        t / (t + 1) * self.ApproxPerm['Svalue_loss'][-2][dataid] \
                        + 1 / (t + 1) * (self.ApproxPerm['utility_loss'][-1][-1] -
                                         self.ApproxPerm['utility_loss'][-1][-2])

                else:
                    self.ApproxPerm['Svalue_loss'][-1][dataid] = self.ApproxPerm['utility_loss'][-1][-1] - \
                                                                     self.ApproxPerm['utility_loss'][-1][-2]

            if t > 5:
                delta = np.max(np.std(np.asarray(self.ApproxPerm['Svalue_loss'])[-5:, :], axis=0))
            print('Permutation {} Delta {}'.format(t+1, delta))

            pickle.dump([self.ApproxPerm['utility_loss']], #, self.ApproxPerm['fitted_para']
                        open(self.PATH + '/Perm_{}_Iter_{}_Base_Loss_Iter_{}.data'.format(Tmax, t, Tmax), 'wb'))

            # if delta < Tol:
            #     print('Reaching tolerance at iteration {:d}'.format(t))
            #     break
            print('Base model time:',self.ApproxPerm['time'][-1])

        # end_time = time.time()
        pickle.dump([self.ApproxPerm['Svalue_loss'], [np.sum(np.array(self.ApproxPerm['time']))/(Tmax*n_data)]],
                    open(self.PATH + '/Base_Shapley_Time_Iter_{}.data'.format(Tmax), 'wb'))


    def _utl_approx_perm(self, deepmodel):
        # calculate the utility estimated by approximate sampling (without training data)
        # directly use deepmodel to predict the estimated parameter
        Tmax = self.config['MaxIter']
        Tol = self.config['Tolerance']

        self.UtlApproxPerm = {}
        self.UtlApproxPerm['utility_loss'] = []
        self.UtlApproxPerm['Svalue_loss'] = []
        self.UtlApproxPerm['perm'] = []  # record the permutation
        all_one_hot = []

        n_data = self.y_train.shape[0]  # number of data points
        t = 0
        delta = 100

        # start_time = time.time()
        for t in tqdm(range(Tmax)):
            np.random.seed(t + 1234)
            perm = np.random.permutation(n_data)
            self.UtlApproxPerm['perm'].append(perm)
            all_one_hot.append([])

            for i in tqdm(range(n_data)):
                dataid = perm[i]
                datasid = np.asarray(perm[:i + 1]).flatten()
                one_hot = np.zeros(len(self.y_train))
                one_hot[datasid] = 1
                all_one_hot[-1].append(one_hot)

        X_test_feature = np.array(all_one_hot).reshape(-1, self.X_train.shape[0])

        start_time = time.time()
        estimation_all = deepmodel.predict(self.X_train, self.y_train_enc,
                                           X_test_feature)
        end_time = time.time()

        estimation_all = np.squeeze(np.array(estimation_all)).reshape((Tmax, self.X_train.shape[0]))

        for t in tqdm(range(Tmax)):
            np.random.seed(t+1234)
            perm = self.UtlApproxPerm['perm'][t]
            self.UtlApproxPerm['utility_loss'].append([-1 * np.log(2)])
            self.UtlApproxPerm['Svalue_loss'].append(np.zeros(n_data))

            for i in range(n_data):
                dataid = perm[i]
                datasid = np.asarray(perm[:i + 1]).flatten()

                X_use, y_use, y_use_org = self.X_train[datasid], self.y_train_enc[datasid], self.y_train[datasid]
                if len(np.unique(y_use_org)) < 10:
                    loss =  np.log(2)

                else:
                    loss = estimation_all[t,i]

                self.UtlApproxPerm['utility_loss'][-1].append(-1*loss)

                if t > 0:
                    self.UtlApproxPerm['Svalue_loss'][-1][dataid] = t / (t + 1) * self.UtlApproxPerm['Svalue_loss'][-2][
                        dataid] + 1 / (t + 1) * (self.UtlApproxPerm['utility_loss'][-1][-1] -
                                                 self.UtlApproxPerm['utility_loss'][-1][-2])

                else:
                    self.UtlApproxPerm['Svalue_loss'][-1][dataid] = self.UtlApproxPerm['utility_loss'][-1][-1] - \
                                                                 self.UtlApproxPerm['utility_loss'][-1][-2]

            # if t > 5:
            #     delta = np.max(np.std(np.asarray(self.UtlApproxPerm['Svalue_loss'])[-5:, :], axis=0))
            # print('Permutation {} Delta {}'.format(t+1, delta))

            pickle.dump([self.UtlApproxPerm['utility_loss']],
                    open(self.PATH + '/Perm_{}_Iter_{}_DeepUtl_Loss_Iter_{}.data'.format(Tmax, t, Tmax), 'wb'))

            # if delta < Tol:
            #     print('Reaching tolerance at iteration {:d}'.format(t))
            #     break

        end_time = time.time()
        print('DeepUtl model avg time:', (end_time-start_time)/(Tmax*n_data))
        pickle.dump([self.UtlApproxPerm['Svalue_loss'], [(end_time-start_time)/(Tmax*n_data)]],
                    open(self.PATH + '/DeepUtl_Shapley_Time_Iter_{}.data'.format(Tmax), 'wb'))


class Iris_Shapley(object):
    def __init__(self, X_train, y_train, X_test, y_test, config):
        self.config = config
        self.seed = self.config['random_state']
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.PATH = self.config['save_path']
        self.C = 1/self.config['lambda']
        self.feature_dim = self.config['dim']
        self.num_class = config['class_number']

        max_loss = np.log(2)
        epsilon = 1 / (len(y_test))
        delta2 = 0.05
        r = 2 * max_loss
        self.Itermax = int(np.ceil((r ** 2) / (2 * (epsilon ** 2)) * np.log(len(y_test) / delta2)))


    def _create_model(self):
        model = LogisticRegression(max_iter=10000, C = self.C, solver='lbfgs')
        return model

    def _deep_approx_perm(self, deepmodel):
        # calculate the utility estimated by approximate sampling (without training data)
        # directly use deepmodel to predict the estimated parameter
        Tol = self.config['Tolerance']
        Tmax = self.config['MaxIter']

        self.DeepApproxPerm = {}
        self.DeepApproxPerm['utility_loss'] = []
        self.DeepApproxPerm['utility_loss_true']=[]
        self.DeepApproxPerm['Svalue_loss'] = []
        self.DeepApproxPerm['perm'] = []  # record the permutation
        all_one_hot = []


        Utility_model = Iris_Utility(self.X_test, self.y_test, self.config, self.seed)
        n_data = self.y_train.shape[0]  # number of data points
        t = 0
        delta = 100

        # start_time = time.time()
        for t in tqdm(range(Tmax)):
            np.random.seed(t + 1234)
            perm = np.random.permutation(n_data)
            self.DeepApproxPerm['perm'].append(perm)
            all_one_hot.append([])

            for i in tqdm(range(n_data)):
                datasid = np.asarray(perm[:i + 1]).flatten()
                one_hot = np.zeros(len(self.y_train))
                one_hot[datasid] = 1
                all_one_hot[-1].append(one_hot)

        X_test_feature = np.array(all_one_hot).reshape(-1, self.X_train.shape[0])

        start_time = time.time()
        estimation_all = deepmodel.predict(self.X_train, self.y_train,
                                           X_test_feature)
        end_time = time.time()

        estimation_all = np.array(estimation_all).reshape((Tmax, self.X_train.shape[0], -1))
        print('Finish Inference')

        for t in tqdm(range(Tmax)):
            np.random.seed(t + 1234)
            perm = self.DeepApproxPerm['perm'][t]
            self.DeepApproxPerm['utility_loss'].append([-1 * np.log(2)])
            self.DeepApproxPerm['Svalue_loss'].append(np.zeros(n_data))

            for i in tqdm(range(n_data)):
                dataid = perm[i]
                datasid = np.asarray(perm[:i + 1]).flatten()
                X_use, y_use= self.X_train[datasid], self.y_train[datasid]
                if len(np.unique(y_use)) == 1:
                    loss = np.log(2)
                else:
                    estimation = estimation_all[t, i, :]
                    estimation = estimation.reshape(1, -1)
                    loss = Utility_model._deep_loss(estimation)

                self.DeepApproxPerm['utility_loss'][-1].append(-1 * loss)

                if t > 0:
                    self.DeepApproxPerm['Svalue_loss'][-1][dataid] = \
                        t / (t + 1) * self.DeepApproxPerm['Svalue_loss'][-2][dataid]+ 1 / (t + 1) \
                        * (self.DeepApproxPerm['utility_loss'][-1][-1] -self.DeepApproxPerm['utility_loss'][-1][-2])

                else:
                    self.DeepApproxPerm['Svalue_loss'][-1][dataid] = self.DeepApproxPerm['utility_loss'][-1][-1] - \
                                                                     self.DeepApproxPerm['utility_loss'][-1][-2]
            # if t > 5:
            #     delta = np.max(np.std(np.asarray(self.DeepApproxPerm['Svalue_loss'])[-5:, :], axis=0))

            # print('Permutation {} Delta {}'.format(t+1, delta))

            pickle.dump([self.DeepApproxPerm['utility_loss']],
                        open(self.PATH + '/Perm_{}_Iter_{}_DeepKKT_Loss_Iter_{}.data'.format(Tmax, t, Tmax), 'wb'))

            # if t > 500 and delta < Tol:
            #     print('Reaching tolerance at iteration {:d}'.format(t))
            #     break

        # end_time = time.time()
        print('DeepKKT model time:', end_time - start_time)
        pickle.dump([self.DeepApproxPerm['Svalue_loss'], [(end_time - start_time) / (Tmax * n_data)]],
                    open(self.PATH + '/DeepKKT_Shapley_Time_Iter_{}.data'.format(Tmax), 'wb'))


    def _approx_perm(self):
        # calculate the utility estimated by approximate sampling (without training data)
        # directly use deepmodel to predict the estimated parameter
        Tol = self.config['Tolerance']
        Tmax = self.config['MaxIter']

        self.ApproxPerm = {}
        self.ApproxPerm['utility_loss'] = []
        self.ApproxPerm['Svalue_loss'] = []
        self.ApproxPerm['perm'] = []  # record the permutation
        self.ApproxPerm['time'] = []


        Utility_model = Iris_Utility(self.X_test, self.y_test, self.config, self.seed)
        n_data = self.y_train.shape[0]  # number of data points
        t = 0
        delta = 100

        # start_time = time.time()
        for t in tqdm(range(Tmax)):
            np.random.seed(t + 1234)
            perm = np.random.permutation(n_data)
            self.ApproxPerm['perm'].append(perm)
            self.ApproxPerm['utility_loss'].append([-1 * np.log(2)])
            self.ApproxPerm['Svalue_loss'].append(np.zeros(n_data))

            for i in range(n_data):
                dataid = perm[i]
                datasid = np.asarray(perm[:i + 1]).flatten()
                one_hot = np.ones(len(self.y_train)) * (-1)
                one_hot[datasid] = 1

                X_use, y_use = self.X_train[datasid], self.y_train[datasid]
                if len(np.unique(y_use)) == 1:
                    true_loss = np.log(2)
                else:
                    start_time = time.time()
                    model = self._create_model()
                    model.fit(X_use, y_use)
                    end_time = time.time()
                    self.ApproxPerm['time'].append(end_time-start_time)

                    true_loss = Utility_model._testing_loss(model)
                    fitted_para = np.concatenate((model.coef_.T, model.intercept_), axis=None)

                self.ApproxPerm['utility_loss'][-1].append(-1 * true_loss)

                if t > 0:
                    self.ApproxPerm['Svalue_loss'][-1][dataid] = \
                        t / (t + 1) * self.ApproxPerm['Svalue_loss'][-2][dataid] \
                        + 1 / (t + 1) * (self.ApproxPerm['utility_loss'][-1][-1] -
                                         self.ApproxPerm['utility_loss'][-1][-2])

                else:
                    self.ApproxPerm['Svalue_loss'][-1][dataid] = self.ApproxPerm['utility_loss'][-1][-1] - \
                                                                     self.ApproxPerm['utility_loss'][-1][-2]

            # if t > 5:
            #     delta = np.max(np.std(np.asarray(self.ApproxPerm['Svalue_loss'])[-5:, :], axis=0))
            # print('Permutation {} Delta {}'.format(t+1, delta))

            pickle.dump([self.ApproxPerm['utility_loss']],
                        open(self.PATH + '/Perm_{}_Iter_{}_Base_Loss_Iter_{}.data'.format(Tmax, t, Tmax), 'wb'))
            # if delta < Tol:
            #     print('Reaching tolerance at iteration {:d}'.format(t))
            #     break

        # end_time = time.time()
        print('Base model avg time:', np.sum(np.array(self.ApproxPerm['time']))/(Tmax*n_data))
        pickle.dump([self.ApproxPerm['Svalue_loss'], [np.sum(np.array(self.ApproxPerm['time']))/(Tmax*n_data)]],
                    open(self.PATH + '/Base_Shapley_Time_Iter_{}.data'.format(Tmax), 'wb'))



    def _utl_approx_perm(self, deepmodel):
        # calculate the utility estimated by approximate sampling (without training data)
        # directly use deepmodel to predict the estimated parameter
        Tmax = self.config['MaxIter']
        Tol = self.config['Tolerance']

        self.UtlApproxPerm = {}
        self.UtlApproxPerm['utility_loss'] = []
        self.UtlApproxPerm['Svalue_loss'] = []
        self.UtlApproxPerm['perm'] = []  # record the permutation

        all_one_hot = []

        n_data = self.y_train.shape[0]  # number of data points
        t = 0
        delta = 100

        # start_time = time.time()
        for t in tqdm(range(Tmax)):
            np.random.seed(t + 1234)
            perm = np.random.permutation(n_data)
            self.UtlApproxPerm['perm'].append(perm)
            all_one_hot.append([])

            for i in tqdm(range(n_data)):
                dataid = perm[i]
                datasid = np.asarray(perm[:i + 1]).flatten()
                one_hot = np.zeros(len(self.y_train))
                one_hot[datasid] = 1
                all_one_hot[-1].append(one_hot)

        X_test_feature = np.array(all_one_hot).reshape(-1, self.X_train.shape[0])

        start_time = time.time()
        estimation_all = deepmodel.predict(self.X_train, self.y_train,
                                           X_test_feature)
        end_time = time.time()

        estimation_all = np.squeeze(np.array(estimation_all)).reshape((Tmax, self.X_train.shape[0]))

        for t in tqdm(range(Tmax)):
            np.random.seed(t + 1234)
            perm = self.UtlApproxPerm['perm'][t]
            self.UtlApproxPerm['utility_loss'].append([-1 * np.log(2)])
            self.UtlApproxPerm['Svalue_loss'].append(np.zeros(n_data))

            for i in range(n_data):
                dataid = perm[i]
                datasid = np.asarray(perm[:i + 1]).flatten()
                X_use, y_use = self.X_train[datasid], self.y_train[datasid]
                if len(np.unique(y_use)) == 1:
                    loss =  np.log(2)


                else:
                    loss = estimation_all[t,i]

                self.UtlApproxPerm['utility_loss'][-1].append(-1*loss)



                if t > 0:
                    self.UtlApproxPerm['Svalue_loss'][-1][dataid] = t / (t + 1) * self.UtlApproxPerm['Svalue_loss'][-2][
                        dataid] + 1 / (t + 1) * (self.UtlApproxPerm['utility_loss'][-1][-1] -
                                                 self.UtlApproxPerm['utility_loss'][-1][-2])

                else:
                    self.UtlApproxPerm['Svalue_loss'][-1][dataid] = self.UtlApproxPerm['utility_loss'][-1][-1] - \
                                                                 self.UtlApproxPerm['utility_loss'][-1][-2]

            # if t > 5:
            #     delta = np.max(np.std(np.asarray(self.UtlApproxPerm['Svalue_loss'])[-5:, :], axis=0))
            # print('Permutation {} Delta {}'.format(t+1, delta))

            pickle.dump([self.UtlApproxPerm['utility_loss']],
                    open(self.PATH + '/Perm_{}_Iter_{}_DeepUtl_Loss_Iter_{}.data'.format(Tmax, t, Tmax), 'wb'))

            # if delta < Tol:
            #     print('Reaching tolerance at iteration {:d}'.format(t))
            #     break

        # end_time = time.time()
        print('DeepUtl model time:', end_time-start_time)
        pickle.dump([self.UtlApproxPerm['Svalue_loss'], [(end_time-start_time)/(Tmax*n_data)]],
                    open(self.PATH + '/DeepUtl_Shapley_Time_Iter_{}.data'.format(Tmax), 'wb'))



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
                    open(self.PATH + '/Influ_Rand_{}_OneEncoding_Iter_{}.data'.format(num, self.config['MaxIter']), 'wb'))
        pickle.dump([self.influ_rand_sets['utility_loss'], self.influ_rand_sets['utility_loss_true']],
                    open(self.PATH + '/Influ_Rand_{}_Deep_TrueLoss_Iter_{}.data'.format(num, self.config['MaxIter']), 'wb'))


