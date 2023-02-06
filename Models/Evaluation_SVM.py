import numpy as np
from tqdm import tqdm
import itertools
import pickle
from sklearn.metrics import accuracy_score, log_loss
from sklearn.svm import LinearSVC
from sklearn import preprocessing
import torch.nn as nn
import torch
from Models.UtilityModel import SVM_bin_Utility
from influence.logistic_reg import bin_svm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import time
from sklearn.metrics import accuracy_score, mean_squared_error
from deltagrad.delta_add_main import *
from deltagrad.delta_del_main import *
from deltagrad.delta_util import *
from copy import deepcopy
import matplotlib.ticker as mtick
#%%

def group_by(a, b):
    # Get argsort indices, to be used to sort a and b in the next steps
    sidx = b.argsort(kind='mergesort')
    a_sorted = a[sidx]
    b_sorted = b[sidx]

    # Get the group limit indices (start, stop of groups)
    cut_idx = np.flatnonzero(np.r_[True,b_sorted[1:] != b_sorted[:-1],True])

    # Split input array with those start, stop ones
    out = [a_sorted[i:j] for i,j in zip(cut_idx[:-1],cut_idx[1:])]
    return out


def para_comp_fig(para_true, para_kkt, para_inf, subset_size, dataset, base_model, n_train):

    dist_kkt, mean_kkt, std_kkt = [], [], []
    # dist_delta, mean_delta, std_delta = [], [], []
    dist_inf, mean_inf, std_inf = [], [], []
    for i in range(len(para_true)):
        dist_kkt.append(np.linalg.norm(np.array(para_kkt[i])-np.array(para_true[i]), ord=2))
        # dist_delta.append(np.linalg.norm(np.array(para_delta[i])-np.array(para_true[i]), ord=2))
        dist_inf.append(np.linalg.norm(np.array(para_inf[i])-np.array(para_true[i]), ord=2))
    mean_dist_KKT = np.mean(np.array(dist_kkt).reshape(10, -1),axis=1)
    mean_dist_INF = np.mean(np.array(dist_inf).reshape(10, -1),axis=1)

    print('KKT parameter testing L2 distance: {}, Std: {}'.format(np.mean(mean_dist_KKT),
                                                                  np.std(mean_dist_KKT)))
    print('Influence parameter testing L2 distance: {}, Std: {}'.format(np.mean(mean_dist_INF),
                                                                  np.std(mean_dist_INF)))

    dist_kkt, mean_kkt, std_kkt = [], [], []
    dist_inf, mean_inf, std_inf = [], [], []
    for i in range(len(para_true)):
        dist_kkt.append(np.linalg.norm(np.array(para_kkt[i])-np.array(para_true[i]), ord=2))
        dist_inf.append(np.linalg.norm(np.array(para_inf[i])-np.array(para_true[i]), ord=2))

    sorted_kkt = group_by(np.array(dist_kkt), np.array(subset_size))
    sorted_inf = group_by(np.array(dist_inf), np.array(subset_size))
    sizes = np.sort(list(set(subset_size)))/n_train*100

    for kkt in sorted_kkt:
        mean_kkt.append(np.mean(kkt))
        std_kkt.append(np.std(kkt))
    mean_kkt, std_kkt = np.array(mean_kkt), np.array(std_kkt)

    for inf in sorted_inf:
        mean_inf.append(np.mean(inf))
        std_inf.append(np.std(inf))
    mean_inf, std_inf = np.array(mean_inf), np.array(std_inf)

    fig, ax = plt.subplots(figsize=(10, 7))#figsize=(20, 20)
    font = {'family':'serif','serif':['Times'], 'weight': 'normal', 'size': 20}
    plt.plot(sizes, mean_inf, label='Influence', color='green', marker='o', markersize=10)
    plt.plot(sizes, mean_kkt, label='OptLearn', color='orange', marker='o', markersize=10)
    plt.fill_between(sizes,mean_kkt+std_kkt, mean_kkt-std_kkt, alpha=0.2, color='orange')
    plt.fill_between(sizes, mean_inf + std_inf, mean_inf - std_inf, alpha=0.2, color='green')

    plt.title('Batch Deletion {} {}'.format(dataset,  base_model))#, fontsize = 15 Deletion

    plt.xticks(np.linspace(50, 100, num=6))
    # plt.xticks(np.linspace(5, 200, num=10))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    plt.ylabel('Distance', fontsize=20)#, fontsize = 15
    plt.xlabel('Size of Remaining Training Subset',fontsize=20)  # , fontsize = 15
    # plt.xlabel('Size of Added Training Subset')#, fontsize = 15
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.legend(loc='upper right', prop = {'size': 20})

    return fig


def para_comp_add_fig(para_true, para_kkt, para_inf, subset_size, dataset, base_model, n_train, n_add):

    dist_kkt, mean_kkt, std_kkt = [], [], []
    dist_inf, mean_inf, std_inf = [], [], []
    for i in range(len(para_true)):
        dist_kkt.append(np.linalg.norm(np.array(para_kkt[i])-np.array(para_true[i]), ord=2))
        dist_inf.append(np.linalg.norm(np.array(para_inf[i])-np.array(para_true[i]), ord=2))

    mean_dist_KKT = np.mean(np.array(dist_kkt).reshape(10, -1),axis=1)
    mean_dist_INF = np.mean(np.array(dist_inf).reshape(10, -1),axis=1)

    print('KKT parameter testing L2 distance: {}, Std: {}'.format(np.mean(mean_dist_KKT),
                                                                  np.std(mean_dist_KKT)))
    print('Influence parameter testing L2 distance: {}, Std: {}'.format(np.mean(mean_dist_INF),
                                                                  np.std(mean_dist_INF)))

    dist_kkt, mean_kkt, std_kkt = [], [], []
    dist_inf, mean_inf, std_inf = [], [], []
    for i in range(len(para_true)):
        dist_kkt.append(np.linalg.norm(np.array(para_kkt[i])-np.array(para_true[i]), ord=2))
        dist_inf.append(np.linalg.norm(np.array(para_inf[i])-np.array(para_true[i]), ord=2))

    sorted_kkt = group_by(np.array(dist_kkt), np.array(subset_size))
    sorted_inf = group_by(np.array(dist_inf), np.array(subset_size))

    subset_size = np.array(subset_size) + n_train
    sizes = np.sort(list(set(subset_size)))/(n_train+n_add)*100

    for kkt in sorted_kkt:
        mean_kkt.append(np.mean(kkt))
        std_kkt.append(np.std(kkt))
    mean_kkt, std_kkt = np.array(mean_kkt), np.array(std_kkt)

    for inf in sorted_inf:
        mean_inf.append(np.mean(inf))
        std_inf.append(np.std(inf))
    mean_inf, std_inf = np.array(mean_inf), np.array(std_inf)

    fig, ax = plt.subplots(figsize=(10, 7))#figsize=(20, 20)
    font = {'family':'serif','serif':['Times'], 'weight': 'normal', 'size': 20}
    plt.plot(sizes, mean_inf, label='Influence', color='green', marker='o', markersize=10)
    plt.plot(sizes, mean_kkt, label='OptLearn', color='orange', marker='o', markersize=10)
    plt.fill_between(sizes,mean_kkt+std_kkt, mean_kkt-std_kkt, alpha=0.2, color='orange')
    plt.fill_between(sizes, mean_inf + std_inf, mean_inf - std_inf, alpha=0.2, color='green')

    plt.title('Batch Addition {} {}'.format(dataset,  base_model))#, fontsize = 15 Deletion

    # plt.xticks(np.linspace(50, 100, num=6))
    plt.xticks(np.linspace(np.min(sizes), np.max(sizes), num=6))
    # plt.xticks(np.linspace(5, 200, num=10))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    plt.ylabel('Distance', fontsize=20)#, fontsize = 15
    plt.xlabel('Size of Added Training Subset', fontsize=20)#, fontsize = 15
    plt.yticks(fontsize = 20)
    plt.xticks(fontsize=20)
    plt.legend(loc='upper right', prop = {'size': 20})

    return fig


def time_comparison(base_time, kkt_time, influ_time):
    base_time = np.array(base_time).reshape(10, -1)
    mean_base_time = np.mean(base_time,axis=1)
    mean_base = np.mean(mean_base_time)
    std_base = np.std(mean_base_time)

    mean_inf = np.mean(np.array(influ_time))
    std_inf = np.std(np.array(influ_time))

    mean_kkt = np.mean(np.array(kkt_time))
    std_kkt = np.std(np.array(kkt_time))

    print('Solver time mean: {}, Std: {}'.format(mean_base,std_base))
    print('KKT time mean: {}, Std: {}'.format(mean_kkt, std_kkt))
    print('Influence time mean: {}, Std: {}'.format(mean_inf, std_inf))


#%%
def all_loss_fig(loss_true, loss_kkt, loss_inf, subset_size):
    mse_kkt, nrmse_kkt, spear_kkt = [], [], []
    for rep in range(10):
        mse_kkt.append(mean_squared_error(np.array(loss_true).reshape(10, -1)[rep], np.array(loss_kkt).reshape(10, -1)[rep]))
        nrmse_kkt.append(np.sqrt(mean_squared_error(np.array(loss_true).reshape(10, -1)[rep], np.array(loss_kkt).reshape(10, -1)[rep]))
                         / (np.max(np.array(loss_true).reshape(10, -1)[rep]) - np.min(np.array(loss_true).reshape(10, -1)[rep])))
        spear_kkt.append(spearmanr(np.array(loss_true).reshape(10, -1)[rep], np.array(loss_kkt).reshape(10, -1)[rep])[0])

    print('KKT utility testing loss MSE: {}, NRMSE: {}, NRMSE std: {}'.format(np.mean(mse_kkt), np.mean(nrmse_kkt), np.std(nrmse_kkt)))
    print('loss spearman: {}, std: {}'.format(np.mean(spear_kkt), np.std(spear_kkt)))


    mse_inf, nrmse_inf, spear_inf = [], [], []
    for rep in range(10):
        mse_inf.append(mean_squared_error(np.array(loss_true).reshape(10, -1)[rep], np.array(loss_inf).reshape(10, -1)[rep]))
        nrmse_inf.append(np.sqrt(mean_squared_error
                                 (np.array(loss_true).reshape(10, -1)[rep] , np.array(loss_inf).reshape(10, -1)[rep]))
                         / (np.max(np.array(loss_true).reshape(10, -1)[rep])- np.min(np.array(loss_true).reshape(10, -1)[rep])))
        spear_inf.append(spearmanr(np.array(loss_true).reshape(10, -1)[rep], np.array(loss_inf).reshape(10, -1)[rep])[0])

    print('Influence utility testing loss MSE: {}, NRMSE: {}, NRMSE std: {}'.format(np.mean(mse_inf), np.mean(nrmse_inf),
                                                                                      np.std(nrmse_inf)))
    print('loss spearman: {}, std: {}'.format(np.mean(spear_inf), np.std(spear_inf)))


    fig, ax = plt.subplots(figsize=(10, 7))#figsize=(20, 20)
    font = {'family':'DejaVu Sans', 'weight': 'normal', 'size': 30}
    plt.rc('font', **font)
    plt.scatter(loss_true, loss_inf, color='green', edgecolors='blue',label = 'Influence function',edgecolor='k',
                marker='o', s=[subset_size[i]**3/3000  for i in range(len(subset_size))], alpha=0.8)
    plt.scatter(loss_true, loss_kkt, color='orange', edgecolors='blue',label = 'OptLearn', edgecolor='k',
                marker='o', s=[subset_size[i]**3/3000  for i in range(len(subset_size))], alpha=0.8)
    plt.plot(loss_true, loss_true, 'r', linewidth=3)
    plt.ylabel("testing loss (predicted)", fontsize = 30)
    plt.xlabel("testing loss (true)", fontsize = 30)
    plt.yticks(fontsize = 30)
    plt.xticks(fontsize=30)
    plt.legend(loc='upper left', prop = {'size': 30})
    # plt.show()


    return fig

#%%
class deletion_eva(object):
    '''
    evaluation the data set deletion
    '''
    def __init__(self, X_train, y_train, X_test, y_test, config, path, seed):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.PATH = path
        self.config = config
        self.C = 1 / self.config['lambda']
        self.feature_dim = self.config['dim']
        self.num_class = config['class_number']
        self.iter = config['iteration']

        # if self.config['class_number'] > 2:
        #     self.y_train_enc = pd.get_dummies(y_train).values # one hot encode
        #     self.y_test_enc = pd.get_dummies(y_test).values  # one hot encode
        #     self.le = preprocessing.LabelEncoder().fit(y_train)

    def _create_random_subsets(self, small):
        subset_index = []
        size_record = []
        n_data = self.X_train.shape[0]
        for i in range(self.iter):
            np.random.seed(i)
            subset_index.append([])
            for size in np.arange(small, self.X_train.shape[0], 5):
                index = np.random.choice(n_data, size, replace=False)
                subset_index[-1].append(index)
                size_record.append(size)

        pickle.dump([subset_index, size_record],
                    open(self.PATH + '/subset_size.data', 'wb'))

        return subset_index, size_record

    def _baseline_result(self, subset_index):
        '''
        record the parameter and the loss on testing
        :param subset_index:
        :return:
        '''
        self.baseline_result = {}
        self.baseline_result['loss'] = []
        self.baseline_result['loss_change'] = []
        self.baseline_result['parameters'] = []
        self.baseline_result['size'] = []
        self.baseline_result['subset'] = []
        self.baseline_result['time'] = []

        model = LinearSVC(max_iter=10000, C=self.C, multi_class='ovr', dual = True,
                          loss = 'squared_hinge', penalty='l2',
                                   random_state=self.config['random_state'])
        # if self.config['class_number'] > 2:
        #     Utility_model = SVM_bin_Utility(self.X_test, self.y_test, self.y_test_enc, self.config, self.seed)
        # else:
        Utility_model = SVM_bin_Utility(self.X_test, self.y_test, self.config, self.seed)
        model.fit(self.X_train, self.y_train)
        loss_all = Utility_model._testing_loss(model)
        start_time = time.time()


        for i in tqdm(range(self.iter)):
            for subset in subset_index[i]:
                X_use, y_use = self.X_train[subset], self.y_train[subset]
                if len(np.unique(y_use)) == self.config['class_number']:
                    start_time1 = time.time()
                    # if self.config['class_number'] > 2:
                    #     y_use_le = self.le.transform(y_use)
                    #     model.fit(X_use, y_use_le)
                    # else:
                    model.fit(X_use, y_use)
                    start_time2 = time.time()
                    self.baseline_result['time'].append(start_time2 - start_time1)
                    # print(start_time2 - start_time1)

                    fitted_para = np.concatenate((model.coef_.reshape(-1, self.feature_dim),
                                                  model.intercept_.reshape(-1, 1)), axis=1).flatten()
                    loss = Utility_model._testing_loss(model)
                    self.baseline_result['subset'].append(subset)
                    self.baseline_result['loss'].append(loss)
                    self.baseline_result['parameters'].append(fitted_para)
                    self.baseline_result['size'].append(len(subset))

                    # loss_change = loss - loss_all
                    loss_change = (loss - loss_all) * self.y_test.shape[0]
                    self.baseline_result['loss_change'].append(loss_change)

        end_time = time.time()

        print('Baseline Time: {}'.format(end_time - start_time))
        avg_time = np.sum(self.baseline_result['time']) / len(self.baseline_result['size'])
        pickle.dump([end_time-start_time, avg_time, self.baseline_result['time']],
                    open(self.PATH+'/baseline_del_time_avg.data', 'wb'))
        pickle.dump([self.baseline_result['subset'], self.baseline_result['loss_change']],
                open(self.PATH + '/baseline_del_subset_losschange.data', 'wb'))
        pickle.dump([self.baseline_result['parameters'], self.baseline_result['loss'], self.baseline_result['size']],
                open(self.PATH + '/baseline_del_para_loss_size.data', 'wb'))


    def _deepset_result(self, subset_index, deepmodel):
        self.deepset_result = {}
        self.deepset_result['loss'] = []
        self.deepset_result['parameters'] = []
        self.deepset_result['size'] = []
        self.deepset_result['time'] = []
        # one_hot_all = []

        # if self.config['class_number'] > 2:
        #     Utility_model = MINST_Utility(self.X_test, self.y_test, self.y_test_enc, self.config, self.seed)
        # else:
        Utility_model = SVM_bin_Utility(self.X_test, self.y_test, self.config, self.seed)

        # start_time = time.time()
        # separate for ten times and record the average time
        rep_length = int(len(subset_index) / 10)
        for i in range(10):
            one_hot_all = []
            for subset in subset_index[rep_length * i:rep_length * (i + 1)]:
                X_use, y_use = self.X_train[subset], self.y_train[subset]
                if len(np.unique(y_use)) == self.config['class_number']:
                    one_hot = np.zeros(len(self.y_train))
                    one_hot[subset] = 1
                    one_hot_all.append(one_hot)

            start_time = time.time()
            X_test_feature = np.array(one_hot_all)
            _ = deepmodel.predict(self.X_train, self.y_train, X_test_feature)
            end_time = time.time()
            self.deepset_result['time'].append((end_time - start_time) / rep_length)

        one_hot_all = []

        for subset in subset_index:
            X_use, y_use = self.X_train[subset], self.y_train[subset]
            if len(np.unique(y_use)) == self.config['class_number']:
                one_hot = np.zeros(len(self.y_train))
                one_hot[subset] = 1
                one_hot_all.append(one_hot)

        start_time = time.time()
        X_test_feature = np.array(one_hot_all)
        estimation_all = deepmodel.predict(self.X_train,
                                           self.y_train, X_test_feature)
        end_time = time.time()
        recorder = 0
        for subset in subset_index:
            X_use, y_use = self.X_train[subset], self.y_train[subset]
            if len(np.unique(y_use)) == self.config['class_number']:
                # start_time1 = time.time()
                # estimation = deepmodel.predict(self.X_train,
                #                                self.y_train, one_hot.reshape(1, -1))
                # # start_time2 = time.time()
                # estimation = np.array(estimation).flatten()
                # # print(start_time2 - start_time1)
                recorder += 1
                estimation = estimation_all[recorder - 1]
                loss = Utility_model._deep_loss(estimation.reshape(1, -1))
                self.deepset_result['loss'].append(loss)
                self.deepset_result['parameters'].append(estimation.flatten())
                self.deepset_result['size'].append(len(subset))

        # end_time = time.time()
        avg_time = (end_time - start_time)/len(self.deepset_result['size'])
        pickle.dump([end_time-start_time, avg_time, self.deepset_result['time']],
                    open(self.PATH+'/KKTUTL_del_time_avg.data', 'wb'))
        print('Deepset Time: {}'.format(end_time - start_time))
        pickle.dump([self.deepset_result['parameters'], self.deepset_result['loss'], self.deepset_result['size']],
                open(self.PATH + '/KKTUTL_del_para_loss_size.data', 'wb'))


    def _influence_result(self, subset_index):
        self.influence_result = {}
        self.influence_result['parameters'] = []
        self.influence_result['loss'] = []
        self.influence_result['size'] = []
        self.influence_result['loss_change'] = []

        start_time = time.time()
        model = LinearSVC(max_iter=10000, C=self.C, multi_class='ovr', dual=True,
                          loss='squared_hinge', penalty='l2',
                          random_state=self.config['random_state'])
        model.fit(self.X_train, self.y_train)
        print(accuracy_score(y_true=self.y_train, y_pred=model.predict(self.X_train)))
        # if self.config['class_number'] > 2:
        #     Utility_model = MINST_Utility(self.X_test, self.y_test, self.y_test_enc, self.config, self.seed)
        #     influ_log = multi_log(1 / self.C)
        #     inf_loss_change_loo = influ_log.get_test_loss_all(model, self.X_train.numpy(), self.y_train_enc,
        #                                                       self.X_test.numpy(), self.y_test_enc)
        # else:
        Utility_model = SVM_bin_Utility(self.X_test, self.y_test, self.config, self.seed)
        influ_log = bin_svm(1 / self.C)
        inf_loss_change_loo = influ_log.get_test_loss_all(model, self.X_train
                                                          , self.y_train,
                                                          self.X_test, self.y_test)
        inf_para_change_loo = influ_log.get_single_para_loo(model, self.X_train, self.y_train)

        loss_all = Utility_model._testing_loss(model)
        para_all = np.concatenate((model.coef_.reshape(-1, self.feature_dim),
                                                  model.intercept_.reshape(-1, 1)), axis=1).flatten()
        # for i in tqdm(range(self.iter)):
        for subset in subset_index:
            X_use, y_use = self.X_train[subset], self.y_train[subset]
            if len(np.unique(y_use)) == self.config['class_number']:
                del_index = np.setdiff1d(np.arange(self.X_train.shape[0]), subset)
                loss_sel = inf_loss_change_loo[del_index]
                loss = (loss_all*self.X_test.shape[0] + np.sum(loss_sel))/ self.X_test.shape[0]
                para_sel = inf_para_change_loo[del_index, :].reshape(-1, para_all.shape[0])
                para = para_all + np.sum(para_sel, axis=0).flatten()
                self.influence_result['loss'].append(loss)
                self.influence_result['size'].append(len(subset))
                self.influence_result['parameters'].append(para)
                loss_change = np.sum(loss_sel)
                self.influence_result['loss_change'].append(loss_change)

        end_time = time.time()
        print('Influence function Time: {}'.format(end_time - start_time))
        avg_time = (end_time - start_time)/len(self.influence_result['size'])
        pickle.dump([end_time-start_time, avg_time],
                    open(self.PATH+'/inlfu_del_time_avg.data', 'wb'))
        pickle.dump([self.influence_result['parameters'], self.influence_result['loss'],
                     self.influence_result['loss_change'], self.influence_result['size']],
                    open(self.PATH + '/inlfu_del_para_loss_change_size.data', 'wb'))
        # return inf_loss_change_loo

class addition_eva(object):
    '''
    evaluation the data set deletion
    '''
    def __init__(self, X_train, y_train, X_test, y_test, X_add, y_add, config, path, seed):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_add = X_add
        self.y_add = y_add
        self.PATH = path
        self.config = config
        self.C = 1 / self.config['lambda']
        self.feature_dim = self.config['dim']
        self.num_class = config['class_number']
        self.iter = config['iteration']

        # if self.config['class_number'] > 2:
        #     self.y_train_enc = pd.get_dummies(y_train).values # one hot encode
        #     self.y_test_enc = pd.get_dummies(y_test).values  # one hot encode
        #     self.y_add_enc = pd.get_dummies(y_add).values  # one hot encode
        #     self.le = preprocessing.LabelEncoder().fit(y_train)

    def _create_random_subsets(self, small):
        subset_index = []
        size_record = []
        n_data = self.X_add.shape[0]
        for i in range(self.iter):
            np.random.seed(i)
            subset_index.append([])
            if self.config['dataset_name'] == 'Iris':
                step = 1
            else:
                step = 5
            for size in np.arange(small, n_data, step):
                index = np.random.choice(n_data, size, replace=False)
                subset_index[-1].append(index)
                size_record.append(size)

        pickle.dump([subset_index, size_record],
                    open(self.PATH + '/subset_size.data', 'wb'))
        return subset_index, size_record


    def _baseline_result(self, subset_index):
        '''
        record the parameter and the loss on testing
        :param subset_index:
        :return:
        '''
        self.baseline_result = {}
        self.baseline_result['loss'] = []
        self.baseline_result['loss_change'] = []
        self.baseline_result['parameters'] = []
        self.baseline_result['size'] = []
        self.baseline_result['subset'] = []
        self.baseline_result['time'] = []

        model = LinearSVC(max_iter=10000, C=self.C, multi_class='ovr', dual=True,
                          loss='squared_hinge', penalty='l2',
                          random_state=self.config['random_state'])

        Utility_model = SVM_bin_Utility(self.X_test, self.y_test, self.config, self.seed)
        model.fit(self.X_train, self.y_train)
        loss_all = Utility_model._testing_loss(model)
        start_time = time.time()

        for i in tqdm(range(self.iter)):
            for subset in subset_index[i]:
                X_add, y_add = self.X_add[subset], self.y_add[subset]
                X_use, y_use = np.concatenate((self.X_train, X_add), axis=0), \
                               np.concatenate((self.y_train, y_add), axis=0)
                if len(np.unique(y_use)) == self.config['class_number']:
                    start_time1 = time.time()
                    model.fit(X_use, y_use)
                    start_time2 = time.time()
                    self.baseline_result['time'].append(start_time2 - start_time1)
                    fitted_para = np.concatenate((model.coef_.reshape(-1, self.feature_dim),
                                                  model.intercept_.reshape(-1, 1)), axis=1).flatten()
                    loss = Utility_model._testing_loss(model)

                    self.baseline_result['subset'].append(subset)
                    self.baseline_result['loss'].append(loss)
                    self.baseline_result['parameters'].append(fitted_para)
                    self.baseline_result['size'].append(subset.shape[0])

                    loss_change = loss - loss_all
                    self.baseline_result['loss_change'].append(loss_change)
        end_time = time.time()

        print('Baseline Time: {}'.format(end_time - start_time))
        avg_time = np.sum(self.baseline_result['time']) / len(self.baseline_result['size'])
        pickle.dump([end_time - start_time, avg_time, self.baseline_result['time']],
                    open(self.PATH + '/baseline_add_time_avg.data', 'wb'))
        pickle.dump([self.baseline_result['subset'], self.baseline_result['loss_change']],
                    open(self.PATH + '/baseline_add_subset_losschange.data', 'wb'))
        pickle.dump([self.baseline_result['parameters'], self.baseline_result['loss'], self.baseline_result['size']],
                open(self.PATH + '/baseline_add_para_loss_size.data', 'wb'))


    def _deepset_result(self, subset_index, deepmodel):
        self.deepset_result = {}
        self.deepset_result['loss'] = []
        self.deepset_result['parameters'] = []
        self.deepset_result['size'] = []
        self.deepset_result['time'] = []

        Utility_model = SVM_bin_Utility(self.X_test, self.y_test, self.config, self.seed)

        # separate for ten times and record the average time
        rep_length = int(len(subset_index)/10)
        for i in range(10):
            one_hot_all = []
            one_hot_org = np.ones(len(self.y_train))
            for subset in subset_index[rep_length*i:rep_length*(i+1)]:
                X_add, y_add = self.X_add[subset], self.y_add[subset]
                X_use, y_use = np.concatenate((self.X_train, X_add), axis=0), np.concatenate((self.y_train, y_add),
                                                                                             axis=0)
                if len(np.unique(y_use)) == self.config['class_number']:
                    one_hot_add = np.zeros(self.y_add.shape[0])
                    one_hot_add[subset] = 1
                    one_hot = np.concatenate((one_hot_org, one_hot_add), axis=0)
                    one_hot_all.append(one_hot)

            X_use, y_use = torch.cat((torch.tensor(self.X_train),
                                      torch.tensor(self.X_add)), dim=0), \
                           torch.cat((torch.tensor(self.y_train),
                                      torch.tensor(self.y_add)), dim=0)

            start_time = time.time()
            X_test_feature = np.array(one_hot_all)
            _ = deepmodel.predict(X_use, y_use, X_test_feature)
            end_time = time.time()
            self.deepset_result['time'].append((end_time-start_time)/rep_length)
        # start_time = time.time()
        one_hot_all = []
        one_hot_org = np.ones(len(self.y_train))

        # for i in tqdm(range(self.iter)):
        for subset in subset_index:
            X_add, y_add = self.X_add[subset], self.y_add[subset]
            X_use, y_use = np.concatenate((self.X_train, X_add), axis=0), np.concatenate((self.y_train, y_add),
                                                                                         axis=0)
            if len(np.unique(y_use)) == self.config['class_number']:
                one_hot_add = np.zeros(self.y_add.shape[0])
                one_hot_add[subset] = 1
                one_hot = np.concatenate((one_hot_org, one_hot_add), axis=0)
                one_hot_all.append(one_hot)

        X_use, y_use = torch.cat((torch.tensor(self.X_train),
                                  torch.tensor(self.X_add)), dim=0),\
                       torch.cat((torch.tensor(self.y_train),
                                  torch.tensor(self.y_add)), dim=0)
        start_time = time.time()
        X_test_feature = np.array(one_hot_all)
        estimation_all = deepmodel.predict(X_use, y_use, X_test_feature)
        end_time = time.time()

        recorder = 0
        for subset in subset_index:
            X_add, y_add = self.X_add[subset], self.y_add[subset]
            X_use, y_use = np.concatenate((self.X_train, X_add), axis=0), np.concatenate((self.y_train, y_add),
                                                                                         axis=0)
            if len(np.unique(y_use)) == self.config['class_number']:

                # # start_time1 = time.time()
                # estimation = deepmodel.predict(X_use,
                #                                y_use, one_hot.reshape(1, -1))
                # # start_time2 = time.time()
                # estimation = np.array(estimation).flatten()
                # # print(start_time2 - start_time1)
                recorder += 1
                estimation = np.array(estimation_all)[recorder - 1].flatten()
                loss = Utility_model._deep_loss(estimation.reshape(1, -1))
                self.deepset_result['loss'].append(loss)
                self.deepset_result['parameters'].append(estimation.flatten())
                # print(subset, subset.shape)
                self.deepset_result['size'].append(len(subset))

        # end_time = time.time()

        print('Deepset Time: {}'.format(end_time - start_time))
        avg_time = (end_time - start_time) / len(self.deepset_result['size'])
        pickle.dump([end_time-start_time, avg_time, self.deepset_result['time']],
                    open(self.PATH+'/KKTUTL_add_time_avg.data', 'wb'))
        pickle.dump([self.deepset_result['parameters'], self.deepset_result['loss'], self.deepset_result['size']],
                open(self.PATH + '/KKTUTL_add_para_loss_size.data', 'wb'))


    def _influence_result(self, subset_index):
        self.influence_result = {}
        self.influence_result['loss'] = []
        self.influence_result['size'] = []
        self.influence_result['parameters'] = []

        start_time = time.time()
        model = LinearSVC(max_iter=10000, C=self.C, multi_class='ovr', dual=True,
                          loss='squared_hinge', penalty='l2',
                          random_state=self.config['random_state'])
        model.fit(self.X_train, self.y_train)
        Utility_model = SVM_bin_Utility(self.X_test, self.y_test, self.config, self.seed)
        influ_log = bin_svm(1 / self.C)
        inf_loss_change_loo = influ_log.get_add_loss_all(model, self.X_train, self.y_train, self.X_add,
                                                             self.y_add,self.X_test, self.y_test)
        inf_para_change_loo = influ_log.get_single_para_add(model, self.X_train, self.y_train,
                                                                self.X_add,  self.y_add)



        loss_all = Utility_model._testing_loss(model)
        para_all = np.concatenate((model.coef_.reshape(-1, self.feature_dim),
                                   model.intercept_.reshape(-1, 1)), axis=1).flatten()
        # for i in tqdm(range(self.iter)):
        for subset in subset_index:
            X_add, y_add = self.X_add[subset], self.y_add[subset]
            X_use, y_use = np.concatenate((self.X_train, X_add), axis=0), np.concatenate((self.y_train, y_add),
                                                                                         axis=0)
            if len(np.unique(y_use)) == self.config['class_number']:
                add_index =subset
                loss_sel = inf_loss_change_loo[add_index]
                loss = (loss_all* self.X_test.shape[0] - np.sum(loss_sel)) / self.X_test.shape[0]
                para_sel = inf_para_change_loo[add_index, :].reshape(-1, para_all.shape[0])
                para = para_all - np.sum(para_sel, axis=0).flatten()
                self.influence_result['loss'].append(loss)
                self.influence_result['size'].append(len(subset))
                self.influence_result['parameters'].append(para)

        end_time = time.time()
        print('Influence function Time: {}'.format(end_time - start_time))
        avg_time = (end_time - start_time)/len(self.influence_result['size'])
        pickle.dump([end_time-start_time, avg_time],
                    open(self.PATH+'/inlfu_add_time_avg.data', 'wb'))
        pickle.dump([self.influence_result['parameters'], self.influence_result['loss'],
                     self.influence_result['size']],
                    open(self.PATH + '/inlfu_add_para_loss_size.data', 'wb'))

