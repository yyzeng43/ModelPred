import numpy as np
from tqdm import tqdm
import itertools
import pickle
from sklearn import datasets as skdataset
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
from Models.ParaModel import Para_KKT_deepset, DeepSet
from Models.Evaluation_SVM import *
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import warnings
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser('OptLearn')
    parser.add_argument('--sampling', default='Perm', type=str, help="sampling method")
    parser.add_argument('--rawdatapath', help="path to retrieve the raw training data")
    parser.add_argument('--modelpath', help="path to retrieve the trained model")
    parser.add_argument('--savepath', help="path to store the results")

    args = parser.parse_args()
    PERMSET = args.sampling
    data_path = args.rawdatapath
    model_path = args.modelpath
    save_path = args.savepath
    torch.cuda.set_device(0)
    np.seterr(all="ignore")
    warnings.filterwarnings('ignore')
    #%% Iris deletion
    config = {}
    config['dataset_name'] = 'Iris'
    config['class_number'] = 2
    config['lambda'] = 1
    config['iteration'] = 10
    config['random_state'] = 33

    ### import data
    iris = skdataset.load_iris()
    X = iris.data
    y = iris.target
    X = X[np.where(y != 2)[0], :]
    y = y[np.where(y != 2)[0]]
    y_1 = 2 * y - 1  # change y from {0,1} to {-1, 1} for the loss computation

    X_mean = np.linalg.norm(X, ord=2, axis=1)
    X_std = X / np.max(X_mean)  # rescale the input features

    SEED = 33
    np.random.seed(SEED)
    X_train, X_test, y_train, y_test = train_test_split(X_std, y_1, test_size=0.33, random_state=SEED)

    config['dim'] = X_train.shape[1]

    SEED = 43
    deletion_evaluation = deletion_eva(X_train, y_train, X_test, y_test, config, save_path, SEED)

    #%% generate random deletion subsets
    Data_Size = X_train.shape[0]
    small_size = int(0.5*Data_Size)
    subset_index, size_record = deletion_evaluation._create_random_subsets(small_size)
    [subset_index, size_record] = pickle.load(open(save_path + '/subset_size.data', 'rb'))

    #%% baseline result
    deletion_evaluation._baseline_result(subset_index)
    [subset_index, true_loss_change] =pickle.load(open(save_path + '/baseline_del_subset_losschange.data', 'rb'))
    #%% deepset result
    Util_dim = 5
    deep_config = {}
    deep_config['set_features'] = 128
    deep_config['hidden_ext'] = 128
    deep_config['hidden_reg'] = 128
    mod_config = {}
    mod_config['util_loss'] = True
    mod_config['deepset'] = 'regressor'
    mod_config['dim'] = config['dim']
    mod_config['model_path'] = None
    mod_config['base_model'] = 'SVM'
    mod_config['dataset'] = 'Iris'
    mod_config['lambda'] = config['lambda']
    mod_config['class_number'] = 2
    mod_config['learning_rate'] = 0.0001

    model_name = '/{}_LR_{}_Del0_KKT_Utl_{}_L1L2_{}_{}_C_{}.state_dict'.\
        format(config['dataset_name'], PERMSET, str(config['util_loss']),
               deep_config['set_features'], deep_config['hidden_ext'],1 / config['lambda'])

    deepmodel = DeepSet(config['dim'], 1, Util_dim, deep_config)
    deepmodel.load_state_dict(torch.load(model_path + model_name))
    Util_deep_mod_load = Para_KKT_deepset(mod_config['dim'], 1,
                                          Util_dim, deep_config, mod_config, model=deepmodel)
    deletion_evaluation._deepset_result(subset_index, Util_deep_mod_load)

    #%% influence result
    rep = 10
    rep_length = int(len(subset_index)/rep)
    influ_time_all = []
    for i in range(rep):
        subset = subset_index[rep_length * i:rep_length * (i + 1)]
        deletion_evaluation._influence_result(subset_index)
        [inf_all, inf_avg] = pickle.load(open(save_path + '/inlfu_del_time_avg.data', 'rb'))
        influ_time_all.append(inf_avg)
    deletion_evaluation._influence_result(subset_index)

    #%% results evaluation

    # baseline
    [para_true, loss_true, subset_size] = pickle.load(open(save_path + '/baseline_del_para_loss_size.data', 'rb'))

    # deepset
    [para_kkt, loss_kkt, subset_kkt] = pickle.load(open(save_path + '/KKTUTL_del_para_loss_size.data', 'rb'))

    # influenced
    [para_inf, loss_inf, inf_change, subset_inf] = pickle.load(open(save_path + '/inlfu_del_para_loss_change_size.data', 'rb'))


    fig_iris = all_loss_fig(loss_true, loss_kkt, loss_inf, subset_size)
    fig_iris.savefig(save_path + '/Loss_compare_del_C1_SVM.png', bbox_inches = 'tight')
    fig_iris.show()

    para_dist = para_comp_fig(para_true, para_kkt,para_inf, subset_size,
                              config['dataset_name'], mod_config['base_model'], X_train.shape[0])
    para_dist.savefig(save_path + '/para_dist_del_C1_SVM.png', bbox_inches = 'tight')
    para_dist.show()


    [exact_all, exact_avg, exact_time_all] = pickle.load(open(save_path + '/baseline_del_time_avg.data', 'rb'))
    [kkt_all, kkt_avg, kkt_time_all] = pickle.load(open(save_path + '/KKTUTL_del_time_avg.data', 'rb'))
    time_comparison(exact_time_all, kkt_time_all, influ_time_all)
    #%% Spam

    config = {}
    config['dataset_name'] = 'spam'
    config['class_number'] = 2
    config['lambda'] = 1
    config['iteration'] = 10
    config['random_state'] = 33

    ### import data
    Data_size = 300
    [X_train, y_train, X_test, y_test] = pickle.load(open(data_path + '/spam_trainXY_testXY_low.data', 'rb'))
    y_train = 2*y_train -1
    y_test = 2*y_test -1
    X_mean = np.linalg.norm(X_train, ord=2, axis=1)
    X_train = X_train / np.max(X_mean)
    X_test = X_test / np.max(X_mean)
    X_train, y_train = X_train[:Data_size], y_train[:Data_size]

    config['dim'] = X_train.shape[1]
    SEED = 43
    deletion_evaluation = deletion_eva(X_train, y_train, X_test, y_test, config, save_path, SEED)
    #%% generate random deletion subsets
    Data_Size = X_train.shape[0]
    small_size = int(0.5 * Data_Size)
    subset_index, size_record = deletion_evaluation._create_random_subsets(small_size)
    [subset_index, size_record] = pickle.load(open(save_path + '/subset_size.data', 'rb'))
    #%% baseline result
    deletion_evaluation._baseline_result(subset_index)
    [subset_index, true_loss_change] =pickle.load(open(save_path + '/baseline_del_subset_losschange.data', 'rb'))
    #%% deepset result
    Util_dim = X_train.shape[1] + 1
    deep_config = {}
    deep_config['set_features'] = 128
    deep_config['hidden_ext'] = 128
    deep_config['hidden_reg'] = 128
    mod_config = {}
    mod_config['util_loss'] = True
    mod_config['deepset'] = 'regressor'
    mod_config['dim'] = config['dim']
    mod_config['model_path'] = None
    mod_config['base_model'] = 'SVM'
    mod_config['dataset'] = 'spam'
    config['dataset_name']= 'spam'
    mod_config['lambda'] = config['lambda']
    mod_config['class_number'] = 2
    mod_config['learning_rate'] = 0.0001

    model_name = '/{}_LR_{}_Del0_KKT_Utl_{}_L1L2_{}_{}_C_{}.state_dict'. \
        format(config['dataset_name'], PERMSET, str(config['util_loss']),
               deep_config['set_features'], deep_config['hidden_ext'], 1 / config['lambda'])

    deepmodel = DeepSet(config['dim'], 1, Util_dim, deep_config)
    deepmodel.load_state_dict(torch.load(model_path + model_name))
    Util_deep_mod_load = Para_KKT_deepset(mod_config['dim'], 1,
                                          Util_dim, deep_config, mod_config, model=deepmodel)
    deletion_evaluation._deepset_result(subset_index, Util_deep_mod_load)

    #%% influence result
    rep = 10
    rep_length = int(len(subset_index)/rep)
    influ_time_all = []
    for i in range(rep):
        subset = subset_index[rep_length * i:rep_length * (i + 1)]
        deletion_evaluation._influence_result(subset_index)
        [inf_all, inf_avg] = pickle.load(open(save_path + '/inlfu_del_time_avg.data', 'rb'))
        influ_time_all.append(inf_avg)
    deletion_evaluation._influence_result(subset_index)
    #%% results evaluation

    # baseline
    [para_true, loss_true, subset_size] = pickle.load(open(save_path + '/baseline_del_para_loss_size.data', 'rb'))

    # deepset
    [para_kkt, loss_kkt, subset_kkt] = pickle.load(open(save_path + '/KKTUTL_del_para_loss_size.data', 'rb'))

    # influence
    [para_inf, loss_inf, inf_change, subset_inf] = pickle.load(open(save_path + '/inlfu_del_para_loss_change_size.data', 'rb'))

    fig_spam = all_loss_fig(loss_true, loss_kkt, loss_inf, subset_size)
    fig_spam.savefig(save_path + '/Loss_compare_del_C1_SVM.png', bbox_inches = 'tight')
    fig_spam.show()

    para_dist = para_comp_fig(para_true, para_kkt, para_inf, subset_size,
                              config['dataset_name'], mod_config['base_model'], X_train.shape[0])
    para_dist.savefig(save_path + '/para_dist_del_C1_SVM.png', bbox_inches = 'tight')
    para_dist.show()

    [exact_all, exact_avg, exact_time_all] = pickle.load(open(save_path + '/baseline_del_time_avg.data', 'rb'))
    [kkt_all, kkt_avg, kkt_time_all] = pickle.load(open(save_path + '/KKTUTL_del_time_avg.data', 'rb'))
    time_comparison(exact_time_all, kkt_time_all, influ_time_all)

    #%% HIGGS
    #
    config = {}
    config['dataset_name'] = 'HIGGS'
    config['class_number'] = 2
    config['lambda'] = 1
    config['iteration'] = 10
    config['random_state'] = 33

    ### import data
    X_train_all = np.load(data_path + '/X_train_all_pos.npy')
    X_test = np.load(data_path + '/X_test_pos.npy')
    y_train_all = np.load(data_path + '/y_train_all_pos.npy')
    y_test = np.load(data_path + '/y_test_pos.npy')
    y_train_all = 2 * y_train_all - 1
    y_test = 2 * y_test - 1

    SEED = 33
    Data_size = 300
    X_train, y_train = X_train_all[:Data_size], y_train_all[:Data_size]

    config['dim'] = X_train.shape[1]
    SEED = 43
    deletion_evaluation = deletion_eva(X_train, y_train, X_test, y_test, config, save_path, SEED)
    #%% generate random deletion subsets
    Data_Size = X_train.shape[0]
    small_size = int(0.5 * Data_Size)
    subset_index, size_record = deletion_evaluation._create_random_subsets(small_size)
    [subset_index, size_record] = pickle.load(open(save_path + '/subset_size.data', 'rb'))
    #%% baseline result
    deletion_evaluation._baseline_result(subset_index)
    [subset_index, true_loss_change] =pickle.load(open(save_path + '/baseline_del_subset_losschange.data', 'rb'))
    #%% deepset result
    Util_dim = X_train.shape[1] + 1
    deep_config = {}
    deep_config['set_features'] = 128
    deep_config['hidden_ext'] = 128
    deep_config['hidden_reg'] = 128
    mod_config = {}
    mod_config['util_loss'] = True
    mod_config['deepset'] = 'regressor'
    mod_config['dim'] = config['dim']
    mod_config['model_path'] = None
    mod_config['base_model'] = 'SVM'
    mod_config['dataset'] = 'HIGGS'
    mod_config['lambda'] = config['lambda']
    mod_config['class_number'] = 2
    mod_config['learning_rate'] = 0.0001

    model_name = '/{}_LR_{}_Del0_KKT_Utl_{}_L1L2_{}_{}_C_{}.state_dict'. \
        format(config['dataset_name'], PERMSET, str(config['util_loss']),
               deep_config['set_features'], deep_config['hidden_ext'], 1 / config['lambda'])

    deepmodel = DeepSet(config['dim'], 1, Util_dim, deep_config)
    deepmodel.load_state_dict(torch.load(model_path + model_name))
    Util_deep_mod_load = Para_KKT_deepset(mod_config['dim'], 1,
                                          Util_dim, deep_config, mod_config, model=deepmodel)
    deletion_evaluation._deepset_result(subset_index, Util_deep_mod_load)

    #%% influence result
    rep = 10
    rep_length = int(len(subset_index)/rep)
    influ_time_all = []
    for i in range(rep):
        subset = subset_index[rep_length * i:rep_length * (i + 1)]
        deletion_evaluation._influence_result(subset_index)
        [inf_all, inf_avg] = pickle.load(open(save_path + '/inlfu_del_time_avg.data', 'rb'))
        influ_time_all.append(inf_avg)
    deletion_evaluation._influence_result(subset_index)

    #%% results evaluation

    # baseline
    [para_true, loss_true, subset_size] = pickle.load(open(save_path + '/baseline_del_para_loss_size.data', 'rb'))

    # deepset
    [para_kkt, loss_kkt, subset_kkt] = pickle.load(open(save_path + '/KKTUTL_del_para_loss_size.data', 'rb'))

    # influence
    [para_inf, loss_inf, inf_change, subset_inf] = pickle.load(open(save_path + '/inlfu_del_para_loss_change_size.data', 'rb'))

    fig_higgs = all_loss_fig(loss_true, loss_kkt, loss_inf, subset_size)
    fig_higgs.savefig(save_path + '/Loss_compare_del_C1_SVM.png', bbox_inches = 'tight')
    fig_higgs.show()

    para_dist = para_comp_fig(para_true, para_kkt, para_inf, subset_size,
                              mod_config['dataset'], mod_config['base_model'], X_train.shape[0])
    para_dist.savefig(save_path + '/para_dist_del_C1_SVM.png', bbox_inches = 'tight')
    para_dist.show()
    #
    #%% compare time and parameter distance
    [exact_all, exact_avg, exact_time_all] = pickle.load(open(save_path + '/baseline_del_time_avg.data', 'rb'))
    [kkt_all, kkt_avg, kkt_time_all] = pickle.load(open(save_path + '/KKTUTL_del_time_avg.data', 'rb'))
    time_comparison(exact_time_all, kkt_time_all, influ_time_all)

