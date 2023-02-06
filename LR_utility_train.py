import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
import torch
import pickle
from Models.ParaModel import Para_deepset, DeepSet
import sys
from scipy.stats import pearsonr, spearmanr
import pandas as pd
from tqdm import tqdm
from Models.UtilityModel import MINST_Utility, Minst_Shapley, Iris_Utility
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser('OptLearn')
    parser.add_argument('--sampling', default='Perm', type=str, help="sampling method")
    parser.add_argument('--rawdatapath', help="path to retrieve the raw training data")
    parser.add_argument('--datapath', help="repository to store the training data")
    parser.add_argument('--modelpath', help="repository to store the trained model")

    args = parser.parse_args()
    PERMSET = args.sampling
    data_path = args.rawdatapath
    save_path = args.datapath
    result_path = args.modelpath

    torch.cuda.set_device(0)

    SEED = 33
    clambda = 1
    #%% MNIST

    [train_feature, test_feature] = pickle.load(open(data_path + '/minst_feature_train_test.data', 'rb'))
    [train_label, test_label] = pickle.load(open(data_path + '/minst_label_train_test.data', 'rb'))

    train_feature_all =  torch.cat(train_feature, dim=0).cpu()
    test_feature_all = torch.cat(test_feature, dim=0).cpu()
    train_label_all, test_label_all = torch.cat(train_label, dim=0), torch.cat(test_label, dim=0).cpu()
    X_mean = np.linalg.norm(train_feature_all, ord=2, axis=1)
    x_train = train_feature_all/np.max(X_mean)
    x_test = test_feature_all/np.max(X_mean)
    y_train, y_test = train_label_all, test_label_all

    x_train, x_test= x_train.cpu(), x_test.cpu()
    y_train, y_test = y_train.cpu(), y_test.cpu()

    # get training data
    clambda = 1
    Data_Size = 300
    Test_Size = 500
    Perm_NUM = 100
    NUM_SAMPLE = 30000
    PERMSET = 'Perm'
    config = {}
    config['saving_path'] = save_path
    config['lambda'] = clambda  # 1 0.01
    config['class_number'] = 10
    config['rand_state'] = 32

    y_train_enc, y_test_enc = pd.get_dummies(y_train).values[:Data_Size, :], pd.get_dummies(y_test).values[:Test_Size,:]
    train_data, train_lab_org, train_lab = x_train[:Data_Size, :], y_train[:Data_Size], y_train_enc[:Data_Size,:]
    x_test, y_test = x_test[:Test_Size],  y_test[:Test_Size]

    num_features = x_train.shape[1]  # 128
    if PERMSET == 'Perm':
        PREM, Iter = 100, 100
        one_hot = []
        parameters = []
        for i in range(Iter):
            [x_one_hot, LR_para] = pickle.load(
                open(save_path + '/Perm_{}_Iter_{}_OneEncoding_FittedPara.data'.format(PREM, i), 'rb'))
            one_hot.append(x_one_hot)
            parameters.append([listpara.flatten() for listpara in LR_para])
        one_hot = np.array(one_hot).reshape((-1, x_one_hot[0].shape[0]))
        parameters = np.array(parameters).reshape((-1, parameters[0][0].shape[0]))

    one_hot = np.array(one_hot)
    parameters = np.squeeze(np.array(parameters))
    Util_dim = parameters.shape[-1]

    zero_index = np.where(~parameters.any(axis=1))[0]
    remaining = [i for i in range(parameters.shape[0]) if i not in list(zero_index)]
    one_hot = one_hot[remaining]
    parameters = parameters[remaining]
    # pickle.dump([one_hot, parameters],
    #             open(data_path + '/{}_{}_OneEncoding_FittedPara_N0.data'.format(PERMSET, NUM_SAMPLE), 'wb'))

    train_ratio = 0.5
    valid_ratio = 0.1
    train_end_index = int(train_ratio * one_hot.shape[0])
    valid_end_index = int((train_ratio + valid_ratio) * one_hot.shape[0])

    X_feature_train = one_hot[:train_end_index]
    X_feature_valid = one_hot[train_end_index:valid_end_index, :]
    X_feature_test = one_hot[valid_end_index:, :]

    Y_feature_train = parameters[:train_end_index]
    Y_feature_valid = parameters[train_end_index:valid_end_index]
    Y_feature_test = parameters[valid_end_index:]

    utility_evaluation = MINST_Utility(train_data, train_lab, y_train_enc, config, SEED)

    loss_true = []
    for i in range(Y_feature_train.shape[0]):
        loss_true.append(utility_evaluation._deep_loss(paras=Y_feature_train[i]))
    np.save(data_path + '/utility_train.npy', loss_true)

    loss_valid = []
    for i in range(Y_feature_valid.shape[0]):
        loss_valid.append(utility_evaluation._deep_loss(paras=Y_feature_valid[i]))
    np.save(data_path + '/utility_valid.npy', loss_valid)

    # Y_feature = np.array(loss_true).reshape(-1, train_data.shape[0])
    Y_feature_train = np.array(loss_true)
    Y_feature_valid = np.array(loss_valid)
    # Y_feature_test = Y_feature[valid_end_index:]

    train_set = (X_feature_train, Y_feature_train)
    valid_set = (X_feature_valid, Y_feature_valid)
    # test_set = (X_feature_test, Y_feature_test)

    X_dim = train_data.shape[1]
    Util_dim = 1

    config = {}
    config['lambda'] = clambda
    config['dataset'] = 'MNIST'
    config['learning_rate'] = 0.0001
    config['base_model'] = 'Logistic Regression'
    config['model_path'] = result_path

    deep_config = {}
    deep_config['set_features'] = 128
    deep_config['hidden_ext'] = 128
    deep_config['hidden_reg'] = 128
    n_epoch = 15

    Util_deep_mod = Para_deepset(X_dim, 10, Util_dim, deep_config, config)  # no KKT, predict on parameter
    Util_deep_mod.fit(train_data, train_lab, train_set, valid_set, 1 / config['lambda'], n_epoch)
    model_savename = result_path + '/UtlModel_Del0_Nepoch{}_L1L2_{}_{}_C_{}.state_dict'.format(
        n_epoch,
        deep_config['set_features'],
        deep_config['hidden_ext'], config['lambda'])
    sys.stdout.flush()
    torch.save(Util_deep_mod.model.state_dict(), model_savename)

    model = DeepSet(X_dim, 10, Util_dim, deep_config)
    model.load_state_dict(torch.load(model_savename))
    Util_deep_mod_load = Para_deepset(X_dim, 1, Util_dim, deep_config, config, model=model)

    #%% Iris
    PERMSET = 'Perm'
    config = {}
    config['saving_path'] = None
    config['lambda'] = clambda  # 1 0.01
    config['class_number'] = 2
    config['rand_state'] = 32

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X = X[np.where(y != 2)[0], :]
    y = y[np.where(y != 2)[0]]
    X_mean = np.linalg.norm(X, ord=2, axis=1)
    X_std = X / np.max(X_mean)  # rescale the input features
    SEED = 33
    np.random.seed(SEED)
    X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.33, random_state=SEED)

    # get training data
    if PERMSET == 'Perm':
        num_features = X_train.shape[1]  # 128
        PREM, Iter = 300, 300
        one_hot = []
        parameters = []
        for i in range(Iter):
            [x_one_hot, LR_para] = pickle.load(
                open(save_path + '/Perm_{}_Iter_{}_OneEncoding_FittedPara.data'.format(PREM, i), 'rb'))
            one_hot.append(x_one_hot)
            parameters.append([listpara.flatten() for listpara in LR_para])
        x_one_hot_approx = np.array(one_hot).reshape((-1, x_one_hot[0].shape[0]))
        estimated_paras = np.array(parameters).reshape((-1, parameters[0][0].shape[0]))

    X_dim = X_std.shape[1]
    utility_evaluation = Iris_Utility(X_train, y_train, config, SEED)
    loss_true = []
    true_paras = np.array(estimated_paras).reshape(-1, X_std.shape[1]+1)
    for i in range(true_paras.shape[0]):
        loss_true.append(utility_evaluation._deep_loss(paras=true_paras[i]))
    Y_feature = np.array(loss_true)
    Util_dim = 1

    train_ratio = 0.5
    valid_ratio = 0.1

    train_end_index = int(train_ratio * np.array(estimated_paras).shape[0])
    valid_end_index = int((train_ratio + valid_ratio) * np.array(estimated_paras).shape[0])

    X_feature_train = np.array(x_one_hot_approx)[:train_end_index, :].reshape(-1, X_train.shape[0])
    X_feature_valid = np.array(x_one_hot_approx)[train_end_index:valid_end_index, :].reshape(-1, X_train.shape[0])
    X_feature_test = np.array(x_one_hot_approx)[valid_end_index:, :].reshape(-1, X_train.shape[0])

    Y_feature_train = Y_feature[:train_end_index]
    Y_feature_valid = Y_feature[train_end_index:valid_end_index]
    Y_feature_test = Y_feature[valid_end_index:]

    train_set = (X_feature_train, Y_feature_train)
    valid_set = (X_feature_valid, Y_feature_valid)
    test_set = (X_feature_test, Y_feature_test)

    config = {}
    config['lambda'] = clambda
    config['dataset'] = 'Iris'
    config['learning_rate'] = 0.0001
    config['base_model'] = 'Logistic Regression'
    config['model_path'] = result_path

    deep_config = {}
    deep_config['set_features'] = 128
    deep_config['hidden_ext'] = 128
    deep_config['hidden_reg'] = 128
    n_epoch = 15

    Util_deep_mod = Para_deepset(X_dim, 1, Util_dim, deep_config, config)  # no KKT, predict on parameter
    Util_deep_mod.fit(X_train, y_train, train_set, valid_set, 1 / config['lambda'], n_epoch)
    model_savename = result_path + '/UtlModel_Del0_Nepoch{}_L1L2_{}_{}_C_{}.state_dict'.format(
        n_epoch,
        deep_config['set_features'],
        deep_config['hidden_ext'], config['lambda'])
    sys.stdout.flush()
    torch.save(Util_deep_mod.model.state_dict(), model_savename)

    model = DeepSet(X_dim, 1, Util_dim, deep_config)
    model.load_state_dict(torch.load(model_savename))
    Util_deep_mod_load = Para_deepset(X_dim, 1, Util_dim, deep_config, config, model=model)

    #%% spam
    PERMSET = 'Perm'
    config = {}
    config['saving_path'] = None
    config['lambda'] = clambda  # 1 0.01
    config['class_number'] = 2
    config['rand_state'] = 32

    Data_size = 300
    [X_train, y_train, X_test, y_test] = pickle.load(open(data_path + '/spam_trainXY_testXY_low.data', 'rb'))
    X_mean = np.linalg.norm(X_train, ord=2, axis=1)
    X_train = X_train / np.max(X_mean)
    X_test = X_test / np.max(X_mean)
    X_train, y_train = X_train[:Data_size], y_train[:Data_size]

    # get training data
    if PERMSET == 'Perm':
        num_features = X_train.shape[1]  # 128
        PREM, Iter = 100, 100
        one_hot = []
        parameters = []
        for i in range(Iter):
            [x_one_hot, LR_para] = pickle.load(
                open(save_path + '/Perm_{}_Iter_{}_OneEncoding_FittedPara.data'.format(PREM, i), 'rb'))
            one_hot.append(x_one_hot)
            parameters.append([listpara.flatten() for listpara in LR_para])
        x_one_hot_approx = np.array(one_hot).reshape((-1, x_one_hot[0].shape[0]))
        estimated_paras = np.array(parameters).reshape((-1, parameters[0][0].shape[0]))

    X_dim = X_train.shape[1]
    utility_evaluation = Iris_Utility(X_train, y_train, config, SEED)
    loss_true = []
    true_paras = np.array(estimated_paras).reshape(-1, X_train.shape[1]+1)
    for i in range(true_paras.shape[0]):
        loss_true.append(utility_evaluation._deep_loss(paras=true_paras[i]))
    Y_feature = np.array(loss_true)
    Util_dim = 1

    train_ratio = 0.5
    valid_ratio = 0.1

    train_end_index = int(train_ratio * np.array(estimated_paras).shape[0])
    valid_end_index = int((train_ratio + valid_ratio) * np.array(estimated_paras).shape[0])

    X_feature_train = np.array(x_one_hot_approx)[:train_end_index, :].reshape(-1, X_train.shape[0])
    X_feature_valid = np.array(x_one_hot_approx)[train_end_index:valid_end_index, :].reshape(-1, X_train.shape[0])
    X_feature_test = np.array(x_one_hot_approx)[valid_end_index:, :].reshape(-1, X_train.shape[0])

    Y_feature_train = Y_feature[:train_end_index]
    Y_feature_valid = Y_feature[train_end_index:valid_end_index]
    Y_feature_test = Y_feature[valid_end_index:]

    train_set = (X_feature_train, Y_feature_train)
    valid_set = (X_feature_valid, Y_feature_valid)
    test_set = (X_feature_test, Y_feature_test)

    config = {}
    config['lambda'] = clambda
    config['dataset'] = 'spam'
    config['learning_rate'] = 0.0001
    config['base_model'] = 'Logistic Regression'
    config['model_path'] = result_path

    deep_config = {}
    deep_config['set_features'] = 128
    deep_config['hidden_ext'] = 128
    deep_config['hidden_reg'] = 128
    n_epoch = 15

    Util_deep_mod = Para_deepset(X_dim, 1, Util_dim, deep_config, config)  # no KKT, predict on parameter
    Util_deep_mod.fit(X_train, y_train, train_set, valid_set, 1 / config['lambda'], n_epoch)
    model_savename = result_path + '/UtlModel_Del0_Nepoch{}_L1L2_{}_{}_C_{}.state_dict'.format(
        n_epoch,
        deep_config['set_features'],
        deep_config['hidden_ext'], config['lambda'])
    sys.stdout.flush()
    torch.save(Util_deep_mod.model.state_dict(), model_savename)

    model = DeepSet(X_dim, 1, Util_dim, deep_config)
    model.load_state_dict(torch.load(model_savename))
    Util_deep_mod_load = Para_deepset(X_dim, 1, Util_dim, deep_config, config, model=model)

    #%% HIGGS
    config = {}
    config['saving_path'] = None
    config['lambda'] = clambda  # 1 0.01
    config['class_number'] = 2
    config['rand_state'] = 32
    config['dataset_name'] = 'HIGGS'

    ### import data
    X_train_all = np.load(data_path + '/X_train_all_pos.npy')
    X_test = np.load(data_path + '/X_test_pos.npy')
    y_train_all = np.load(data_path + '/y_train_all_pos.npy')
    y_test = np.load(data_path + '/y_test_pos.npy')
    Data_size = 300
    X_train, y_train = X_train_all[:Data_size], y_train_all[:Data_size]

    config['dim'] = X_train.shape[1]
    # get training data
    if PERMSET == 'Perm':
        num_features = X_train.shape[1]  # 128
        PREM, Iter = 100, 100
        one_hot = []
        parameters = []
        for i in range(Iter):
            [x_one_hot, LR_para] = pickle.load(
                open(save_path + '/Perm_{}_Iter_{}_OneEncoding_FittedPara.data'.format(PREM, i), 'rb'))
            one_hot.append(x_one_hot)
            parameters.append([listpara.flatten() for listpara in LR_para])
        x_one_hot_approx = np.array(one_hot).reshape((-1, x_one_hot[0].shape[0]))
        estimated_paras = np.array(parameters).reshape((-1, parameters[0][0].shape[0]))

    one_hot = np.array(x_one_hot_approx)
    parameters = np.squeeze(np.array(estimated_paras))


    zero_index = np.where(~parameters.any(axis=1))[0]
    remaining = [i for i in range(parameters.shape[0]) if i not in list(zero_index)]
    one_hot = one_hot[remaining]
    parameters = parameters[remaining]
    X_dim = X_train.shape[1]
    utility_evaluation = Iris_Utility(X_train, y_train, config, SEED)
    loss_true = []
    true_paras = estimated_paras
    for i in range(true_paras.shape[0]):
        loss_true.append(utility_evaluation._deep_loss(paras=true_paras[i]))
    Y_feature = np.array(loss_true)
    Util_dim = 1

    train_end_index = int(train_ratio * Y_feature.shape[0])
    valid_end_index = int((train_ratio + valid_ratio) *Y_feature.shape[0])

    X_feature_train = one_hot[:train_end_index, :].reshape(-1, X_train.shape[0])
    X_feature_valid = one_hot[train_end_index:valid_end_index, :].reshape(-1, X_train.shape[0])
    X_feature_test = one_hot[valid_end_index:, :].reshape(-1, X_train.shape[0])

    Y_feature_train = parameters[:train_end_index, :]
    Y_feature_valid = parameters[train_end_index:valid_end_index, :]
    Y_feature_test = parameters[valid_end_index:, :]

    train_set = (X_feature_train, Y_feature_train)
    valid_set = (X_feature_valid, Y_feature_valid)
    test_set = (X_feature_test, Y_feature_test)

    config = {}
    config['lambda'] = clambda
    config['dataset'] = 'HIGGS'
    config['learning_rate'] = 0.0001
    config['base_model'] = 'Logistic Regression'
    config['model_path'] = result_path

    deep_config = {}
    deep_config['set_features'] = 128
    deep_config['hidden_ext'] = 128
    deep_config['hidden_reg'] = 128
    n_epoch = 15

    Util_deep_mod = Para_deepset(X_dim, 1, Util_dim, deep_config, config)  # no KKT, predict on parameter
    Util_deep_mod.fit(X_train, y_train, train_set, valid_set, 1 / config['lambda'], n_epoch)
    model_savename = result_path + '/UtlModel_Del0_Nepoch{}_L1L2_{}_{}_C_{}.state_dict'.format(
        n_epoch,
        deep_config['set_features'],
        deep_config['hidden_ext'], config['lambda'])
    sys.stdout.flush()
    torch.save(Util_deep_mod.model.state_dict(), model_savename)

    model = DeepSet(X_dim, 1, Util_dim, deep_config)
    model.load_state_dict(torch.load(model_savename))
    Util_deep_mod_load = Para_deepset(X_dim, 1, Util_dim, deep_config, config, model=model)
