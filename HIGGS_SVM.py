import numpy as np
import time
import tensorflow as tf
from sklearn import datasets
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
import torch
import pickle
import sys
from Models.ParaModel import Para_KKT_deepset, DeepSet, DeepNalu
from data_util.dataset_train_test_generation import SVM_multiclass_preparation
import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser('OptLearn')
    parser.add_argument('--sampling', default='Perm', type=str, help="sampling method")
    parser.add_argument('--datapath', help="repository to store the training data")
    parser.add_argument('--modelpath', help="repository to store the trained model")

    args = parser.parse_args()
    PERMSET = args.sampling
    save_path = args.datapath
    result_path = args.modelpath

    torch.cuda.set_device(0)

    data_path = './data/'
    X_train_all = np.load(data_path + 'X_train_all_pos.npy')
    X_test = np.load(data_path + 'X_test_pos.npy')
    y_train_all = np.load(data_path + 'y_train_all_pos.npy')
    y_test = np.load(data_path + 'y_test_pos.npy')
    y_train_all = 2 * y_train_all - 1
    y_test = 2 * y_test - 1

    SEED = 33
    Data_size = 300
    X_train, y_train = X_train_all[:Data_size], y_train_all[:Data_size]

    config = {}
    config['saving_path'] = save_path
    config['lambda'] = 1 #1
    config['class_number'] = 2
    config['rand_state'] = 33
    NUM_SAMPLE = 10000
    Perm_NUM = 100

    SVM_data = SVM_multiclass_preparation(config)
    if PERMSET == 'Perm':
        SVM_data.permutation_generation(X_train, y_train, Perm_NUM)
    elif PERMSET == 'RAND':
        SVM_data.random_sampling_generation(X_train, y_train, NUM_SAMPLE, 3, X_train.shape[0], SEED)
# %% load and train the utility meta model
    train_ratio = 0.5
    valid_ratio = 0.1
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

    elif PERMSET == 'RAND':
        [x_one_hot_approx, estimated_paras] = pickle.load(
            open(save_path + '/Rand_{}_OneEncoding_FittedPara.data'.format(NUM_SAMPLE), 'rb'))

    # Util_dim = len(parameters[0])
    one_hot = np.array(x_one_hot_approx)
    parameters = np.squeeze(np.array(estimated_paras))
    Util_dim = parameters.shape[-1]

    zero_index = np.where(~parameters.any(axis=1))[0]
    remaining = [i for i in range(parameters.shape[0]) if i not in list(zero_index)]
    one_hot = one_hot[remaining]
    parameters = parameters[remaining]
    pickle.dump([one_hot, parameters],
                open(data_path + '/{}_{}_OneEncoding_FittedPara_N0.data'.format(PERMSET, NUM_SAMPLE), 'wb'))

    X_dim = X_train.shape[1]

    train_end_index = int(train_ratio * np.array(estimated_paras).shape[0])
    valid_end_index = int((train_ratio + valid_ratio) * np.array(estimated_paras).shape[0])

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
    config['lambda'] = 1 #1
    config['dataset'] = 'HIGGS'
    config['learning_rate'] = 0.0001
    config['base_model'] = 'SVM'
    config['model_path'] = result_path
    config['util_loss'] = True
    config['deepset'] = 'regressor'

    deep_config = {}
    deep_config['set_features'] = 128
    deep_config['hidden_ext'] = 128
    deep_config['hidden_reg'] = 128
    n_epoch = 15

    Util_deep_mod = Para_KKT_deepset(X_dim, 1, Util_dim, deep_config, config)
    train_loss, valid_loss = Util_deep_mod.fit(X_train, y_train, train_set, valid_set, 1 / config['lambda'], n_epoch)
    model_savename = result_path + '/HIGGS_SVM_{}_Del0_KKT_Utl_{}_L1L2_{}_{}_C_{}.state_dict'.\
        format(PERMSET,str(config['util_loss']),deep_config['set_features'],deep_config['hidden_ext'],1/config['lambda'])
    sys.stdout.flush()
    torch.save(Util_deep_mod.model.state_dict(), model_savename)

    model = DeepSet(X_dim, 1, Util_dim, deep_config)
    model.load_state_dict(torch.load(model_savename))
    Util_deep_mod_load = Para_KKT_deepset(X_dim, 1, Util_dim, deep_config, config, model=model)

    Y_feature_pred = Util_deep_mod_load.predict(X_train, y_train, X_feature_test)
    Y_feature_pred = np.squeeze(np.array(Y_feature_pred))
    MSE_test = mean_squared_error(Y_feature_test, Y_feature_pred)
    NRMSE_test = np.sqrt(MSE_test) / (np.max(Y_feature_test) - np.min(Y_feature_test))
    print('Parameter estimation testing MSE: {:.6f}'.format(MSE_test))

    Y_feature_pred = Util_deep_mod_load.predict(X_train, y_train, X_feature_train)
    Y_feature_pred = np.squeeze(np.array(Y_feature_pred))
    MSE_train = mean_squared_error(Y_feature_train, Y_feature_pred)
    NRMSE_train = np.sqrt(MSE_test) / (np.max(Y_feature_train) - np.min(Y_feature_train))
    print('Parameter estimation training MSE: {:.6f}'.format(MSE_train))

    # %% evaluate the utility performance
    from Models.UtilityModel import SVM_bin_Utility
    from scipy.stats import pearsonr, spearmanr

    utility_evaluation = SVM_bin_Utility(X_train, y_train, config, SEED)

    Y_feature_pred = Util_deep_mod_load.predict(X_train, y_train, X_feature_train)
    Y_feature_pred = np.squeeze(np.array(Y_feature_pred))
    loss_pred = []
    loss_true = []
    for i in range(Y_feature_pred.shape[0]):
        loss_pred.append(utility_evaluation._deep_loss(paras=Y_feature_pred[i]))
        loss_true.append(utility_evaluation._deep_loss(paras=Y_feature_train[i]))

    MSE_train = mean_squared_error(loss_true, loss_pred)
    print('Loss estimation training MSE: {:.6f}'.format(MSE_train))
    pearson_corr = pearsonr(loss_true, loss_pred)
    spear_corr = spearmanr(loss_true, loss_pred)
    print(pearson_corr[0], spear_corr[0])

    Y_feature_pred = Util_deep_mod_load.predict(X_train, y_train, X_feature_test)
    Y_feature_pred = np.squeeze(np.array(Y_feature_pred))
    loss_pred = []
    loss_true = []
    for i in range(Y_feature_pred.shape[0]):
        loss_pred.append(utility_evaluation._deep_loss(paras=Y_feature_pred[i]))
        loss_true.append(utility_evaluation._deep_loss(paras=Y_feature_test[i]))

    MSE_test = mean_squared_error(loss_true, loss_pred)
    print('Loss estimation testing MSE: {:.6f}'.format(MSE_test))
    pearson_corr = pearsonr(loss_true, loss_pred)
    spear_corr = spearmanr(loss_true, loss_pred)
    print(pearson_corr[0], spear_corr[0])




