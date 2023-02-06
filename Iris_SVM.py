import numpy as np
import time
import tensorflow as tf
from sklearn import datasets
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
import torch
import pickle
import sys
from Models.ParaModel import Para_KKT_deepset, DeepSet
from data_util.dataset_train_test_generation import SVM_multiclass_preparation
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


    # import data and select two classes
    iris = datasets.load_iris()
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
    C = 1
    #%%  sampling
    config = {}
    config['saving_path'] = save_path
    config['lambda'] = 1 #1
    config['class_number'] = 2
    config['rand_state'] = 32
    NUM_SAMPLE = 10000
    Perm_NUM = 300

    SVM_data = SVM_multiclass_preparation(config)
    SVM_data.permutation_generation(X_train, y_train, Perm_NUM)

    # %% load and train the utility meta model
    train_ratio = 0.5
    valid_ratio = 0.1

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

    elif PERMSET == 'RAND':
        [x_one_hot_approx, estimated_paras] = pickle.load(open(save_path + '/Rand_{}_OneEncoding_FittedPara.data'.format(NUM_SAMPLE), 'rb'))

    X_dim = X_std.shape[1]
    Y_feature = np.squeeze(np.array(estimated_paras))
    Util_dim = Y_feature.shape[1]

    train_end_index = int(train_ratio * np.array(estimated_paras).shape[0])
    valid_end_index = int((train_ratio + valid_ratio) * np.array(estimated_paras).shape[0])

    X_feature_train = np.array(x_one_hot_approx)[:train_end_index, :].reshape(-1, X_train.shape[0])
    X_feature_valid = np.array(x_one_hot_approx)[train_end_index:valid_end_index, :].reshape(-1, X_train.shape[0])
    X_feature_test = np.array(x_one_hot_approx)[valid_end_index:, :].reshape(-1, X_train.shape[0])

    Y_feature_train = Y_feature[:train_end_index, :]
    Y_feature_valid = Y_feature[train_end_index:valid_end_index, :]
    Y_feature_test = Y_feature[valid_end_index:, :]

    train_set = (X_feature_train, Y_feature_train)
    valid_set = (X_feature_valid, Y_feature_valid)
    test_set = (X_feature_test, Y_feature_test)

    #%% SVM training need +-1 for y (loss)
    config = {}
    config['lambda'] = 1 #
    config['dataset'] = 'IRIS'
    config['learning_rate'] = 0.0001
    config['base_model'] = 'SVM'
    config['model_path'] = result_path
    config['util_loss'] = True
    config['deepset'] = 'regressor' #regressor naludeep nalu

    deep_config = {}
    deep_config['nalu_layers'] = 3
    deep_config['set_features'] = 128
    deep_config['hidden_ext'] = 128
    deep_config['hidden_reg'] = 128
    n_epoch = 15
    torch.cuda.set_device(1)

    Util_deep_mod = Para_KKT_deepset(X_dim, 1, Util_dim, deep_config, config)

    train_loss, valid_loss = Util_deep_mod.fit(X_train, y_train, train_set, valid_set, 1 / config['lambda'], n_epoch)
    model_savename = result_path + '/Iris_SVM_{}_{}_Del0_KKT_Utl_{}_L1L2_{}_{}_C_{}.state_dict'.\
        format(PERMSET,config['deepset'],str(config['util_loss']),
               deep_config['set_features'],deep_config['hidden_ext'],1/config['lambda'])
    sys.stdout.flush()
    torch.save(Util_deep_mod.model.state_dict(), model_savename)
    # %% testing accuracy
    model = DeepSet(X_dim, 1, Util_dim, deep_config)
    model.load_state_dict(torch.load(model_savename))
    Util_deep_mod_load = Para_KKT_deepset(X_dim, 1, Util_dim, deep_config, config, model=model)

    Y_feature_pred = Util_deep_mod_load.predict(X_train, y_train, X_feature_test)
    Y_feature_pred = np.squeeze(np.array(Y_feature_pred))
    MSE_test = mean_squared_error(Y_feature_test, Y_feature_pred)
    NRMSE_test = np.sqrt(MSE_test) / (np.max(Y_feature_test) - np.min(Y_feature_test))
    print('Parameter estimation testing MSE: {:.6f}, NRMSE: {:.6f}'.format(MSE_test, NRMSE_test))

    Y_feature_pred = Util_deep_mod_load.predict(X_train, y_train, X_feature_train)
    Y_feature_pred = np.squeeze(np.array(Y_feature_pred))
    MSE_train = mean_squared_error(Y_feature_train, Y_feature_pred)
    NRMSE_train = np.sqrt(MSE_test) / (np.max(Y_feature_train) - np.min(Y_feature_train))
    print('Parameter estimation training MSE: {:.6f}, NRMSE: {:.6f}'.format(MSE_train, NRMSE_train))

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
