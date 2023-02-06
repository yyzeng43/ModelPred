# minst feature extracted by CNN (128 * 10）
import pickle
import numpy as np
import pandas as pd
import time
import sys
import torch
from sklearn.metrics import mean_squared_error, accuracy_score
from Models.ParaModel import Para_KKT_deepset, DeepSet
from data_util.dataset_train_test_generation import LR_multiclass_preparation
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
    # get all data
    feature_path = './data'
    [train_feature, test_feature] = pickle.load(open(feature_path + '/minst_feature_train_test.data', 'rb'))
    [train_label, test_label] = pickle.load(open(feature_path + '/minst_label_train_test.data', 'rb'))

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
    Data_Size = 500
    Perm_NUM = 100
    config = {}
    config['saving_path'] = save_path
    config['lambda'] = 1 #1
    config['class_number'] = 10
    config['rand_state'] = 32

    NUM_SAMPLE = 30000
    SEED = 33
    train_data, train_lab = x_train[:Data_Size, :], y_train[:Data_Size]
    y_train_enc, y_test_enc = pd.get_dummies(y_train).values[:Data_Size, :], pd.get_dummies(y_test).values

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1, random_state=33, max_iter=10000)
    clf.fit(train_data, train_lab)
    y_pred = clf.predict(x_test)
    print(accuracy_score(y_test, y_pred))

    LR_data = LR_multiclass_preparation(config)
    LR_data = LR_multiclass_preparation(config)
    if PERMSET == 'Perm':
        LR_data.permutation_generation(train_data, train_lab, Perm_NUM)
    elif PERMSET == 'RAND':
        LR_data.random_sampling_generation(train_data, train_lab, NUM_SAMPLE, 3, train_data.shape[0], SEED)

    #%%
    num_features = x_train.shape[1]   # 128
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
    elif PERMSET == 'RAND':
        [one_hot, parameters] = pickle.load(
            open(save_path + '/Rand_{}_OneEncoding_FittedPara.data'.format(NUM_SAMPLE), 'rb'))

    # Util_dim = len(parameters[0])
    one_hot = np.array(one_hot)
    parameters = np.squeeze(np.array(parameters))
    Util_dim = parameters.shape[-1]

    zero_index = np.where(~parameters.any(axis=1))[0]
    remaining = [i for i in range(parameters.shape[0]) if i not in list(zero_index)]
    one_hot = one_hot[remaining]
    parameters = parameters[remaining]
    pickle.dump([one_hot, parameters],
                open(save_path + '/{}_{}_OneEncoding_FittedPara_N0.data'.format(PERMSET, NUM_SAMPLE), 'wb'))


    #%% random sampling

    # [one_hot, parameters] = pickle.load(
    #     open(data_path + '/Rand_{}_OneEncoding_FittedPara_N0.data'.format(NUM_SAMPLE), 'rb'))


    parameters = parameters.reshape((parameters.shape[0],-1))
    Util_dim = parameters.shape[-1]
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

    train_set = (X_feature_train, Y_feature_train)
    valid_set = (X_feature_valid, Y_feature_valid)
    test_set = (X_feature_test, Y_feature_test)

    # %% deepset training
    config = {}
    config['lambda'] = 1
    config['class_number'] = 10
    config['dim'] = num_features
    config['dataset'] = 'MNIST'
    config['learning_rate'] = 0.0001
    config['base_model'] = 'Logistic Regression'
    config['model_path'] = result_path
    config['util_loss'] = True
    config['deepset'] = 'regressor'  # regressor

    deep_config = {}
    deep_config['set_features'] = 128
    deep_config['hidden_ext'] = 128
    deep_config['hidden_reg'] = 128
    n_epoch = 15

    Util_deep_mod = Para_KKT_deepset(num_features, 10, Util_dim, deep_config, config)
    Util_deep_mod.fit(train_data, train_lab, train_set, valid_set,
                      1 / config['lambda'], n_epoch, kkt_lambda=1)
    model_savename = result_path + \
                     '/MNIST_LR_{}_{}_Extr_Del0_KKT_0_Utl_{}_Nepoch{}_L1L2_{}_{}_C_{}.state_dict'.\
                         format(PERMSET,config['deepset'], n_epoch,str(config['util_loss']),
                                deep_config['set_features'],deep_config['hidden_ext'],1/config['lambda'])
    sys.stdout.flush()
    torch.save(Util_deep_mod.model.state_dict(), model_savename)

    # %% test the performance
    # model_savename = result_path + '/UtlExtr_Del0_KKT_1_Nepoch15_L1L2_128_128.state_dict'

    model = DeepSet(num_features, 10, Util_dim, deep_config)
    model.load_state_dict(torch.load(model_savename))
    Util_deep_mod_load = Para_KKT_deepset(num_features, 10, Util_dim, deep_config, config, model=model)

    Y_feature_pred = Util_deep_mod_load.predict(train_data, train_lab, X_feature_test)
    Y_feature_pred = np.squeeze(np.array(Y_feature_pred))
    MSE_test = mean_squared_error(Y_feature_test, Y_feature_pred)
    NRMSE_test = np.sqrt(MSE_test) / (np.max(Y_feature_test) - np.min(Y_feature_test))
    print('Parameter estimation MSE: {:.6f}'.format(MSE_test))

    Y_feature_pred = Util_deep_mod_load.predict(train_data, train_lab, X_feature_train)
    Y_feature_pred = np.squeeze(np.array(Y_feature_pred))
    MSE_test = mean_squared_error(Y_feature_train, Y_feature_pred)
    NRMSE_test = np.sqrt(MSE_test) / (np.max(Y_feature_train) - np.min(Y_feature_train))
    print('Parameter estimation MSE: {:.6f}'.format(MSE_test))

    # %% evaluate the utility performance
    from Models.UtilityModel import MINST_Utility
    SEED = 32

    y_train_onehot = torch.nn.functional.one_hot(train_lab, config['class_number'])
    utility_evaluation = MINST_Utility(train_data, y_train_onehot, y_train_enc,config, SEED)

    Y_feature_pred = Util_deep_mod_load.predict(train_data, train_lab, X_feature_train)
    Y_feature_pred = np.squeeze(np.array(Y_feature_pred))
    loss_pred = []
    loss_true = []
    for i in range(Y_feature_pred.shape[0]):
        loss_pred.append(utility_evaluation._deep_loss(paras=Y_feature_pred[i]))
        loss_true.append(utility_evaluation._deep_loss(paras=Y_feature_train[i]))
        print('Iter {}'.format(i))

    MSE_train = mean_squared_error(loss_true, loss_pred)
    print('Loss estimation training MSE: {:.6f}'.format(MSE_train))


    Y_feature_pred = Util_deep_mod_load.predict(train_data, train_lab, X_feature_test)
    Y_feature_pred = np.squeeze(np.array(Y_feature_pred))
    loss_pred = []
    loss_true = []
    for i in range(Y_feature_pred.shape[0]):
        loss_pred.append(utility_evaluation._deep_loss(paras=Y_feature_pred[i]))
        loss_true.append(utility_evaluation._deep_loss(paras=Y_feature_test[i]))

    MSE_test = mean_squared_error(loss_true, loss_pred)
    print('Loss estimation testing MSE: {:.6f}'.format(MSE_test))

    #%%
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(20, 20))
    font = {'family': 'serif', 'serif': ['Times New Roman'], 'weight': 'normal', 'size': 30}
    plt.rc('font', **font)
    # plt.scatter(np.concatenate((loss_true.reshape(-1,1), loss_pred.reshape(-1,1)), axis=1),
    #             marker='o', s=range(len(loss_true)))
    plt.scatter(loss_true, loss_pred, color = 'orange', edgecolors='blue',
                marker='o', s=[np.sum(X_feature_test[i])*10 for i in range(len(X_feature_test))], alpha=0.8)
    plt.plot(loss_true, loss_true, 'r')
    plt.ylabel("testing loss (predicted)")
    plt.xlabel("testing loss (true)")
    plt.show()

    from scipy.stats import pearsonr, spearmanr
    pearson_corr = pearsonr(loss_true, loss_pred)
    spear_corr = spearmanr(loss_true, loss_pred)
    print(pearson_corr[0], spear_corr[0])


    #%% train the parameter model
    config = {}
    config['lambda'] = 1
    config['class_number'] = 10
    config['dim'] = num_features
    config['dataset'] = 'MNIST'
    config['learning_rate'] = 0.0001
    config['base_model'] = 'Logistic Regression'
    config['model_path'] = result_path
    config['util_loss'] = False
    config['deepset'] = 'regressor'  # regressor

    deep_config = {}
    deep_config['set_features'] = 128
    deep_config['hidden_ext'] = 128
    deep_config['hidden_reg'] = 128
    n_epoch = 15

    Util_deep_mod = Para_KKT_deepset(num_features, 10, Util_dim, deep_config, config)
    Util_deep_mod.fit(train_data, train_lab, train_set, valid_set,
                      1 / config['lambda'], n_epoch, kkt_lambda=0)
    model_savename = result_path + \
                     '/MNIST_LR_Perm_regressor_Extr_Del0_Utl_15_NepochFalse_L1L2_128_128_C_1.0.state_dict'
    sys.stdout.flush()
    torch.save(Util_deep_mod.model.state_dict(), model_savename)




