import numpy as np
from tqdm import tqdm
import itertools
import pickle
from sklearn import datasets as skdataset
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
from Models.ParaModel import Para_KKT_deepset, DeepSet, Para_deepset
from Shapley.Valuation import *
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser('OptLearn')
    parser.add_argument('--sampling', default='Perm', type=str, help="sampling method")
    parser.add_argument('--rawdatapath', help="path to retrieve the raw training data")
    parser.add_argument('--modelpath', help="path to retrieve the trained model")
    parser.add_argument('--savepath', help="path to store the results")
    parser.add_argument('--maxiter', help="maximum iteration")

    args = parser.parse_args()
    PERMSET = args.sampling
    data_path = args.rawdatapath
    model_path = args.modelpath
    save_path = args.savepath
    torch.cuda.set_device(0)
    np.seterr(all="ignore")

    # set the number of iterations
    MaxIter = args.maxiter  #10,50, 100, 500(200), 1000

    #%%  MNIST Shapley
    config = {}
    config['dataset_name'] = 'MNIST'
    config['class_number'] = 10
    config['lambda'] = 1
    config['base_model'] = 'Logistic Regression'
    config['Tolerance'] = 1e-3
    config['MaxIter']= 200
    config['save_path'] = save_path
    config['random_state'] = 33


    ### import data
    [train_feature, test_feature] = pickle.load(open(data_path + '/minst_feature_train_test.data', 'rb'))
    [train_label, test_label] = pickle.load(open(data_path + '/minst_label_train_test.data', 'rb'))

    train_feature_all = torch.cat(train_feature, dim=0).cpu()
    test_feature_all = torch.cat(test_feature, dim=0).cpu()
    train_label_all, test_label_all = torch.cat(train_label, dim=0), torch.cat(test_label, dim=0).cpu()
    X_mean = np.linalg.norm(train_feature_all, ord=2, axis=1)
    x_train = train_feature_all/np.max(X_mean)
    x_test = test_feature_all/np.max(X_mean)
    y_train, y_test = train_label_all, test_label_all

    x_train, x_test= x_train.cpu(), x_test.cpu()
    y_train, y_test = y_train.cpu(), y_test.cpu()

    config['dim'] = x_train.shape[1]

    ### training data testing data
    Data_Size = 300
    Test_Size = 500
    x_test, y_test = x_test[:Test_Size], y_test[:Test_Size]
    y_train_enc, y_test_enc = pd.get_dummies(y_train).values, pd.get_dummies(y_test).values
    train_data, train_lab_org, train_lab = x_train[150:150+Data_Size, :], y_train[150:150+Data_Size], y_train_enc[150:150+Data_Size,:]

    #%% baseline
    shapley_evaluation = Minst_Shapley(train_data, train_lab_org, train_lab, x_test, y_test, y_test_enc, config)
    shapley_evaluation._approx_perm()
    #%% deepKKT

    Util_dim = 1290
    deep_config = {}
    deep_config['set_features'] = 128
    deep_config['hidden_ext'] = 128
    deep_config['hidden_reg'] = 128
    mod_config = {}
    mod_config['util_loss'] = True
    mod_config['deepset'] = 'regressor'
    mod_config['dim'] = config['dim']
    mod_config['model_path'] = None
    mod_config['base_model'] = 'Logistic Regression'
    mod_config['dataset'] = 'MNIST'
    mod_config['lambda'] = config['lambda']
    mod_config['class_number'] = 10
    mod_config['learning_rate'] = 0.0001

    model_name = '/{}_LR_{}_Del0_KKT_Utl_{}_L1L2_{}_{}_C_{}.state_dict'.\
        format(config['dataset_name'], PERMSET, str(config['util_loss']),
               deep_config['set_features'], deep_config['hidden_ext'],1 / config['lambda'])
    #
    deepmodel = DeepSet(config['dim'], config['class_number'], Util_dim, deep_config)
    deepmodel.load_state_dict(torch.load(model_path + model_name))
    Util_deep_mod_load = Para_KKT_deepset(mod_config['dim'], mod_config['class_number'],
                                          Util_dim, deep_config, mod_config, model=deepmodel)
    shapley_evaluation._deep_approx_perm(Util_deep_mod_load)

    #%% utility training and result
    clambda = 1
    config = {}
    config['lambda'] = clambda
    config['dataset'] = 'MNIST'
    config['learning_rate'] = 0.0001
    config['base_model'] = 'Logistic Regression'
    config['model_path'] = None
    config['MaxIter']= 200

    deep_config = {}
    deep_config['set_features'] = 128
    deep_config['hidden_ext'] = 128
    deep_config['hidden_reg'] = 128
    n_epoch = 15
    Util_dim = 1
    model_name = '/UtlModel_Del0_Nepoch15_L1L2_128_128_C_1.state_dict'

    X_dim = train_data.shape[1]
    model = DeepSet(X_dim, 10, Util_dim, deep_config)
    model.load_state_dict(torch.load(model_path + model_name))
    Util_deep_mod_load = Para_deepset(X_dim, 10, Util_dim, deep_config, config, model=model)

    shapley_evaluation._utl_approx_perm(Util_deep_mod_load)

    #%% compare the results
    # baseline
    [shapley_true, exact_time] = pickle.load(open(save_path + '/Base_Shapley_Time_Iter_{}.data'.format(config['MaxIter']), 'rb'))

    # DeepKKT
    [shapley_kkt, kkt_time] = pickle.load(open(save_path + '/DeepKKT_Shapley_Time_Iter_{}.data'.format(config['MaxIter']), 'rb'))

    # Utility
    [shapley_utl, utl_time] = pickle.load(open(save_path + '/DeepUtl_Shapley_Time_Iter_{}.data'.format(config['MaxIter']), 'rb'))

    MNIST_shapely_fig = all_shapley_fig(shapley_true[-1], shapley_kkt[-1], shapley_utl[-1])
    MNIST_shapely_fig.savefig(save_path + '/MNIST_shapley_LR_Iter_{}.png'.format(config['MaxIter']), bbox_inches = 'tight')
    MNIST_shapely_fig.show()

    print(' MNIST: Exact time:', exact_time)
    print('KKT time:', kkt_time)
    print('Utility model time:', utl_time)
    #%% Iris
    config = {}
    config['dataset_name'] = 'Iris'
    config['class_number'] = 2
    config['lambda'] = 1
    config['Tolerance'] = 1e-4
    config['MaxIter']= MaxIter
    config['random_state'] = 33
    config['save_path'] = save_path

    ### import data
    iris = skdataset.load_iris()
    X = iris.data
    y = iris.target
    X = X[np.where(y != 2)[0], :]
    y = y[np.where(y != 2)[0]]

    X_mean = np.linalg.norm(X, ord=2, axis=1)
    X_std = X / np.max(X_mean)  # rescale the input features
    SEED = 15
    X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.33, random_state=SEED)

    config['dim'] = X_train.shape[1]

    #%% baseline
    shapley_evaluation = Iris_Shapley(X_train, y_train, X_test, y_test, config)
    shapley_evaluation._approx_perm()

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
    mod_config['base_model'] = 'Logistic Regression'
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
    shapley_evaluation._deep_approx_perm(Util_deep_mod_load)

    #%% utility training and result
    clambda = 1
    config = {}
    config['lambda'] = clambda
    config['dataset'] = 'Iris'
    config['learning_rate'] = 0.0001
    config['base_model'] = 'Logistic Regression'
    config['model_path'] = None
    config['MaxIter'] = MaxIter

    deep_config = {}
    deep_config['set_features'] = 128
    deep_config['hidden_ext'] = 128
    deep_config['hidden_reg'] = 128
    n_epoch = 15
    Util_dim = 1
    model_name = '/UtlModel_Del0_Nepoch15_L1L2_128_128_C_1.state_dict'

    X_dim = X_train.shape[1]
    model = DeepSet(X_dim, 1, Util_dim, deep_config)
    model.load_state_dict(torch.load(model_path + model_name))
    Util_deep_mod_load = Para_deepset(X_dim, 1, Util_dim, deep_config, config, model=model)

    shapley_evaluation._utl_approx_perm(Util_deep_mod_load)

    #%% compare the results
    # baseline
    [shapley_true, exact_time] = pickle.load(open(save_path + '/Base_Shapley_Time_Iter_{}.data'.format(config['MaxIter']), 'rb'))

    # DeepKKT
    [shapley_kkt, kkt_time] = pickle.load(open(save_path + '/DeepKKT_Shapley_Time_Iter_{}.data'.format(config['MaxIter']), 'rb'))

    # Utility
    [shapley_utl, utl_time] = pickle.load(open(save_path + '/DeepUtl_Shapley_Time_Iter_{}.data'.format(config['MaxIter']), 'rb'))

    iris_shapely_fig = all_shapley_fig(shapley_true[-1], shapley_kkt[-1], shapley_utl[-1])
    iris_shapely_fig.savefig(save_path + '/Iris_shapley_LR_Iter_{}.png'.format(config['MaxIter']), bbox_inches = 'tight')
    iris_shapely_fig.show()

    print(' Iris: Exact time:', exact_time)
    print('KKT time:', kkt_time)
    print('Utility model time:', utl_time)

    #%% spam
    config = {}
    config['dataset_name'] = 'spam'
    config['class_number'] = 2
    config['lambda'] = 1
    config['Tolerance'] = 1e-3
    config['MaxIter']= MaxIter
    config['random_state'] = 33
    config['save_path'] = save_path
    [X_train, y_train, X_test, y_test] = pickle.load(open(data_path + '/spam_trainXY_testXY_low.data', 'rb'))


    X_mean = np.linalg.norm(X_train, ord=2, axis=1)
    X_train = X_train / np.max(X_mean)
    X_test = X_test / np.max(X_mean)

    Data_size = 300
    X_train, y_train = X_train[150:150+Data_size], y_train[150:150+Data_size]
    config['dim'] = X_train.shape[1]

    #%% baseline
    shapley_evaluation = Iris_Shapley(X_train, y_train, X_test, y_test, config)
    shapley_evaluation._approx_perm()

    #%% deepset result
    Util_dim = config['dim'] + 1
    deep_config = {}
    deep_config['set_features'] = 128
    deep_config['hidden_ext'] = 128
    deep_config['hidden_reg'] = 128
    mod_config = {}
    mod_config['util_loss'] = False
    mod_config['deepset'] = 'regressor'
    mod_config['dim'] = config['dim']
    mod_config['model_path'] = None
    mod_config['base_model'] = 'Logistic Regression'
    mod_config['dataset'] = 'spam'
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
    shapley_evaluation._deep_approx_perm(Util_deep_mod_load)

    #%% utility training and result
    clambda = 1
    utl_config = {}
    utl_config['lambda'] = clambda
    utl_config['dataset'] = 'spam'
    utl_config['learning_rate'] = 0.0001
    utl_config['base_model'] = 'Logistic Regression'
    utl_config['model_path'] = None

    deep_config = {}
    deep_config['set_features'] = 128
    deep_config['hidden_ext'] = 128
    deep_config['hidden_reg'] = 128
    n_epoch = 15
    Util_dim = 1
    model_name = '/UtlModel_Del0_Nepoch15_L1L2_128_128_C_1.state_dict'

    X_dim = X_train.shape[1]
    model = DeepSet(X_dim, 1, Util_dim, deep_config)
    model.load_state_dict(torch.load(model_path + model_name))
    Util_deep_mod_load = Para_deepset(X_dim, 1, Util_dim, deep_config, utl_config, model=model)

    shapley_evaluation._utl_approx_perm(Util_deep_mod_load)

    #%% compare the results
    # baseline
    [shapley_true, exact_time] = pickle.load(open(save_path + '/Base_Shapley_Time_Iter_{}.data'.format(config['MaxIter']), 'rb'))

    # DeepKKT
    [shapley_kkt, kkt_time] = pickle.load(open(save_path + '/DeepKKT_Shapley_Time_Iter_{}.data'.format(config['MaxIter']), 'rb'))

    # Utility
    [shapley_utl, utl_time] = pickle.load(open(save_path + '/DeepUtl_Shapley_Time_Iter_{}.data'.format(config['MaxIter']), 'rb'))

    spam_shapely_fig = all_shapley_fig(shapley_true[-1], shapley_kkt[-1], shapley_utl[-1])
    spam_shapely_fig.savefig(save_path + '/spam_shapley_LR_Iter_{}.png'.format(config['MaxIter']), bbox_inches = 'tight')
    spam_shapely_fig.show()
    print('spam: Exact time:', exact_time)
    print('KKT time:', kkt_time)
    print('Utility model time:', utl_time)

    #%% HIGGS
    config = {}
    config['dataset_name'] = 'HIGGS'
    config['class_number'] = 2
    config['lambda'] = 1
    config['Tolerance'] = 1e-4
    config['MaxIter']= MaxIter
    config['random_state'] = 33
    config['save_path'] = save_path


    ### import data
    Data_size = 300
    X_train_all = np.load(data_path + '/X_train_all_pos.npy')
    X_test = np.load(data_path + '/X_test_pos.npy')
    y_train_all = np.load(data_path + '/y_train_all_pos.npy')
    y_test = np.load(data_path + '/y_test_pos.npy')
    X_train, y_train = X_train_all[150:150+Data_size], y_train_all[150:150+Data_size]

    config['dim'] = X_train.shape[1]

    #%% baseline
    shapley_evaluation = Iris_Shapley(X_train, y_train, X_test, y_test, config)
    shapley_evaluation._approx_perm()

    #%% deepset result
    Util_dim = config['dim'] + 1
    deep_config = {}
    deep_config['set_features'] = 128
    deep_config['hidden_ext'] = 128
    deep_config['hidden_reg'] = 128
    mod_config = {}
    mod_config['util_loss'] = True
    mod_config['deepset'] = 'regressor'
    mod_config['dim'] = config['dim']
    mod_config['model_path'] = None
    mod_config['base_model'] = 'Logistic Regression'
    mod_config['dataset'] = 'HIGGS'
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
    shapley_evaluation._deep_approx_perm(Util_deep_mod_load)

    #%% utility training and result
    clambda = 1
    config = {}
    config['lambda'] = clambda
    config['dataset'] = 'spam'
    config['learning_rate'] = 0.0001
    config['base_model'] = 'Logistic Regression'
    config['model_path'] = None
    config['MaxIter']= MaxIter

    deep_config = {}
    deep_config['set_features'] = 128
    deep_config['hidden_ext'] = 128
    deep_config['hidden_reg'] = 128
    n_epoch = 15
    Util_dim = 1
    model_name = '/UtlModel_Del0_Nepoch15_L1L2_128_128_C_1.state_dict'

    X_dim = X_train.shape[1]
    model = DeepSet(X_dim, 1, Util_dim, deep_config)
    model.load_state_dict(torch.load(model_path + model_name))
    Util_deep_mod_load = Para_deepset(X_dim, 1, Util_dim, deep_config, config, model=model)

    shapley_evaluation._utl_approx_perm(Util_deep_mod_load)

    #%% compare the results
    # baseline
    [shapley_true, exact_time] = pickle.load(open(save_path + '/Base_Shapley_Time_Iter_{}.data'.format(config['MaxIter']), 'rb'))

    # DeepKKT
    [shapley_kkt, kkt_time] = pickle.load(open(save_path + '/DeepKKT_Shapley_Time_Iter_{}.data'.format(config['MaxIter']), 'rb'))

    # Utility
    [shapley_utl, utl_time] = pickle.load(open(save_path + '/DeepUtl_Shapley_Time_Iter_{}.data'.format(config['MaxIter']), 'rb'))

    HIGGS_shapely_fig = all_shapley_fig(shapley_true[-1], shapley_kkt[-1], shapley_utl[-1])
    HIGGS_shapely_fig.savefig(save_path + '/HIGGS_shapley_LR_Iter_{}.png'.format(config['MaxIter']), bbox_inches = 'tight')
    HIGGS_shapely_fig.show()
    print(' HIGGS: Exact time:', exact_time)
    print('KKT time:', kkt_time)
    print('Utility model time:', utl_time)