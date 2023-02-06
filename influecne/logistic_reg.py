import sklearn
from sklearn import linear_model
from sklearn import datasets
import numpy as np
import copy
from functools import partial
from tqdm import tqdm

# import jax
# import jax.numpy as jnp
# from jax import custom_jvp, custom_vjp
# from jax import lax
# from jax import jit, vmap
from scipy.special import softmax

class bin_log(object):
    def __init__(self, C):
        self.C = C

    def sigma(self,theta, xi):
        zi = np.dot(theta, xi)
        return 1 / (1 + np.exp(-zi))

    def gradient_single(self, theta, xi, yi):
        return self.C * xi * (self.sigma(theta, xi) - yi)
        # return  xi * (self.sigma(theta, xi) - yi)


    def hessian_single(self, theta, xi):
        return np.outer(xi, xi) * self.sigma(theta, xi) * (1 - self.sigma(theta, xi))

    def hessian_inverse(self, theta, X_syn_aug):
        n, d = X_syn_aug.shape
        hessian = np.zeros((d, d))
        hess_reg =  np.eye(theta.shape[0])
        hess_reg[-1,-1]=0

        for i in range(n):
            hessian += self.hessian_single(theta, X_syn_aug[i])
        hessian = self.C*hessian + hess_reg

        try:
            hessian_inv = np.linalg.inv(hessian)
        except Exception:
            hessian_inv = np.linalg.inv(hessian + np.random.random(hessian.shape)*(1e-5))

        return hessian_inv

    def inf_test_loss_single(self, theta, hessian_inv, X_train, Y_train, x_test, y_test, n_test, n=None):
        if n == None:
            n = len(Y_train)
        inf_loss_lst = np.zeros(len(Y_train))
        grad_x_train = np.zeros((len(Y_train),X_train.shape[0]))
        for i in range(len(Y_train)):
            xi, yi = X_train[i], Y_train[i]
            # inf_loss = np.dot(np.dot(hessian_inv, self.gradient_single(theta, xi, yi)),
            #                   self.gradient_single(theta, x_test, y_test))
            reg_theta = copy.deepcopy(theta)
            reg_theta[-1] = 0
            inf_loss = np.dot(np.dot(hessian_inv, self.gradient_single(theta, xi, yi)+reg_theta/n),
                              self.gradient_single(theta, x_test, y_test)+reg_theta/n_test)
            inf_loss_lst[i] = inf_loss

        return inf_loss_lst

    def aug_x(self, X):
        b = np.ones((X.shape[0], 1))
        X_aug = np.hstack((X, b))
        return X_aug

    def get_test_loss_all(self, model, X_train, Y_train, X_test, Y_test):
        theta = np.concatenate((model.coef_[0], model.intercept_))

        X_train_aug = self.aug_x(X_train)
        X_test_aug = self.aug_x(X_test)
        n_test = X_test_aug.shape[0]

        hessian_inv = self.hessian_inverse(theta, X_train_aug)

        n = len(Y_train)
        inf_loss_lst = np.zeros(n)

        for x_test, y_test in  tqdm(zip(X_test_aug, Y_test)):
            inf_loss_lst += self.inf_test_loss_single(theta, hessian_inv, X_train_aug, Y_train, x_test, y_test, n_test)
        return inf_loss_lst

    def get_add_loss_all(self, model, X_train, Y_train, X_add, Y_add, X_test, Y_test):
        theta = np.concatenate((model.coef_[0], model.intercept_))
        X_use = np.concatenate((X_train, X_add), axis=0)
        Y_use = np.concatenate((Y_train, Y_add), axis=0)
        X_train_aug = self.aug_x(X_train)
        X_use_aug = self.aug_x(X_use)
        X_test_aug = self.aug_x(X_test)
        n_test = X_test_aug.shape[0]

        hessian_inv = self.hessian_inverse(theta, X_train_aug)
        n = len(Y_add)
        inf_loss_lst = np.zeros(n)

        X_add_aug = self.aug_x(X_add)

        for x_test, y_test in  tqdm(zip(X_test_aug, Y_test)):
            inf_loss_lst += self.inf_test_loss_single(theta, hessian_inv,
                                                      X_add_aug, Y_add, x_test,
                                                      y_test, n_test, n=X_train.shape[0])#, n=X_use.shape[0]

        return inf_loss_lst


    def get_index_after_removal(self, n_tot, remove_ind):
        ind_bin = np.ones(n_tot)
        ind_bin[remove_ind] = 0
        return np.nonzero(ind_bin)[0].astype(int)

    def get_single_para_loo(self, model, X_train, Y_train):
        n = X_train.shape[0]
        theta = np.concatenate((model.coef_[0], model.intercept_))
        X_train_aug = self.aug_x(X_train)
        hessian_inv = self.hessian_inverse(theta, X_train_aug)
        para_del = np.zeros((len(Y_train),X_train.shape[1] + 1))

        for i in range(len(Y_train)):
            xi, yi = X_train_aug[i], Y_train[i]
            reg_theta = copy.deepcopy(theta)
            reg_theta[-1] = 0
            para_del[i,:] = np.dot(hessian_inv, self.gradient_single(theta, xi, yi) + reg_theta / n).reshape(1,-1)
        return para_del

    def get_single_para_add(self, model, X_train, Y_train, X_add, Y_add):
        n = X_train.shape[0]
        theta = np.concatenate((model.coef_[0], model.intercept_))
        X_train_aug = self.aug_x(X_train)
        X_add_aug = self.aug_x(X_add)
        hessian_inv = self.hessian_inverse(theta, X_train_aug)
        para_del = np.zeros((len(Y_add),X_train.shape[1] + 1))

        for i in range(len(Y_add)):
            xi, yi = X_add_aug[i], Y_add[i]
            reg_theta = copy.deepcopy(theta)
            reg_theta[-1] = 0
            para_del[i,:] = np.dot(hessian_inv, self.gradient_single(theta, xi, yi) + reg_theta / n).reshape(1,-1)
        return para_del




class multi_log(object):
    def __init__(self, C):
        self.C = C

    def get_softmax(self,theta, xi):
        zi = np.dot(xi,theta.T)
        if len(zi.shape)>1:
            return softmax(zi,axis=1)
        else:
            return softmax(zi,axis=0)

    def gradient_single(self, theta, xi, yi):
        return self.C * np.kron((self.get_softmax(theta, xi) - yi).reshape(-1,1), xi.reshape(1,-1)).reshape((1290,1))
        # return theta + self.C * xi * (self.sigma(theta, xi) - yi)

    def hessian_inverse(self, theta, X_syn_aug):
        n, d = X_syn_aug.shape
        y_hat = self.get_softmax(theta, X_syn_aug)
        hessian_no_reg = np.kron(np.diagonal(y_hat.T)-np.dot(y_hat.T, y_hat), np.dot(X_syn_aug.T, X_syn_aug))

        hess_reg = np.eye(theta.shape[0]*theta.shape[1])
        hess_reg = hess_reg.reshape((10, 129, 10, 129))
        hess_reg[:, 128,:,128] = 0
        hess_reg = hess_reg.reshape((1290, 1290))
        # hessian = hessian_no_reg
        hessian = self.C *hessian_no_reg + hess_reg

        try:
            hessian_inv = np.linalg.inv(hessian)
        except Exception:
            hessian_inv = np.linalg.inv(hessian + np.random.random(hessian.shape)*(1e-5))

        return hessian_inv

    def inf_test_loss_single(self, theta, hessian_inv, X_train, Y_train, x_test, y_test, n_test, n=None):
        if n == None:
            n = len(Y_train)
        inf_loss_lst = np.zeros(X_train.shape[0])
        for i in range(X_train.shape[0]):
            xi, yi = X_train[i], Y_train[i]
            # inf_loss = np.dot(np.dot(hessian_inv, self.gradient_single(theta, xi, yi)).reshape(1,-1),
            #                   self.gradient_single(theta, x_test, y_test))
            reg_theta = copy.deepcopy(theta)
            reg_theta[:, -1] = 0
            inf_loss = np.dot(np.dot(hessian_inv, self.gradient_single(theta, xi, yi)+reg_theta.reshape(-1,1)/n).reshape(1,-1),
                              self.gradient_single(theta, x_test, y_test)+reg_theta.reshape(-1,1)/n_test)
            inf_loss_lst[i] = inf_loss

        return inf_loss_lst

    def aug_x(self, X):
        b = np.ones((X.shape[0], 1))
        X_aug = np.hstack((X, b))
        return X_aug

    def get_test_loss_all(self, model, X_train, Y_train, X_test, Y_test):
        theta = np.concatenate((model.coef_, model.intercept_.reshape(-1,1)), axis=1)

        X_train_aug = self.aug_x(X_train)
        X_test_aug = self.aug_x(X_test)
        n_test = X_test_aug.shape[0]

        hessian_inv = self.hessian_inverse(theta, X_train_aug)

        n = len(Y_train)
        inf_loss_lst = np.zeros(n)

        for x_test, y_test in  tqdm(zip(X_test_aug, Y_test)):
            inf_loss_lst += self.inf_test_loss_single(theta, hessian_inv, X_train_aug, Y_train, x_test, y_test, n_test)

        return inf_loss_lst


    def get_add_loss_all(self, model, X_train, Y_train, X_add, Y_add, X_test, Y_test):
        theta = np.concatenate((model.coef_, model.intercept_.reshape(-1, 1)), axis=1)
        X_use = np.concatenate((X_train, X_add), axis=0)
        Y_use = np.concatenate((Y_train, Y_add), axis=0)
        X_train_aug = self.aug_x(X_train)

        X_use_aug = self.aug_x(X_use)
        X_test_aug = self.aug_x(X_test)
        n_test = X_test_aug.shape[0]
        hessian_inv = self.hessian_inverse(theta, X_train_aug)
        # hessian_inv = self.hessian_inverse(theta, X_use_aug)
        n = len(Y_add)
        inf_loss_lst = np.zeros(n)

        X_add_aug = self.aug_x(X_add)

        for x_test, y_test in  tqdm(zip(X_test_aug, Y_test)):
            inf_loss_lst += self.inf_test_loss_single(theta, hessian_inv, X_add_aug,
                                                      Y_add, x_test, y_test, n_test, n=X_train.shape[0])

        return inf_loss_lst

    def get_single_para_loo(self, model, X_train, Y_train):
        n = X_train.shape[0]
        theta = np.concatenate((model.coef_, model.intercept_.reshape(-1, 1)), axis=1)
        X_train_aug = self.aug_x(X_train)
        hessian_inv = self.hessian_inverse(theta, X_train_aug)
        para_del = np.zeros((len(Y_train), 1290))

        for i in range(len(Y_train)):
            xi, yi = X_train_aug[i], Y_train[i]
            reg_theta = copy.deepcopy(theta)
            reg_theta[:, -1] = 0
            para_del[i,:] = np.dot(hessian_inv, self.gradient_single(theta, xi, yi) + reg_theta.reshape(-1,1) / n).reshape(1,-1)
        return para_del

    def get_single_para_add(self, model, X_train, Y_train, X_add, Y_add):
        n = X_train.shape[0]
        theta = np.concatenate((model.coef_, model.intercept_.reshape(-1, 1)), axis=1)
        X_train_aug = self.aug_x(X_train)
        X_add_aug = self.aug_x(X_add)
        hessian_inv = self.hessian_inverse(theta, X_train_aug)
        para_del = np.zeros((len(Y_add), 1290))
        for i in range(len(Y_add)):
            xi, yi = X_add_aug[i], Y_add[i]
            reg_theta = copy.deepcopy(theta)
            reg_theta[:, -1] = 0
            para_del[i,:] = np.dot(hessian_inv, self.gradient_single(theta, xi, yi) + reg_theta.reshape(-1,1) / n).reshape(1,-1)
        return para_del


    def get_index_after_removal(self, n_tot, remove_ind):
        ind_bin = np.ones(n_tot)
        ind_bin[remove_ind] = 0
        return np.nonzero(ind_bin)[0].astype(int)



class bin_svm(object):
    # squared hinge loss
    def __init__(self, C):
        self.C = C

    def gradient_single(self, theta, xi, yi):
        zi = np.dot(theta, xi)
        if 1-yi*zi>0:
            return -1*self.C * yi* xi *2*(1-yi*zi)
        else:
            return np.zeros_like(xi)

    def hessian_single(self, theta, xi, yi):
        zi = np.dot(theta, xi)
        if 1 - yi * zi > 0:
            return np.outer(xi, xi) * 2* self.C * yi**2
        else:
            return np.zeros_like(np.outer(xi, xi))

    def hessian_inverse(self, theta, X_syn_aug, y):
        n, d = X_syn_aug.shape
        hessian = np.zeros((d, d))
        hess_reg = np.eye(theta.shape[0])
        # hessian += hess_reg
        for i in range(n):
            hessian += self.hessian_single(theta, X_syn_aug[i], y[i])
        hessian = self.C *hessian + hess_reg
        try:
            hessian_inv = np.linalg.inv(hessian)
        except Exception:
            hessian_inv = np.linalg.inv(hessian + np.random.random(hessian.shape)*(1e-5))

        return hessian_inv

    def inf_test_loss_single(self, theta, hessian_inv, X_train, Y_train, x_test, y_test, n_test, n=None):
        if n == None:
            n = len(Y_train)
        inf_loss_lst = np.zeros(len(Y_train))
        grad_x_train = np.zeros((len(Y_train),X_train.shape[0]))
        for i in range(len(Y_train)):
            xi, yi = X_train[i], Y_train[i]
            # inf_loss = np.dot(np.dot(hessian_inv, self.gradient_single(theta, xi, yi)),
            #                   self.gradient_single(theta, x_test, y_test))
            inf_loss = np.dot(np.dot(hessian_inv, self.gradient_single(theta, xi, yi)+theta/n),
                              self.gradient_single(theta, x_test, y_test)+theta/n_test)
            inf_loss_lst[i] = inf_loss

        return inf_loss_lst

    def aug_x(self, X):
        b = np.ones((X.shape[0], 1))
        X_aug = np.hstack((X, b))
        return X_aug

    def get_test_loss_all(self, model, X_train, Y_train, X_test, Y_test):
        theta = np.concatenate((model.coef_[0], model.intercept_))

        X_train_aug = self.aug_x(X_train)
        X_test_aug = self.aug_x(X_test)
        n_test = X_test_aug.shape[0]

        hessian_inv = self.hessian_inverse(theta, X_train_aug, Y_train)

        n = len(Y_train)
        inf_loss_lst = np.zeros(n)

        for x_test, y_test in  tqdm(zip(X_test_aug, Y_test)):
            inf_loss_lst += self.inf_test_loss_single(theta, hessian_inv, X_train_aug, Y_train, x_test, y_test, n_test)
        return inf_loss_lst

    def get_add_loss_all(self, model, X_train, Y_train, X_add, Y_add, X_test, Y_test):
        theta = np.concatenate((model.coef_[0], model.intercept_))
        X_use = np.concatenate((X_train, X_add), axis=0)
        Y_use = np.concatenate((Y_train, Y_add), axis=0)
        X_train_aug = self.aug_x(X_train)
        X_use_aug = self.aug_x(X_use)
        X_test_aug = self.aug_x(X_test)
        n_test = X_test_aug.shape[0]

        hessian_inv = self.hessian_inverse(theta, X_train_aug, Y_train)
        n = len(Y_add)
        inf_loss_lst = np.zeros(n)

        X_add_aug = self.aug_x(X_add)

        for x_test, y_test in  tqdm(zip(X_test_aug, Y_test)):
            inf_loss_lst += self.inf_test_loss_single(theta, hessian_inv,
                                                      X_add_aug, Y_add, x_test,
                                                      y_test, n_test, n=X_train.shape[0]) #, n=X_use.shape[0]

        return inf_loss_lst

    def get_single_para_loo(self, model, X_train, Y_train):
        n = X_train.shape[0]
        theta = np.concatenate((model.coef_[0], model.intercept_))
        X_train_aug = self.aug_x(X_train)
        hessian_inv = self.hessian_inverse(theta, X_train_aug, Y_train)
        para_del = np.zeros((len(Y_train),X_train.shape[1] + 1))

        for i in range(len(Y_train)):
            xi, yi = X_train_aug[i], Y_train[i]
            reg_theta = copy.deepcopy(theta)
            reg_theta[-1] = 0
            para_del[i,:] = np.dot(hessian_inv, self.gradient_single(theta, xi, yi) + reg_theta / n).reshape(1,-1)
        return para_del

    def get_single_para_add(self, model, X_train, Y_train, X_add, Y_add):
        n = X_train.shape[0]
        theta = np.concatenate((model.coef_[0], model.intercept_))
        X_train_aug = self.aug_x(X_train)
        X_add_aug = self.aug_x(X_add)
        hessian_inv = self.hessian_inverse(theta, X_train_aug, Y_train)
        para_del = np.zeros((len(Y_add),X_train.shape[1] + 1))

        for i in range(len(Y_add)):
            xi, yi = X_add_aug[i], Y_add[i]
            reg_theta = copy.deepcopy(theta)
            reg_theta[-1] = 0
            para_del[i,:] = np.dot(hessian_inv, self.gradient_single(theta, xi, yi) + reg_theta / n).reshape(1,-1)
        return para_del

    def get_index_after_removal(self, n_tot, remove_ind):
        ind_bin = np.ones(n_tot)
        ind_bin[remove_ind] = 0
        return np.nonzero(ind_bin)[0].astype(int)