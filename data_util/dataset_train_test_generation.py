# use selected data set to generate parameter estimation result with permutation sampling for the training
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from tqdm import tqdm

class LR_multiclass_preparation(object):
    def __init__(self, config):
        '''
        :param config: hyperparameters of logistic regression
        '''
        self.config = config
        self.PATH = config['saving_path']
        self.C = 1/self.config['lambda']
        self.num_class = config['class_number']

    def _create_model(self):
        model = LogisticRegression(max_iter=10000, C = self.C, solver='lbfgs', random_state=self.config['rand_state'])
        return model

    def permutation_generation(self, train_data, train_lab, perm_num):
        # record the permutation encoding and estimation results
        # the data has been normalized

        self.PermResult = {}
        self.PermResult['one_hot'] = []
        self.PermResult['fitted_para'] = []


        n_data = train_data.shape[0] # number of all data points
        feature_dim = train_data.shape[-1]

        for t in tqdm(range(perm_num)):
            np.random.seed(t)
            perm = np.random.permutation(n_data)
            self.PermResult['one_hot'] = []
            self.PermResult['fitted_para'] = []

            for i in tqdm(range(n_data)):
                dataid = perm[i]
                datasid = np.asarray(perm[:i + 1]).flatten()
                one_hot = np.ones(len(train_lab)) * (-1)
                one_hot[datasid] = 1

                self.PermResult['one_hot'].append(one_hot)
                X_use, y_use = train_data[datasid], train_lab[datasid]
                if len(np.unique(y_use)) < self.num_class:
                    if self.num_class == 2:
                        self.PermResult['fitted_para'].append(np.zeros(feature_dim + 1))
                    else:
                        self.PermResult['fitted_para'].append(np.zeros((self.num_class, feature_dim + 1)))
                else:
                    model = self._create_model()
                    model.fit(X_use, y_use)
                    filename = self.PATH + '/ LR_model_n_{}_Iter_{}.sav'.format(i, t)
                    pickle.dump(model, open(filename, 'wb'))
                    fitted_para = np.concatenate((model.coef_, model.intercept_.reshape(-1, 1)), axis=1)
                    self.PermResult['fitted_para'].append(fitted_para)

            pickle.dump([self.PermResult['one_hot'], self.PermResult['fitted_para']],
                    open(self.PATH + '/Perm_{}_Iter_{}_OneEncoding_FittedPara.data'.format(perm_num, t), 'wb'))


    def rep_permutation_generation(self, train_data, train_lab, perm_num, rep_num):
        # add replications to test the result
        self.PermResult = {}
        self.PermResult['one_hot'] = []
        self.PermResult['fitted_para'] = []

        n_data = train_data.shape[0]  # number of all data points
        feature_dim = train_data.shape[-1]

        for r in tqdm(range(rep_num)):
            for t in tqdm(range(perm_num)):
                np.random.seed(t)
                perm = np.random.permutation(n_data)
                self.PermResult['one_hot'] = []
                self.PermResult['fitted_para'] = []

                for i in tqdm(range(n_data)):
                    dataid = perm[i]
                    datasid = np.asarray(perm[:i + 1]).flatten()
                    one_hot = np.ones(len(train_lab)) * (-1)
                    one_hot[datasid] = 1

                    self.PermResult['one_hot'].append(one_hot)
                    X_use, y_use = train_data[datasid], train_lab[datasid]
                    if len(np.unique(y_use)) < self.num_class:
                        self.PermResult['fitted_para'].append(np.zeros((self.num_class, feature_dim + 1)))
                    else:
                        model = self._create_model()
                        model.fit(X_use, y_use)
                        fitted_para = np.concatenate((model.coef_, model.intercept_.reshape(-1, 1)), axis=1)
                        self.PermResult['fitted_para'].append(fitted_para)

                pickle.dump([self.PermResult['one_hot'], self.PermResult['fitted_para']],
                            open(self.PATH + '/Comparison/Rep_{}_Iter_{}_OneEnc_FittedPara.data'.format(r, t), 'wb'))

    def random_sampling_generation(self, train_data, train_lab, num, size_min, size_max, random_state):

        self.rand_sets = {}
        self.rand_sets['one_hot'] = []
        self.rand_sets['fitted_para'] = []

        N = train_data.shape[0]
        feature_dim = train_data.shape[-1]
        np.random.seed(random_state)

        for i in tqdm(range(num)):
            n_select = np.random.choice(range(size_min, size_max))
            subset_index = np.random.choice(range(N), n_select, replace=False)

            one_hot = np.ones(len(train_lab)) * (-1)
            one_hot[subset_index] = 1
            self.rand_sets['one_hot'].append(one_hot)
            X_use, y_use = train_data[subset_index], train_lab[subset_index]
            if len(np.unique(y_use)) < self.num_class:
                if self.num_class == 2:
                    self.rand_sets['fitted_para'].append(np.zeros((1,feature_dim + 1)))
                else:
                    self.rand_sets['fitted_para'].append(np.zeros((self.num_class, feature_dim + 1)))
            else:
                model = self._create_model()
                model.fit(X_use, y_use)
                fitted_para = np.concatenate((model.coef_, model.intercept_.reshape(-1, 1)), axis=1)
                self.rand_sets['fitted_para'].append(fitted_para)

        pickle.dump([self.rand_sets['one_hot'], self.rand_sets['fitted_para']],
                    open(self.PATH + '/Rand_{}_OneEncoding_FittedPara.data'.format(num), 'wb'))



class SVM_multiclass_preparation(object):

    def __init__(self, config):
        '''
        :param config: hyperparameters of logistic regression
        '''
        self.config = config
        self.PATH = config['saving_path']
        self.C = 1 / self.config['lambda']
        self.num_class = config['class_number']

    def _create_model(self):
        model = LinearSVC(max_iter=10000, C=self.C, multi_class='ovr', dual = True,
                          loss = 'squared_hinge', penalty='l2',
                                   random_state=self.config['rand_state'])
        return model

    def permutation_generation(self, train_data, train_lab, perm_num):
        # record the permutation encoding and estimation results
        # the data has been normalized

        self.PermResult = {}
        self.PermResult['one_hot'] = []
        self.PermResult['fitted_para'] = []


        n_data = train_data.shape[0] # number of all data points
        feature_dim = train_data.shape[-1]

        for t in tqdm(range(perm_num)):
            np.random.seed(t)
            perm = np.random.permutation(n_data)
            self.PermResult['one_hot'] = []
            self.PermResult['fitted_para'] = []

            for i in tqdm(range(n_data)):
                dataid = perm[i]
                datasid = np.asarray(perm[:i + 1]).flatten()
                one_hot = np.ones(len(train_lab)) * (-1)
                one_hot[datasid] = 1

                self.PermResult['one_hot'].append(one_hot)
                X_use, y_use = train_data[datasid], train_lab[datasid]
                if len(np.unique(y_use)) < self.num_class:
                    if self.num_class == 2:
                        self.PermResult['fitted_para'].append(np.zeros(feature_dim + 1))
                    else:
                        self.PermResult['fitted_para'].append(np.zeros((self.num_class, feature_dim + 1)))
                else:
                    model = self._create_model()
                    model.fit(X_use, y_use)
                    # filename = self.PATH + '/ SVM_model_n_{}_Iter_{}.sav'.format(i, t)
                    # pickle.dump(model, open(filename, 'wb'))
                    fitted_para = np.concatenate((model.coef_, model.intercept_.reshape(-1, 1)), axis=1)
                    self.PermResult['fitted_para'].append(fitted_para)



            pickle.dump([self.PermResult['one_hot'], self.PermResult['fitted_para']],
                    open(self.PATH + '/Perm_{}_Iter_{}_OneEncoding_FittedPara.data'.format(perm_num, t), 'wb'))


    def rep_permutation_generation(self, train_data, train_lab, perm_num, rep_num):
        # add replications to test the result
        self.PermResult = {}
        self.PermResult['one_hot'] = []
        self.PermResult['fitted_para'] = []

        n_data = train_data.shape[0]  # number of all data points
        feature_dim = train_data.shape[-1]

        for r in tqdm(range(rep_num)):
            for t in tqdm(range(perm_num)):
                np.random.seed(t)
                perm = np.random.permutation(n_data)
                self.PermResult['one_hot'] = []
                self.PermResult['fitted_para'] = []

                for i in tqdm(range(n_data)):
                    dataid = perm[i]
                    datasid = np.asarray(perm[:i + 1]).flatten()
                    one_hot = np.ones(len(train_lab)) * (-1)
                    one_hot[datasid] = 1

                    self.PermResult['one_hot'].append(one_hot)
                    X_use, y_use = train_data[datasid], train_lab[datasid]
                    if len(np.unique(y_use)) < self.num_class:
                        self.PermResult['fitted_para'].append(np.zeros((self.num_class, feature_dim + 1)))
                    else:
                        model = self._create_model()
                        model.fit(X_use, y_use)
                        fitted_para = np.concatenate((model.coef_, model.intercept_.reshape(-1, 1)), axis=1)
                        self.PermResult['fitted_para'].append(fitted_para)

                pickle.dump([self.PermResult['one_hot'], self.PermResult['fitted_para']],
                            open(self.PATH + '/Comparison/Rep_{}_Iter_{}_OneEnc_FittedPara.data'.format(r, t), 'wb'))

    def random_sampling_generation(self, train_data, train_lab, num, size_min, size_max, random_state):

        self.rand_sets = {}
        self.rand_sets['one_hot'] = []
        self.rand_sets['fitted_para'] = []

        N = train_data.shape[0]
        feature_dim = train_data.shape[-1]
        np.random.seed(random_state)

        for i in tqdm(range(num)):
            n_select = np.random.choice(range(size_min, size_max))
            subset_index = np.random.choice(range(N), n_select, replace=False)

            one_hot = np.ones(len(train_lab)) * (-1)
            one_hot[subset_index] = 1
            self.rand_sets['one_hot'].append(one_hot)
            X_use, y_use = train_data[subset_index], train_lab[subset_index]
            if len(np.unique(y_use)) < self.num_class:
                if self.num_class == 2:
                    self.rand_sets['fitted_para'].append(np.zeros((1,feature_dim + 1)))
                else:
                    self.rand_sets['fitted_para'].append(np.zeros((self.num_class, feature_dim + 1)))
            else:
                model = self._create_model()
                model.fit(X_use, y_use)
                fitted_para = np.concatenate((model.coef_, model.intercept_.reshape(-1, 1)), axis=1)
                self.rand_sets['fitted_para'].append(fitted_para)

        pickle.dump([self.rand_sets['one_hot'], self.rand_sets['fitted_para']],
                    open(self.PATH + '/Rand_{}_OneEncoding_FittedPara.data'.format(num), 'wb'))

























