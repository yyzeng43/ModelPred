import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

data_path = './data/'
train_df = pd.read_csv(data_path +'training.csv')

# Check which columns have missing data
for cols in train_df:
    test_missing = (train_df[cols] == -999.000)
    print('Column {} has {} missing data'.format(cols, test_missing.sum()))


def replace_missing_data_by_frequent_value(x_train):
    for i in range(x_train.shape[1]):
        if np.any(x_train[:, i] == -999):

            # Collect all the indices of non -999 value
            temp_train = (x_train[:, i] != -999)

            # Calculate frequency
            values, counts = np.unique(x_train[temp_train, i], return_counts=True)
            # Replace -999 by the most frequent value of the columns if there exits at least one non -999 value
            if (len(values) > 1):
                x_train[~temp_train, i] = values[np.argmax(counts)]

            else:
                x_train[~temp_train, i] = 0

    return x_train

def standardize(x, mean = None, std = None):
    # if mean is None:
    #     mean = np.mean(x, axis=0)
    # x = x - mean
    #
    # if std is None:
    #     std = np.std(x, axis=0)
    # x[:,std > 0] = x[:,std > 0] / std[std > 0]

    X_mean = np.linalg.norm(x, ord=2, axis=1)
    X_std = x / np.max(X_mean)

    return X_std


def process_data(x_train):
    x_train = replace_missing_data_by_frequent_value(x_train)
    # Delete 5 phi features
    x_train = np.delete(x_train, [15,18,20,25,28], 1)
    # invese logarithm for all the positive value columns
    # log_index = [0, 1, 2, 5, 7, 9, 10, 13, 16, 19, 21, 23, 26]
    # x_train_log = np.log(1 / (1 + x_train[:, log_index]))
    # x_train = np.hstack((x_train, x_train_log))
    # Standardization
    x_train = standardize(x_train)

    return x_train


train_df['Label'] = pd.factorize(train_df['Label'])[0]
y = train_df['Label'].values
X = train_df.drop(['Label', 'Weight', 'EventId'], axis=1).values

from sklearn.utils import shuffle
X_new, y_new = shuffle(X[:1100], y[:1100], random_state=32)
X_pre = process_data(X_new)

X_train_all, X_test = X_pre[:600], X_pre[600: 1100, :]
y_train_all, y_test = y_new[:600], y_new[600: 1100]

clf = LogisticRegression(C=1, solver='lbfgs', max_iter=10000, random_state=30)
clf.fit(X_train_all[300:], y_train_all[300:])
y_pred = clf.predict(X_test)


clf2 = LinearSVC(C=1, random_state=30)
clf2.fit(X_train_all[300:], y_train_all[300:])
y_pred2 = clf2.predict(X_test)


print(accuracy_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred2))

np.save(data_path + 'X_train_all_pos.npy', X_train_all)
np.save(data_path + 'X_test_pos.npy', X_test)
np.save(data_path + 'y_train_all_pos.npy', y_train_all)
np.save(data_path + 'y_test_pos.npy', y_test)
