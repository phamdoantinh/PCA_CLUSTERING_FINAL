import pandas as pd
import numpy as np


def read_data(path: str) -> pd.DataFrame:
    data_df = pd.read_csv(path)
    return data_df


def load_csv(the_file):
    the_array = np.genfromtxt(the_file, delimiter=";", skip_header=0)
    return the_array


def load_data(train_file, test_file, delim, skiphd, num_feature=589):
    train = np.genfromtxt(train_file, delimiter=delim, skip_header=skiphd)
    test = np.genfromtxt(test_file, delimiter=delim, skip_header=skiphd)
    X_train = train[:, 0:num_feature]
    X_train[X_train == 100] = 0
    Y_train = train[:, num_feature:591]
    X_test = test[:, 0:num_feature]
    X_test[X_test == 100] = 0
    Y_test = test[:, num_feature:591]

    return X_train, Y_train, X_test, Y_test
