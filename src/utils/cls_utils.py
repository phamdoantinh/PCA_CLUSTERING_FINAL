import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score
from copy import deepcopy
from src.utils.models import save_model_sklearn, load_model_sklearn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier

# constant valud
RANDOM_SEED = 2023
NUM_TREES = 100
MAX_DEPTH = 5

# Toàn bộ các model chạy thực nghiệm training phần classification
models_trial = {
    "knn": KNeighborsClassifier(),
    "logistic": LogisticRegression(),
    "sgd": SGDClassifier(),
    "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_SEED),
    "random_forest": RandomForestClassifier(n_estimators=NUM_TREES, max_depth=MAX_DEPTH, random_state=RANDOM_SEED),
    "svc_1": SVC(kernel="linear", C=0.025, random_state=RANDOM_SEED),
    "svc_2": SVC(gamma=2, C=1, random_state=RANDOM_SEED),
    "svc_3": SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED),
    "mlp": MLPClassifier(random_state=RANDOM_SEED),
    "naive_bayes": GaussianNB(),
    "linear_discriminant": LinearDiscriminantAnalysis(),
    "quadratic_discriminant": QuadraticDiscriminantAnalysis(),
    "adaboost": AdaBoostClassifier(random_state=RANDOM_SEED),
    "gradientboost": GradientBoostingClassifier(n_estimators=NUM_TREES, learning_rate=1.0, max_depth=MAX_DEPTH),
    "xgboost": XGBClassifier(n_estimators=NUM_TREES, random_state=RANDOM_SEED),
}


def train_single_model(model, train_X, train_y, test_X, test_y):
    """
    Training single classification model
    :param model: A model from sklearn library
    :param train_X: X_train (format numpy)
    :param train_y: Label train (format numpy)
    :param test_X: X_test (format numpy)
    :param test_y: Label test (format numpy)
    :return: predictions and accuracy for test data aftering training model
    """
    print(f"==> Single Model: {model.__class__.__name__}")
    model.fit(train_X, train_y)
    predictions = model.predict(test_X)
    acc = accuracy_score(test_y, predictions)
    return predictions, acc


def train_multiple_model(models, X_train, Y_train, X_test, Y_test):
    """
    Training more classification models for choice the best model that has the highest score
    :param models: List of models from sklearn library
    :param X_train: X_train (format numpy)
    :param Y_train: Label train (format numpy)
    :param X_test: X_test (format numpy)
    :param Y_test: Label test (format numpy)
    :return: The best model and the highest score
    """
    best_acc = 0.0
    best_model = None
    for model in models:
        _, acc = train_single_model(model, X_train, Y_train, X_test, Y_test)
        if acc > best_acc:
            best_acc = acc
            best_model = deepcopy(model)
    return best_model, best_acc


class Ensemble_Classic_Model():
    """
    Ensemble là 1 class phục vụ training các model con của các cluster
    (Classic -> Thể hiện việc training các model cluster con này sẽ dùng là Machine Learning (không phải là Deep))
    """
    def __init__(
            self,
            models=None,
            ensemble_model_path=None):
        """
        :param models: List of models from sklearn library
        :param ensemble_model_path: path for save all model classifier
        """
        self.models = models
        self.ensemble_model_path = ensemble_model_path

    def train_student(
            self,
            traindata_X, traindata_Y, train_cluster_label,
            validdata_X, validdata_Y, valid_cluster_label):

        for cluster_id in np.unique(train_cluster_label):
            X_train = traindata_X[train_cluster_label == cluster_id]
            Y_train = traindata_Y[train_cluster_label == cluster_id]

            X_valid = validdata_X[valid_cluster_label == cluster_id]
            Y_valid = validdata_Y[valid_cluster_label == cluster_id]

            print(f"==== Train cluster {cluster_id} regression =====")
            weight_path = os.path.join(self.ensemble_model_path, f'model_{cluster_id}.sav')
            best_model, best_acc = train_multiple_model(self.models, X_train, Y_train, X_valid, Y_valid)
            print(f"[ACC]: {best_acc} | [MODEL] {best_model.__class__.__name__}")

            save_model_sklearn(weight_path, best_model)


def classic_model_evaluation(
        # model_total_path=None,
        model_dict_path=None,
        model_cluster_classifier_path=None,
        # model_cluster_path=None,
        X_test=None,
        Y_test=None):
    # load model
    # model_total = load_model_sklearn(model_total_path)
    model_dict = [load_model_sklearn(path) for path in model_dict_path]
    model_cluster_classifier = load_model_sklearn(model_cluster_classifier_path)
    # model_cluster = load_model_sklearn(model_cluster_path)

    Z_test = model_cluster_classifier.predict(X_test)
    Y_true = []
    Y_pred = []

    for i in np.unique(Z_test):
        test_data_clusters = X_test[Z_test == i]
        test_label_clusters = Y_test[Z_test == i]

        # output_regr_total = model_total.predict(test_data_clusters)
        output_clf_cluter = model_dict[i].predict(test_data_clusters)

        # out_cluster_1 = model_cluster.predict(output_regr_cluter)
        # out_cluster_2 = model_cluster.predict(output_regr_total)
        # check = (out_cluster_1 != out_cluster_2)
        # check = np.vstack([check, check])
        # check = np.transpose(check.astype(int))
        # final_output = (1.0 - check) * output_regr_cluter + check * ((output_regr_total + output_regr_cluter) / 2)

        Y_true.append(test_label_clusters)
        Y_pred.append(output_clf_cluter)

    Y_true = np.hstack(Y_true)
    Y_pred = np.hstack(Y_pred)
    acc = accuracy_score(Y_true, Y_pred)
    print(f"Total Accuracy: {acc}")
