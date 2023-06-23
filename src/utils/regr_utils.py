import numpy as np
from copy import deepcopy
import pickle


def train_single_model(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train)
    return model.score(X_test, Y_test)


def train_multiple_models(models, X_train, Y_train, X_test, Y_test):
    best_score = 100000.0
    best_model = None

    for model in models:
        score = train_single_model(model, X_train, Y_train, X_test, Y_test)
        if score < best_score:
            best_score = score
            best_model = deepcopy(model)

    return best_model


def train_multiple_labels(models, X_train, Y_train, X_test, Y_test, weight_path):
    best_model_1 = train_multiple_models(models, X_train, Y_train[:, 0], X_test, Y_test[:, 0])
    best_model_2 = train_multiple_models(models, X_train, Y_train[:, 1], X_test, Y_test[:, 1])
    print(f"Best model X: {best_model_1.__class__.__name__} | Best model Y: {best_model_2.__class__.__name__}")
    pickle.dump((best_model_1, best_model_2), open(weight_path, 'wb'))


def predict_multiplt_models(weight_path, X_test):
    (model_1, model_2) = pickle.load(open(weight_path, 'rb'))
    a = model_1.predict(X_test)
    b = model_2.predict(X_test)
    return np.stack((a, b), axis=1)
