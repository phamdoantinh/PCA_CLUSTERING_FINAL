import numpy as np
import os
import pickle
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score
from sklearn.utils import compute_class_weight

from src.utils.loss import MeanLoss
from src.utils.dataset import RSSIDataset
from src.utils.helper import clip_gradient, get_current_date, mean_error
import src.utils.regr_utils as regr


class RegressionNet(nn.Module):
    def __init__(self, cluster=True, n_cluster=2, num_in=100):
        super(RegressionNet, self).__init__()

        self.fc_in = nn.Sequential(
            nn.Linear(num_in, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.init_weight()

        if cluster:
            num_classes = 2
        else:
            num_classes = 2
        self.fc_out = nn.Linear(16, num_classes)

    def init_weight(self):
        for m in self.modules():
            if m.__class__.__name__.startswith('Linear'):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.0001)

    def forward(self, x):
        x = self.fc_in(x)
        x = F.relu(self.fc_out(x))
        return x


# regression model for each cluster
class Ensemble_Model():
    def __init__(
            self,
            n_cluster=2,
            num_in=100,
            teacher=None,
            cluster_model=None,
            ensemble_model_path=None,
            weight_sharing=0.95):

        self.cluster_model = cluster_model
        self.n_cluster = n_cluster
        self.num_in = num_in
        self.ensemble_model_path = ensemble_model_path

        self.teacher = teacher
        self.students = {}

        # n_cluster mean number of cluster or number of student
        for i in range(n_cluster):
            self.students[i] = self.init_model()

        self.loss = MeanLoss()
        self.weight_sharing = weight_sharing

    def init_model(self):
        return RegressionNet(cluster=True, n_cluster=self.n_cluster, num_in=self.num_in)

    @torch.no_grad()
    def share_knowledge(self, student):
        if self.teacher is not None:
            for param_t, param_s in zip(self.teacher.parameters(), student.parameters()):
                param_s.data = param_s.data * self.weight_sharing + param_t.data * (1.0 - self.weight_sharing)

    def train_student(
            self,
            traindata_X, traindata_Y, train_cluster_label,
            validdata_X, validdata_Y, valid_cluster_label):

        for cluster_id in np.unique(train_cluster_label):
            train_data_clusters = traindata_X[train_cluster_label == cluster_id]
            train_label_clusters = traindata_Y[train_cluster_label == cluster_id]
            train_data = RSSIDataset(train_data_clusters, train_label_clusters)
            train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True, drop_last=True)

            valid_data_clusters = validdata_X[valid_cluster_label == cluster_id]
            valid_label_clusters = validdata_Y[valid_cluster_label == cluster_id]
            valid_data = RSSIDataset(valid_data_clusters, valid_label_clusters)
            valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False)

            print("\n",
                  "=" * 10,
                  f"Training cluster {cluster_id} with dataset contain {len(train_data)} train samples and "
                  f"{len(valid_data)} valid samples ==========\n")

            weight_path = os.path.join(self.ensemble_model_path, f'model_{cluster_id}.pth')

            train_regression(
                model=self.students[cluster_id],
                metric=self.loss,
                train_dataloader=train_dataloader,
                valid_dataloader=valid_dataloader,
                weight_path=weight_path)


class Ensemble_Classic_Model():
    def __init__(
            self,
            models=None,
            ensemble_model_path=None):
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

            regr.train_multiple_labels(self.models, X_train, Y_train, X_valid, Y_valid, weight_path)


class ClassificationNet(nn.Module):
    def __init__(self, num_in=100, num_out=2):
        super(ClassificationNet, self).__init__()

        self.fc_in = nn.Sequential(
            nn.Linear(num_in, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            # nn.Linear(64, 32),
            # nn.BatchNorm1d(32),
            # nn.ReLU(),
            # nn.Dropout(0.2),
            #
            # nn.Linear(32, 16),
            # nn.BatchNorm1d(16),
            # nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(64, num_out),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        :param x: Dạng torch
        :return:
        """
        x = self.fc_in(x)
        return x

    def predict(self, x):
        """
        :param x: Dạng numpy
        :return:
        """
        x = torch.from_numpy(x).float()
        result = self.forward(x)
        _, predicted = torch.max(result.data, 1)
        return predicted.detach().numpy()

    def score(self, X_test, Y_test):
        """
        :param X_test: Dạng numpy
        :param Y_test: Dạng numpy
        :return:
        """
        Y_predict = self.predict(X_test)
        return accuracy_score(Y_test, Y_predict)


def train_regression(
        model=None,
        metric=None,
        train_dataloader=None,
        valid_dataloader=None,
        weight_path=None):
    # model = RegressionNet(cluster=cluster)
    # epochs = 50
    epochs = 50
    loss_list = []
    loss_eval = []
    best_test_loss = 1000000.0
    clip = 0.5

    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.004, momentum=0.9, nesterov=True)
    # optimizer = optim.Adam(model.parameters(), lr=0.004, weight_decay=0.001)
    decay_rate = 0.96
    my_lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_rate)

    for epoch in range(1, epochs + 1):
        total_loss = []
        # train phrase
        # model.train()
        with tqdm(train_dataloader, unit="batch") as train_tqdm:
            for i, (data, target) in enumerate(train_tqdm):
                train_tqdm.set_description(f"Epoch {epoch} train")

                optimizer.zero_grad()
                output = model(data.float())
                loss = metric(output.float(), target.float())
                loss.backward()
                clip_gradient(optimizer, clip)
                optimizer.step()

                total_loss.append(loss.item())
                train_tqdm.set_postfix(AVG_LOSS=sum(total_loss) / len(total_loss), LOSS_ITER=loss.item())

        my_lr_scheduler.step()
        loss_list.append(sum(total_loss) / len(total_loss))

        # validation phrase
        if epoch % 1 == 0:
            test_loss = 0
            with torch.no_grad():
                model.eval()
                with tqdm(valid_dataloader, unit="batch") as valid_tqdm:
                    for inputs, labels in valid_tqdm:
                        valid_tqdm.set_description(f"Epoch {epoch} val")

                        output = model(inputs.float())
                        loss = metric(output.float(), target.float())
                        test_loss += loss.item()

                        valid_tqdm.set_postfix(AVG_LOSS=test_loss / len(valid_dataloader), LOSS_ITER=loss.item())

                    test_loss /= len(valid_dataloader)
                    loss_eval.append(test_loss)

                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    torch.save(model.state_dict(), weight_path)
                    print('  ' * 50, 'Best valid loss: {:.4f}'.format(best_test_loss))

    return loss_list, loss_eval, weight_path


def train_classification(
        model=None,
        X_train=None,
        Y_train=None,
        X_valid=None,
        Y_valid=None,
        weight_path=None):
    '''
    :param model: model classification
    :param X_train: Dạng numpy
    :param Y_train: Dạng numpy
    :param X_valid: Dạng numpy
    :param Y_valid: Dạng numpy
    :param weight_path: str
    :return:
    '''

    train_dataset = RSSIDataset(X_train, Y_train)
    valid_dataset = RSSIDataset(X_valid, Y_valid)

    train_dataloader = DataLoader(train_dataset, batch_size=16, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1)

    epochs = 50
    loss_train = []
    loss_eval = []
    accuracy_train = []
    accuracy_eval = []
    best_accuracy = 0.0
    clip = 0.5

    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(Y_train), y=Y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    metric = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=0.004, momentum=0.9, nesterov=True)
    # optimizer = optim.Adam(model.parameters(), lr=0.004, weight_decay=0.001)
    decay_rate = 0.96
    my_lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_rate)

    # Define your execution device
    # device = torch.device("cpu")
    # model.to(device)

    model.train()
    for epoch in range(1, epochs + 1):  # loop over the dataset multiple times
        running_train_loss = []
        running_train_acc = 0.0
        total_train_acc = 0

        # Training pharse
        # model.train()
        with tqdm(train_dataloader, unit="batch") as train_tqdm:
            for i, (data, target) in enumerate(train_tqdm):
                train_tqdm.set_description(f"Epoch {epoch} train")

                optimizer.zero_grad()
                output = model(data.float())
                loss = metric(output.float(), target.long())
                _, predicted = torch.max(output.data, 1)
                loss.backward()
                clip_gradient(optimizer, clip)
                optimizer.step()

                running_train_loss.append(loss.item())
                total_train_acc += target.size(0)
                running_train_acc += (predicted == target).sum().item()

                train_tqdm.set_postfix(AVG_LOSS=sum(running_train_loss) / len(running_train_loss),
                                       LOSS_ITER=loss.item(), ACC_ITER=running_train_acc)

        my_lr_scheduler.step()
        loss_train.append(sum(running_train_loss) / len(running_train_loss))
        accuracy_train.append(100 * running_train_acc / total_train_acc)

        # Validation pharse
        if epoch % 1 == 0:
            running_eval_loss = 0.0
            running_eval_acc = 0.0
            total_eval_acc = 0

            with torch.no_grad():
                model.eval()
                with tqdm(valid_dataloader, unit="batch") as valid_tqdm:
                    for inputs, labels in valid_tqdm:
                        valid_tqdm.set_description(f"Epoch {epoch} val")

                        output = model(inputs.float())
                        _, predicted = torch.max(output.data, 1)
                        loss = metric(output.float(), labels.long())

                        running_eval_loss += loss.item()
                        total_eval_acc += labels.size(0)
                        running_eval_acc += (predicted == labels).sum().item()

                        valid_tqdm.set_postfix(AVG_LOSS=running_eval_loss / len(valid_dataloader),
                                               LOSS_ITER=loss.item(), ACC_ITER=running_eval_acc)

                    running_eval_loss /= len(valid_dataloader)
                    loss_eval.append(running_eval_loss)
                    acc = 100 * running_eval_acc / total_eval_acc
                    accuracy_eval.append(acc)

                    if acc > best_accuracy:
                        best_accuracy = acc
                        torch.save(model.state_dict(), weight_path)
                        print('  ' * 50, 'Best valid accuracy: {:.4f}'.format(best_accuracy))

    return loss_train, loss_eval, accuracy_train, accuracy_eval


def load_model_regression(model_path=None, cluster=False, n_cluster=2, num_in=100):
    model = RegressionNet(cluster=cluster, n_cluster=n_cluster, num_in=num_in)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def load_model_classification(model_path=None, num_in=100, num_out=2):
    model = ClassificationNet(num_in=num_in, num_out=num_out)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def save_model_sklearn(model_path, model):
    pickle.dump(model, open(model_path, 'wb'))
    print("===== Save model sklearn successfully =====")


def load_model_sklearn(model_path):
    loaded_model = pickle.load(open(model_path, 'rb'))
    print("===== Load model sklearn successfully =====")
    return loaded_model


def model_evaluation(
        model_total_path=None,
        model_dict_path=None,
        model_cluster_classifier_path=None,
        model_cluster_path=None,
        metric=None,
        num_in=100,
        X_test=None,
        Y_test=None):
    # load model
    model_total = load_model_regression(model_path=model_total_path, cluster=False, num_in=num_in)
    model_dict = [load_model_regression(model_path=path, cluster=True, num_in=num_in) for path in model_dict_path]
    model_cluster_classifier = load_model_sklearn(model_cluster_classifier_path)
    model_cluster = load_model_sklearn(model_cluster_path)

    total_loss = 0
    Z_test = model_cluster_classifier.predict(X_test)
    for i in np.unique(Z_test):
        test_data_clusters = X_test[Z_test == i]
        test_label_clusters = Y_test[Z_test == i]

        test_data = RSSIDataset(test_data_clusters, test_label_clusters)
        test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

        model_dict_regr = model_dict[i]
        l = 0
        for inputs, labels in test_dataloader:
            output_regr_total = model_total(inputs.float()).long().detach().numpy()
            output_regr_cluter = model_dict_regr(inputs.float()).long().detach().numpy()

            out_cluster_1 = model_cluster.predict(output_regr_cluter)
            out_cluster_2 = model_cluster.predict(output_regr_total)

            if out_cluster_1 != out_cluster_2:
                final_output = (output_regr_total + output_regr_cluter) / 2
            else:
                final_output = output_regr_cluter

            # check = (out_cluster_1 != out_cluster_2)
            # check = np.vstack([check, check])
            # check = np.transpose(check.astype(int))
            # final_output = (1.0 - check) * output_regr_cluter + check * ((output_regr_total + output_regr_cluter) / 2)

            loss = metric(torch.from_numpy(final_output), labels)
            l += loss.detach().numpy()
            total_loss += loss.detach().numpy()
        print("Cluster {} loss: ".format(i), l / len(test_dataloader))

    print(f"total loss: {total_loss / len(X_test)}")


def classic_model_evaluation(
        model_total_path=None,
        model_dict_path=None,
        model_cluster_classifier_path=None,
        model_cluster_path=None,
        X_test=None,
        Y_test=None):
    # load model
    model_cluster_classifier = load_model_sklearn(model_cluster_classifier_path)
    model_cluster = load_model_sklearn(model_cluster_path)

    Z_test = model_cluster_classifier.predict(X_test)
    Y_true = []
    Y_pred = []

    for i in np.unique(Z_test):
        test_data_clusters = X_test[Z_test == i]
        test_label_clusters = Y_test[Z_test == i]

        output_regr_total = regr.predict_multiplt_models(model_total_path, test_data_clusters)
        output_regr_cluter = regr.predict_multiplt_models(model_dict_path[i], test_data_clusters)

        out_cluster_1 = model_cluster.predict(output_regr_cluter)
        out_cluster_2 = model_cluster.predict(output_regr_total)

        check = (out_cluster_1 != out_cluster_2)
        check = np.vstack([check, check])
        check = np.transpose(check.astype(int))

        final_output = (1.0 - check) * output_regr_cluter + check * ((output_regr_total + output_regr_cluter) / 2)

        Y_true.append(test_label_clusters)
        Y_pred.append(final_output)

    Y_true = np.vstack(Y_true)
    Y_pred = np.vstack(Y_pred)
    loss_total = mean_error(Y_true, Y_pred)
    print(f"total loss: {loss_total[0]}")
