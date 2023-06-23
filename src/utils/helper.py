import os.path
import re
import time
import uuid
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from yellowbrick.cluster import SilhouetteVisualizer
from datetime import datetime
from typing import List
import itertools
from sklearn.cluster import KMeans

# Set the default text font size
plt.rc('font', size=16)
# Set the axes title font size
plt.rc('axes', titlesize=16)
# Set the axes labels font size
plt.rc('axes', labelsize=16)
# Set the font size for x tick labels
plt.rc('xtick', labelsize=16)
# Set the font size for y tick labels
plt.rc('ytick', labelsize=16)
# Set the legend font size
plt.rc('legend', fontsize=18)
# Set the font size of the figure title
plt.rc('figure', titlesize=20)


def gen_name_image():
    return f"{uuid.uuid4()}.png"


# setup random seed
def set_seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_current_date():
    curr_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    date = curr_time.split('-')[0]
    return date


def detect_num_feature_rssi(data: pd.DataFrame) -> int:
    columns = data.columns
    pattern = r"^WAP\d+"
    count = 0
    for col in columns:
        check = re.match(pattern, col)
        if check is not None:
            count += 1
    return count


def constant_columns(data: pd.DataFrame, threshold: int = 2, column_range: int = 589) -> List:
    result = []
    for column in data.columns[:column_range]:
        value = pd.unique(data[column]).shape[0]
        if value < threshold:
            result.append(column)

    return result


def normalize_data(data: pd.DataFrame,
                   num_feature=589,
                   removed_columns: list = None,
                   constant_columns=None) -> pd.DataFrame:
    cell = data.iloc[:, 0:num_feature]
    data.iloc[:, 0:num_feature] = np.where(cell <= 0, (cell + 100) / 100, 0)

    # Remove columns
    if removed_columns is not None:
        data = data.drop(columns=removed_columns, axis=1)
    if constant_columns is not None:
        data = data.drop(columns=constant_columns, axis=1)

    return data


def separates_data_uts(data: pd.DataFrame):
    data["Floor_ID"] = data["Floor_ID"].apply(lambda x: x + 3)

    _X = data.drop(['Pos_x', 'Pos_y', 'Floor_ID', 'Building_ID'], axis=1)
    # _y = data[['Pos_x', 'Pos_y', 'Floor_ID', 'Building_ID']]
    _y = data[['Pos_x', 'Pos_y', 'Floor_ID']]
    # _z = data[['Floor_ID']]
    _X = _X.to_numpy()
    _y = _y.to_numpy()
    # _z = _z.to_numpy()

    # Dữ liệu _y gồm [Pos_x, Pos_y, Floor_ID]
    return _X, _y  # , _z.reshape(-1, ).astype('int64') + 3


def separates_data_uji(data: pd.DataFrame):
    _X = data.drop(['LONGITUDE', 'LATITUDE', 'FLOOR'], axis=1)
    # _y = data[['Pos_x', 'Pos_y', 'Floor_ID', 'Building_ID']]
    _y = data[['LONGITUDE', 'LATITUDE', 'FLOOR', "BUILDINGID"]]
    # _z = data[['Floor_ID']]
    _X = _X.to_numpy()
    _y = _y.to_numpy()
    # _z = _z.to_numpy()

    # Dữ liệu _y gồm ['LONGITUDE', 'LATITUDE', 'FLOOR']
    return _X, _y  # , _z.reshape(-1, ).astype('int64') + 3


def separates_data_tampere(data: pd.DataFrame):
    _X = data.drop(['Pos_x', 'Pos_y', 'Floor_ID'], axis=1)
    _y = data[['Pos_x', 'Pos_y', 'Floor_ID']]
    _X = _X.to_numpy()
    _y = _y.to_numpy()

    # Dữ liệu _y gồm [Pos_x, Pos_y, Floor_ID]
    return _X, _y


def mean_error(A, B):
    tot = 0.
    err_a = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        mtot = np.linalg.norm((A[i] - B[i]))
        err_a[i] = mtot
    return err_a.mean(), err_a


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_data_npz(data_info):
    for data_path, data in data_info:
        np.savez_compressed(data_path, data)


def load_data_npz(data_path_list):
    X_train = np.load(data_path_list[0])['arr_0']
    Y_train = np.load(data_path_list[1])['arr_0']
    X_test = np.load(data_path_list[2])['arr_0']
    Y_test = np.load(data_path_list[3])['arr_0']
    X_valid = np.load(data_path_list[4])['arr_0']
    Y_valid = np.load(data_path_list[5])['arr_0']
    return (X_train, Y_train), (X_test, Y_test), (X_valid, Y_valid)


def draw_distribution_cluster(data, data_cluster, title=None, xlabel=None, ylabel=None):
    marker = itertools.cycle(('+', '*', 'o', '.', ','))
    for cluster_id in np.unique(data_cluster):
        plt.scatter(
            data[data_cluster == cluster_id, 0],
            data[data_cluster == cluster_id, 1],
            s=40, marker=next(marker),
            label=f"cluster_{cluster_id}")

    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.xlabel(ylabel)

    plt.legend()
    plt.show()


def draw_kmean_distortion(data, k_ranges, image_save_path="../../images"):
    distortions = []
    for k in k_ranges:
        kmean_model = KMeans(n_clusters=k)
        kmean_model.fit(data)
        distortions.append(kmean_model.inertia_)

    plt.figure(figsize=(16, 8))
    plt.plot(k_ranges, distortions, color='blue', linestyle='solid', marker='o', markerfacecolor='red', markersize=10)
    plt.xlabel("K")
    plt.ylabel("Distortion")
    plt.title("The Elbow Method showing the optimal k")
    plt.savefig(os.path.join(image_save_path, gen_name_image()), bbox_inches='tight', dpi=800)
    plt.show()


def draw_kmean_silhouette(data, k_ranges, image_save_path="../../images"):
    fig, ax = plt.subplots(2, 2, figsize=(15, 8))
    for i in k_ranges:
        km = KMeans(n_clusters=i, init="k-means++", n_init=10, max_iter=100, random_state=2023)
        q, mod = divmod(i, 2)

        visualizer = SilhouetteVisualizer(km, colors="yellowbrick", ax=ax[q - 1][mod])
        visualizer.fit(data)

    fig.savefig(os.path.join(image_save_path, gen_name_image()), bbox_inches='tight', dpi=800)
    visualizer.show()


def get_result_floor_classifier(models, X_train, Y_train, X_test, Y_test, data_type, result_path):
    log_cols = ["index", "model_name", "accuracy", "training_time", "data_type"]
    log = pd.DataFrame(columns=log_cols)

    for idx in models:

        model = models.get(idx)
        time_start = time.time()
        print(model)
        model.fit(X_train, Y_train)
        acc = model.score(X_test, Y_test)
        training_time = time.time() - time_start

        print(f"\tScore: {acc} | Training Time: {training_time}")

        log_entry = pd.DataFrame([[idx, idx, acc, training_time, data_type]], columns=log_cols)
        log = log.append(log_entry, ignore_index=True)

    # save log
    if result_path is not None:
        log.to_csv(result_path, index=False)
