# IPS_PCA_CLUSTERING

## Requirements

## How to use

## Folder structure

```
IPS_PCA_CLUSTERING
  │
  ├── configs/
  ├── data/
  ├── draft/
  ├── images/
  ├── references/
  ├── results/
  ├── src/
  ├── .gitignore
  ├── LICENSE
  └── README.md
```

- **Data Source**: 

:one: [UTS - University of Technology Sydney](https://github.com/XudongSong/UTSIndoorLoc-dataset)

:two:  [UJI - Universitat Jaume I](http://archive.ics.uci.edu/ml/datasets/UJIIndoorLoc)

:three:  [TUT - Technical University of Tampere](https://zenodo.org/record/1001662)

# Illustration Proposal Idea
:one: **Location Regression**
![](images/train_eval/train_regr.jpg?raw=true)
![](images/train_eval/eval_regr.jpg?raw=true)


:two: **Floor (Building) Classification** 
![](images/train_eval/train_eval_clfs.jpg?raw=true)

# Step by Step
## 1. Install enviroment Python
* Create env python by Conda
* Install Anaconda (or Miniconda) (follow below instructions)
    * [Anaconda Installation](https://docs.anaconda.com/anaconda/install/index.html)
    * [Miniconda Installation](https://docs.conda.io/en/main/miniconda.html)

```commandline
git clone https://github.com/phamdoantinh/IPS_PCA_CLUSTERING.git
cd IPS_PCA_CLUSTERING

conda create -n RSSI python=3.7
conda activate RSSI
pip install -r requirements.txt
```

## 2. Run code
🤘 **Step 1**: Data Preparation
  - Run code from :point_right::point_right::point_right: [FILE](./src/notebook/data_notebook.ipynb) :point_left::point_left::point_left:  
  - From data raw from data folder (`data/uts/raw/`), the pre-process it. Such as: concating raw RSSI and label files, normalizing data, removing nullable values, removing constant columns, ... The output of pre-processing data is `data/.../processed/`.
  - Data Decomposition (PCA, KernelPCA, AutoEncoder, VAE, ...). The output of decomposition phase is `data/uts/pca/`.


🤘 **Step 2**: Train & Evaluation Position (X, Y) Model
  - Run code from :point_right::point_right::point_right: [FILE](./src/notebook/loc_notebook.ipynb) :point_left::point_left::point_left:
  - Training Model is saved `models/uts_models/...` folder.

🤘 **Step 3**: Train & Evaluation Floor (Building) Model
  - Run code from :point_right::point_right::point_right: [FILE](./src/notebook/loc_notebook.ipynb) :point_left::point_left::point_left:
  - Training Model is saved `models/uts_models/...` folder.
  - Evaluation Result if saved `results/floor_classification/uts/...`

# Contact
:boom: [Pham Doan Tinh](https://github.com/phamdoantinh) :boom:

:boom: [Bui Huy Hoang](https://github.com/AustrianOakvn) :boom:

:boom: [Nguyen Van Nam](https://github.com/ngnambka00-github) :boom:


```
          _/﹋\_
          (҂`_´)
          <,︻╦╤─ ҉ – – 🍎
          _/﹋\_
```
