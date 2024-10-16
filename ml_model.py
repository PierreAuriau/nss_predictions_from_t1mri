"""
We aim at giving learning curves of classical ML models on AUSZ for:
* Prediction of NSS
* VBM pre-processing

"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import re
import itertools

import nibabel as nib
from collections import defaultdict
from multiprocessing import Pool

from scipy.ndimage import binary_dilation, gaussian_filter

import sklearn.linear_model as lm
from sklearn import preprocessing
import sklearn.svm as svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, \
                            r2_score, roc_auc_score, balanced_accuracy_score

from config import Config

config = Config()

def load_data(preproc, texture=None):
    
    if preproc == "vbm":
        arr = np.load(os.path.join(config.path2interim, "ausz_t1mri_mwp1_gs-raw_data64.npy"), mmap_mode="r")
        m_vbm = nib.load(os.path.join(config.path2brainmasks, "mni_cerebrum-gm-mask_1.5mm.nii.gz"))
        brain_mask = (m_vbm.get_fdata() != 0).astype(bool)
    elif preproc == "skeleton":
        arr_raw = np.load(os.path.join(config.path2interim, "ausz_t1mri_skeleton_data32.npy"), mmap_mode="r")
        arr = np.zeros_like(arr_raw)
        for i, img in enumerate(arr_raw):  
            arr[i, 0, ...] = gaussian_filter(img[0], sigma=(1.0, 1.0, 1.0), mode="constant", cval=0.0, radius=(2, 2, 2))
        m_vbm = nib.load(os.path.join(config.path2brainmasks, "mni_cerebrum-gm-mask_1.5mm.nii.gz"))
        brain_mask = (m_vbm.get_fdata() != 0).astype(bool)
        brain_mask = np.pad(brain_mask, pad_width = ((3, 4), (3, 4), (3, 4)))
        brain_mask = binary_dilation(brain_mask, iterations=4, border_value=0, origin=0)
    elif preproc == "freesurfer":
        arr = np.load(os.path.join(config.path2interim, "ausz_freesurfer_thickness_curv_area_sulc.npy"), mmap_mode="r")
        with open(os.path.join(config.path2interim, "channels.txt"), "r") as f:
            channels = [re.search("texture-([a-z]+)", l).group(1) for l in f.readlines()]
        arr = arr[:, channels.index(texture), ...]
        brain_mask = None
    df = pd.read_csv(os.path.join(config.tmp, "processed", "ausz_t1mri_participants.csv"), dtype=config.id_types)
    scheme = pd.read_csv(os.path.join(config.tmp, "processed", "nss_stratified_10_fold_ausz.csv"), dtype=config.id_types)

    print(f"Nb of subjects in scheme: {len(scheme)}")
    print(f"Nb of subjects with metadata: {len(df)}")
    print(f"Data shape : {arr.shape}")
    if brain_mask is not None:
        print(f"Brain mask shape : {brain_mask.shape}")
        arr = arr[:, 0, brain_mask].astype(np.float32)
        print(f"Data shape after applying mask: {arr.shape}")
    assert (df[["participant_id", "session"]] == scheme[["participant_id", "session"]]).all().all(), \
           print("Scheme and participant dataframe do not have same order.")
    return arr, df, scheme

def get_model_cv(model, scaler, cv_train=3):
    steps = []
    if scaler:
        if model in ("logreg", "lrl2", "lrenet"):
            steps.append(preprocessing.StandardScaler())
        elif model in ("svmrbf", "forest", "gb"):
            steps.append(preprocessing.MinMaxScaler())
    models_cv = {
    "logreg_cv": GridSearchCV(lm.LogisticRegression(max_iter=1000),
                              param_grid={"C": 10. ** np.arange(-1, 3)},
                              cv=cv_train, n_jobs=1),
    "lrl2_cv": GridSearchCV(lm.Ridge(),
                            param_grid={'alpha': 10. ** np.arange(-1, 3)},
                            cv=cv_train, n_jobs=1),

    "lrenet_cv": GridSearchCV(lm.ElasticNet(max_iter=1000),
                     param_grid={'alpha': 10. ** np.arange(-1, 2),
                                 'l1_ratio': [.1, .5]},
                     cv=cv_train, n_jobs=1),

    "svmrbf_cv": GridSearchCV(svm.SVR(),
                     # {'kernel': ['poly', 'rbf'], 'C': 10. ** np.arange(-3, 3)},
                     param_grid={'kernel': ['poly', 'rbf'],
                                 'C': 10. ** np.arange(-1, 2)},
                     cv=cv_train, n_jobs=1),

    "forest_cv": GridSearchCV(RandomForestRegressor(random_state=1),
                     param_grid={"n_estimators": [100]},
                     cv=cv_train, n_jobs=1),

    "gb_cv": GridSearchCV(estimator=GradientBoostingRegressor(random_state=1),
                     param_grid={"n_estimators": [100],
                                 "subsample":[1, .5],
                                 "learning_rate": [.1, .5]
                                 },
                     cv=cv_train, n_jobs=1)}
    steps.append(models_cv[model + "_cv"])
    return make_pipeline(*steps)

def train(arr, df, scheme, model, label, preproc, scaler, 
          saving_dir, save_y_pred=False, texture=None):

    model_cv = get_model_cv(model, scaler)
    
    nb_folds = 10
    logs = defaultdict(list)
    
    for fold in range(nb_folds):
        print(f"# Fold: {fold}")

        # 0) Load data
        train_mask = scheme[f"fold{fold}"] == "train"
        train_data = arr[train_mask]
        y_train = df.loc[train_mask, label]

        test_mask = scheme[f"fold{fold}"] == "test"
        test_data = arr[test_mask]
        
        if label in ["sex", "diagnosis"]:
            y_train = y_train.replace(config.label_mapping).values.astype(int)
            y_test = y_test.replace(config.label_mapping).values.astype(int)
            
            metrics = {
                "roc_auc": lambda y_pred, y_true: roc_auc_score(y_true=y_true, y_score=y_pred[:, 1]),
                "bacc": lambda y_pred, y_true: balanced_accuracy_score(y_true=y_true, y_pred=(y_pred[:, 1] > 0.5).astype(int))
            }
            
        else:
            y_train = y_train.values.astype(np.float32)
            y_test = y_test.values.astype(np.float32)
            
            metrics = {
                "r2": r2_score,
                "rmse": lambda y_pred, y_true: mean_squared_error(y_pred=y_pred, y_true=y_true, squared=False),
                "mae": mean_absolute_error
            }

        # 2) Training    
        model_cv.fit(train_data, y_train)

        # 3) Testing
        print(f"Score on test set: {model_cv.score(test_data, y_test)}")
        for split, data, y_true in zip(["train", "test"], 
                                       [train_data, test_data], 
                                       [y_train, y_test]):
            if label in ["sex", "diagnosis"]:
                y_pred = model_cv.predict_proba(data)
            else:
                y_pred = model_cv.predict(data)

            for name, metric in metrics.items():
                try:
                    value = metric(y_true=y_true, y_pred=y_pred)
                except ValueError as e:
                    print(f"ValueError during evaluation: {e}")
                    continue

                # 4) Saving
                if save_y_pred:
                    if texture is None:
                        filename = f"y_pred_label-{label}_preproc-{preproc}_fold-{fold}.npy"
                    else:
                        filename = f"y_pred_label-{label}_preproc-{preproc}_texture-{texture}_fold-{fold}.npy"
                    np.save(os.path.join(saving_dir, filename), y_pred)

                logs["label"].append(label)
                logs["fold"].append(fold)
                logs["split"].append(split)
                logs["model"].append(model)
                logs["scaler"].append(scaler)
                logs["preproc"].append(preproc)
                logs["metric"].append(name)
                logs["value"].append(value)
                if texture is not None:
                    logs["texture"].append(texture)
    
    logs_df = pd.DataFrame(logs)
    if texture is None:
        filename = f"preproc-{preproc}_model-{model}_label-{label}.csv"
    else:
        filename = f"preproc-{preproc}_texture-{texture}_model-{model}_label-{label}.csv"
    logs_df.to_csv(os.path.join(saving_dir, filename), sep=",", index=False)
