# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import logging
import argparse
from multiprocessing import Pool

from config import Config
from ml_model import load_data, train

config = Config()

def predict_nss_from_skeleton(saving_folder):
    """ Function to train a model to predict NSS from skeleton pre-processing.
    """
    preproc = "skeleton"
    saving_dir = os.path.join(config.path2models, preproc, saving_folder)
    os.makedirs(saving_dir, exist_ok=True)
    arr, df, scheme = load_data(preproc)
    train(arr, df, scheme, model="lrl2", label="NSS", preproc=preproc, scaler=False, 
          saving_dir=saving_dir, save_y_pred=True)

def predict_nss_from_vbm(saving_folder):
    """ Function to train a model to predict NSS from VBM pre-processing.
    """
    preproc = "vbm"
    saving_dir = os.path.join(config.path2models, preproc, saving_folder)
    os.makedirs(saving_dir, exist_ok=True)
    arr, df, scheme = load_data(preproc)
    train(arr, df, scheme, model="lrl2", label="NSS", preproc=preproc, scaler=True, 
          saving_dir=saving_dir, save_y_pred=True)

def predict_metadata_from_skeleton(saving_folder):
    """ Function to train a model to predict sex, age, diagnosis and TIV 
    from skeleton pre-processing.
    """
    preproc = "skeleton"
    saving_dir = os.path.join(config.path2models, preproc, saving_folder)
    os.makedirs(saving_dir, exist_ok=True)
    scaler = False
    arr, df, scheme = load_data(preproc)
    labels = ["age", "sex", "diagnosis", "tiv"]
    models = ["logreg" if l in ["diagnosis", "sex"] else "lrl2" for l in labels]
    list_args = [(arr, df, scheme, m, l, preproc, scaler, saving_dir) for m, l in zip(models, labels)]
    with Pool() as pool:
        pool.starmap(train, list_args)

def predict_metadata_from_vbm(saving_folder):
    """ Function to train a model to predict sex, age, diagnosis and TIV 
    from VBM pre-processing.
    """
    preproc = "vbm"
    saving_dir = os.path.join(config.path2models, preproc, saving_folder)
    os.makedirs(saving_dir, exist_ok=True)
    scaler = True
    arr, df, scheme = load_data(preproc)
    labels = ["age", "sex", "diagnosis", "tiv"]
    models = ["logreg" if l in ["diagnosis", "sex"] else "lrl2" for l in labels]
    list_args = [(arr, df, scheme, m, l, preproc, scaler, saving_dir) for m, l in zip(models, labels)]
    with Pool() as pool:
        pool.starmap(train, list_args)

def predict_nss_from_random_models(saving_folder):
    """ Function to train models to predict random permutation of NSS scores
    from skeleton pre-processing.
    NB : permutation-0 is the true NSS scores (no permutation).
    """
    preproc = "skeleton"
    saving_dir = os.path.join(config.path2models, preproc, saving_folder)
    os.makedirs(saving_dir, exist_ok=True)
    scaler = False
    arr, df, scheme = load_data(preproc)
    n_permutations = 1000
    label = "NSS"
    permutations = {}
    permutations[f"{label}_permutation-0"] = df[label].values
    for i in range(1, n_permutations):
        permutations[f"{label}_permutation-{i}"] = np.random.permutation(df[label].values)
    permutations_df = pd.DataFrame(permutations)
    df = pd.concat((df, permutations_df), axis=1)
    labels = [f"{label}_permutation-{i}" for i in range(n_permutations)]
    model = "lrl2"
    list_args = [(arr, df, scheme, model, l, preproc, scaler, saving_dir) for l in labels]
    with Pool() as pool:
        pool.starmap(train, list_args)

def predict_nss_from_freesurfer_textures(saving_folder):
    """ Function to train a model to predict NSS from freesurfer textures:
    area, thickness, curv and sulc.
    """
    preproc = "freesurfer"
    saving_dir = os.path.join(config.path2models, preproc, saving_folder)
    os.makedirs(saving_dir, exist_ok=True)
    scaler = True
    label = "NSS"
    model = "lrl2"
    save_y_pred = True
    list_args = []
    for texture in ["area", "thickness", "curv", "sulc"]:
        arr, df, scheme = load_data(preproc, texture=texture)
        list_args.append((arr, df, scheme, model, label, preproc, scaler,
                          saving_dir, save_y_pred, texture))
    with Pool() as pool:
        pool.starmap(train, list_args)

def predict_nss_with_several_ml_models(saving_folder):
    """ Function to train several ml models to predict NSS from skeleton 
    pre-processing. Models are : Ridge, Elastic Net, SVM with RBF kernel,
    random forest and gradient boosting.
    """
    preproc = "skeleton"
    saving_dir = os.path.join(config.path2models, preproc, saving_folder)
    os.makedirs(saving_dir, exist_ok=True)
    scaler = False
    arr, df, scheme = load_data(preproc)
    label = "NSS"
    models = ["lrl2", "lrenet", "svmrbf", "forest", "gb"]
    list_args = [(arr, df, scheme, m, label, preproc, scaler, saving_dir) for m in models]
    with Pool() as pool:
        pool.starmap(train, list_args)

"""
def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--saving_dir", required=True, type=str, help="Saving directory.")
    parser.add_argument("--models", type=str, nargs="+", default=["lrl2"], help="Models to trained", 
                         choices=["lrl2", "lrenet", "svmrbf", "forest", "gb"])
    parser.add_argument("--preproc", required=True, type=str, choices=["vbm", "skeleton"], help="Data preprocessing")
    parser.add_argument("--scaler", action="store_true", help="Scale the data before training.")
    parser.add_argument("--labels", type=str, nargs="+", default=["NSS"], help="Labels to predict", 
                         choices=["NSS", "age", "sex", "diagnosis", "tiv", "skeleton_size"])
    parser.add_argument("--n_permutations", type=int, default=1000)
    args = parser.parse_args(argv)

    return args

if __name__ == "__main__":

    args = parse_args(sys.argv[1:])
    saving_dir = os.path.join(config.path2models, args.preproc, args.saving_dir)
    os.makedirs(saving_dir, exist_ok=True)
    scaler = args.scaler
    labels = args.labels
    preproc = args.preproc
    print("Data Loading")
    arr, df, scheme = load_data(args.preproc)
    n_permutations = args.n_permutations + 1
    if n_permutations > 1:
        label = "NSS"
        permutations = {}
        permutations[f"{label}_permutation-0"] = df[label].values
        for i in range(1, n_permutations):
            permutations[f"{label}_permutation-{i}"] = np.random.permutation(df[label].values)
        permutations_df = pd.DataFrame(permutations)
        df = pd.concat((df, permutations_df), axis=1)
        labels += [f"{label}_permutation-{i}" for i in range(n_permutations)]
    models = ["logreg" if l in ["diagnosis", "sex"] else "lrl2" for l in labels] # add several models
    list_args = [(arr, df, scheme, m, l, preproc, scaler, saving_dir) for m, l in zip(models, labels)]
    print(len(list_args))
    with Pool() as pool:
        pool.starmap(train, list_args)
"""