# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Imports
import os
import pickle
import logging
import bisect
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict

from typing import Callable, List, Type, Sequence, Dict, Union
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms

from config import Config

logger = logging.getLogger("dataset")
config = Config()


class MRIDataset(Dataset):

    def __init__(self, root: str, preproc: Union[str, List[str]] = 'vbm', target: Union[str, List[str]] = 'diagnosis',
                 split: str = 'train', transforms: Callable[[np.ndarray], np.ndarray] = None,
                 fold: int = 0, two_views: bool = True):
        """
        :param root: str, path to the root directory containing the different .npy and .csv files
        :param preproc: str, must be either VBM ('vbm'), Quasi-Raw ('quasi_raw') or Skeleton ('skeleton')
        :param target: str or [str], either 'dx' or 'site'.
        :param split: str, either 'train', 'val', 'test' (inter) or (eventually) 'test_intra'
        :param transforms (callable, optional): A function/transform that takes in
            a 3D MRI image and returns a transformed version.
        """
        # 0) Check parameters and set attributes
        # preproc
        if isinstance(preproc, str):
            assert preproc in {'vbm', 'quasi_raw', 'skeleton'}, f"Unknown preproc: {preproc}"
            self.preproc = [preproc]
        elif isinstance(preproc, list):
            assert set(preproc) <= {'vbm', 'quasi_raw', 'skeleton'}, f"Unknown preproc: {preproc}"
            self.preproc = preproc
        # root
        self.root = root
        if not self._check_integrity():
            raise FileNotFoundError(f"Files not found. Check the the root directory {root}")
        # split
        assert split in ['train', 'val', 'test', 'test_intra', 'validation'], f"Unknown split: {split}"
        if split == "val":
            self.split = "validation"
        else:
            self.split = split
        self.fold = fold
        self.scheme = self.load_scheme()

        # target
        # FIXME : find a solution for NSS
        if isinstance(target, str):
            self.target_name = [target]
        elif isinstance(target, list):
            self.target_name = target
        self.transforms = transforms
        # transforms
        self.transforms = dict()
        for pr in self.preproc:
            if transforms is None:
                self.transforms[pr] = None
            elif pr not in transforms.keys():
                self.transforms[pr] = None
            else:
                self.transforms[pr] = transforms[pr]
        # two views
        self.two_views = two_views
                
        # 1) Loads globally all the data
        df = pd.concat([pd.read_csv(os.path.join(root, self._get_pd_files % db), dtype=self._id_types) 
                        for db in self._studies],
                       ignore_index=True, sort=False)
        # FIXME
        data = {pr: [np.load(os.path.join(root, self._get_npy_files[pr] % db), mmap_mode='r')
                     for db in self._studies]
                for pr in self.preproc}
                 
        cumulative_sizes = {pr: np.cumsum([len(db) for db in data[pr]]) for pr in self.preproc}
        # check if all cumulative sizes are equals
        assert all([np.array_equal(cumulative_sizes[self.preproc[0]], arr) for arr in cumulative_sizes.values()]), \
        f"All npy files of the different preprocessings do not have same number of subjects."

        # 2) Selects the data to load in memory according to selected scheme
        mask = self._extract_mask(df, unique_keys=self._unique_keys, check_uniqueness=self._check_uniqueness)

        # Get the labels to predict
        assert set(self.target_name) <= set(df.keys()), \
            f"Inconsistent files: missing {self.target_name} in participants DataFrame"
        self.target = df[mask][self.target_name]
        assert self.target.isna().sum().sum() == 0, f"Missing values for {self.target_name} label"
        self.target = self.target.apply(self.target_transform_fn, axis=1, raw=True).values.astype(np.float32)

        # Prepares private variables to build mapping target_idx -> img_idx
        self.shape = {pr: (mask.sum(), *data[pr][0][0].shape) for pr in self.preproc}
        self._mask_indices = np.arange(len(df))[mask]
        self._cumulative_sizes = cumulative_sizes[self.preproc[0]]
        self._data = data

    @property
    def _studies(self) -> List[str]:
        return ["abide1", "abide2", "biobd", "bsnip1", "candi", "cnp", "schizconnect-vip-prague"]

    @property
    def _train_val_test_scheme(self) -> str:
        return "train_val_test_test-intra_stratified.csv"

    @property
    def _unique_keys(self) -> List[str]:
        return ["participant_id", "session", "run", "study"]

    @property
    def _target_mappings(self) -> Dict[str, Dict[str, int]]:
        # NB; no target in MRI dataset
        return dict()

    @property
    def _check_uniqueness(self) -> bool:
        return True

    @property
    def _get_npy_files(self) -> Dict[str, str]:
        return {"vbm": "%s_t1mri_mwp1_gs-raw_data64.npy",
                "quasi_raw": "%s_t1mri_quasi_raw_data32_1.5mm_skimage.npy",
                "skeleton": "%s_t1mri_skeleton_data32.npy"}

    @property
    def _get_pd_files(self) -> Dict[str, str]:
        return "%s_t1mri_participants.csv"
    
    @property
    def _id_types(self):
        return {"participant_id": str,
                "session": int,
                "acq": int,
                "run": int}

    def _check_integrity(self):
        """
        Check the integrity of root dir (including the directories/files required). It does NOT check their content.
        Should be formatted as:
        /root
            <train_val_test_split.pkl>
            [cohort]_t1mri_mwp1_gs-raw_data64.npy
            [cohort]_t1mri_skeleton_data32.npy
            [cohort]_t1mri_participants.csv
        """
        is_complete = os.path.isdir(self.root)
        is_complete &= os.path.isfile(os.path.join(self.root, self._train_val_test_scheme))

        for pr in self.preproc:
            files = [self._get_pd_files, self._get_npy_files[pr]]
            for file in files:
                for db in self._studies:
                    is_complete &= os.path.isfile(os.path.join(self.root, file % db))
        return is_complete

    def _extract_mask(self, df: pd.DataFrame, unique_keys: Sequence[str], check_uniqueness: bool = True):
        """
        :param df: pandas DataFrame
        :param unique_keys: list of str
        :param check_uniqueness: if True, check the unique_keys identified uniquely an image in the dataset
        :return: a binary mask indicating, for each row, if the participant belongs to the current scheme or not.
        """
        _source_keys = df[unique_keys].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        if check_uniqueness:
            assert len(set(_source_keys)) == len(_source_keys), f"Multiple identique identifiers found"
        _target_keys = self.scheme[unique_keys].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        mask = _source_keys.isin(_target_keys).values.astype(bool)
        return mask
    
    def target_transform_fn(self, target):
        """ Transforms the target according to mapping site <-> int and dx <-> int """
        target = target.copy()
        for i, name in enumerate(self.target_name):
            if name in self._target_mappings.keys():
                target[i] = self._target_mappings[name][target[i]]
        return target

    def load_pickle(self, path: str):
        with open(path, 'rb') as f:
            pkl = pickle.load(f)
        return pkl
    
    def load_scheme(self):
        df = pd.read_csv(os.path.join(self.root, self._train_val_test_scheme))
        return df.loc[df[f"fold_{self.fold}"] == self.split, self._unique_keys]
    
    def _mapping_idx(self, idx: int):
        """
        :param idx: int ranging from 0 to len(dataset)-1
        :return: integer that corresponds to the original image index to load
        """
        idx = self._mask_indices[idx]
        dataset_idx = bisect.bisect_right(self._cumulative_sizes, idx)
        sample_idx = idx - self._cumulative_sizes[dataset_idx - 1] if dataset_idx > 0 else idx
        return dataset_idx, sample_idx
    
    def __getitem__(self, idx: int):
        sample = dict()
        (dataset_idx, sample_idx) = self._mapping_idx(idx)
        for pr in self.preproc:
            sample[pr] = self._data[pr][dataset_idx][sample_idx]
        for t, tgt in enumerate(self.target_name):
            sample[tgt] = self.target[idx][t]
        for pr in self.preproc:
            if self.two_views:
                view_1 = self.transforms[pr](sample[pr])
                view_2 = self.transforms[pr](sample[pr])
                sample[pr] = (view_1, view_2)
            else:
                if self.transforms[pr] is not None:
                    sample[pr] = self.transforms[pr](sample[pr])
        return sample
    
    def __len__(self):
        return len(self.target)

    def __str__(self):
        return f"{type(self).__name__}-{tuple(self.preproc)}-{self.split}"


class SCZDataset(MRIDataset):

    @property
    def _studies(self):
        return ["schizconnect-vip-prague", "bsnip1", "cnp", "candi"]

    @property
    def _train_val_test_scheme(self):
        return "train_val_test_test-intra_scz_stratified.pkl"

    @property
    def _unique_keys(self):
        return ["participant_id", "session", "study"]

    @property
    def _dx_site_mappings(self):
        return dict(diagnosis={"control": 0, "schizophrenia": 1, "scz": 1},
                    site=self._site_mapping)

    def load_scheme(self):
        return self.load_pickle(os.path.join(self.root, self._train_val_test_scheme))[self.split]

    def _check_integrity(self):
        return super()._check_integrity() & os.path.isfile(os.path.join(self.root, "mapping_site_name-class_scz.pkl"))

    def __init__(self, root: str, *args, **kwargs):
        self._site_mapping = self.load_pickle(os.path.join(root, "mapping_site_name-class_scz.pkl"))
        super().__init__(root, *args, **kwargs)

    def __str__(self):
        return "SCZDataset"


class BDDataset(MRIDataset):

    @property
    def _studies(self):
        return ["biobd", "bsnip1", "cnp", "candi"]

    @property
    def _train_val_test_scheme(self):
        return "train_val_test_test-intra_bd_stratified.pkl"

    @property
    def _unique_keys(self):
        return ["participant_id", "session", "study"]

    @property
    def _dx_site_mappings(self):
        return dict(diagnosis={"control": 0, "bipolar": 1, "bipolar disorder": 1, "psychotic bipolar disorder": 1, "bd": 1, "psychotic bd": 1},
                    site=self._site_mapping)
    
    def load_scheme(self):
        return self.load_pickle(os.path.join(self.root, self._train_val_test_scheme))[self.split]

    def _check_integrity(self):
        return super()._check_integrity() & os.path.isfile(os.path.join(self.root, "mapping_site_name-class_bd.pkl"))

    def __init__(self, root: str, *args, **kwargs):
        self._site_mapping = self.load_pickle(os.path.join(root, "mapping_site_name-class_bd.pkl"))
        super().__init__(root, *args, **kwargs)
    
    def __str__(self):
        return "BDDataset"


class ASDDataset(MRIDataset):

    @property
    def _studies(self):
        return ["abide1", "abide2"]

    @property
    def _train_val_test_scheme(self):
        return "train_val_test_test-intra_asd_stratified.pkl"

    @property
    def _unique_keys(self):
        return ["participant_id", "session", "study", "run"]

    @property
    def _dx_site_mappings(self):
        return dict(diagnosis={"control": 0, "autism": 1, "asd": 1},
                    site=self._site_mapping)
    
    def load_scheme(self):
        return self.load_pickle(os.path.join(self.root, self._train_val_test_scheme))[self.split]

    def _check_integrity(self):
        return super()._check_integrity() & os.path.isfile(os.path.join(self.root, "mapping_site_name-class_asd.pkl"))

    def __init__(self, root: str, *args, **kwargs):
        self._site_mapping = self.load_pickle(os.path.join(root, "mapping_site_name-class_asd.pkl"))
        super().__init__(root, *args, **kwargs)
    
    def __str__(self):
        return "ASDDataset"
"""
class NSSDataset(MRIDataset):

    @property
    def _studies(self):
        return ["ausz"]

    @property
    def _train_val_test_scheme(self):
        return "stratified_10_fold_ausz.csv"

    @property
    def _unique_keys(self):
        return ["participant_id", "session", "study"]

    @property
    def _target_mappings(self):
        return dict(diagnosis={"control": 0, "asd": 1, "scz-asd": 2, "scz": 3},
                    sex={"M": 0, "m": 0, "f": 1, "F": 1},
                    site=self._site_mapping)
    @property
    def _site_mapping(self):
        return {"AUSZ": 0}
    
    def _check_integrity(self):
        is_complete = os.path.isdir(self.root)
        print("root", is_complete)
        is_complete &= os.path.isfile(os.path.join(self.root, self._train_val_test_scheme))
        print("scheme", is_complete)
        print(os.path.join(self.root, self._train_val_test_scheme))
        for pr in self.preproc:
            files = [self._get_pd_files, self._get_npy_files[pr]]
            for file in files:
                for db in self._studies:
                    is_complete &= os.path.isfile(os.path.join(self.root, file % db))
                    print(pr, file % db, is_complete)
        return is_complete
        #return super()._check_integrity()

    def __init__(self, root: str, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
    
    def load_scheme(self):
        df = pd.read_csv(os.path.join(self.root, self._train_val_test_scheme))
        return df.loc[df[f"fold{self.fold}"] == self.split, self._unique_keys]
    
    def __str__(self):
        return "NSSDataset"
"""

class NSSDataset(Dataset):
    id_types = {"participant_id": str,
                "session": int,
                "acq": int,
                "run": int}

    def __init__(self, fold, split, preproc, target):

        self.preproc = preproc
        if preproc == "skeleton":
            self.imgs = np.load(os.path.join(config.home_local, "ausz_t1mri_skeleton_data32.npy"), mmap_mode="r")
        elif preproc == "vbm":  
            self.imgs = np.load(os.path.join(config.home_local, "ausz_t1mri_mwp1_gs-raw_data64.npy"), mmap_mode="r")

        self.scheme = pd.read_csv(os.path.join(config.home_local, "nss_stratified_10_fold_ausz.csv"), dtype=self.id_types)
        print("WARNING: you are using the home local directory !")
        print("WARNING: you are using the scheme: nss_stratified_10_fold_ausz !")

        self.mask = self.scheme[f"fold{fold}"] == split

        self.metadata = pd.read_csv(os.path.join(config.home_local, "ausz_t1mri_participants.csv"), dtype=self.id_types)
        self.label = target
        self.targets = self.metadata[target].values.astype(np.float32)

    def set_split_fold(self, split=None, fold=None):
        if fold is not None:
            self.fold = fold 
        if split is not None:
            self.split = split
        self.mask =  self.scheme[f"fold{self.fold}"] == self.split
        
    def __getitem__(self, idx):
        img = self.imgs[self.mask][idx]
        img = torch.Tensor(img)
        label = self.targets[self.mask][idx]

        return {self.preproc: img,
                self.label: label}
    
    def __len__(self):
        return self.mask.sum()



if __name__ == "__main__":
    dataset = NSSDataset(fold=8, split="train", target="NSS")
    
    sample = dataset[12]
    print(sample["vbm"].shape, sample["NSS"], np.unique(sample["skeleton"]), sample["skeleton"].shape)
    print(len(dataset))
