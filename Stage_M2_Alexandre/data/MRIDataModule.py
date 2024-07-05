"""
Author: Alexandre LECLERCQ
Licence: MIT
"""
import os
import pandas as pd
import torch
import lightning as L
from torch.utils.data import random_split, DataLoader
import numpy as np
from typing import Literal
from torchvision.transforms import Compose, ToTensor, ConvertImageDtype, Lambda, Resize

from data.MRI_Dataset import MRI_Dataset


def get_threshold_set_values(dataset, margin: float=0.002):
    """
    return the max value in which a specific % of all the images of the train set are include
    margin: % of value which are not include in the range returned
    """
    histo = torch.zeros((1, 65535))
    for sample in dataset:
        histo += torch.histc(sample, min=0, max=65535, bins=65535) / (512 * 512)
    histo /= len(dataset)

    sum_proportion_level = .0
    level = 0
    for i, bin in enumerate(histo[0]):
        sum_proportion_level += bin
        if sum_proportion_level > (histo.sum().item() - margin):
            level = i
            break
    print(f"{(1 - margin) * 100:.2f}% values are contain between [0, {level}]")
    return level


class MRIDataModule(L.LightningDataModule):

    def __init__(self, dataset_path: str,
                 manifest_filename: str,
                 batch_size: int,
                 seed: int,
                 task: Literal["autoencoder", "generation", "conditional_generation"],
                 crop_size: int,
                 train_val_test_split: tuple = (0.8, 0.1, 0.1),
                 num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.task = task
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.prepare_data_per_node = False
        self.dataset_path = dataset_path
        self.manifest_filename = manifest_filename
        self.crop_size = crop_size
        self.train_val_test_split = train_val_test_split
        # We initiate each set as an empty list. Their corresponding dataloader will be of len 0 if they stay empty.
        self.train_set, self.val_set, self.test_set = [], [], []
        self.transform = None
        self.reverse_transform = None
        np.random.seed(seed)

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        """
        load dataset
        apply transform
        split dataset
        """
        data = pd.read_csv(os.path.join(self.dataset_path, self.manifest_filename))
        patients = data['patient'].unique()

        p_train, p_val, p_test = self.train_val_test_split
        assert p_train + p_val + p_test == 1.0

        indices_split = np.random.rand(len(patients))
        train_mask = indices_split < p_train
        val_mask = (p_train < indices_split) & (indices_split < p_train + p_val)
        test_mask = (p_train + p_val < indices_split) & (indices_split < p_train + p_val + p_test)

        train_patients = patients[train_mask]
        val_patients = patients[val_mask]
        test_patients = patients[test_mask]

        self.train_set = MRI_Dataset(dataset=data[data['patient'].isin(train_patients)],
                                     dataset_path=self.dataset_path,
                                     task=self.task,
                                     transform=Compose([ToTensor()]))

        threshold_value = get_threshold_set_values(dataset=self.train_set, margin=0.005)

        self.transform = Compose([
            ToTensor(),
            Resize(self.crop_size),
            ConvertImageDtype(torch.float32),
            Lambda(lambda x: x / threshold_value),  # [0, range] --> ~ [0, 1]
            Lambda(lambda x: x * 2 - 1)   # [0, 1] --> [-1, 1]
        ])

        self.reverse_transform = Compose([
            Lambda(lambda x: (x + 1) / 2),
            Lambda(lambda x: x * threshold_value),
        ])

        self.train_set = MRI_Dataset(dataset=data[data['patient'].isin(train_patients)],
                                     dataset_path=self.dataset_path,
                                     task=self.task,
                                     transform=self.transform)

        self.val_set = MRI_Dataset(dataset=data[data['patient'].isin(val_patients)],
                                   dataset_path=self.dataset_path,
                                   task=self.task,
                                   transform=self.transform)

        self.test_set = MRI_Dataset(dataset=data[data['patient'].isin(test_patients)],
                                    dataset_path=self.dataset_path,
                                    task=self.task,
                                    transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            # num_workers=self.num_workers,
            # persistent_workers=True,
            pin_memory=True)

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            # num_workers=self.num_workers,
            # persistent_workers=True,
            pin_memory=True)

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            # num_workers=self.num_workers,
            # persistent_workers=True,
            pin_memory=True)
