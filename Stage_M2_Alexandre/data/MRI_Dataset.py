"""
Author: Alexandre LECLERCQ
Licence: MIT
"""

import os
from typing import Literal
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset, dataset
from torchvision import transforms


class MRI_Dataset(Dataset):

    def __init__(self, dataset: pd.DataFrame, dataset_path,
                 task: Literal["autoencoder", "generation", "conditional_generation"],
                 transform=None) -> None:
        self.dataset_path = dataset_path
        self.data = dataset
        if task == 'autoencoder':
            assert 'MRI_img' in self.data.columns
        elif task == 'generation':
            assert all(elem in self.data.columns for elem in ['preMRI', 'postMRI'])
        elif task == 'conditional_generation':
            assert all(elem in self.data.columns for elem in ['preMRI', 'treatment', 'postMRI'])
        else:
            raise NotImplementedError

        self.task = task
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple:
        """
        return a tupple of our 3 images
        """
        patient = str(self.data.iloc[idx]['patient'])
        slice_ = str(self.data.iloc[idx]['slice'])


        if self.task == 'autoencoder':
            MRI_img_path = os.path.join(self.dataset_path, 'data', patient, slice_, self.data.iloc[idx]['MRI_img'])

            MRI_img = nib.load(MRI_img_path).get_fdata()

            if self.transform:
                MRI_img = self.transform(MRI_img)
            return MRI_img

        elif self.task == 'generation':
            preMRI_path = os.path.join(self.dataset_path, 'data', patient, slice_, self.data.iloc[idx]['preMRI'])
            postMRI_path = os.path.join(self.dataset_path, 'data', patient, slice_, self.data.iloc[idx]['postMRI'])

            preMRI_img = nib.load(preMRI_path).get_fdata()
            postMRI_img = nib.load(postMRI_path).get_fdata()

            if self.transform:
                preMRI_img = self.transform(preMRI_img)
                postMRI_img = self.transform(postMRI_img)
            return preMRI_img, postMRI_img

        elif self.task == 'conditional_generation':
            preMRI_path = os.path.join(self.dataset_path, 'data', patient, slice_, self.data.iloc[idx]['preMRI'])
            treatment_path = os.path.join(self.dataset_path, 'data', patient, slice_, self.data.iloc[idx]['treatment'])
            postMRI_path = os.path.join(self.dataset_path, 'data', patient, slice_, self.data.iloc[idx]['postMRI'])

            preMRI_img = nib.load(preMRI_path).get_fdata()
            treatment_img = nib.load(treatment_path).get_fdata()
            postMRI_img = nib.load(postMRI_path).get_fdata()

            if self.transform:
                preMRI_img = self.transform(preMRI_img)
                treatment_img = self.transform(treatment_img)
                postMRI_img = self.transform(postMRI_img)
            return preMRI_img, treatment_img, postMRI_img