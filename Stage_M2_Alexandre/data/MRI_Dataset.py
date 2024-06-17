"""
Author: Alexandre LECLERCQ
Licence: MIT
"""

import os
from os.path import isfile, join
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MRI_Dataset(Dataset):

    def __init__(self, dataset_path: str, manifest_filename: str = None, transform=None) -> None:
        assert manifest_filename is not None
        self.data = pd.read_csv(os.path.join(dataset_path, manifest_filename))
        self.dataset_path = dataset_path
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple:
        """
        return a tupple of our 3 images
        """
        patient = str(self.data.iloc[idx]['patient'])
        slice_ = str(self.data.iloc[idx]['slice'])
        preMRI_path = os.path.join(self.dataset_path, patient, slice_, self.data.iloc[idx]['preMRI'])
        treatment_path = os.path.join(self.dataset_path, patient, slice_, self.data.iloc[idx]['treatment'])
        postMRI_path = os.path.join(self.dataset_path, patient, slice_, self.data.iloc[idx]['postMRI'])

        preMRI_img = nib.load(preMRI_path).get_fdata()
        treatment_img = nib.load(treatment_path).get_fdata()
        postMRI_img = nib.load(postMRI_path).get_fdata()

        if self.transform:
            preMRI_img = self.transform(preMRI_img)
            treatment_img = self.transform(treatment_img)
            postMRI_img = self.transform(postMRI_img)
        return preMRI_img, treatment_img, postMRI_img