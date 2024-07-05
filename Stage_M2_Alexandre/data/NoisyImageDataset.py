"""
Author: Alexandre LECLERCQ
Licence: MIT
"""

import os
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset


class NoisyImageDataset(Dataset):

    def __init__(self, dataset_path: str, annotations_file: str, transform=None) -> None:
        self.img_annotations = pd.read_csv(os.path.join(dataset_path, annotations_file))
        self.dataset_path = dataset_path
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_path, self.img_annotations.iloc[idx, 1])
        image = read_image(img_path)
        noisy_image = add_noises(image)
        if self.transform:
            image = self.transform(image)
            noisy_image = self.transform(noisy_image)
        return image, noisy_image


def add_noises(image: torch.Tensor, mean=0, std=1):
    """
    Add a Gaussian Noise to the image
    :param image:
    :param mean:
    :param std:
    :return:
    """
    return image + torch.randn_like(image) * std + mean
