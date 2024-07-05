"""
Author: Alexandre LECLERCQ
Licence: MIT
"""

import os
from os.path import isfile, join
from typing import Optional
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms


class GenerationImageDataset(Dataset):

    def __init__(self, dataset_path: str, annotations_file: Optional[str] = None, transform=None) -> None:
        if annotations_file is None:  # load all the image path from a directory
            self.img_paths = [file for file in os.listdir(dataset_path) if isfile(join(dataset_path, file))]
        else:  # load image path thanks to an annotations_file
            self.img_paths = pd.read_csv(os.path.join(dataset_path, annotations_file))['filepath'].tolist()
        self.dataset_path = dataset_path
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_path, self.img_paths[idx])
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image
