"""
Author: Alexandre LECLERCQ
Licence: MIT
"""

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings


class BaseTrainer:

    def __init__(self,
                 model: torch.nn.Module,
                 train_dataset: Dataset,
                 test_dataset: Dataset,
                 loss_fn,
                 optimizer: torch.optim,
                 batch_size=1,
                 save_best=False
                 ) -> None:


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f'device: {self.device}')

        self.model = model
        self.model = self.model.to(self.device)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.save_best = save_best
        self.metric_values = dict()

    def train(self, num_epochs: int) -> list:
        raise NotImplementedError
