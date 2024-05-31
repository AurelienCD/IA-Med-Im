"""
Author: Alexandre LECLERCQ
Licence: MIT
"""

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
from utils.trainer.BaseTrainer import BaseTrainer


class GenerationTrainer(BaseTrainer):

    def __init__(self,
                 model: torch.nn.Module,
                 train_dataset: Dataset,
                 test_dataset: Dataset,
                 loss_fn,
                 optimizer: torch.optim,
                 batch_size=1,
                 save_best=False
                 ) -> None:
        super().__init__(model, train_dataset, test_dataset, loss_fn, optimizer, batch_size, save_best)

    def train(self, num_epochs: int) -> list:
        training_batches_progress = tqdm(range(len(self.train_loader)), desc="Training progress")

        losses = []

        for epoch in range(num_epochs):
            training_batches_progress.reset()
            losses.append(0.0)

            for images in self.train_loader:
                images = images.to(self.device).to(torch.float32)
                self.optimizer.zero_grad()
                outputs = self.model(images)

                loss = self.loss_fn(outputs, images.squeeze(1).to(torch.long))

                loss.backward()
                self.optimizer.step()

                losses[-1] += loss.item()
                training_batches_progress.update()
                # training_batches_progress.set_postfix_str(f'Loss current batch: {loss}')

        return losses
