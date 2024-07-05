"""
Author: Alexandre LECLERCQ
Licence: MIT
"""

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import timeit
import warnings
from torcheval.metrics import PeakSignalNoiseRatio
from utils.trainer.BaseTrainer import BaseTrainer


class DiffusionModelTrainer:

    def __init__(self,
                 model,
                 train_dataset: Dataset,
                 test_dataset: Dataset,
                 loss_fn,
                 optimizer: torch.optim,
                 batch_size=1,
                 save_best=False
                 ) -> None:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.save_best = save_best
        self.metric_values = dict()

    def train(self, num_epochs: int):

        self.metric_values['training_loss'] = []

        for epoch in range(num_epochs):
            training_batches_progress = tqdm(range(len(self.train_loader)),
                                             desc='Epoch {}/{}'.format(epoch + 1, num_epochs))
            self.metric_values['training_loss'].append(0.0)

            for images in self.train_loader:
                images, _ = images
                images = images.to(self.device)

                self.optimizer.zero_grad()
                loss = self.model.compute_loss(images)
                loss.backward()
                self.optimizer.step()

                self.metric_values['training_loss'][-1] += len(images) * loss.item()
                training_batches_progress.update()

            self.metric_values['training_loss'][-1] /= len(self.train_dataset)

            training_batches_progress.set_postfix_str(str(f'Loss: {self.metric_values["training_loss"][-1]:.4f}'))
            training_batches_progress.close()

            if self.save_best and self.metric_values['training_loss'][-1] >= max(self.metric_values['training_loss']):
                print('Saving best model')
                self.model.save()

        return self.metric_values

    def evaluate(self):
        loss = 0.0
        psnr_metric = PeakSignalNoiseRatio()

        testing_batches_progress = tqdm(range(len(self.test_loader)), desc="Testing progress")

        self.model.eval()
        with torch.no_grad():
            for images, _ in self.test_loader:
                images = images.to(self.device).to(torch.float32)
                noisy_images = self.noise(images)
                outputs = self.model(noisy_images)
                loss += self.loss_fn(outputs, images)
                psnr_metric.update(outputs, images)
                testing_batches_progress.update()
            loss = loss / len(self.test_dataset)
            return loss, psnr_metric.compute()
