"""
Author: Alexandre LECLERCQ
Licence: MIT
"""

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
from torcheval.metrics import PeakSignalNoiseRatio
from utils.RandomNoise import AddGaussianNoise
from utils.trainer.BaseTrainer import BaseTrainer


class DenoysingTrainer(BaseTrainer):
    
    def __init__(self, 
                 model: torch.nn.Module,
                 train_dataset: Dataset,
                 test_dataset: Dataset,
                 noise: AddGaussianNoise,
                 loss_fn,
                 optimizer: torch.optim,
                 batch_size=1,
                 save_best=False
                 ) -> None:
        super().__init__(model, train_dataset, test_dataset, loss_fn, optimizer, batch_size, save_best)
        self.noise = noise

    def train(self, num_epochs: int):

        self.metric_values['training_loss'] = []
        self.metric_values['training_psnr'] = []

        
        for epoch in range(num_epochs):
            training_batches_progress = tqdm(range(len(self.train_loader)), desc='Epoch {}/{}'.format(epoch + 1, num_epochs))
            self.metric_values['training_loss'].append(0.0)
            psnr_metric = PeakSignalNoiseRatio()
            
            for images, _ in self.train_loader:

                images = images.to(self.device).to(torch.float32)
                noisy_images = self.noise(images)
                self.optimizer.zero_grad()

                outputs = self.model(noisy_images)
                
                loss = self.loss_fn(outputs, images)  # comparison between denoised images and real images
                psnr_metric.update(outputs, images)
                loss.backward()
                self.optimizer.step()
                
                self.metric_values['training_loss'][-1] += len(images) * loss.item()
                training_batches_progress.update()
            self.metric_values['training_loss'][-1] /= len(self.train_loader)
            self.metric_values['training_psnr'].append(psnr_metric.compute())
            training_batches_progress.set_postfix_str(str(f'Loss: {self.metric_values["training_loss"][-1]:.4f}, PSNR: {self.metric_values["training_psnr"][-1]:.4f}'))
            training_batches_progress.close()

            if self.save_best and self.metric_values['training_psnr'][-1] >= max(self.metric_values['training_psnr']):
                print('Saving best model')
                self.model.save()

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
