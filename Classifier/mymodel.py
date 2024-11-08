import torch
import torch.nn as nn
import lightning as L
from torchmetrics import Accuracy, F1Score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import timm

class MyModel(L.LightningModule):
    def __init__(self,
                 lr: float = 0.001,
                 weight_decay: float = 1e-4,
                 num_class: int = 3,
                 class_counts= [135, 205, 580],
                 *args,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        total_count = sum(class_counts)
        class_weights = [total_count / count for count in class_counts]
        class_weights = torch.tensor(class_weights, dtype=torch.float32)

        self.model = timm.create_model(
            'resnet51q.ra2_in1k',
            pretrained=True,
            pretrained_cfg_overlay=dict(file='../Timm/resnet51q.safetensors'),
            in_chans=1,
            num_classes=self.hparams.num_class)

        in_features = self.model.get_classifier().in_features
        self.model.fc = nn.Linear(in_features, self.hparams.num_class)

        self.loss = nn.CrossEntropyLoss(weight=class_weights)

        self.acc = Accuracy(task="multiclass", num_classes=self.hparams.num_class)

        self.f1score = F1Score(task='multiclass', num_classes=self.hparams.num_class, average='weighted')

        self.train_labels = []
        self.train_preds = []

    def forward(self, x):
        return self.model(x)

    def get_input(self, batch):
        return batch['preMRI'], batch['class']

    def training_step(self, batch, batch_idx):
        x, y = self.get_input(batch)

        y_hat = self(x)
        loss = self.loss(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)

        acc = self.acc(preds, y)

        f1score = self.f1score(preds, y)

        self.log('train/loss', loss, logger=True, batch_size=x.shape[0])
        self.log('train/acc', acc, logger=True, batch_size=x.shape[0])
        self.log('train/f1score', f1score, logger=True, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self.get_input(batch)

        y_hat = self(x)
        loss = self.loss(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)

        acc = self.acc(preds, y)

        f1score = self.f1score(preds, y)

        self.log('val/loss', loss, logger=True, batch_size=x.shape[0])
        self.log('val/acc', acc, logger=True, batch_size=x.shape[0])
        self.log('val/f1score', f1score, logger=True, batch_size=x.shape[0])
        return loss

    def test_step(self, batch, batch_idx):
        x, y = self.get_input(batch)

        y_hat = self(x)
        loss = self.loss(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)

        acc = self.acc(preds, y)

        f1score = self.f1score(preds, y)

        self.log('test/loss', loss, logger=True, batch_size=x.shape[0])
        self.log('test/acc', acc, logger=True, batch_size=x.shape[0])
        self.log('test/f1score', f1score, logger=True, batch_size=x.shape[0])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss',
            }
        }