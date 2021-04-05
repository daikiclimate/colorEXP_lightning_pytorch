import os
from glob import glob

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.utils.data as data
from torchvision import transforms
from torchvision import datasets, models, transforms

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from PIL import Image
import matplotlib.pyplot as plt

from dataset import TrainDatasets

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class CoolSystem(pl.LightningModule):
    def __init__(self, num_class):
        super(CoolSystem, self).__init__()
        self._model = models.resnet18(pretrained=True)
        num_features = self._model.fc.in_features
        self._model.fc = nn.Linear(num_features, num_class)  

    def forward(self, x):
        x = self._model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {"train_loss": loss}
        # return {"loss": loss, "log": tensorboard_logs}
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        out = self.forward(x)
        loss = F.cross_entropy(out, y)

        return {"val_loss": F.cross_entropy(out, y)}
        # self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        # REQUIRED
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        # optimizer = torch.optim.Adam(self.parameters(), lr=0.02)
        return optimizer

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_dataset = TrainDatasets(
            mode="train", transform=transform
        )

        train_loader = DataLoader(train_dataset,num_workers = 4, shuffle=True, batch_size=32)

        return train_loader

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        val_dataset = TrainDatasets(
            mode="test", transform=transform
            )

        val_loader = DataLoader(val_dataset,num_workers = 4,  batch_size=32)

        return val_loader

def main():
    num_classes = 10
    cool_model = CoolSystem(num_classes)
    trainer = pl.Trainer(gpus=1)
    trainer.fit(cool_model)

if __name__=="__main__":
    main()
