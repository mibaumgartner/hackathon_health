import timm

from typing import Optional
from abc import abstractmethod
import torch
import pytorch_lightning as pl

import torch.nn as nn
import torch.optim as optim

# Template starte from:
# https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/04-inception-resnet-densenet.html


class ClassificationModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = self.create_model()
        self.loss = self.create_loss()

        # TODO: single channel vs gray scale multichannel
        self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)

    def forward(self, imgs):
        return self.model(imgs)

    def training_step(self, batch, batch_idx: Optional[int] = None):
        imgs, labels = batch # TODO: adjust accordingly
        
        preds = self.model(imgs)
        loss = self.loss(preds, labels)

        acc = (preds.argmax(dim=-1) == labels).float().mean()
        self.log("train/acc", acc, on_step=False, on_epoch=True)
        self.log("train/loss", loss)
        return loss 

    def validation_step(self, batch, batch_idx: Optional[int] = None):
        imgs, labels = batch  # TODO: adjust accordingly
        
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        self.log("val/acc", acc)

    @abstractmethod
    def create_model(self) -> torch.nn.Module:
        raise NotImplementedError

    @abstractmethod
    def create_loss(self) -> torch.nn.Module:
        raise NotImplementedError

    @abstractmethod
    def get_num_classes(self) -> torch.nn.Module:
        raise NotImplementedError


class BaselineClassification(ClassificationModule):
    def create_loss(self) -> torch.nn.Module:
        return nn.CrossEntropyLoss()

    def get_num_classes(self) -> torch.nn.Module:
        return 2

    def create_model(self) -> torch.nn.Module:
        model = timm.create_model('resnet18', pretrained=True, num_classes=2)
        return model

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [scheduler]
