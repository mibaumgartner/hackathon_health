import timm

from typing import Optional
from abc import abstractmethod
import torch
import pytorch_lightning as pl

import torch.nn as nn
import torch.optim as optim

# Template starte from:
# https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/04-inception-resnet-densenet.html

# Models
# Efficient Net
# https://github.com/rwightman/pytorch-image-models/blob/07379c6d5dbb809b3f255966295a4b03f23af843/timm/models/efficientnet.py#L348-L351
# https://github.com/rwightman/pytorch-image-models/blob/07379c6d5dbb809b3f255966295a4b03f23af843/timm/models/efficientnet.py#L361-L364
# https://github.com/rwightman/pytorch-image-models/blob/07379c6d5dbb809b3f255966295a4b03f23af843/timm/models/efficientnet.py#L378-L381
# https://github.com/rwightman/pytorch-image-models/blob/07379c6d5dbb809b3f255966295a4b03f23af843/timm/models/efficientnet.py#L322-L326
# https://github.com/rwightman/pytorch-image-models/blob/07379c6d5dbb809b3f255966295a4b03f23af843/timm/models/efficientnet.py#L109-L110

# ResNet
# https://github.com/rwightman/pytorch-image-models/blob/07379c6d5dbb809b3f255966295a4b03f23af843/timm/models/resnet.py#L39-L41

# RegNet
# https://arxiv.org/pdf/2003.13678.pdf
# https://github.com/rwightman/pytorch-image-models/blob/07379c6d5dbb809b3f255966295a4b03f23af843/timm/models/regnet.py#L77

# MobileNetV3
# https://github.com/rwightman/pytorch-image-models/blob/07379c6d5dbb809b3f255966295a4b03f23af843/timm/models/mobilenetv3.py#L50-L52


class ClassificationModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = self.create_model()
        self.loss = self.create_loss()

        self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)

    def forward(self, imgs):
        return self.model(imgs)

    def training_step(self, batch, batch_idx: Optional[int] = None):
        imgs, labels = batch

        preds = self.model(imgs)
        loss = self.loss(preds, labels)

        acc = (preds.argmax(dim=-1) == labels).float().mean()
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx: Optional[int] = None):
        imgs, labels = batch

        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        self.log("val/acc", acc, prog_bar=True, logger=True)

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
        model = timm.create_model('resnet18', pretrained=True, num_classes=self.get_num_classes())
        return model

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=3e-4, weight_decay=0.001, amsgrad=True)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [scheduler]
