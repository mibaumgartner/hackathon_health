import timm

from typing import Optional, List
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


class BaseClassificationModule(pl.LightningModule):
    def __init__(
        self,
        run_name: str,
        architecture: str,
        pretrained: bool,
        epochs: int,
        init_learning_rate: float,
        weight_decay: float,
        loss: str,
    ):
        super().__init__()
        self.epochs = epochs
        self.architecture = architecture
        self.init_lr = init_learning_rate
        self.wd = weight_decay

        self.model = self.create_module(architecture, pretrained)
        self.loss = self.create_loss(loss)

        self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
        self.save_hyperparameters({"n_classes": self.get_num_classes()})

    def forward(self, imgs):
        return self.model(imgs)

    def training_step(self, batch, batch_idx: Optional[int] = None):
        imgs, labels = batch

        preds = self.model(imgs)
        loss = self.loss(preds, labels)

        acc = (preds.argmax(dim=-1) == labels).float().mean()
        self.log(
            "train/acc", acc, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )
        self.log("train/loss", loss, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx: Optional[int] = None):
        imgs, labels = batch
        preds = self.model(imgs)

        acc = (labels == preds.argmax(dim=-1)).float().mean()
        val_loss = self.loss(preds, labels)

        self.log("val/acc", acc, prog_bar=False, logger=True)
        self.log("val/loss", val_loss, prog_bar=False, logger=True)

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> List[torch.Tensor]:
        imgs = batch
        preds = self.model(imgs)
        return preds.argmax(dim=-1)
    
    def get_num_classes(self) -> int:
        """
        Depends on the type of loss used.
        Can be overloaded to replace the default of 2.
        """
        return 2

    def create_module(self, architecture, pretrained):
        return timm.create_model(
            architecture, pretrained=pretrained, num_classes=self.get_num_classes()
        )

    def create_loss(self, loss_name: str) -> torch.nn.Module:
        if loss_name == "CE":
            return nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Not implemented yet.")

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.init_lr, weight_decay=self.wd, amsgrad=True
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,
            eta_min=1e-9,
        )
        return [optimizer], [scheduler]
