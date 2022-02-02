from pathlib import Path

import cv2

from medhack.dataset import CovidImageDataset
import albumentations as A
import pytorch_lightning as pl

from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
import torch.distributed as dist

from medhack.distributed_sampler import WeightedDistributedRandomSampler

mean = 0.8184206354986654  # Gimme for Gregor
std = 0.03884859786640268  # Wait for Gregor


class BasicDataModule(pl.LightningDataModule):
    def __init__(self, root_dir: Path,gpu_num: int , num_workers: int = 8, batch_size: int = 32,
                 ):
        super(BasicDataModule, self).__init__()
        train_csv = "train.csv"
        val_csv = "valid.csv"

        # Hyperparameters of Dataloader
        self.gpu_num: int = gpu_num
        self.batch_size: int = batch_size
        self.shuffle: bool = True
        self.drop_last: bool = False
        self.persistent_workers: bool = True  # Probably useful
        self.num_workers = num_workers
        self.pin_memory = True

        train_transforms = A.Compose(
            [
                # A.Normalize(mean, std),
                A.VerticalFlip(),
                A.Affine(scale=(0.80, 1.20),  # 0.5 == 50% zoomed out
                         rotate=30,
                         shear=(10, 10),
                         interpolation=cv2.INTER_CUBIC,
                         mode=cv2.BORDER_REFLECT,
                         p=1.0,
                         ),
                A.RandomContrast(0.1),
                A.GaussianBlur()
            ]
        )
        val_transforms = A.Compose(
            [
                # A.Normalize(mean, std),
                A.VerticalFlip(),
            ]
        )
        self.train_dataset = CovidImageDataset(train_csv, root_dir, train_transforms)
        self.valid_dataset = CovidImageDataset(val_csv, root_dir, val_transforms)
        
        if self.gpu_num == 1:
            rank = 0
        else:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = None
        
        self.training_data_sampler = WeightedDistributedRandomSampler(
            weights=self.train_dataset.get_data_weights(),
            num_samples=len(self.train_dataset),
            replacement=True,
            rank=rank
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=False,
            sampler=self.training_data_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.valid_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.persistent_workers,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass
