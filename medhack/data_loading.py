from pathlib import Path

import cv2

from medhack.dataset import CovidImageDataset
import albumentations as A
import pytorch_lightning as pl

from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, WeightedRandomSampler

mean = ()  # Gimme for Gregor
std = ()  # Wait for Gregor


class BasicDataModule(pl.LightningDataModule):
    def __init__(self, root_dir: Path):
        super(BasicDataModule, self).__init__()
        train_csv = "train.csv"
        val_csv = "valid.csv"

        # Hyperparameters of Dataloader
        self.batch_size: int = 256
        self.shuffle: bool = True
        self.drop_last: bool = False
        self.persistent_workers: bool = True  # Probably useful
        self.num_workers = 0
        self.pin_memory = True

        train_transforms = A.Compose(
            [
                A.HorizontalFlip(),
                A.Affine(scale=(0.95, 1.10),  # 0.5 == 50% zoomed out
                         rotate=10,
                         shear=(10, 10),
                         interpolation=cv2.INTER_CUBIC,
                         mode = cv2.BORDER_REFLECT,
                         p=0.2
                ),
                A.RandomContrast(0.1),
                A.GaussianBlur()
            ]
        )
        val_transforms = A.Compose(
            [
                A.HorizontalFlip(),
            ]
        )
        self.train_dataset = CovidImageDataset(train_csv, root_dir, train_transforms)
        self.valid_dataset = CovidImageDataset(val_csv, root_dir, val_transforms)

        self.training_data_sampler = WeightedRandomSampler(
            self.train_dataset.get_data_weights(),
            num_samples=len(self.train_dataset),
            replacement=True,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            self.shuffle,
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
            self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.persistent_workers,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass
