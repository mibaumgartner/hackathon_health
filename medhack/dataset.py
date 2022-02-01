import csv
from pathlib import Path
from typing import List, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CovidImageDataset(Dataset):
    def __init__(self, csv_file: str, root_dir: Path, transform: A.Compose = None):
        """
        :param csv_file: Name of the csv file ("train.csv" or "valid.csv")
        :param root_dir: Path to the root directory of the data.
         Expects: "train.csv" in there and the direcotry "imgs" in root_dir/imgs
        :param transform: albumentations transformations -- ToTensorV2 already included
        (Has to be preprocessing trans for test cases)
        """

        self.data_label_pairs: List[Tuple[str, int]] = []
        with open(root_dir / csv_file, "r") as f:
            csv_reader = csv.reader(f)
            next(csv_reader, None)  # Skip header
            for row in csv_reader:
                self.data_label_pairs.append(
                    (
                        str(root_dir / "imgs" / (row[0][:-4] + ".npy")),
                        0 if row[1] == "negative" else 1,
                    )
                )
        self.negative_data_samples: List[Tuple[str, int]] = [
            d for d in self.data_label_pairs if d[1] == 0
        ]
        self.positive_data_samples: List[Tuple[str, int]] = [
            d for d in self.data_label_pairs if d[1]
        ]
        self.transform = transform

    def __len__(self):
        return len(self.data_label_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        data = self.data_label_pairs[idx]
        image: np.ndarray = np.load(data[0])
        label: int = data[1]
        # gray_image_np = np.expand_dims(self.transform(image=image)["image"], 0)
        # rgb_image_np = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # rgb_image_torch = A.Compose([ToTensorV2()])(image=rgb_image_np)["image"]

        rgb_image_np = self.transform(image=image)["image"].repeat(3, axis=0)
        rgb_image_torch = torch.tensor(rgb_image_np, dtype=torch.float)
        return rgb_image_torch, label

    def get_data_weights(self) -> List[float]:
        labels: List[int] = [d[1] for d in self.data_label_pairs]
        positives = sum(labels)
        total_samples = len(labels)
        prob_positive = 1 - positives / total_samples
        prob_negatives = positives / total_samples
        return [prob_positive if l == 1 else prob_negatives for l in labels]
