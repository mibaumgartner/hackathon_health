import csv
import os
from tqdm import tqdm
from typing import List, Tuple
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
import numpy as np


root_dir = "/hkfs/work/workspace/scratch/im9193-health_challenge/data/"
csv_train = "train.csv"
csv_val = "valid.csv"

target_dir = "/hkfs/work/workspace/scratch/im9193-H1/preprocessed_data/"


data_label_pairs_train: List[Tuple[str, int]] = []
with open(os.path.join(root_dir, csv_train), "r") as f:
    csv_reader = csv.reader(f)
    next(csv_reader, None)  # Skip header
    for row in csv_reader:
        data_label_pairs_train.append(
            (os.path.join(root_dir, "imgs", row[0]), 0 if row[1] == "negative" else 1)
        )

data_label_pairs_val: List[Tuple[str, int]] = []
with open(os.path.join(root_dir, csv_val), "r") as f:
    csv_reader = csv.reader(f)
    next(csv_reader, None)  # Skip header
    for row in csv_reader:
        data_label_pairs_val.append(
            (os.path.join(root_dir, "imgs", row[0]), 0 if row[1] == "negative" else 1)
        )

img_filenames_train_val = [tup[0] for tup in data_label_pairs_train] + [
    tup[0] for tup in data_label_pairs_val
]

transforms = Compose([Resize((224, 224)), ToTensor()])


for file in tqdm(img_filenames_train_val):
    img = Image.open(file)
    npy_arr = transforms(img).numpy()
    np.save(
        os.path.join(target_dir, "imgs", file.split("/")[-1][:-4] + ".npy"), npy_arr
    )
