import csv
import os
from tqdm import tqdm
from typing import List, Tuple
import numpy as np


root_dir = "/hkfs/work/workspace/scratch/im9193-H1/preprocessed_data/"
csv_train = "train.csv"
csv_val = "valid.csv"

data_label_pairs_train: List[Tuple[str, int]] = []
with open(os.path.join(root_dir, csv_train), "r") as f:
    csv_reader = csv.reader(f)
    next(csv_reader, None)  # Skip header
    for row in csv_reader:
        data_label_pairs_train.append(
            (os.path.join(root_dir, "imgs", row[0]), 0 if row[1] == "negative" else 1))

data_label_pairs_val: List[Tuple[str, int]] = []
with open(os.path.join(root_dir, csv_val), "r") as f:
    csv_reader = csv.reader(f)
    next(csv_reader, None)  # Skip header
    for row in csv_reader:
        data_label_pairs_val.append(
            (os.path.join(root_dir, "imgs", row[0]), 0 if row[1] == "negative" else 1))

img_filenames_train_val = [tup[0] for tup in data_label_pairs_train] + [tup[0] for tup in data_label_pairs_val]

img_data = np.zeros((len(img_filenames_train_val), 224, 224))

for i, file in enumerate(tqdm(img_filenames_train_val)):
    img_arr = np.load(file[:-4] + ".npy")
    img_data[i] = img_arr


import IPython
IPython.embed()
