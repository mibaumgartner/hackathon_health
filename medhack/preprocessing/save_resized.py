import csv
import os
from typing import List, Tuple

root_dir = "/hkfs/work/workspace/scratch/im9193-health_challenge/data/"
csv_train = "train.csv"
csv_val = "valid.csv"


data_label_pairs_train: List[Tuple[str, int]] = []
with open(os.path.join(root_dir, csv_train), "r") as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
        data_label_pairs_train.append(
            (os.path.join(root_dir, "imgs", row[0]), 0 if row[1] == "negative" else 1))
        
data_label_pairs_val: List[Tuple[str, int]] = []
with open(os.path.join(root_dir, csv_val), "r") as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
        data_label_pairs_val.append(
            (os.path.join(root_dir, "imgs", row[0]), 0 if row[1] == "negative" else 1))
        
import IPython
IPython.embed()
