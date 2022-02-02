import csv
import os
import sys
from argparse import ArgumentParser
from typing import List, Tuple, Dict
from itertools import chain

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor
from pathlib import Path
from medhack.ptmodule.module import BaseClassificationModule


class DummyDataset(Dataset):
    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(np.random.random(size=(3, 224, 224))).half()

    def __len__(self):
        return 20000


class CovidInferenceImageDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        """
        Expects the images to be located in {root_dir}/imgs
        """
        self.test_image_names: List[str] = []
        with open(os.path.join(root_dir, csv_file), "r") as f:
            csv_reader = csv.reader(f)
            next(csv_reader, None)  # Skip header
            for row in csv_reader:
                self.test_image_names.append(os.path.join(root_dir, "imgs", row[0]))
        if csv_file == "valid.csv":
            self.test_image_names = self.test_image_names # * int(200000 / len(self.test_image_names))

        self.transform = Compose([
            Resize((224, 224)),
            ToTensor()])

    def __len__(self):
        return len(self.test_image_names)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        img_name = self.test_image_names[int(idx)]
        image = Image.open(img_name)
        image = torch.repeat_interleave(self.transform(image), 3, dim=0)
        return img_name, image


if __name__ == "__main__":
    parser = ArgumentParser()
    # Here we have potentially multiple checkpoints (4 since each model has one)
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Directory where weights and results are saved",
        default="/hkfs/work/workspace/scratch/im9193-H1/eval_data",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory containing the CSV file you want to predict "
             "and the imgs dir is!",
        default="/hkfs/work/workspace/scratch/im9193-H1/eval_data",
    )

    parser.add_argument("-nw", "--num_workers", default=32, type=int, nargs="?")
    parser.add_argument("-bs", "--batch_size", default=32, type=int, nargs="?")
    parser.add_argument("-ngpu", "--num_gpu", default=4, type=int, nargs="?")

    args = parser.parse_args()

    # GPUS: int = args.num_gpu  # N_GPUS
    WORKERS: int = args.num_workers  # 152/4 38 --> 32
    BS: int = args.batch_size  # BatchSize
    ACCELERATOR = "gpu"
    GPUS = 1
    PRECISION = 16
    BENCHMARK = True
    DETERMINISTIC = False

    save_dir = args.save_dir  # Determines where to save the .csv
    data_dir = args.data_dir  # Decides if test/train is used.
    # data_dir = os.getcwd()
    os.makedirs(save_dir, exist_ok=True)

    test_run = "test_data" in data_dir

    # load model with pretrained weights
    final_model_arch = "resnet18"
    final_model_ckpt_path = "/hkfs/work/workspace/scratch/im9193-H1/checkpoints/logs/resnet18_pre_noNormAugPlusPostFix_ddp/version_1/" \
                            "checkpoints/epoch=99-step=49399.ckpt"
    model = BaseClassificationModule(run_name="Nobody_cares",
                                     architecture=final_model_arch,
                                     pretrained=False,
                                     epochs=0,
                                     init_learning_rate=0,
                                     weight_decay=0,
                                     loss="None",
                                     is_testing=True
                                     )

    # dataloader
    print(f"Running inference on {'test' if test_run else 'validation'} data")

    if test_run:
        # dataset = DummyDataset()
        dataset = CovidInferenceImageDataset("test.csv", root_dir=data_dir)
    else:
        # dataset = DummyDataset()
        dataset = CovidInferenceImageDataset("valid.csv", root_dir=data_dir)

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        accelerator=ACCELERATOR,
        precision=PRECISION,
        benchmark=True,
        enable_model_summary=False,
        gpus=GPUS,
        reload_dataloaders_every_n_epochs=False,
        enable_progress_bar=False,
        deterministic=False,
        replace_sampler_ddp=True,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=BS,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=True,
    )

    predictions: List[int]
    with torch.inference_mode():
        outputs: List[Tuple[List[str], torch.Tensor]] = trainer.predict(
            model=model,
            dataloaders=dataloader,
            return_predictions=True,
            ckpt_path=final_model_ckpt_path
        )
        img_names = list(chain.from_iterable([o[0] for o in outputs]))
        predictions = list(torch.cat([o[1] for o in outputs], dim=0).detach().cpu().numpy())

    with open(os.path.join(data_dir, "predictions.csv"), "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["image", "prediction"])
        for img_name, prediction in zip(img_names, predictions):
            csv_writer.writerow([Path(img_name).name, prediction])

    if test_run == "test":
        sys.exit()
    else:
        print("Done! The result is saved in {}".format(save_dir))

        groundtruths: Dict[str, int] = {}
        with open(os.path.join(data_dir, "valid_with_labels.csv"), "r") as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                groundtruths[row["image"]] = 0 if row["label"] == 'negative' else 1

        read_predictions: Dict[str, int] = {}
        with open(os.path.join(save_dir, "predictions.csv"), "r") as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                read_predictions[row["image"]] = row["prediction"]

        is_true: List[bool] = []
        for key in groundtruths.keys():
            gt = groundtruths[key]
            pd = read_predictions[key]
            is_true.append(int(gt) == int(pd))
            
        acc = float(sum(is_true)) / float(len(is_true))
        print(f"Accuracy read from predictions.csv: {acc}")
