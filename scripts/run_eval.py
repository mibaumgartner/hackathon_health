import csv
import os
import sys
from argparse import ArgumentParser
from typing import List, Tuple, Dict
from itertools import chain

import pickle
import torch.distributed as dist
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor
from pathlib import Path
from medhack.ptmodule.module import BaseClassificationModule


class CovidInferenceImageDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        """
        
        Expects the images to be located in {root_dir}/imgs
        :param csv_file: Filepath of the CSV file
        :Param root_dir: Path to the directory containing the data.
        """
        self.test_image_names: List[str] = []
        with open(csv_file, "r") as f:
            csv_reader = csv.reader(f)
            next(csv_reader, None)  # Skip header
            for row in csv_reader:
                self.test_image_names.append(os.path.join(root_dir, row[0]))

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
        help="Directory where predictions will be saved to as 'predictions.csv'",
        default="/hkfs/work/workspace/scratch/im9193-H1/eval_data",
    )
    
    parser.add_argument(
        "--csv_filepath",
        type=str,
        help="Path to the csv file",
        default="/hkfs/work/workspace/scratch/im9193-H1/eval_data",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory to the folder containing the images of the csv file.)",
        default="/hkfs/work/workspace/scratch/im9193-H1/eval_data",
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/hkfs/work/workspace/scratch/im9193-H1/checkpoints/logs/"
        "resnet18_pre_noNormAugPlusPostFix_ddp/version_1/checkpoints/epoch=99-step=49399.ckpt",
    )

    args = parser.parse_args()

    # GPUS: int = args.num_gpu  # N_GPUS
    WORKERS: int = 32
    BS: int = 64
    ACCELERATOR = "gpu"
    GPUS = 4
    PRECISION = 16
    BENCHMARK = True
    DETERMINISTIC = False

    save_dir = args.save_dir  # Determines where to save the .csv
    data_dir = args.data_dir  # Decides if test/train is used.
    ckpt_path = args.ckpt_path
    csv_filepath = args.csv_filepath

    # data_dir = os.getcwd()
    os.makedirs(save_dir, exist_ok=True)

    test_run = True
    # "test_data" in data_dir

    # load model with pretrained weights
    final_model_arch = "resnet18"
    final_model_ckpt_path = ckpt_path

    model = BaseClassificationModule(run_name="Nobody_cares",
                                     architecture=final_model_arch,
                                     pretrained=False,
                                     epochs=0,
                                     init_learning_rate=0,
                                     weight_decay=0,
                                     loss="None",
                                     is_testing=True,
                                     output_path=save_dir
                                     )

    # dataloader
    # print(f"Running inference on {'test' if test_run else 'validation'} data")

    dataset = CovidInferenceImageDataset(csv_filepath, root_dir=data_dir)

    trainer = pl.Trainer(
        logger=False,
        strategy="ddp" if GPUS > 1 else None,
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
    all_preds = list(
        torch.cat([o[1] for o in outputs], dim=0).detach().cpu().numpy()
    )

    if GPUS > 1:
        rank = dist.get_rank()
    else:
        rank = 0

    output_path = Path(save_dir) / f"gpu_{rank}_prediction.csv"
    with open(output_path, "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["image", "prediction"])
        for img_name, prediction in zip(img_names, all_preds):
            csv_writer.writerow([Path(img_name).name, prediction])

    # expected_outputs = [save_dir / f"gpu_{rank}_prediction.csv" for rank in range(4)]
    # all_image_names = []
    # all_preds = []
    # for output_file in expected_outputs:
    #     with open(output_file, "r") as f:
    #         csv_reader = csv.DictReader(f)
    #         for row in csv_reader:
    #             all_image_names.append(row["image"])
    #             all_preds.append(row["prediction"])

    if GPUS > 1:
        dist.barrier()
    if GPUS == 1 or dist.get_rank() == 0:
        expected_outputs = [Path(save_dir) / f"gpu_{rank}_prediction.csv" for rank in range(4)]
        all_image_names = []
        all_preds = []
        for output_file in expected_outputs:
            with open(output_file, "r") as f:
                csv_reader = csv.DictReader(f)
                for row in csv_reader:
                    all_image_names.append(row["image"])
                    all_preds.append(row["prediction"])

        # all_image_names = list(chain.from_iterable([o[0] for o in outputs]))
        # all_preds = list(
        #     torch.cat([o[1] for o in outputs], dim=0).detach().cpu().numpy()
        # )

        with open(os.path.join(save_dir, "predictions.csv"), "w") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["image", "prediction"])
            for img_name, prediction in zip(all_image_names, all_preds):
                csv_writer.writerow([img_name, prediction])

        sys.exit()