import csv
import os
import sys
from typing import List

import pytorch_lightning as pl
from argparse import ArgumentParser
import torch
import timm
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from torchvision.transforms import Compose, Resize, ToTensor


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
            self.test_image_names = self.test_image_names * int(200000 / len(self.test_image_names))

        self.transform = Compose([Resize((224, 224)), ToTensor()])

    def __len__(self):
        return len(self.test_image_names)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_name = self.test_image_names[int(idx)]
        image = Image.open(img_name)
        image = self.transform(image)
        return image

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--weights_path",
        type=str,
        default="/hkfs/work/workspace/scratch/im9193-health_challenge_baseline/saved_models/vgg_baseline.pt",
        help="Model weights path",
    )  # TODO: adapt to your model weights path in the bash script

    # Here we have potentially multiple checkpoints (4 since each model has one)
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Directory where weights and results are saved",
        default="/hkfs/work/workspace/scratch/im9193-health_challenge_baseline/submission_test",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory containing the data you want to predict",
        default="/hkfs/work/workspace/scratch/im9193-health_challenge",
    )
    
    parser.add_argument("-nw", "--num_workers", default=16, type=int, nargs="?")
    parser.add_argument("-bs", "--batch_size", default=32, type=int, nargs="?")
    parser.add_argument("-ngpu", "--num_gpu", default=1, type=int, nargs="?")

    args = parser.parse_args()

    GPUS: int = args.num_gpu  # N_GPUS
    WORKERS: int = args.num_workers  # 152/4 38 --> 32
    BS: int = args.batch_size  # BatchSize
    ACCELERATOR = "cpu"  # ToDo: Move to GPU once CPU tested!
    PRECISION = 16
    BENCHMARK = True
    DETERMINISTIC = False

    weights_path = Path(args.weights_path)
    save_dir = args.save_dir
    data_dir = args.data_dir  # Decides if test/train is used.

    os.makedirs(save_dir, exist_ok=True)

    check_script = "test_data" not in data_dir

    # load model with pretrained weights
    final_model_arch = "regnety002"
    final_model_chkpt_path = "/hkfs/work/workspace/scratch/im9193-H1/checkpoints/" \
                             "logs/regnety002_normMoreAug_ddp_lowLR/version_0/" \
                             "checkpoints/epoch=19-step=9879.ckpt"
    model = timm.create_model(final_model_arch, checkpoint_path=final_model_chkpt_path)
    model.eval()

    # dataloader
    data_split = "validation" if check_script else "test"
    print("Running inference on {} data".format(data_split))

    if check_script:
        # ToDo: Test that samples are augmented identically"!
        dataset = CovidInferenceImageDataset("test.csv", root_dir=data_dir)
    else:
        dataset = CovidInferenceImageDataset("valid.csv", root_dir=data_dir)

    trainer = pl.Trainer(
        logger=False,
        checkpoint_callback=False,
        enable_checkpointing=False,
        accelerator=ACCELERATOR,
        precision=PRECISION,
        benchmark=True,
        weights_summary=None,
        gpus=GPUS,
        reload_dataloaders_every_n_epochs=False,
        enable_progress_bar=False,
        deterministic=False,
        replace_sampler_ddp=True,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=64,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=True,
    )

    predictions: List[int]
    with torch.no_grad():
        prediction_batches = trainer.predict(
            model=model, dataloaders=dataloader, return_predictions=True
        )
        predictions = torch.cat(prediction_batches, dim=0).detach().cpu().numpy()

    img_names: List[str] = dataset.test_image_names
    with open(os.path.join(data_dir, "predictions.csv"), "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["image", "prediction"])
        for img_name, prediction in zip(img_names, predictions):
            csv_writer.writerow([img_name, prediction])

    if check_script == "test":
        sys.exit()
    else:
        print("Done! The result is saved in {}".format(save_dir))
        groundtruths: dict[str:int] = {}
        with open(os.path.join(data_dir, "valid.csv"), "w") as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                groundtruths[row["image"]] = row["label"]
        read_predictions: dict[str:int] = {}
        with open(os.path.join(data_dir, "predictions.csv"), "w") as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                read_predictions[row["image"]] = row["label"]

        is_true = []
        for key in groundtruths.keys():
            groundtruth = groundtruths[key]
            prediction = predictions[key]
            is_true.append(groundtruth == prediction)
        acc = float(sum(is_true) / len(is_true))
        print("Accuracy read from predictions.csv")
