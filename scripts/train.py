import argparse
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from medhack.data_loading import BasicDataModule
from medhack.ptmodule.module import (
    BaseClassificationModule,
)
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
GPUS = 1
ACCELERATOR = "gpu"
PRECISION = 16
BENCHMARK = True
DETERMINISTIC = False

TRAIN_DIR = "/hkfs/work/workspace/scratch/im9193-H1/checkpoints"
ROOT_DIR = "/hkfs/work/workspace/scratch/im9193-H1/preprocessed_data"

# TRAINING PARAMS

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, help="Experiment Name")
    parser.add_argument(
        "-a",
        "--architecture",
        default="resnet18",
        nargs="?",
        help="Name of the Timm Architecture.",
    )
    parser.add_argument(
        "-l", "--loss_name", default="CE", choices=["CE"], nargs="?"
    )
    parser.add_argument("-wd", "--weight_decay", default=1e-5, nargs="?")
    parser.add_argument("-lr", "--learning_rate", default=1e-4, nargs="?")
    parser.add_argument("-loss", "--loss_name", default="CE", nargs="?")
    parser.add_argument("-pt",
                        "--pretrained",
                        default=True,
                        type=str2bool,
                        nargs="?")
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Name of the ModelTrainer, basically the different Dataset splits.",
    )

    args = parser.parse_args()
    name = args.name
    ARCH = args.architecture
    MAX_EPOCHS = args.epochs
    LR = args.learning_rate
    WD = args.weight_decay
    LOSS_NAME = args.loss_name
    PRETRAINED = args.pretrained
    

    root_dir = Path(ROOT_DIR)
    train_dir = Path(TRAIN_DIR) / name

    if train_dir.is_dir():
        raise RuntimeError("Train dir already exists")
    else:
        train_dir.mkdir(parents=True)

    print("Setup data")
    datamodule = BasicDataModule(root_dir=root_dir)
    module = BaseClassificationModule(run_name=name,
                                      architecture=ARCH,
                                      pretrained=PRETRAINED,
                                      epochs=MAX_EPOCHS,
                                      init_learning_rate=LR,
                                      weight_decay=WD,
                                      loss=LOSS_NAME)

    print("Setup Callbacks and Logging")
    callbacks = []
    checkpoint_cb = ModelCheckpoint(
        dirpath=train_dir,
        filename="model_best",
        save_last=True,
        save_top_k=3,
        monitor="val/acc",
        mode="max",
    )
    callbacks.append(checkpoint_cb)

    log_dir = Path(TRAIN_DIR) / "logs"
    log_dir.mkdir()
    mllogger = TensorBoardLogger(
        save_dir=str(log_dir),
        name=name,
    )

    print("Start Training")
    trainer = pl.Trainer(
        gpus=GPUS,
        accelerator=ACCELERATOR,
        precision=PRECISION,
        benchmark=BENCHMARK,
        deterministic=DETERMINISTIC,
        # callbacks=callbacks,
        logger=mllogger,
        max_epochs=MAX_EPOCHS,
        progress_bar_refresh_rate=None,
        reload_dataloaders_every_epoch=False,
        num_sanity_val_steps=10,
        weights_summary="full",
        # plugins=plugins,
        move_metrics_to_cpu=False,
    )
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
