import argparse
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from medhack.data_loading import BasicDataModule
from medhack.ptmodule.module import (
    BaselineClassification,
)

GPUS = 1
ACCELERATOR = "gpu"
PRECISION = 16
BENCHMARK = True
DETERMINISTIC = False

TRAIN_DIR = "/hkfs/work/workspace/scratch/im9193-H1/checkpoints"
ROOT_DIR = "/hkfs/work/workspace/scratch/im9193-H1/preprocessed_data"

# TRAINING PARAMS
MAX_EPOCHS = 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, help="Experiment Name")
    args = parser.parse_args()
    name = args.name

    root_dir = Path(ROOT_DIR)
    train_dir = Path(TRAIN_DIR) / name

    if train_dir.is_dir():
        raise RuntimeError("Train dir already exists")
    else:
        train_dir.mkdir(parents=True)

    datamodule = BasicDataModule(root_dir=root_dir)
    module = BaselineClassification()

    callbacks = []
    checkpoint_cb = ModelCheckpoint(
        dirpath=train_dir,
        filename='model_best',
        save_last=True,
        save_top_k=3,
        monitor="val/acc",
        mode="max",
    )
    callbacks.append(checkpoint_cb)

    log_dir = train_dir / "logs"
    log_dir.mkdir()
    mllogger = TensorBoardLogger(
        save_dir=log_dir,
    )

    trainer = pl.Trainer(
        gpus=list(range(GPUS)) if GPUS > 1 else GPUS,
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
        weights_summary='full',
        # plugins=plugins,
        terminate_on_nan=True,
        move_metrics_to_cpu=False,
    )
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    pass
