import argparse
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from medhack.data_loading import BasicDataModule
from medhack.ptmodule.ft_callback import FeatureExtractorFreezeUnfreeze
from medhack.ptmodule.ft_module import (
    FineTuneClassificationModule,
)
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
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
        type=str,
        nargs="?",
        help="Name of the Timm Architecture.",
    )
    parser.add_argument("-wd", "--weight_decay", default=1e-5, type=float, nargs="?")
    parser.add_argument("-lr", "--learning_rate", default=1e-4, type=float, nargs="?")
    parser.add_argument("-loss", "--loss_name", default="CE", type=str, nargs="?")
    parser.add_argument("-pt", "--pretrained", default=True, type=str2bool, nargs="?")
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
    )
    parser.add_argument("-nw", "--num_workers", default=16, type=int, nargs="?")
    parser.add_argument("-bs", "--batch_size", default=32, type=int, nargs="?")
    parser.add_argument("-ngpu", "--num_gpu", default=1, type=int, nargs="?")

    args = parser.parse_args()
    name = args.name
    ARCH = args.architecture
    MAX_EPOCHS = args.epochs
    LR = args.learning_rate
    WD = args.weight_decay
    LOSS_NAME = args.loss_name
    PRETRAINED = args.pretrained
    NUM_WORKERS = args.num_workers
    BATCH_SIZE = args.batch_size
    GPUS = args.num_gpu

    ACCELERATOR = "gpu" if GPUS > 0 else "cpu"

    root_dir = Path(ROOT_DIR)
    train_dir = Path(TRAIN_DIR) / name

    train_dir.mkdir(parents=True, exist_ok=True)

    print("Setup data")
    datamodule = BasicDataModule(
        root_dir=root_dir, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, gpu_num=GPUS
    )
    module = FineTuneClassificationModule(
        run_name=name,
        architecture=ARCH,
        pretrained=PRETRAINED,
        epochs=MAX_EPOCHS,
        init_learning_rate=LR,
        weight_decay=WD,
        loss=LOSS_NAME,
        is_testing=False,
    )

    print("Setup Callbacks and Logging")
    callbacks = []
    # checkpoint_cb = ModelCheckpoint(
    #     dirpath=train_dir,
    #     filename="model_best",
    #     save_last=True,
    #     save_top_k=3,
    #     monitor="val/acc",
    #     mode="max",
    # )
    # callbacks.append(checkpoint_cb)

    finetune_cb = FeatureExtractorFreezeUnfreeze(
        train_last_n_modules=1, unfreeze_at_epoch=100
    )
    callbacks.append(finetune_cb)

    log_dir = Path(TRAIN_DIR) / "logs"
    log_dir.mkdir(exist_ok=True)
    mllogger = TensorBoardLogger(
        save_dir=str(log_dir),
        name=name,
    )

    kwargs = {}
    if GPUS > 1:
        kwargs["strategy"] = "ddp"

    print("Start Training")

    # TODO:
    # gradient_clip_val=12.0 ?
    # reload_dataloaders_every_n_epochs: int = 0 by default, means reloading every epoch - change?
    # sync_batchnorm False by default, change?
    # add tracking of grad norm? -> track_grad_norm 2 for l2
    # maybe sync validation logging:
    # https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html#synchronize-validation-and-test-logging

    # # Simulate DDP for debugging on your GPU-less laptop
    # trainer = Trainer(accelerator="cpu", strategy="ddp", num_processes=2)

    trainer = pl.Trainer(
        gpus=GPUS,
        accelerator=ACCELERATOR,
        precision=PRECISION,
        benchmark=BENCHMARK,
        deterministic=DETERMINISTIC,
        callbacks=callbacks,  # TODO: add top3 weights callback?
        logger=mllogger,
        max_epochs=MAX_EPOCHS,
        progress_bar_refresh_rate=None,
        reload_dataloaders_every_epoch=False,
        num_sanity_val_steps=10,
        weights_summary="full",
        sync_batchnorm=True,
        # plugins=plugins,
        move_metrics_to_cpu=False,
        replace_sampler_ddp=False if GPUS > 1 else True,
        # find_unused_parameters=False,
        # track_grad_norm=2,
        **kwargs,
    )
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
