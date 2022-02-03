from pytorch_lightning.callbacks import BasePredictionWriter


class CSVWriter(BasePredictionWriter):
    def write_on_epoch_end(
        self, trainer, pl_module: 'LightningModule', predictions, batch_indices,
    ):
        breakpoint()
