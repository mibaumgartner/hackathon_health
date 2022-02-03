import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import BaseFinetuning


class FeatureExtractorFreezeUnfreeze(BaseFinetuning):
    def __init__(self, train_last_n_modules, unfreeze_at_epoch=100):
        super().__init__()
        self._train_last_n_modules = train_last_n_modules
        self._unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, pl_module):
        sub_modules = [module for module in pl_module.model.children()]
        for module in sub_modules[: -self._train_last_n_modules]:
            self.freeze(module)

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
        if current_epoch == self._unfreeze_at_epoch:
            sub_modules = [module for module in pl_module.model.children()]
            for module in sub_modules[: -self._train_last_n_modules]:
                self.unfreeze_and_add_param_group(
                    modules=pl_module.feature_extractor,
                    optimizer=optimizer,
                    train_bn=True,
                )
