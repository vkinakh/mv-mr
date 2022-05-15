from pytorch_lightning.callbacks import Callback
from pytorch_lightning.core.saving import save_hparams_to_yaml


class OnCheckpointHparams(Callback):
    def on_save_checkpoint(self, trainer, pl_module):
        # only do this 1 time
        if trainer.current_epoch == 0:
            file_path = f"{trainer.logger.log_dir}/hparams.yaml"
            print(f"Saving hparams to file_path: {file_path}")
            save_hparams_to_yaml(config_yaml=file_path, hparams=pl_module.hparams)
