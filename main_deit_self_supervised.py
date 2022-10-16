import argparse
from pathlib import Path
from datetime import datetime
import yaml

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from src.utils import get_config
from src.model import DeiTSelfSupervisedModule, OnlineFineTuner


def main(args) -> None:
    config_path = args.config

    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    config = get_config(config_path)
    logger = pl_loggers.TensorBoardLogger(save_dir='lightning_logs', name=f"{date_time}_{config['comment']}")

    precision = 16 if config['fp16'] else 32
    epochs = config['epochs']
    eval_every = config['eval_every']

    module = DeiTSelfSupervisedModule(config)

    callback_lr = LearningRateMonitor('step')
    callback_best_ckpt = ModelCheckpoint(every_n_epochs=1, filename='best_{epoch}_{step}', monitor='val/acc',
                                         mode='max')
    callback_last_ckpt = ModelCheckpoint(every_n_epochs=1, filename='last_{epoch}_{step}')

    encoder_dim = module.num_features
    n_classes = config['dataset']['n_classes']
    callback_finetuner = OnlineFineTuner(encoder_dim, n_classes)

    # checkpoint
    path_checkpoint = config['fine_tune_from']

    trainer = pl.Trainer(logger=logger,
                         callbacks=[callback_lr,
                                    callback_best_ckpt,
                                    callback_last_ckpt,
                                    callback_finetuner
                                    ],
                         gpus=-1, auto_select_gpus=True,
                         max_epochs=epochs,
                         check_val_every_n_epoch=eval_every,
                         strategy='ddp',
                         precision=precision)
    trainer.fit(module, ckpt_path=path_checkpoint)

    # save config to file
    save_path = Path(logger.experiment.get_logdir()) / Path(config_path).name
    with open(save_path, 'w') as f:
        yaml.dump(config, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    args = parser.parse_args()
    main(args)
