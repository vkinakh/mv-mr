import argparse
from pathlib import Path
from datetime import datetime
import yaml

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from src.utils import get_config
from src.model import SelfSupervisedModule, OnlineFineTuner


def main(args) -> None:
    config_path = args.config
    auto_bs = args.auto_bs
    auto_lr = args.auto_lr

    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    config = get_config(config_path)
    logger = pl_loggers.TensorBoardLogger(save_dir='lightning_logs', name=f"{date_time}_{config['comment']}")

    precision = 16 if config['fp16'] else 32
    accumulate_grad_batches = 1 if not config['accumulate_grad_batches'] else config['accumulate_grad_batches']
    epochs = config['epochs']
    eval_every = config['eval_every']

    module = SelfSupervisedModule(config)

    # configure callbacks
    callback_lr = LearningRateMonitor('step')
    callback_best_ckpt = ModelCheckpoint(every_n_epochs=1, filename='best_{epoch}_{step}', monitor='val/acc',
                                         mode='max')
    callback_last_ckpt = ModelCheckpoint(every_n_epochs=1, filename='last_{epoch}_{step}')

    encoder_dim = module.num_features
    n_classes = config['dataset']['n_classes']
    callback_finetuner = OnlineFineTuner(encoder_dim, n_classes)

    # checkpoint
    path_checkpoint = config['fine_tune_from']
    grad_clip_val = config['gradient_clip_val'] if 'gradient_clip_val' in config else None

    trainer = pl.Trainer(logger=logger,
                         callbacks=[callback_lr,
                                    callback_best_ckpt,
                                    callback_last_ckpt,
                                    callback_finetuner
                                    ],
                         gpus=-1, auto_select_gpus=True,
                         auto_scale_batch_size=auto_bs,
                         max_epochs=epochs,
                         check_val_every_n_epoch=eval_every,
                         strategy='ddp',
                         log_every_n_steps=config['log_every'],
                         precision=precision,
                         accumulate_grad_batches=accumulate_grad_batches,
                         gradient_clip_val=grad_clip_val)

    if auto_lr:
        lr_finder = trainer.tuner.lr_find(module, min_lr=1e-5, max_lr=1e-1,)
        lr = lr_finder.suggestion()
        print(f'LR: {lr}')
        module.hparams.lr = lr
        # save suggested lr and bs to config
        config['lr_suggested'] = lr

    trainer.tune(module)

    if auto_bs:
        config['batch_size_suggested'] = module.hparams.batch_size

    trainer.fit(module, ckpt_path=path_checkpoint)

    # save config to file
    save_path = Path(logger.experiment.get_logdir()) / Path(config_path).name
    with open(save_path, 'w') as f:
        yaml.dump(config, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c',
                        type=str,
                        help='path to config file')
    parser.add_argument('--auto_bs',
                        action='store_true')
    parser.add_argument('--auto_lr',
                        action='store_true')
    args = parser.parse_args()
    main(args)
