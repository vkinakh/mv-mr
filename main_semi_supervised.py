from pathlib import Path
from datetime import datetime
import yaml
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from src.model import SemiSupervisedModule, ResnetMultiProj
from src.utils import get_config


def main(args) -> None:
    config = get_config(args.config)
    auto_bs = args.auto_bs
    auto_lr = args.auto_lr
    path_ckpt = args.path_ckpt

    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")

    # load encoder
    encoder = ResnetMultiProj(**config['encoder']).eval()
    encoder.load_state_dict(torch.load(path_ckpt)['encoder'])
    encoder = encoder.backbone

    # configure semi-supervised module
    module_semi = SemiSupervisedModule(encoder=encoder, config=config)

    # configure training for semi-supervised module
    logger = pl_loggers.TensorBoardLogger(save_dir='lightning_logs', name=f"{date_time}_{config['comment']}")
    precision = 16 if config['fp16'] else 32
    accumulate_grad_batches = 1 if not config['accumulate_grad_batches'] else config['accumulate_grad_batches']
    epochs = config['epochs']
    eval_every = config['eval_every']

    # configure callbacks
    callback_lr = LearningRateMonitor('step')
    callback_best_ckpt = ModelCheckpoint(every_n_epochs=1, filename='best_{epoch}_{step}', monitor='val/acc',
                                         mode='max')
    callback_last_ckpt = ModelCheckpoint(every_n_epochs=1, filename='last_{epoch}_{step}')

    # train semi-supervised module
    trainer = pl.Trainer(logger=logger,
                         callbacks=[callback_lr, callback_best_ckpt, callback_last_ckpt],
                         gpus=-1, auto_select_gpus=True,
                         auto_scale_batch_size=auto_bs,
                         max_epochs=epochs,
                         check_val_every_n_epoch=eval_every,
                         strategy='ddp',
                         precision=precision,
                         accumulate_grad_batches=accumulate_grad_batches)

    if auto_lr:
        lr_finder = trainer.tuner.lr_find(module_semi, min_lr=1e-5, max_lr=1e-1,)
        lr = lr_finder.suggestion()
        print(f'LR: {lr}')
        module_semi.hparams.lr = lr
        # save suggested lr and bs to config
        config['lr_suggested'] = lr

    trainer.tune(module_semi)

    if auto_bs:
        config['batch_size_suggested'] = module_semi.hparams.batch_size

    # semi-supervised module checkpoint
    path_checkpoint = config['fine_tune_from']

    trainer.fit(module_semi, ckpt_path=path_checkpoint)

    # save config to file
    save_path = Path(logger.experiment.get_logdir()) / Path(args.config).name
    with open(save_path, 'w') as f:
        yaml.dump(config, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help='path to config file')
    parser.add_argument('--path_ckpt', '-p', type=str, required=True, help='path to checkpoint file')
    parser.add_argument('--auto_bs',
                        action='store_true',
                        help='auto-select batch size')
    parser.add_argument('--auto_lr',
                        action='store_true',
                        help='auto-select learning rate')
    args = parser.parse_args()
    main(args)
