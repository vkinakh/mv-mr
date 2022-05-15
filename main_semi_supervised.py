from pathlib import Path
from datetime import datetime
import yaml
import argparse

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from src.model import SemiSupervisedModule, SelfSupervisedModule
from src.utils import get_config


def main(args) -> None:
    path_config_self = args.config_self_supervised
    path_config_semi = args.config_semi_supervised
    path_self = args.path_self_supervised
    auto_bs = args.auto_bs
    auto_lr = args.auto_lr

    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    config_self = get_config(path_config_self)
    config_semi = get_config(path_config_semi)

    # load pretrained self-supervised module
    module_self = SelfSupervisedModule.load_from_checkpoint(checkpoint_path=path_self, config=config_self)
    encoder = module_self.encoder.backbone  # select backbone from ssl-encoder

    # configure semi-supervised module
    module_semi = SemiSupervisedModule(encoder=encoder, config=config_semi)

    # configure training for semi-supervised module
    logger = pl_loggers.TensorBoardLogger(save_dir='lightning_logs', name=f"{date_time}_{config_semi['comment']}")
    precision = 16 if config_semi['fp16'] else 32
    accumulate_grad_batches = 1 if not config_semi['accumulate_grad_batches'] \
        else config_semi['accumulate_grad_batches']
    epochs = config_semi['epochs']
    eval_every = config_semi['eval_every']

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
        config_semi['lr_suggested'] = lr

    trainer.tune(module_semi)

    if auto_bs:
        config_semi['batch_size_suggested'] = module_semi.hparams.batch_size

    # semi-supervised module checkpoint
    path_checkpoint = config_semi['fine_tune_from']

    trainer.fit(module_semi, ckpt_path=path_checkpoint)

    # save config to file
    save_path = Path(logger.experiment.get_logdir()) / Path(path_config_semi).name
    with open(save_path, 'w') as f:
        yaml.dump(config_semi, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_self_supervised',
                        type=str, required=True,
                        help='path to config file for self-supervised module')
    parser.add_argument('--config_semi_supervised',
                        type=str, required=True,
                        help='path to config file for semi-supervised module')
    parser.add_argument('--path_self_supervised',
                        type=str, required=True,
                        help='path to checkpoint file for self-supervised module')
    parser.add_argument('--auto_bs',
                        action='store_true',
                        help='auto-select batch size')
    parser.add_argument('--auto_lr',
                        action='store_true',
                        help='auto-select learning rate')
    args = parser.parse_args()
    main(args)
