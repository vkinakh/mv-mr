from pathlib import Path
from datetime import datetime
import argparse
import yaml

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from src.model import VocLinearEvalModule, SelfSupervisedModule
from src.utils import get_config


def main(args) -> None:
    path_config_self = args.config_self
    path_config_voc = args.config_voc
    path_self = args.path_self
    auto_bs = args.auto_bs
    auto_lr = args.auto_lr

    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    config_self = get_config(path_config_self)
    config_voc = get_config(path_config_voc)

    # load pretrained self-supervised model
    module = SelfSupervisedModule.load_from_checkpoint(checkpoint_path=path_self, config=config_self)
    # configure model for VOC evaluation
    module_voc = VocLinearEvalModule(encoder=module.encoder.backbone, config=config_voc)

    # configure training for semi-supervised model
    logger = pl_loggers.TensorBoardLogger(save_dir='lightning_logs', name=f"{date_time}_{config_voc['comment']}")
    precision = 16 if config_voc['fp16'] else 32
    accumulate_grad_batches = 1 if not config_voc['accumulate_grad_batches'] \
        else config_voc['accumulate_grad_batches']
    epochs = config_voc['epochs']
    eval_every = config_voc['eval_every']

    # configure callbacks
    callback_lr = LearningRateMonitor('step')
    callback_best_ckpt = ModelCheckpoint(every_n_epochs=1, filename='best_{epoch}_{step}',
                                         monitor='val/average_precision', mode='max')
    callback_last_ckpt = ModelCheckpoint(every_n_epochs=1, filename='last_{epoch}_{step}')

    # train VOC model
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
        lr_finder = trainer.tuner.lr_find(module_voc, min_lr=1e-5, max_lr=1e-1,)
        lr = lr_finder.suggestion()
        print(f'LR: {lr}')
        module_voc.hparams.lr = lr
        # save suggested lr and bs to config
        config_voc['lr_suggested'] = lr

    trainer.tune(module_voc)

    if auto_bs:
        config_voc['batch_size_suggested'] = module_voc.hparams.batch_size

    # semi-supervised module checkpoint
    path_checkpoint = config_voc['fine_tune_from']

    trainer.fit(module_voc, ckpt_path=path_checkpoint)

    # save config to file
    save_path = Path(logger.experiment.get_logdir()) / Path(path_config_voc).name
    with open(save_path, 'w') as f:
        yaml.dump(config_voc, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_voc', type=str, required=True,
                        help='path to config file for VOC evaluation')
    parser.add_argument('--config_self', type=str, required=True,
                        help='path to config file for self-supervised model')
    parser.add_argument('--path_self', type=str, required=True,
                        help='path to checkpoint file for self-supervised model')
    parser.add_argument('--auto_bs', action='store_true',
                        help='automatically determine batch size')
    parser.add_argument('--auto_lr', action='store_true',
                        help='automatically determine learning rate')
    args = parser.parse_args()
    main(args)
