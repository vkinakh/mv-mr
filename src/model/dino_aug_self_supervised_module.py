from typing import Dict

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
import pytorch_lightning as pl

from src.model import ResnetMultiProj, DeiTMultiProj, CaiTMultiProj
from src.loss import DistanceCorrelation
from src.transform import DataAugmentationDINO
from src.utils import get_params_groups


np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.benchmark = True


class DINOAugSelfSupervisedModule(pl.LightningModule):

    def __init__(self, config: Dict):
        super().__init__()
        self._config = config
        params_enc = config['encoder']

        if config['encoder_type'] == 'resnet':
            self._encoder = ResnetMultiProj(**params_enc)
        elif config['encoder_type'] == 'deit':
            self._encoder = DeiTMultiProj(**params_enc)
        elif config['encoder_type'] == 'cait':
            self._encoder = CaiTMultiProj(**params_enc)
        else:
            raise NotImplementedError(f"Encoder type {config['encoder_type']} not implemented")

        self._loss_dc = self.get_loss()
        self._margin_std = self.config['std_margin']
        self.hparams.batch_size = config['batch_size']  # set start batch_size
        self.hparams.lr = eval(config['lr'])
        self.online_finetuner = nn.Linear(self._encoder.num_features, config['dataset']['n_classes'])

    @property
    def encoder(self):
        return self._encoder

    @property
    def num_features(self) -> int:
        return self._encoder.num_features

    def get_loss(self):
        return DistanceCorrelation()

    def get_dataloader(self, stage: str) -> DataLoader:
        is_train = stage == 'train'

        bs = self.hparams.batch_size
        n_workers = self.config['n_workers']

        split = 'train+unlabeled' if is_train else 'test'
        trans = DataAugmentationDINO(**self.config['transform'])
        ds = datasets.STL10('./data', split=split, download=True, transform=trans)
        dl = DataLoader(ds, batch_size=bs, shuffle=is_train, num_workers=n_workers, pin_memory=is_train,
                        drop_last=True)
        return dl

    def train_dataloader(self):
        return self.get_dataloader('train')

    def val_dataloader(self):
        return self.get_dataloader('val')

    def configure_optimizers(self):
        lr = self.hparams.lr
        wd = eval(self.config['wd'])

        opt_type = self.config['optimizer']

        if opt_type == 'adam':
            opt = optim.Adam(self._encoder.parameters(), lr=lr, weight_decay=wd)
        elif opt_type == 'adamw':

            if self.config['encoder_type'] in ['deit', 'cait']:
                params = get_params_groups(self._encoder)
            else:
                params = self._encoder.parameters()
            opt = optim.AdamW(params, lr=lr, weight_decay=wd)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(self.train_dataloader()))
        return [opt], [sched]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def step(self, batch, batch_idx: int, *, stage: str) -> torch.Tensor:
        imgs, _ = batch

        img_ga = imgs[0]
        img_gb = imgs[1]

        imgs_loc = imgs[2:]

        h_ga, z_ga = self._encoder(img_ga)
        h_gb, z_gb = self._encoder(img_gb)

        if self.config['normalize_z']:
            z_ga = F.normalize(z_ga, dim=1)
            z_gb = F.normalize(z_gb, dim=1)

        # trick for FP16 training, scaling doesn't affect dist corr
        z_ga_scaled = z_ga / 32
        z_gb_scaled = z_gb / 32

        # MSE loss
        loss_mse = F.mse_loss(z_ga, z_gb)
        # STD loss
        std_za = z_ga.std(dim=0)
        std_zb = z_gb.std(dim=0)
        loss_std = torch.relu(self._margin_std - std_za).mean() + torch.relu(self._margin_std - std_zb).mean()
        # distance correlation on z
        loss_dc_zz = 1 - self._loss_dc(z_ga_scaled, z_gb_scaled)
        loss = loss_mse + loss_std + loss_dc_zz

        loss_dc_loc = 0
        for img_loc in imgs_loc:
            # h_loc, z_loc = self._encoder(img_loc)
            # if self.config['normalize_z']:
            #     z_loc = F.normalize(z_loc, dim=1)
            # z_loc_scaled = z_loc / 32

            feat = img_loc.flatten(start_dim=1) / 32
            loss_dc_zz_loc = 1 - self._loss_dc(z_ga_scaled, feat)
            loss_dc_loc += loss_dc_zz_loc

        res_dict = {
            f'{stage}/loss': loss.item(),
            f'{stage}/MSE': loss_mse.item(),
            f'{stage}/STD': loss_std.item(),
            f'{stage}/DC_zz': loss_dc_zz.item(),
            f'{stage}/DC_loc': loss_dc_loc.item(),
        }

        self.log_dict(res_dict)

        # manual lr scheduling
        if self.current_epoch >= self.config['warmup_epochs']:
            sch = self.lr_schedulers()
            sch.step()
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, stage='train')

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, stage='val')

    @property
    def config(self) -> Dict:
        return self._config

    def forward(self, x: torch.Tensor):
        return self._encoder(x)
