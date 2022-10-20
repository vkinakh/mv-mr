from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import open_clip

import numpy as np

from src.model import ResnetMultiProj, DeiTMultiProj, CaiTMultiProj
from src.loss import DistanceCorrelation
from src.data import DatasetSSL
from src.transform import AugTransform, ValTransform
from src.utils import get_params_groups


np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.benchmark = True


class CLIPSelfSupervisedModule(pl.LightningModule):

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

        # CLIP feature extractor
        self.clip = open_clip.create_model(**config['clip'], device='cuda', jit=False)
        self.clip.eval()

    @property
    def encoder(self):
        return self._encoder

    @property
    def num_features(self) -> int:
        return self._encoder.num_features

    def get_loss(self):
        return DistanceCorrelation()

    def train_dataloader(self) -> DataLoader:
        bs = self.hparams.batch_size
        n_workers = self.config['n_workers']

        # dataset params
        name = self.config['dataset']['name']
        size = self.config['dataset']['size']
        path = self.config['dataset']['path']
        aug_policy = self.config['dataset']['aug_policy']
        n_aug = 1 if 'n_aug' not in self.config['dataset'] else self.config['dataset']['n_aug']

        trans = AugTransform(name, size, policy=aug_policy)
        trans_orig = ValTransform(name, size)

        ds = DatasetSSL(dataset_name=name, trans=trans, trans_orig=trans_orig,
                        path=path, train=True, unlabeled=True, n_aug=n_aug)
        shuffle = n_aug == 1
        dl = DataLoader(ds, bs, shuffle=shuffle, drop_last=True, num_workers=n_workers,
                        pin_memory=True)
        return dl

    def val_dataloader(self) -> DataLoader:
        bs = self.hparams.batch_size
        n_workers = self.config['n_workers']

        # dataset params
        name = self.config['dataset']['name']
        size = self.config['dataset']['size']
        path = self.config['dataset']['path']
        aug_policy = self.config['dataset']['aug_policy']

        trans = AugTransform(name, size, policy=aug_policy)
        trans_orig = ValTransform(name, size)

        ds = DatasetSSL(dataset_name=name, trans=trans, trans_orig=trans_orig,
                        path=path, train=False, unlabeled=False)
        dl = DataLoader(ds, bs, drop_last=True, num_workers=n_workers)
        return dl

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
        im_orig, im, _ = batch

        h, z = self._encoder(im)
        h_orig, z_orig = self._encoder(im_orig)

        if self.config['normalize_z']:
            z = F.normalize(z, dim=1)
            z_orig = F.normalize(z_orig, dim=1)

        # trick for FP16 training, scaling doesn't affect dist corr
        z_scaled = z / 32
        z_orig_scaled = z_orig / 32

        # MSE loss
        loss_mse = F.mse_loss(z_orig, z)
        # STD loss
        std_z = z.std(dim=0)
        std_z_orig = z_orig.std(dim=0)
        loss_std = torch.relu(self._margin_std - std_z).mean() + torch.relu(self._margin_std - std_z_orig).mean()
        # distance correlation on z
        loss_dc_zz = 1 - self._loss_dc(z_orig_scaled, z_scaled)
        loss = loss_mse + loss_std + loss_dc_zz

        # CLIP loss
        with torch.no_grad():
            im_orig = F.interpolate(im_orig, size=(224, 224), mode='bilinear', align_corners=False)
            clip_z_orig = self.clip.encode_image(im_orig) / 32

        loss_dc_clip = 1 - self._loss_dc(z_scaled, clip_z_orig)
        loss += loss_dc_clip

        res_dict = {
            f'{stage}/loss': loss.item(),
            f'{stage}/MSE': loss_mse.item(),
            f'{stage}/STD': loss_std.item(),
            f'{stage}/DC_zz': loss_dc_zz.item(),
            f'{stage}/DC_clip': loss_dc_clip.item(),
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
