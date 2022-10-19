from typing import Dict
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from kymatio.torch import Scattering2D
from einops import rearrange
import torch_dct as dct

from src.model import DeiTMultiProj, CaiTMultiProj, HOGLayer
from src.model import WarmupCosineSchedule, CosineWDSchedule, MomentumScheduler
from src.loss import DistanceCorrelation
from src.data import DatasetSSL
from src.transform import AugTransform, ValTransform
from src.utils import std_filter_torch, split_into_patches


def init_opt(
    encoder,
    iterations_per_epoch: int,
    start_lr: float,
    ref_lr: float,
    warmup: int,
    num_epochs: int,
    wd: float = 1e-6,
    final_wd: float = 1e-6,
    final_lr: float = 0.0
):
    """Init optimizer and schedulers

    Args:
        encoder: encoder to optimize
        iterations_per_epoch: number of iterations per epoch
        start_lr: start learning rate
        ref_lr: reference learning rate
        warmup: number of warmup epochs
        num_epochs: number of epochs
        wd: weight decay
        final_wd: final weight decay
        final_lr: final learning rate

    Returns:
        optimizer, lr_scheduler, wd_scheduler
    """

    param_groups = [
        {'params': (p for n, p in encoder.named_parameters()
                    if ('bias' not in n) and ('bn' not in n) and len(p.shape) != 1)},
        {'params': (p for n, p in encoder.named_parameters()
                    if ('bias' in n) or ('bn' in n) or (len(p.shape) == 1)),
         'WD_exclude': True,
         'weight_decay': 0}
    ]
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup*iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(1.25*num_epochs*iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(1.25*num_epochs*iterations_per_epoch))
    return optimizer, scheduler, wd_scheduler


def init_model(encoder_type: str, params: Dict):

    if encoder_type == 'deit':
        encoder = DeiTMultiProj(**params)
    elif encoder_type == 'cait':
        encoder = CaiTMultiProj(**params)
    else:
        raise ValueError(f'Unknown encoder type {encoder_type}')

    target_encoder = copy.deepcopy(encoder)
    target_encoder.eval()

    for p in target_encoder.parameters():
        p.requires_grad = False
    return encoder, target_encoder


def get_loss():
    return DistanceCorrelation()


class DeiTSelfSupervisedModule(pl.LightningModule):

    def __init__(self, config: Dict):
        super().__init__()

        self.hparams.batch_size = config['batch_size']  # set start batch_size

        # load encoders
        self._config = config
        params_enc = config['encoder']
        self.encoder, self.target_encoder = init_model(config['encoder_type'], params_enc)

        self._loss_dc = get_loss()
        self._margin_std = self.config['std_margin']
        self._scatnet, self._hog, self._transform = self.get_feature_extraction()

        # dataloaders
        self._train_dl = self.get_dataloader('train')
        self._val_dl = self.get_dataloader('val')

        # make optimization manually
        self.automatic_optimization = False

        # finetuner head
        self.online_finetuner = nn.Linear(self.encoder.num_features, config['dataset']['n_classes'])

        # momentum scheduler
        self.momentum_scheduler = MomentumScheduler(0.996, 1.0, config['epochs'], len(self._train_dl))

    @property
    def num_features(self):
        return self.encoder.num_features

    @property
    def config(self) -> Dict:
        return self._config

    def get_feature_extraction(self):
        # feature extraction methods
        params_scat = self.config['scatnet']
        scatnet = Scattering2D(**params_scat).to(self.device).eval()

        hog_params = self.config['hog']
        hog = HOGLayer(**hog_params).to(self.device).eval()

        size = self.config['dataset']['size']
        blur_kernel_size = 2 * int(.05 * size) + 1
        color = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)

        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=blur_kernel_size),
        ])
        return scatnet, hog, transform

    def get_dataloader(self, stage: str = 'train'):
        is_train = stage == 'train'

        bs = self.config['batch_size']
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
                        path=path, train=is_train, unlabeled=is_train, n_aug=n_aug)
        shuffle = n_aug == 1 and is_train
        dl = DataLoader(ds, bs, shuffle=shuffle, drop_last=True, num_workers=n_workers,
                        pin_memory=is_train)
        return dl

    def train_dataloader(self) -> DataLoader:
        return self._train_dl

    def val_dataloader(self) -> DataLoader:
        return self._val_dl

    def configure_optimizers(self):
        opt_params = self.config['optimizer']

        optimizer, scheduler, wd_scheduler = init_opt(
            self.encoder,
            iterations_per_epoch=len(self._train_dl),
            **opt_params
        )
        return [optimizer], [scheduler, wd_scheduler]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def _step_dc(self, im_orig: torch.Tensor, z: torch.Tensor, method_dcorr: str):
        if method_dcorr == 'full_img':
            feat = im_orig.flatten(start_dim=1)
            feat = feat / 32  # trick for FP16 training
        elif method_dcorr == 'full_img_scatnet':
            with torch.no_grad():
                feat = self._scatnet(im_orig).flatten(start_dim=1)
                feat = feat / 32  # trick for FP16 training
        elif method_dcorr == 'aug_scatnet':
            with torch.no_grad():
                feat = self._scatnet(self._transform(im_orig)).flatten(start_dim=1)
        elif method_dcorr == 'full_img_hog':
            b = im_orig.shape[0]
            im_orig_r = rearrange(im_orig, 'b c h w -> (b c) 1 h w')
            im_orig_hog = self._hog(im_orig_r)
            im_orig_hog = rearrange(im_orig_hog, '(b c) c1 h w -> b (c c1) h w', b=b)
            feat = im_orig_hog.flatten(start_dim=1)
        elif method_dcorr == 'aug_hog':
            im_aug = self._transform(im_orig)
            b = im_orig.shape[0]
            im_orig_r = rearrange(im_aug, 'b c h w -> (b c) 1 h w')
            im_orig_hog = self._hog(im_orig_r)
            im_orig_hog = rearrange(im_orig_hog, '(b c) c1 h w -> b (c c1) h w', b=b)
            feat = im_orig_hog.flatten(start_dim=1)
        elif method_dcorr == 'full_img_std':
            feat = std_filter_torch(im_orig, (3, 3)).flatten(start_dim=1)
        elif method_dcorr == 'aug_std':
            feat = std_filter_torch(self._transform(im_orig), (3, 3)).flatten(start_dim=1)
        elif method_dcorr == 'full_img_augment':
            feat = self._transform(im_orig).flatten(start_dim=1)
            feat = feat / 32  # trick for FP16 training
        elif method_dcorr == 'full_img_dct':
            feat = dct.dct_2d(im_orig).flatten(start_dim=1)
        elif method_dcorr == 'patch_scatnet':
            n_p = 4
            im_patch = split_into_patches(im_orig, n_p, n_p)
            im_patch = rearrange(im_patch, 'b p c s1 s2 -> (b p) c s1 s2')
            with torch.no_grad():
                im_patch_scatnet = self._scatnet(im_patch)
            feat = rearrange(im_patch_scatnet, '(b p) c1 c2 s1 s2 -> b (p c1 c2 s1 s2)',
                             b=im_orig.shape[0])
        else:
            raise ValueError('Incorrect method')
        loss_dc = self._loss_dc(feat, z)
        return loss_dc

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        opt = self.optimizers()

        im_orig, im, _ = batch

        h, z = self.encoder(im)
        with torch.no_grad():
            h_orig, z_orig = self.target_encoder(im_orig)

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

        res_dict = {
            'train/MSE': loss_mse.item(),
            'train/STD': loss_std.item(),
            'train/DC_zz': loss_dc_zz.item(),
        }

        # distance correlation on representations
        representations = ['full_img', 'full_img_scatnet', 'full_img_augment', 'full_img_hog', 'full_img_std']

        for representation in representations:
            loss_dc = self._step_dc(im_orig, z_scaled, representation)
            res_dict[f'train/DC_{representation}'] = loss_dc.item()
            loss += loss_dc

        res_dict['train/loss'] = loss.item()
        self.log_dict(res_dict)
        self.manual_backward(loss)

        if (batch_idx + 1) % self.config['accumulate_grad_batches'] == 0:
            clip_grad = 3.0

            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip_grad)
            opt.step()
            opt.zero_grad()

            self.momentum_scheduler(self.encoder, self.target_encoder)

        lr_sch, wd_sch = self.lr_schedulers()
        lr_sch.step()
        wd_sch.step()
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        im_orig, im, _ = batch

        h, z = self.encoder(im)
        with torch.no_grad():
            h_orig, z_orig = self.target_encoder(im_orig)

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

        res_dict = {
            'val/MSE': loss_mse.item(),
            'val/STD': loss_std.item(),
            'val/DC_zz': loss_dc_zz.item(),
        }

        # distance correlation on representations
        representations = ['full_img', 'full_img_scatnet', 'full_img_augment', 'full_img_hog', 'full_img_std']

        for representation in representations:
            loss_dc = self._step_dc(im_orig, z_scaled, representation)
            res_dict[f'val/DC_{representation}'] = loss_dc.item()
            loss += loss_dc

        res_dict['val/loss'] = loss.item()
        self.log_dict(res_dict)
        return loss

    def forward(self, x: torch.Tensor):
        return self.target_encoder(x)
