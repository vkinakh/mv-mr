from typing import Dict, Sequence, Tuple, Union

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import Callback
import torchmetrics
from kymatio.torch import Scattering2D
from einops import rearrange
import torch_dct as dct

from src.model import ResnetMultiProj, DeiTMultiProj, CaiTMultiProj, HOGLayer
from src.loss import DistanceCorrelation
from src.data import DatasetSSL
from src.transform import AugTransform, ValTransform
from src.utils import std_filter_torch, split_into_patches, infinite_loader


np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.benchmark = True


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


class SelfSupervisedModule(pl.LightningModule):

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
        self._scatnet, self._hog, self._transform = self.get_feature_extraction()
        self._identity = nn.Identity()
        self.hparams.batch_size = config['batch_size']  # set start batch_size
        self.hparams.lr = eval(config['lr'])
        self.online_finetuner = nn.Linear(self._encoder.num_features, config['dataset']['n_classes'])

    @property
    def encoder(self):
        return self._encoder

    @property
    def num_features(self) -> int:
        return self._encoder.num_features

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

        res_dict = {
            f'{stage}/MSE': loss_mse.item(),
            f'{stage}/STD': loss_std.item(),
            f'{stage}/DC_zz': loss_dc_zz.item(),
        }

        # distance correlation on representations
        representations = ['full_img', 'full_img_scatnet', 'full_img_augment', 'full_img_hog', 'full_img_std']

        for representation in representations:
            loss_dc = self._step_dc(im_orig, z_scaled, representation)
            res_dict[f'{stage}/DC_{representation}'] = loss_dc.item()
            loss += loss_dc

        res_dict[f'{stage}/loss'] = loss.item()
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


class OnlineFineTuner(Callback):
    def __init__(
        self,
        encoder_output_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.optimizer: torch.optim.Optimizer
        self.train_dataloader: DataLoader
        self.val_dataloader: DataLoader

        self.encoder_output_dim = encoder_output_dim
        self.num_classes = num_classes
        self.train_accuracy = torchmetrics.Accuracy()
        self.train_accuracy_top_5 = torchmetrics.Accuracy(top_k=5)
        self.val_accuracy = torchmetrics.Accuracy()
        self.val_accuracy_top_5 = torchmetrics.Accuracy(top_k=5)

    @staticmethod
    def get_train_dataloader(pl_module: pl.LightningModule) -> DataLoader:
        bs = pl_module.hparams.batch_size
        n_workers = pl_module.config['n_workers']

        # dataset params
        name = pl_module.config['dataset']['name']
        size = pl_module.config['dataset']['size']
        path = pl_module.config['dataset']['path']

        trans = AugTransform(name, size)
        trans_orig = ValTransform(name, size)

        ds = DatasetSSL(dataset_name=name, trans=trans, trans_orig=trans_orig,
                        path=path, train=True, unlabeled=False)
        dl = DataLoader(ds, bs, shuffle=True, drop_last=True, num_workers=n_workers)
        return infinite_loader(dl)

    @staticmethod
    def get_val_dataloader(pl_module: pl.LightningModule) -> DataLoader:
        bs = pl_module.hparams.batch_size
        n_workers = pl_module.config['n_workers']

        # dataset params
        name = pl_module.config['dataset']['name']
        size = pl_module.config['dataset']['size']
        path = pl_module.config['dataset']['path']

        trans = AugTransform(name, size)
        trans_orig = ValTransform(name, size)

        ds = DatasetSSL(dataset_name=name, trans=trans, trans_orig=trans_orig,
                        path=path, train=False, unlabeled=False)
        dl = DataLoader(ds, bs, drop_last=True, num_workers=n_workers)
        return infinite_loader(dl)

    def on_pretrain_routine_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:

        # add linear_eval layer and optimizer

        if not pl_module.online_finetuner:
            pl_module.online_finetuner = nn.Linear(self.encoder_output_dim, self.num_classes).to(pl_module.device)
        self.optimizer = torch.optim.Adam(pl_module.online_finetuner.parameters(), lr=1e-4)
        self.train_dataloader = self.get_train_dataloader(pl_module)
        self.val_dataloader = self.get_val_dataloader(pl_module)

    def extract_online_finetuning_view(
        self, batch: Sequence, device: Union[str, torch.device]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        finetune_view, im, y = batch
        finetune_view = finetune_view.to(device)
        y = y.to(device)

        return finetune_view, y

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:

        batch = next(self.train_dataloader)
        x, y = self.extract_online_finetuning_view(batch, pl_module.device)

        with torch.no_grad():
            feats, _ = pl_module(x)

        feats = feats.detach()
        preds = pl_module.online_finetuner(feats)
        loss = F.cross_entropy(preds, y)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        acc = self.train_accuracy(F.softmax(preds, dim=1), y)
        acc_5 = self.train_accuracy_top_5(F.softmax(preds, dim=1), y)
        pl_module.log('train/acc', acc, on_step=True, on_epoch=False)
        pl_module.log('train/acc_5', acc_5, on_step=True, on_epoch=False)
        pl_module.log('train/loss_bce', loss, on_step=True, on_epoch=False)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        batch = next(self.val_dataloader)
        x, y = self.extract_online_finetuning_view(batch, pl_module.device)

        with torch.no_grad():
            feats, _ = pl_module(x)

        feats = feats.detach()
        preds = pl_module.online_finetuner(feats)
        loss = F.cross_entropy(preds, y)

        acc = self.val_accuracy(F.softmax(preds, dim=1), y)
        acc_5 = self.val_accuracy_top_5(F.softmax(preds, dim=1), y)
        pl_module.log('val/acc', acc, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log('val/acc_5', acc_5, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log('val/loss_bce', loss, on_step=False, on_epoch=True, sync_dist=True)

    def on_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.train_accuracy = self.train_accuracy.to(pl_module.device)
        self.train_accuracy_top_5 = self.train_accuracy_top_5.to(pl_module.device)
        self.val_accuracy = self.val_accuracy.to(pl_module.device)
        self.val_accuracy_top_5 = self.val_accuracy_top_5.to(pl_module.device)

        self.train_accuracy.reset()
        self.train_accuracy_top_5.reset()
        self.val_accuracy.reset()
        self.val_accuracy_top_5.reset()

    def on_epoch_end(self,
                     trainer: pl.Trainer,
                     pl_module: pl.LightningModule) -> None:
        if self.train_accuracy.mode:
            pl_module.log('train/total_acc', self.train_accuracy.compute(),
                          on_step=False, on_epoch=True, sync_dist=True)
        if self.train_accuracy_top_5.mode:
            pl_module.log('train/total_acc_5', self.train_accuracy_top_5.compute(),
                          on_step=False, on_epoch=True, sync_dist=True)
        if self.val_accuracy.mode:
            pl_module.log('val/total_acc', self.val_accuracy.compute(),
                          on_step=False, on_epoch=True, sync_dist=True)
        if self.val_accuracy_top_5.mode:
            pl_module.log('val/total_acc_5', self.val_accuracy_top_5.compute(),
                          on_step=False, on_epoch=True, sync_dist=True)
