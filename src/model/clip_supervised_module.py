from typing import Dict, Callable

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics.functional import accuracy
import pytorch_lightning as pl
import open_clip

from src.model import ResnetMultiProj
from src.model import CosineAnnealingWarmupRestarts
from src.loss import DistanceCorrelation, DistillKL
from src.data import DatasetSSL
from src.transform import AugTransform, ValTransform


np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.benchmark = True


class CLIPSupervisedModule(pl.LightningModule):

    """CLIP Supervised Module"""

    def __init__(self, config: Dict):
        super().__init__()

        self._config = config
        params_enc = config['encoder']

        if config['encoder_type'] == 'resnet':
            self._encoder = ResnetMultiProj(**params_enc)
        else:
            raise NotImplementedError(f"Encoder type {config['encoder_type']} not implemented")

        self._classifier = nn.Linear(self._encoder.num_features, config['dataset']['n_classes'])

        self._loss_dc = DistanceCorrelation()
        self._loss_ce = nn.CrossEntropyLoss()
        self._margin_std = self.config['std_margin']
        self.hparams.batch_size = config['batch_size']  # set start batch_size
        self.hparams.lr = eval(config['lr'])

        # CLIP feature extractor
        self.clip = open_clip.create_model(**config['clip'], device='cuda', jit=False).visual
        self.clip.eval()
        self.clip.requires_grad_(False)

        self._teacher_cls = nn.Sequential(
            nn.Linear(self.clip.output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, config['dataset']['n_classes'])
        )
        # load teacher weights
        self._teacher_cls.load_state_dict(torch.load(config['teacher_path']))
        self._teacher_cls.eval()
        self._teacher_cls.requires_grad_(False)

        self._loss_kl = DistillKL(4)

        # transform
        self.transform = self.get_transform()

    @property
    def config(self) -> Dict:
        return self._config

    @property
    def encoder(self):
        return self._encoder

    @property
    def num_features(self) -> int:
        return self._encoder.num_features

    def get_transform(self) -> Callable:

        size = self.config['dataset']['size']
        blur_kernel_size = 2 * int(.05 * size) + 1
        color = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)

        # transform
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=blur_kernel_size),
        ])
        return transform

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

        params = list(self._encoder.parameters()) + list(self._classifier.parameters())
        if opt_type == 'adam':
            opt = optim.Adam(params, lr=lr, weight_decay=wd)
        elif opt_type == 'adamw':
            opt = optim.AdamW(params, lr=lr, weight_decay=wd)
        elif opt_type == 'sgd':
            opt = optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)

        sched_type = self.config['scheduler']
        if sched_type == 'cosine':
            epochs = self.config['epochs']
            scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(self.train_dataloader()),
                                                             eta_min=1e-6)
            sched = {
                'scheduler': scheduler,
                'interval': 'step',  # The scheduler will be updated per step
                'frequency': 1,
            }

        elif sched_type == 'multistep':
            steps = self.config['milestones']
            sched = optim.lr_scheduler.MultiStepLR(opt, milestones=steps, gamma=0.1)
        if sched_type == 'warmup_cosine':
            epochs = self.config['epochs']
            warmup_epochs = self.config['warmup_epochs']
            warmup_steps = warmup_epochs * len(self.train_dataloader())
            total_steps = epochs * len(self.train_dataloader())
            scheduler = CosineAnnealingWarmupRestarts(opt, total_steps, max_lr=lr, min_lr=1e-6, warmup_steps=warmup_steps)

            sched = {
                'scheduler': scheduler,
                'interval': 'step',  # The scheduler will be updated per step
                'frequency': 1,
            }

        return [opt], [sched]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def step(self, batch, batch_idx: int, *, stage: str) -> torch.Tensor:
        im_orig, im, label = batch

        z, _ = self._encoder(im)
        z_orig, _ = self._encoder(im_orig)

        logits = self._classifier(z)
        with torch.no_grad():
            logits_orig = self._classifier(z_orig)

        if self.config['normalize_z']:
            z = F.normalize(z, dim=1)
            # z_orig = F.normalize(z_orig, dim=1)

        # trick for FP16 training, scaling doesn't affect dist corr
        z_scaled = z / 32
        # z_orig_scaled = z_orig / 32

        # MSE Loss
        # loss_mse = F.mse_loss(z, z_orig)

        # STD loss
        # std_z = z.std(dim=0)
        # std_z_orig = z_orig.std(dim=0)
        # loss_std = torch.relu(self._margin_std - std_z).mean() + torch.relu(self._margin_std - std_z_orig).mean()

        # distance correlation on z
        # loss_dc_zz = 1 - self._loss_dc(z_scaled, z_orig_scaled)

        # CLIP loss
        with torch.no_grad():
            im_orig = F.interpolate(im_orig, size=(224, 224), mode='bilinear', align_corners=False)
            clip_z_orig = self.clip(im_orig)
            logits_clip = self._teacher_cls(clip_z_orig)

        loss_dc_clip = (1 - self._loss_dc(z_scaled, clip_z_orig))

        # supervised losses
        loss_ce = self._loss_ce(logits, label)
        loss_kl = self._loss_kl(logits, logits_clip)
        # loss_ce_orig = self._loss_ce(logits_orig, label)

        loss = loss_ce + loss_kl + 10 * loss_dc_clip

        # compute accuracy
        acc_aug = accuracy(logits, label)
        acc_orig = accuracy(logits_orig, label)

        res_dict = {
            f'{stage}/loss': loss,
            # f'{stage}/MSE': loss_mse,
            # f'{stage}/STD': loss_std,
            # f'{stage}/DC_zz': loss_dc_zz,
            f'{stage}/DC_clip': loss_dc_clip,
            f'{stage}/CE': loss_ce,
            f'{stage}/KL': loss_kl,
            # f'{stage}/CE_orig': loss_ce_orig,
            f'{stage}/acc': acc_orig,
            f'{stage}/acc_aug': acc_aug,
        }
        self.log_dict(res_dict)
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, stage='train')

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, stage='val')

    def forward(self, x: torch.Tensor):
        return self._encoder(x)
