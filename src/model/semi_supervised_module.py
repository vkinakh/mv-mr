from typing import Dict

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

from src.data import get_dataset
from src.transform import AugTransform, ValTransform


np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.benchmark = True


class SemiSupervisedModule(pl.LightningModule):

    def __init__(self, encoder: nn.Module, config: Dict):
        super().__init__()

        self.config = config
        self.encoder = encoder
        self.loss = nn.CrossEntropyLoss()
        self.classifier = nn.Linear(config['num_features'], config['dataset']['n_classes'])

        self.hparams.batch_size = config['batch_size']  # set start batch_size
        self.hparams.lr = eval(config['lr'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))

    def get_dataloader(self, mode: str) -> DataLoader:
        bs = self.config['batch_size']
        n_workers = self.config['n_workers']

        train = mode == 'train'

        # dataset params
        name = self.config['dataset']['name']
        size = self.config['dataset']['size']
        path = self.config['dataset']['path']
        percentage = self.config['dataset']['percentage']

        if train:
            trans = AugTransform(name, size)
        else:
            trans = ValTransform(name, size)

        ds = get_dataset(name, train=train, transform=trans, path=path, download=True, unlabeled=False)
        if train:
            n = len(ds)
            indices = np.random.randint(0, n, size=int(percentage * n / 100) + 1)
            ds = Subset(ds, indices)
        dl = DataLoader(ds, batch_size=bs, shuffle=train, num_workers=n_workers, drop_last=True, pin_memory=train)
        return dl

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader('train')

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader('val')

    def configure_optimizers(self):
        lr = self.hparams.lr
        wd = eval(self.config['wd'])

        opt = optim.Adam(list(self.encoder.parameters()) + list(self.classifier.parameters()),
                         lr=lr, weight_decay=wd)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(self.train_dataloader()))
        return [opt], [sched]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def step(self, batch, batch_idx: int, *, stage: str) -> torch.Tensor:
        im, lbl = batch
        pred = self(im)
        loss = self.loss(pred, lbl)

        lbl_pred = pred.argmax(dim=1)
        acc = (lbl_pred == lbl).sum().item() / im.shape[0]
        self.log_dict({f'{stage}/acc': acc, f'{stage}/loss': loss.item()})
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, stage='train')

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, stage='val')
