from typing import Dict, List
import os
from pathlib import Path
from functools import partial

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchmetrics.functional import average_precision
import pytorch_lightning as pl


np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.benchmark = True


def get_categories(labels_dir):
    """
    Get the object categories

    Args:
        labels_dir: Directory that contains object specific label as .txt files
    Raises:
        FileNotFoundError: If the label directory does not exist
    Returns:
        Object categories as a list
    """

    if not os.path.isdir(labels_dir):
        raise FileNotFoundError

    else:
        categories = []

        for file in os.listdir(labels_dir):
            if file.endswith("_train.txt"):
                categories.append(file.split("_")[0])

        return categories


def encode_labels(target, object_categories: List[str]):
    """
    Encode multiple labels using 1/0 encoding

    Args:
        target: xml tree file
    Returns:
        torch tensor encoding labels as 1/0 vector
    """

    ls = target['annotation']['object']

    j = []
    if type(ls) == dict:
        if int(ls['difficult']) == 0:
            j.append(object_categories.index(ls['name']))

    else:
        for i in range(len(ls)):
            if int(ls[i]['difficult']) == 0:
                j.append(object_categories.index(ls[i]['name']))

    k = np.zeros(len(object_categories))
    k[j] = 1

    return torch.from_numpy(k)


class VocLinearEvalModule(pl.LightningModule):

    def __init__(self, encoder: nn.Module, config: Dict):
        super().__init__()

        self.config = config
        self.encoder = encoder.eval()
        self.classifier = nn.Linear(2048, self.config['num_classes'])
        self.hparams.batch_size = config['batch_size']  # set start batch_size
        self.hparams.lr = eval(config['lr'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = self.encoder(x)
        return self.classifier(h)

    def get_trans(self, stage: str):
        # VOC2012 stats
        mean_val = (0.45704722, 0.43824774, 0.4061733)
        std_val = (0.23908591, 0.23509644, 0.2397309)
        stats = (mean_val, std_val)

        if stage == 'train':
            return transforms.Compose([
                transforms.Resize((300, 300)),
                transforms.RandomChoice([
                    transforms.ColorJitter(brightness=(0.80, 1.20)),
                    transforms.RandomGrayscale(p=0.25)
                ]),
                transforms.RandomHorizontalFlip(p=0.25),
                transforms.RandomRotation(25),
                transforms.ToTensor(),
                transforms.Normalize(*stats, inplace=True)
            ])
        else:
            return transforms.Compose([
                transforms.Resize(330),
                transforms.CenterCrop(300),
                transforms.ToTensor(),
                transforms.Normalize(*stats)
            ])

    def get_dataloader(self, stage: str):
        bs = self.hparams.batch_size
        n_workers = self.config['n_workers']

        path = Path('./data/VOCdevkit/VOC2007/ImageSets/Main')
        if not path.exists():
            # download dataset
            datasets.VOCDetection(root='./data', year='2007', image_set=stage, download=True)
        object_categories = get_categories(path)
        target_transform = partial(encode_labels, object_categories=object_categories)

        ds = datasets.VOCDetection(root='./data', year='2007', image_set=stage, download=False,
                                   transform=self.get_trans(stage), target_transform=target_transform)
        print(f'stage: {stage}, N: {len(ds)}')
        train = stage == 'train'
        dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=n_workers, pin_memory=train)
        return dl

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader('train')

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader('val')

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader('val')

    def configure_optimizers(self):
        lr = self.hparams.lr
        wd = eval(self.config['wd'])
        opt = optim.Adam(self.classifier.parameters(), lr=lr, weight_decay=wd)
        return [opt]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def step(self, batch, batch_idx: int, *, stage: str) -> torch.Tensor:
        x, y = batch
        pred = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(pred, y)

        self.log_dict({
            f'{stage}/loss': loss.item(),
            f'{stage}/average_precision': average_precision(pred, y)

        })

        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, stage='train')

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, stage='val')

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, stage='val')
