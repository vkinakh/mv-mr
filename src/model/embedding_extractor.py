from typing import Tuple

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data import get_dataset
from src.transform import ValTransform


class EmbeddingExtractor:

    """Extracts embeddings from images using model"""

    def __init__(self, encoder: nn.Module,
                 dataset_name: str,
                 size: int,
                 device: str,
                 batch_size: int,
                 path: str = None,
                 embedding_type: str = 'h',
                 full_test: bool = False):
        """
        Args:
            encoder: model to compute embeddings
            dataset_name: dataset to compute embeddings
            path: path to dataset (ignored in STL10, CIFAR)
            size: input image size
            device: device to load data
            batch_size: batch size
            embedding_type: type of embeddings to compute. Choices: `h`, `z`, `concat`
            full_test: (only for Imagenet) if True, full dataset will be used for training and validation
        """

        if embedding_type not in ['h', 'z', 'concat']:
            raise ValueError('Incorrect embedding type')

        self._encoder = encoder
        self._dataset_name = dataset_name
        self._path = path
        self._size = size
        self._device = device
        self._batch_size = batch_size
        self._embedding_type = embedding_type
        self._full_test = full_test

    def get_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Computes embeddings, that will be used for downstream tasks

        Returns:
            Tuple: train features, train labels, test features, test labels
        """

        trans = ValTransform(self._dataset_name, self._size)
        ds_train = get_dataset(self._dataset_name, train=True,
                               transform=trans, path=self._path,
                               download=True, unlabeled=False)
        ds_test = get_dataset(self._dataset_name, train=False,
                              transform=trans, path=self._path,
                              download=True, unlabeled=False)

        # if imagenet, select 5% of training samples and 20% of test samples
        if self._dataset_name == 'imagenet' and not self._full_test:
            n_train = int(0.05 * len(ds_train))
            indices_train = np.random.randint(0, len(ds_train), size=n_train)
            ds_train = Subset(ds_train, indices_train)

            n_test = int(0.2 * len(ds_test))
            indices_test = np.random.randint(0, len(ds_test), size=n_test)
            ds_test = Subset(ds_test, indices_test)

        dl_train = DataLoader(ds_train, batch_size=self._batch_size, num_workers=16)
        dl_test = DataLoader(ds_test, batch_size=self._batch_size, num_workers=16)

        features_train, lbl_train = self._compute_embeddings(dl_train)
        features_test, lbl_test = self._compute_embeddings(dl_test)
        return features_train, lbl_train, features_test, lbl_test

    @torch.no_grad()
    def _compute_embeddings(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        features = []
        lbls = []

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self._device)
            lbls.extend(batch_y)

            with torch.no_grad():
                h, z = self._encoder(batch_x)

            if self._embedding_type == 'h':
                features.extend(h.cpu().detach().numpy())
            elif self._embedding_type == 'z':
                features.extend(z.cpu().detach().numpy())
            elif self._embedding_type == 'concat':
                f = torch.cat((h, z), 1)
                features.extend(f.cpu().detach().numpy())

        features = np.array(features)
        lbls = np.array(lbls)
        return features, lbls
