from typing import Callable
from pathlib import Path

from torch.utils.data import Dataset
from torchvision import datasets

from src.data import ImageNetSubset


def get_dataset(dataset: str,
                train: bool,
                transform=None,
                path: str = None,
                download: bool = False,
                unlabeled: bool = False) -> Dataset:
    """Returns dataset

    Args:
        dataset: dataset name

        train: if True, then train split will be returned

        transform: transform to apply to images

        path: path to dataset (ignored for STL10, CIFAR)

        download: if True, then dataset will be downloaded, if not downloaded

        unlabeled: if True unlabeled split will be returned. Only for STL10

    Returns:
        Dataset: dataset

    Raises:
        ValueError: if the dataset is unsupported
    """

    if dataset == 'stl10':

        if train and unlabeled:
            split = 'train+unlabeled'
        elif train:
            split = 'train'
        elif unlabeled:
            split = 'unlabeled'
        else:
            split = 'test'

        return datasets.STL10('./data', split=split, download=download, transform=transform)
    elif dataset in ['imagenet', 'tiny-imagenet']:
        path = Path(path)
        if train:
            return datasets.ImageFolder(path / 'train', transform=transform)
        else:
            return datasets.ImageFolder(path / 'val', transform=transform)
    elif dataset in ['imagenet50', 'imagenet100', 'imagenet200']:
        n_classes = dataset.split('imagenet')[1]

        subset_file = f'./data/imagenet_subsets/imagenet_{n_classes}.txt'
        split = 'train' if train else 'val'
        return ImageNetSubset(subset_file, path, split, transform)
    else:
        raise ValueError('Unsupported dataset')


class DatasetSSL(Dataset):

    """Dataset for self-supervised learning.
    Dataset returns original image and it's augmented version
    """

    def __init__(self,
                 dataset_name: str,
                 trans: Callable,
                 trans_orig: Callable,
                 path: str = None,
                 train: bool = True,
                 unlabeled: bool = True,
                 n_aug: int = 1):
        """
        Args:
            dataset_name: name of the dataset to be loaded
            path: path to dataset
            trans: transform to apply to get augmented image
            trans_orig: transform to apply to original image
            train: if True, train split will be loaded
            unlabeled: if True, unlabeled split will be loaded
            n_aug: number of augmented images to return (default: 1).
                   If n_aug > 1, then dataset will return n_aug augmented images
                   Note: use n_aug > 1 only when shuffle=False
        """

        self._dataset = get_dataset(dataset_name, train, path=path, download=True,
                                    unlabeled=unlabeled)
        self._trans = trans
        self._trans_orig = trans_orig
        self._n_aug = n_aug

    def __len__(self) -> int:
        return len(self._dataset) * self._n_aug

    def __getitem__(self, i: int):
        idx = int(i / self._n_aug)
        im, lbl = self._dataset[idx]
        im_orig = self._trans_orig(im)
        im = self._trans(im)
        return im_orig, im, lbl
