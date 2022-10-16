from typing import Tuple
import yaml
import random
import os

import numpy as np

import torch
from torch.utils.data import DataLoader
from kornia.filters import box_blur
from einops import rearrange


def get_device() -> str:
    """Returns available torch device

    Returns:
        str: available torch device
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def std_filter_torch(im: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Applies std filter to the image

    Args:
        im: input image. Should be of the shape (B, C, H, W)
        size: size of the filter

    Returns:
        torch.Tensor: std filtered image
    """

    eps = 1e-8

    im_mu = box_blur(im, size)
    im_mu2 = box_blur(im**2, size)
    im_sigma = torch.sqrt(im_mu2 - im_mu**2 + eps)
    im_sigma[torch.isnan(im_sigma)] = 0
    return im_sigma


def split_into_patches(img: torch.Tensor,
                       n_h: int,
                       n_w: int) -> torch.Tensor:
    """Splits image into patches

    Args:
        img: image to split into patches, should be of the shape (B, C, H, W)
        n_h: number of vertical patches
        n_w: number of horizontal patches

    Returns:
        torch.Tensor: image split into patches. Output shape: (B, n_h * n_w, C, H, W)
    """

    *_, h, w = img.shape

    hp = h // n_h
    wp = w // n_w
    patches = img.unfold(2, hp, wp).unfold(3, hp, wp)
    patches = rearrange(patches, 'b c c1 c2 h w -> b (c1 c2) c h w')
    return patches


def infinite_loader(data_loader: DataLoader):
    """Infinitely returns batches from the data loader.
    Useful for training GANs

    Args:
        data_loader: data loader to load from

    Yields:
        batch
    """

    while True:
        for batch in data_loader:
            yield batch


def seed_everything(seed: int) -> None:
    """Sets seed for all random generators

    Args:
        seed: seed to set
    """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f'Setting all seeds to be {seed} to reproduce...')
