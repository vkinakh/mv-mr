from typing import Callable

import os
from glob import glob
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


class ImageNetSubset(Dataset):

    """Custom Imagenet dataset with selected classes
    Adapted from https://github.com/wvangansbeke/Unsupervised-Classification/blob/master/data/imagenet.py
    """

    def __init__(self, subset_file: str,
                 root: str,
                 split: str = 'train',
                 transform: Callable = None):
        super(ImageNetSubset, self).__init__()

        self.root = os.path.join(root, split)
        self.transform = transform
        self.split = split

        # Read the subset of classes to include (sorted)
        with open(subset_file, 'r') as f:
            result = f.read().splitlines()
        subdirs, class_names = [], []
        for line in result:
            subdir, class_name = line.split(' ', 1)
            subdirs.append(subdir)
            class_names.append(class_name)

        # Gather the files (sorted)
        imgs = []
        for i, subdir in enumerate(subdirs):
            files = sorted(glob(os.path.join(self.root, subdir, '*.JPEG')))
            for f in files:
                imgs.append((f, i))
        self.imgs = imgs
        self.classes = class_names

        # Resize
        self.resize = transforms.Resize(256)

    def get_image(self, index: int):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.resize(img)
        return img

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, index: int):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.resize(img)

        if self.transform is not None:
            img = self.transform(img)
        return img, target
