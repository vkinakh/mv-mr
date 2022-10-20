from PIL import ImageFilter, ImageOps, Image
import random

import torch
from torchvision import transforms


IMAGENET_STATS = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

CIFAR_STATS = {
    'mean': [0.491, 0.482, 0.447],
    'std': [0.247, 0.243, 0.261]
}

DATASET_STATS = {
    'stl10': IMAGENET_STATS,
    'cifar10': CIFAR_STATS,
    'cifar20': CIFAR_STATS,
    'cifar100': CIFAR_STATS,
    'imagewang': IMAGENET_STATS,
    'imagenet': IMAGENET_STATS,
    'imagenet50': IMAGENET_STATS,
    'imagenet100': IMAGENET_STATS,
    'imagenet200': IMAGENET_STATS,
    'tiny-imagenet': IMAGENET_STATS
}


class AugTransform:

    """Applies augmentation to the image"""

    def __init__(self, dataset: str, size: int, policy: str = 'custom'):

        stats = DATASET_STATS[dataset]

        if policy == 'custom':
            blur_kernel_size = 2 * int(.05 * size) + 1
            color = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)

            trans = [
                transforms.RandomResizedCrop(size=size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=blur_kernel_size)
            ]
        elif policy == 'randaugment':
            trans = [
                transforms.RandomResizedCrop(size=size),
                transforms.RandAugment(num_ops=4, magnitude=10),
            ]
        elif policy == 'autoaugment':
            trans = [
                transforms.RandomResizedCrop(size=size),
                transforms.AutoAugment()
            ]
        else:
            raise ValueError('Incorrect policy type')
        trans.extend([transforms.ToTensor(), transforms.Normalize(mean=stats['mean'], std=stats['std'])])
        self._augmentations = transforms.Compose(trans)

    def __call__(self, im) -> torch.Tensor:
        return self._augmentations(im)


class ValTransform:

    """Applied valid transform to the image"""

    def __init__(self, dataset: str, size: int):
        stats = DATASET_STATS[dataset]

        if dataset in ['imagewang', 'imagenet', 'imagenet50', 'imagenet100', 'imagenet200']:
            self._augmentations = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=stats['mean'], std=stats['std'])
            ])
        else:  # STL10, CIFAR
            self._augmentations = transforms.Compose([
                transforms.Resize(size=size),
                transforms.ToTensor(),
                transforms.Normalize(mean=stats['mean'], std=stats['std'])
            ])

    def __call__(self, im) -> torch.Tensor:
        return self._augmentations(im)


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class DataAugmentationDINO(object):
    def __init__(self, global_crop_size: int,
                 local_crop_size: int,
                 global_crops_scale: float,
                 local_crops_scale: float,
                 local_crops_number: int):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(global_crop_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(global_crop_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(local_crop_size, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops
