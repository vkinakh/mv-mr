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
    'imagenet200': IMAGENET_STATS
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
