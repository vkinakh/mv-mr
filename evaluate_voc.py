from typing import Dict
from pathlib import Path
from functools import partial
import argparse

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchmetrics import AveragePrecision

from src.model import ResnetMultiProj
from src.model.voc_linear_eval_module import get_categories, encode_labels
from src.utils import get_config, get_device


def get_dl(config: Dict):
    bs = config['batch_size']
    n_workers = config['n_workers']
    stage = 'val'

    mean_val = (0.45704722, 0.43824774, 0.4061733)
    std_val = (0.23908591, 0.23509644, 0.2397309)
    stats = (mean_val, std_val)

    trans = transforms.Compose([
        transforms.Resize(330),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    path = Path('./data/VOCdevkit/VOC2007/ImageSets/Main')
    if not path.exists():
        # download dataset
        datasets.VOCDetection(root='./data', year='2007', image_set=stage, download=True)

    object_categories = get_categories(path)
    target_transform = partial(encode_labels, object_categories=object_categories)

    ds = datasets.VOCDetection(root='./data', year='2007', image_set=stage, download=False,
                               transform=trans, target_transform=target_transform)
    print(f'stage: {stage}, N: {len(ds)}')
    train = stage == 'train'
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=n_workers, pin_memory=train)
    return dl


def evaluate(args) -> None:
    # load config
    path_config = args.config
    config = get_config(path_config)
    device = get_device()

    # load checkpoint
    path_ckpt = args.ckpt
    ckpt = torch.load(path_ckpt, map_location=device)

    # load encoder
    encoder = ResnetMultiProj(**config['encoder'])
    n_features = encoder.num_features
    encoder = encoder.backbone
    encoder.to(device)
    encoder.load_state_dict(ckpt['encoder'])
    encoder.eval()

    # load classifier
    classifier = nn.Linear(n_features, config['dataset']['n_classes'])
    classifier.to(device)
    classifier.load_state_dict(ckpt['classifier'])
    classifier.eval()

    dl_val = get_dl(config)

    average_precision = AveragePrecision()
    for batch_x, batch_y in tqdm(dl_val):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        with torch.no_grad():
            pred = classifier(encoder(batch_x))
        curr_avg_prec = average_precision(pred, batch_y)

    print(f'Average precision: {average_precision.compute()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to the config file', type=str, required=True)
    parser.add_argument('--ckpt', help='Path to the checkpoint file', type=str, required=True)
    args = parser.parse_args()
    evaluate(args)
