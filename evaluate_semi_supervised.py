import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics

from src.model import ResnetMultiProj
from src.data import get_dataset
from src.transform import ValTransform
from src.utils import get_config, get_device


def evaluate(args):
    """Evaluate Semi-Supervised model on validation set"""

    path_config = args.config
    config = get_config(path_config)
    device = get_device()

    # load checkpoint
    path_ckpt = args.ckpt
    ckpt = torch.load(path_ckpt, map_location=device)

    # load encoder
    encoder = ResnetMultiProj(**config['encoder']).backbone
    encoder = encoder.to(device)
    encoder.load_state_dict(ckpt['encoder'])
    encoder.eval()

    # load classifier
    classifier = nn.Linear(2048, config['dataset']['n_classes'])
    classifier = classifier.to(device)
    classifier.load_state_dict(ckpt['classifier'])
    classifier.eval()

    # get dataset and dataloader
    ds_name = config['dataset']['name']
    ds_path = config['dataset']['path']
    img_size = config['dataset']['size']

    trans = ValTransform(ds_name, img_size)
    ds = get_dataset(ds_name, train=False, transform=trans, path=ds_path)
    dl = DataLoader(ds, batch_size=config['batch_size'], shuffle=False, num_workers=config['n_workers'])

    acc = torchmetrics.Accuracy().to(device)
    acc_top5 = torchmetrics.Accuracy(top_k=5).to(device)

    for batch_x, batch_y in tqdm(dl):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        with torch.no_grad():
            logits = classifier(encoder(batch_x))
        curr_acc = acc(logits, batch_y)
        curr_acc_top5 = acc_top5(logits, batch_y)

    print(f'Acc Top 1: {acc.compute()}')
    print(f'Acc Top 5: {acc_top5.compute()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='Path to the config',
                        required=True, type=str)
    parser.add_argument('--ckpt',
                        help='Path to the checkpoint',
                        required=True, type=str)
    args = parser.parse_args()
    evaluate(args)
