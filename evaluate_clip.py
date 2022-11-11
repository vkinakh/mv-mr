import argparse
from pathlib import Path

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torchmetrics
import open_clip

from src.data import get_dataset
from src.transform import ValTransform, DATASET_STATS
from src.utils import get_config, get_device


def evaluate_linear(args):
    out_path = Path(args.out_path)
    out_path.mkdir(exist_ok=True, parents=True)

    epochs = args.epochs
    config = get_config(args.config)
    device = get_device()

    clip = open_clip.create_model(**config['clip'], device=device, jit=False).visual
    clip.eval()

    # classifier
    n_classes = config['dataset']['n_classes']
    classifier = nn.Sequential(nn.Linear(clip.output_dim, n_classes)).to(device)

    # load train dataset
    ds_name = config['dataset']['name']
    size = config['dataset']['size']
    path = config['dataset']['path']
    ds_stats = DATASET_STATS[ds_name]

    train_trans = transforms.Compose([
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(ds_stats['mean'], ds_stats['std'])
    ])
    val_trans = ValTransform(ds_name, size)

    train_ds = get_dataset(ds_name, train=True, path=path, transform=train_trans)
    val_ds = get_dataset(ds_name, train=False, path=path, transform=val_trans)

    batch_size = config['batch_size']
    n_workers = config['n_workers']
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    # optimizer
    opt = optim.Adam(classifier.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=0)

    best_acc = 0
    best_epoch = 0
    for i in range(epochs):
        classifier.train()
        pbar = tqdm(train_dl)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                h = clip(x)
            h = h.detach()
            y_hat = classifier(h)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_description(f'Epoch: {i}. Loss: {loss.item():.3f}')
        scheduler.step()
        acc = torchmetrics.Accuracy().to(device)
        for x, y in tqdm(val_dl):
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                h = clip(x)
            y_hat = classifier(h)
            acc(y_hat, y)
        curr_acc = acc.compute()
        print(f'Epoch: {i}, Acc: {curr_acc}')
        if curr_acc > best_acc:
            best_acc = curr_acc
            best_epoch = i
            print(f'Best acc: {best_acc} at epoch {best_epoch}')

            torch.save({'classifier': classifier.state_dict()}, f'./{args.out_path}/{ds_name}_clip_classifier.pth')

    print(f'Best acc: {best_acc} at epoch {best_epoch}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to the config file.')
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--out_path', '-o', type=str, default='.')
    args = parser.parse_args()

    evaluate_linear(args)
