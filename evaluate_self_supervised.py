import argparse

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import ten_crop
import torchmetrics

from src.model import ResnetMultiProj
from src.data import get_dataset
from src.transform import ValTransform, DATASET_STATS
from src.utils import get_config, get_device


def evaluate_retrain(args):
    epochs = args.epochs
    config = get_config(args.config)
    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)

    # load train dataset
    ds_name = config['dataset']['name']
    size = config['dataset']['size']
    path = config['dataset']['path']
    n_classes = config['dataset']['n_classes']
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

    encoder = ResnetMultiProj(**config['encoder']).to(device)
    encoder.load_state_dict(ckpt['encoder'])
    encoder = encoder.backbone
    encoder.eval()

    finetuner = nn.Linear(encoder.num_features, n_classes).to(device)

    if 'online_finetuner' in ckpt.keys():
        finetuner.load_state_dict(ckpt['online_finetuner'])

    # optimizer
    opt = optim.Adam(
        finetuner.parameters(),
        lr=0.0001
    )

    best_acc = 0
    best_epoch = 0
    for i in range(epochs):
        finetuner.train()
        pbar = tqdm(train_dl)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                h = encoder(x)
            h = h.detach()
            y_hat = finetuner(h)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_description(f'Epoch: {i}. Loss: {loss.item():.3f}')

        finetuner.eval()
        acc = torchmetrics.Accuracy().to(device)
        for x, y in tqdm(val_dl):
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                h = encoder(x)
            y_hat = finetuner(h)
            acc(y_hat, y)
        curr_acc = acc.compute()
        print(f'Epoch: {i}, Acc: {curr_acc}')
        if curr_acc > best_acc:
            best_acc = curr_acc
            best_epoch = i

            torch.save(finetuner.state_dict(), f'finetuner_{ds_name}_{size}.pth')

    print(f'Best epoch: {best_epoch}, Best acc: {best_acc}')


def evaluate_finetuner(args):
    config = get_config(args.config)
    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)

    # load train dataset
    ds_name = config['dataset']['name']
    size = config['dataset']['size']
    path = config['dataset']['path']
    n_classes = config['dataset']['n_classes']

    val_trans = ValTransform(ds_name, size)
    val_ds = get_dataset(ds_name, train=False, path=path, transform=val_trans)
    batch_size = config['batch_size']
    n_workers = config['n_workers']
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    encoder = ResnetMultiProj(**config['encoder']).eval().to(device)
    encoder.load_state_dict(ckpt['encoder'])

    finetuner = nn.Linear(encoder.num_features, n_classes).to(device)
    finetuner.load_state_dict(ckpt['online_finetuner'])

    acc = torchmetrics.Accuracy().to(device)
    acc_top5 = torchmetrics.Accuracy(top_k=5).to(device)

    for (batch_x, batch_y) in tqdm(val_dl, desc='Evaluating'):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        batch_x_ten = torch.cat(ten_crop(batch_x, (size, size)))

        with torch.no_grad():
            h, _ = encoder(batch_x_ten)
            logits = finetuner(h)

        logits = logits.view(10, -1, logits.shape[-1])
        logits_avg = logits.mean(dim=0)

        preds = torch.argmax(logits, dim=-1)
        mode, _ = torch.mode(preds, dim=0)

        curr_acc = acc(logits_avg, batch_y)
        curr_acc_5 = acc_top5(logits_avg, batch_y)

    print(f'Acc Top 1: {acc.compute()}, acc Top 5: {acc_top5.compute()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c',
                        help='Path to config',
                        type=str)
    parser.add_argument('--ckpt',
                        help='Path to checkpoint',
                        type=str)
    parser.add_argument('--epochs', '-e',
                        help='Number of epochs',
                        type=int, default=100)
    parser.add_argument('--retrain',
                        action='store_true',
                        help='If true, linear classifier will be retrained')
    args = parser.parse_args()

    if args.retrain:
        evaluate_retrain(args)
    else:
        evaluate_finetuner(args)
