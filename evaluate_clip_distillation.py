import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torchmetrics

from src.model import CLIPSupervisedModule
from src.data import get_dataset
from src.transform import ValTransform
from src.utils import get_config, get_device


def evaluate(args):

    path_config = args.config
    config = get_config(path_config)
    device = get_device()

    # load from checkpoint
    model = CLIPSupervisedModule.load_from_checkpoint(args.ckpt, config=config)
    model = model.to(device)
    model.eval()

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
            logits = model(batch_x)
        curr_acc = acc(logits, batch_y)
        curr_acc_top5 = acc_top5(logits, batch_y)

    print(f'Acc Top 1: {acc.compute()}')
    print(f'Acc Top 5: {acc_top5.compute()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/clip_supervised.yaml')
    parser.add_argument('--ckpt', type=str, default='logs/clip_supervised/epoch=19.ckpt')
    args = parser.parse_args()

    evaluate(args)
