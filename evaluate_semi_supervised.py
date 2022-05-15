import argparse
from tqdm import tqdm

import torch
import torchmetrics

from src.model import SemiSupervisedModule, ResnetMultiProj
from src.utils import get_config, get_device


def evaluate(args):

    path_config_self = args.config_self
    path_config_semi = args.config_semi
    path_model = args.path_model

    config_self = get_config(path_config_self)
    config_semi = get_config(path_config_semi)
    device = get_device()

    encoder = ResnetMultiProj(**config_self['encoder']).backbone
    encoder = encoder.to(device)
    module = SemiSupervisedModule.load_from_checkpoint(config=config_semi, encoder=encoder, checkpoint_path=path_model)
    module = module.eval()

    module = module.to(device)
    val_dl = module.val_dataloader()

    acc = torchmetrics.Accuracy().to(device)
    acc_top5 = torchmetrics.Accuracy(top_k=5).to(device)

    for batch_x, batch_y in tqdm(val_dl):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        with torch.no_grad():
            logits = module(batch_x)
        curr_acc = acc(logits, batch_y)
        curr_acc_top5 = acc_top5(logits, batch_y)

    print(f'Acc Top 1: {acc.compute()}')
    print(f'Acc Top 5: {acc_top5.compute()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_self',
                        help='Path to self-supervised config',
                        type=str)
    parser.add_argument('--config_semi',
                        help='Path to semi-supervised config',
                        type=str)
    parser.add_argument('--path_model',
                        help='Path to the semi-supervised model',
                        type=str)
    args = parser.parse_args()
    evaluate(args)
