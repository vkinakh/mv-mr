import argparse

from tqdm import tqdm

import torch
from torchmetrics import AveragePrecision

from src.model import VocLinearEvalModule, ResnetMultiProj
from src.utils import get_config, get_device


def evaluate(args) -> None:
    path_config_self = args.config_self
    path_config_voc = args.config_voc
    path_voc = args.path_voc

    config_self = get_config(path_config_self)
    config_voc = get_config(path_config_voc)
    device = get_device()

    encoder = ResnetMultiProj(**config_self['encoder']).backbone
    encoder.to(device)
    encoder.eval()
    # configure model for VOC evaluation
    module = VocLinearEvalModule.load_from_checkpoint(encoder=encoder, config=config_voc, checkpoint_path=path_voc)
    module = module.eval()
    module = module.to(device)

    dl_val = module.val_dataloader()

    average_precision = AveragePrecision()
    for batch_x, batch_y in tqdm(dl_val):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        with torch.no_grad():
            pred = module(batch_x)
        curr_avg_prec = average_precision(pred, batch_y)

    print(f'Average precision: {average_precision.compute()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_self',
                        help='Path to the config file for the self-supervised model',
                        type=str, required=True)
    parser.add_argument('--config_voc',
                        help='Path to the config file for the VOC model',
                        type=str, required=True)
    parser.add_argument('--path_voc',
                        help='Path to the checkpoint file for the VOC model',
                        type=str, required=True)
    args = parser.parse_args()
    evaluate(args)
