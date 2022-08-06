from typing import Dict
import argparse

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchmetrics

from src.model import SelfSupervisedModule, EmbeddingExtractor, LinearEvaluator
from src.utils import get_config, get_device


def linear_evaluation(encoder: nn.Module, config: Dict, emb_type: str = 'h',  epochs: int = 100) -> float:
    """Runs linear evaluation on the pretrained encoder

    Args:
        encoder: pretrained encoder
        config: configs
        emb_type: type of embeddings used for classification. Choices: `h`, `z`, `concat`
        epochs: number of epochs to train linear evaluation

    Returns:
        float: classification accuracy
    """

    encoder.eval()

    n_classes = config['dataset']['n_classes']
    batch_size = config['batch_size']
    size = config['dataset']['size']
    path = config['dataset']['path']
    name = config['dataset']['name']
    device = get_device()

    encoder = encoder.to(device)

    print('Start embedding extraction')
    extractor = EmbeddingExtractor(encoder, device=device,
                                   dataset_name=name, size=size,
                                   batch_size=config['batch_size'],
                                   path=path,
                                   embedding_type=emb_type)
    train_data, train_labels, test_data, test_labels = extractor.get_features()
    print('Finish embedding extraction')

    evaluator = LinearEvaluator(n_features=train_data.shape[1],
                                n_classes=n_classes, device=device,
                                batch_size=batch_size)
    accuracy = evaluator.run_evaluation(train_data, train_labels, test_data, test_labels, epochs)
    return accuracy


def evaluate(args):
    config = get_config(args.config)
    model = SelfSupervisedModule(config=config)

    ckpt = torch.load(args.ckpt, map_location=get_device())
    encoder = model.encoder.eval()
    encoder.load_state_dict(ckpt['encoder'])

    for emb_type in ['h']:  #, 'z', 'concat']:
        print(f'Evaluating {emb_type}')
        acc, acc_5 = linear_evaluation(encoder, config, emb_type, epochs=200)
        print(f'Emb type: {emb_type}, Acc: {acc}, Acc 5: {acc_5}')


def evaluate_finetuner(args):
    config = get_config(args.config)
    device = get_device()
    model = SelfSupervisedModule(config=config)

    ckpt = torch.load(args.ckpt, map_location=device)

    encoder = model.encoder.eval().to(device)
    encoder.load_state_dict(ckpt['encoder'])
    finetuner = model.online_finetuner.eval().to(device)
    finetuner.load_state_dict(ckpt['online_finetuner'])

    batch_size = config['batch_size']
    size = config['dataset']['size']
    path = config['dataset']['path']
    name = config['dataset']['name']
    emb_type = 'h'

    extractor = EmbeddingExtractor(encoder, device=device,
                                   dataset_name=name, size=size,
                                   batch_size=config['batch_size'],
                                   path=path,
                                   embedding_type=emb_type)
    train_data, train_labels, test_data, test_labels = extractor.get_features()

    test = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_labels).type(torch.long))
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    acc = torchmetrics.Accuracy().to(device)
    acc_top5 = torchmetrics.Accuracy(top_k=5).to(device)

    for batch_x, batch_y in tqdm(test_loader, desc='Evaluating'):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        with torch.no_grad():
            logits = finetuner(batch_x)
        pred = F.softmax(logits, dim=1)
        curr_acc = acc(pred, batch_y)
        curr_acc_top5 = acc_top5(pred, batch_y)

    print(f'Acc Top 1: {acc.compute()}')
    print(f'Acc Top 5: {acc_top5.compute()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c',
                        help='Path to config',
                        type=str)
    parser.add_argument('--ckpt',
                        help='Path to checkpoint',
                        type=str)
    parser.add_argument('--retrain',
                        action='store_true',
                        help='If true, linear classifier will be retrained')
    args = parser.parse_args()

    if args.retrain:
        evaluate(args)
    else:
        evaluate_finetuner(args)
