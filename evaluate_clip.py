import argparse
from typing import Dict

from tqdm import tqdm

import torch
import torch.nn as nn
import torchmetrics
import open_clip

from src.model import EmbeddingExtractor, LinearEvaluator
from src.data import dataset_labels, get_dataset
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
    size = 224
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


def evaluate_zero_shot(args):
    config = get_config(args.config)
    device = get_device()

    clip, _, trans = open_clip.create_model_and_transforms(**config['clip'], device=device, jit=False)
    clip.eval()

    # get text labels
    dataset_name = config['dataset']['name']
    zero_shot_labels = [f'a photo of a {label}' for label in dataset_labels[dataset_name]]
    text_tokens = open_clip.tokenize(zero_shot_labels).to(device)

    path = config['dataset']['path']
    dataset = get_dataset(dataset_name, train=False, transform=trans, path=path, download=True, unlabeled=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    with torch.no_grad():
        text_feat = clip.encode_text(text_tokens)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)

    acc = torchmetrics.Accuracy().to(device)
    acc_top5 = torchmetrics.Accuracy(top_k=5).to(device)
    for batch in tqdm(dataloader):
        im_orig, label = batch
        im_orig = im_orig.to(device)
        label = label.to(device)

        with torch.no_grad():
            img_feat = clip.encode_image(im_orig)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)

        text_probs = (img_feat @ text_feat.T).softmax(dim=-1)
        curr_acc = acc(text_probs, label)
        curr_acc_top5 = acc_top5(text_probs, label)

    print(f'Acc: {curr_acc}, Acc 5: {curr_acc_top5}')


def evaluate_linear(args):
    config = get_config(args.config)
    device = get_device()

    clip = open_clip.create_model(**config['clip'], device=device, jit=False).visual
    clip.eval()

    acc1, acc5 = linear_evaluation(clip, config, emb_type='h', epochs=200)
    print(f'Acc: {acc1}, Acc 5: {acc5}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to the config file.')
    parser.add_argument('--mode', '-m', type=str, required=True, choices=['linear', 'zero_shot'])
    args = parser.parse_args()

    if args.mode == 'linear':
        evaluate_linear(args)
    elif args.mode == 'zero_shot':
        evaluate_zero_shot(args)
