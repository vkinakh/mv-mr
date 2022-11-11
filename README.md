![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg)

# MV-MR: Bridging multi-views and multi-representations for self-supervised learning based on dependence maximization

This repo contains official Pytorch implementations of the paper:
**Bridging multi-views and multi-representations for self-supervised learning based on dependence maximization**


# Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Results](#results)
6. [Citation](#citation)

## Introduction

We present a new self-supervised learning method **MV-MR**, which is based on the maximization of several dependency
measures between two embeddings obtained from views with and without augmentations and multiple representations
extracted from a non-augmented view. We use an upper bound on mutual information for the estimation of dependencies
between the embeddings of the same dimensions along with distance covariance that in addition handles the dependence
estimation for the representations of different dimensions.
In contrast to the state-of-the-art self-supervising techniques, no contrastive learning with negative pairs, 
clustering, stop gradients, regularization of independence between the embedding dimensions or constraints on 
covariance matrices of embeddings, etc. are used.  MV-MR provides the state-of-the-art performance on several 
datasets such as STL10, ImageNet50, and comparable results for ImageNet-1K. MV-MR is a generic framework allowing 
for flexibly incorporating various constraints onto the structure of the embedding space via the usage of 
hand-crafted image multi-representations as regularizers. These regularizers maximize the dependence between these 
multi-representations and the targeted embedding.

## Installation

### Conda installation
```commandline
conda env create -f environment.yml
```

## Training

### Training of the self-supervised model

To run the training of the self-supervised model, first fill the **config file**. Examples of the config files: 
`configs/stl10_self_supervised.yaml`, `configs/imagenet_self_supervised.yaml`.

Then run
```commandline
python main_self_supervised.py --config <path to config file>
```

If you want to automatically select the batch size, add `--auto_bs` flag. If you want to automatically select learning 
rate, add `--auto_lr` flag.

### Training semi-supervised model

To run the training of the semi-supervised model (fine-tuning the pretrained self-supervised model), first fill the 
**config file**. Examples of the config files: `configs/stl10_semi_supervised.yaml`, 
`configs/imagenet_semi_supervised.yaml`.

Then run
```commandline
python main_semi_supervised \ 
--config <path to semi-supervised config> \ 
--path_ckpt <path to pretrained self-supervised model>
```
If you want to automatically select the batch size, add `--auto_bs` flag. If you want to automatically select learning 
rate, add `--auto_lr` flag.

### Training of the distillation of CLIP into ResNet50

To run the self-supervised distillation of CLIP into ResNet50, first fill the **config** file. Example of the 
config file: `configs/imagenet_clip_self_supervised.yaml`.

```commandline
python main_clip_self_supervised.py --config <path to the config>
```
If you want to automatically select the batch size, add `--auto_bs` flag. If you want to automatically select learning 
rate, add `--auto_lr` flag.

### Training multiclass classification model on VOC07

Multiclass classification on VOC07 is one of the ways to evaluate the pretrained self-supervised models. The idea is to 
train liner model on top of the frozen embeddings from pretrained encoder.

To run the training of multiclass classification model on VOC07 dataset, first fill the **config file**. Example of 
config file: `configs/imagenet_voc.yaml`.

Then run
```commandline
python main_voc.py  --config_voc <path to VOC config> \ 
--config_self <path to self-supervised config> \ 
--path_self <path to pretrained self-supervised model>
```

If you want to automatically select the batch size, add `--auto_bs` flag. If you want to automatically select learning 
rate, add `--auto_lr` flag.

## Evaluation

### Evaluate self-supervised model
Self-supervised model evaluation follows linear evaluation protocol: linear classifier is trained on top of frozen 
embeddings from the pretrained encoder. Script will process validation set and display Top-1 and Top-5 accuracies.

To run self-supervised model evaluation:

```commandline
python evaluate_self_supervised.py --config <path to self-supervised config> \ 
--ckpt <path to model to evaluate> \ 
--epochs <number of epochs to filetune>
```

By default it will evaluate the linear classifier (called online finetuner), that is trained alongside the self-supervised 
encoder, If you want to retrain the linear classifier from scratch, add `--retrain` flag. Retraining might take some 
time, but it generally provides higher accuracy. 

### Evaluate semi-supervised model
Semi-supervised model evaluation simply loads model and processes validation set.

To run semi-supervised model evaluation:

```commandline
python evaluate_semi_supervised.py --config <path to semi-supervised config> --ckpt <path to train semi-supervised model>
```

### Evaluate distillation
To run the evaluation of the ResNet50 distilled from CLIP, simply run self-supervised model avaluation:

```commandline
python evaluate_self_supervised.py --config <path to the config> \ 
--ckpt <path to model to evaluate> \ 
--epochs <number of epochs to filetune>
```

By default it will evaluate the linear classifier (called online finetuner), that is trained alongside the encoder.
If you want to retrain the linear classifier from scratch, add `--retrain` flag. Retraining might take some 
time, but it generally provides higher accuracy. 


### Evaluate multiclass classification on VOC07

To run multiclass classification evaluation on VOC07:

```commandline
python evaluate_voc.py --config <path to VOC config> --ckpt <path to model trained on VOC>
```

## Results

### Self-supervised models
| Dataset     | Top-1 accuracy | Top-5 accuracy | Download link |
|-------------|----------------|----------------|-------|
| STL10       | 89.67%         | 99.46%         | Coming soon |
| ImageNet-1K | 74.5%          | 92.1%          | Coming soon | 

 
### Semi-supervised models
| Dataset     | Top-1 accuracy | Top-5 accuracy | Percentage of labels | Download link |
|-------------|----------------|----------------|----------------------|--------------|
| ImageNet-1K | 56.1%          | 79.4%          | 1%                   |Coming soon    | 
| ImageNet-1K | 69.9%          | 89.5%          | 10%                  |Coming soon    | 

### Distillation
| Dataset     | Top-1 accuracy | Download link |
|-------------|----------------|----------|
| ImageNet-1K | 75.3%          |  Coming soon |
| STL10 | 95.6%| Coming soon |

## Citation
Coming soon