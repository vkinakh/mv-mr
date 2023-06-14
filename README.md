![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mv-mr-multi-views-and-multi-representations/self-supervised-learning-on-stl-10)](https://paperswithcode.com/sota/self-supervised-learning-on-stl-10?p=mv-mr-multi-views-and-multi-representations)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mv-mr-multi-views-and-multi-representations/unsupervised-image-classification-on-stl-10)](https://paperswithcode.com/sota/unsupervised-image-classification-on-stl-10?p=mv-mr-multi-views-and-multi-representations)

# MV-MR: multi-views and multi-representations for self-supervised learning and knowledge distillation

This repo contains official Pytorch implementations of the paper:
**MV-MR: multi-views and multi-representations for self-supervised learning and knowledge distillation**

[Paper](https://arxiv.org/abs/2303.12130v1)


# Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Results](#results)
6. [Citation](#citation)

## Introduction

We present a new method of self-supervised learning and knowledge distillation based on the multi-views and 
multi-representations (MV-MR). The MV-MR is based on the maximization of dependence between learnable embeddings from 
augmented and non-augmented views, jointly with the maximization of dependence between learnable embeddings from 
augmented view and multiple non-learnable representations from non-augmented view. We show that the proposed method 
can be used for efficient self-supervised classification and model-agnostic knowledge distillation. 
Unlike other self-supervised techniques, our approach does not use any contrastive learning, clustering, 
or stop gradients. MV-MR is a generic framework allowing the incorporation of constraints on the learnable embeddings 
via the usage of image multi-representations as regularizers. Along this line, knowledge distillation is considered as 
a particular case of such a regularization. MV-MR provides the state-of-the-art performance on the STL10, CIFAR20 and 
ImageNet-1K datasets among non-contrastive and clustering-free methods. We show that a lower complexity ResNet50 model 
pretrained using proposed knowledge distillation based on the CLIP ViT model achieves state-of-the-art performance on STL10 linear evaluation

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

## Convert pytorch-lightning weight to pytorch format
See `scripts/` folder.

## Results

### Self-supervised models
| Dataset     | Top-1 accuracy | Top-5 accuracy | Download link                                                                                    |
|-------------|-------------|----------------|--------------------------------------------------------------------------------------------------|
| STL10       | 89.67%      | 99.46%         | [Download](https://drive.google.com/drive/folders/1ljf0ZHDZSTsB-4aKTCrYIoyFS9gANSQR?usp=sharing) |
| ImageNet-1K | 74.5%       | 92.1%          | [Download](https://drive.google.com/drive/folders/1Ck0MzmXsu--m8vzNRRD2oW3Qi9qDRYJr?usp=sharing) | 
 | CIFAR20    | 73.2%       | 95.6%          | [Download](https://drive.google.com/drive/folders/1lUhgq5ZGV0_wklWao1_0jlFueFHQd-yD?usp=sharing) |

 
### Semi-supervised models
| Dataset     | Top-1 accuracy | Top-5 accuracy | Percentage of labels | Download link                                                                                    |
|-------------|----------------|----------------|----------------------|--------------------------------------------------------------------------------------------------|
| ImageNet-1K | 56.1%          | 79.4%          | 1%                   | [Download](https://drive.google.com/drive/folders/1BugB2eAw3taII3Ug2vPI-6jVnqfw58I6?usp=sharing) | 
| ImageNet-1K | 69.9%          | 89.5%          | 10%                  | [Download](https://drive.google.com/drive/folders/1Y9s_iVVI_6o9vqNTW3v_gaDYOLbFjPyY?usp=sharing) | 


### Transfer learning on VOC
| Pretrain dataset | Finetune dataset | mAP  | Download link                                                                                    |
|------------------|------------------|------|--------------------------------------------------------------------------------------------------|
 | ImageNet-1k     | VOC2007          | 87.1 | [Download](https://drive.google.com/drive/folders/13dsE_rIu2_wJVddcdEPMHTsZIuhHQMHB?usp=sharing) |

### Distillation
| Dataset     | Top-1 accuracy | Download link                                                                                    |
|-------------|----------------|--------------------------------------------------------------------------------------------------|
| ImageNet-1K | 75.3%          | [Download](https://drive.google.com/drive/folders/1LYR_U683CT7xVP9__DjMIjR9lMaX8jhQ?usp=sharing)                                                                                     |
| STL10 | 95.6%          | [Download](https://drive.google.com/drive/folders/1lCGvOZvoJ8CLNoPLPEoQAJ76sWZtCClB?usp=sharing) |
 | CIFAR100 | 78.6%          | [Download](https://drive.google.com/drive/folders/1gBMqJ4dvVp2wiNnCo1Uzz5rXC9rqmq53?usp=sharing) | 

## Citation
```
@article{kinakh2023mv,
  title={MV-MR: multi-views and multi-representations for self-supervised learning and knowledge distillation},
  author={Kinakh, Vitaliy and Drozdova, Mariia and Voloshynovskiy, Slava},
  journal={arXiv preprint arXiv:2303.12130},
  year={2023}
}
```