from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import open_clip

from src.data import DatasetSSL
from src.transform import AugTransform, ValTransform


def train_eval():
    config_clip = {
        'model_name': 'ViT-L-14',
        'pretrained': 'laion2b_s32b_b82k'
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset_name = 'cifar100'
    n_classes = 100
    size = 32
    batch_size = 256
    n_workers = 16
    n_epochs = 300

    model_clip = open_clip.create_model(**config_clip, device=device, jit=False).visual
    model_clip.eval()
    model_clip.requires_grad_(False)

    # create linear classifier
    classifier = nn.Linear(model_clip.output_dim, n_classes).to(device)

    # create 2 layer classifier
    # classifier = nn.Sequential(
    #     nn.Linear(model_clip.output_dim, 512),
    #     nn.ReLU(),
    #     nn.Linear(512, n_classes)
    # ).to(device)

    # create 3 layer classifier with batchnorm
    # classifier = nn.Sequential(
    #     nn.Linear(model_clip.output_dim, 512),
    #     nn.BatchNorm1d(512),
    #     nn.ReLU(),
    #     nn.Linear(512, 512),
    #     nn.BatchNorm1d(512),
    #     nn.ReLU(),
    #     nn.Linear(512, n_classes)
    # ).to(device)

    # create 3 layer classifier
    # classifier = nn.Sequential(
    #     nn.Linear(model_clip.output_dim, 512),
    #     nn.ReLU(),
    #     nn.Linear(512, 512),
    #     nn.ReLU(),
    #     nn.Linear(512, n_classes)
    # ).to(device)

    # optimizer
    optimizer = optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 250], gamma=0.1)

    # loss
    criterion = nn.CrossEntropyLoss()

    # dataset
    trans = ValTransform(dataset_name, size)
    trans_aug = AugTransform(dataset_name, size, policy='custom')
    dataset_train = DatasetSSL(dataset_name=dataset_name, trans=trans_aug, trans_orig=trans, train=True)
    dl_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=n_workers)

    dataset_val = DatasetSSL(dataset_name, trans=trans, trans_orig=trans, train=False, unlabeled=False)
    dl_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    best_acc = 0

    # train
    for epoch in range(n_epochs):
        for batch in tqdm(dl_train, desc=f'Train epoch {epoch + 1}'):
            im_orig, im, y = batch
            im = im.to(device)
            y = y.to(device)

            with torch.no_grad():
                im_up = F.interpolate(im, size=(224, 224), mode='bilinear', align_corners=False)
                emb = model_clip(im_up)

            logits = classifier(emb)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # validation
        acc = 0
        total = 0

        for batch in tqdm(dl_val, desc=f'Val epoch {epoch + 1}'):
            im_orig, im, y = batch
            im_orig = im_orig.to(device)

            with torch.no_grad():
                im_up = F.interpolate(im_orig, size=(224, 224), mode='bilinear', align_corners=False)
                emb = model_clip(im_up)
                logits = classifier(emb)

            acc += (logits.argmax(dim=-1) == y.to(device)).sum().item()
            total += len(y)

        acc /= total
        print(f'Epoch {epoch} acc: {acc}')

        if acc > best_acc:
            print(f'Best model updated: {best_acc} -> {acc}')
            best_acc = acc
            torch.save(classifier.state_dict(), f'clip_{config_clip["model_name"]}_linear_classifier_best_{epoch}.pth')

    torch.save(classifier.state_dict(), f'clip_{config_clip["model_name"]}_linear_classifier_last.pth')


def eval_classifier():
    config_clip = {
        'model_name': 'ViT-L-14',
        'pretrained': 'laion2b_s32b_b82k'
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset_name = 'cifar100'
    n_classes = 100
    size = 32
    batch_size = 512
    n_workers = 16

    model_clip = open_clip.create_model(**config_clip, device=device, jit=False).visual
    model_clip.eval()
    model_clip.requires_grad_(False)

    # create linear classifier
    classifier = nn.Linear(model_clip.output_dim, n_classes).to(device)

    # create 2 layer classifier
    # classifier = nn.Sequential(
    #     nn.Linear(model_clip.output_dim, 512),
    #     nn.ReLU(),
    #     nn.Linear(512, n_classes)
    # ).to(device)

    # create 3 layer classifier with batchnorm
    # classifier = nn.Sequential(
    #     nn.Linear(model_clip.output_dim, 512),
    #     nn.BatchNorm1d(512),
    #     nn.ReLU(),
    #     nn.Linear(512, 512),
    #     nn.BatchNorm1d(512),
    #     nn.ReLU(),
    #     nn.Linear(512, n_classes)
    # ).to(device)

    # create 3 layer classifier
    # classifier = nn.Sequential(
    #     nn.Linear(model_clip.output_dim, 512),
    #     nn.ReLU(),
    #     nn.Linear(512, 512),
    #     nn.ReLU(),
    #     nn.Linear(512, n_classes)
    # ).to(device)

    # load weights
    classifier.load_state_dict(torch.load('./lightning_logs/cifar100_vitb16_linear_classifier_sgd/clip_ViT-B-16_classifier_best_179.pth'))

    trans = ValTransform(dataset_name, size)
    dataset_val = DatasetSSL(dataset_name, trans=trans, trans_orig=trans, train=False, unlabeled=False)
    dl_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    # run validation
    acc = 0
    total = 0

    for batch in tqdm(dl_val, desc='Val'):
        im_orig, im, y = batch
        im_orig = im_orig.to(device)

        with torch.no_grad():
            im_up = F.interpolate(im_orig, size=(224, 224), mode='bilinear', align_corners=False)
            emb = model_clip(im_up)
            logits = classifier(emb)

        acc += (logits.argmax(dim=-1) == y.to(device)).sum().item()
        total += len(y)

    acc /= total
    print(f'Acc: {acc}')


if __name__ == '__main__':
    train_eval()
    # eval_classifier()
