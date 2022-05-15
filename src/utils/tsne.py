import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def tsne_display_tensorboard(embeddings, c_vector=None) -> np.ndarray:
    fig = plt.figure()
    if c_vector is not None:
        plt.scatter(embeddings[:, 0], embeddings[:, 1], c=c_vector)
    else:
        plt.scatter(embeddings[:, 0], embeddings[:, 1])
    plt.colorbar()
    fig.canvas.draw()

    img_plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    img_plot = img_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,)) / 255
    img_plot = cv2.flip(img_plot, 0)
    img_plot = cv2.rotate(img_plot, cv2.ROTATE_90_CLOCKWISE)
    img_plot = np.swapaxes(img_plot, 0, 2)

    return img_plot


def run_tsne(model: nn.Module, loader: DataLoader, device):

    model.eval()
    h_vector = []
    z_vector = []
    c_vector = []

    for img, c in loader:
        img = img.to(device)

        with torch.no_grad():
            h, z = model(img)
            h = F.normalize(h, dim=1)
            z = F.normalize(z, dim=1)

        h_vector.extend(h.cpu().detach().numpy())
        z_vector.extend(z.cpu().detach().numpy())
        c_vector.extend(c.cpu().detach().numpy())

    model.train()

    h_vector = np.array(h_vector)
    z_vector = np.array(z_vector)
    c_vector = np.array(c_vector)
    embeddings_h = TSNE(n_components=2, n_jobs=16).fit_transform(h_vector)
    embeddings_z = TSNE(n_components=2, n_jobs=16).fit_transform(z_vector)

    return tsne_display_tensorboard(embeddings_h, c_vector), tsne_display_tensorboard(embeddings_z, c_vector)
