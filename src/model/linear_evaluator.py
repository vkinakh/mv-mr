from typing import Tuple

from tqdm import trange
import numpy as np
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchmetrics


class LinearClassifier(nn.Module):

    """One layer linear classifier"""

    def __init__(self, n_features: int, n_classes: int):
        """
        Args:
            n_features: number of input features
            n_classes: number of output classes
        """

        super(LinearClassifier, self).__init__()
        self.model = nn.Linear(n_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class LinearEvaluator:

    """Linear evaluator for the Self-Supervised models"""

    def __init__(self,
                 n_features: int,
                 n_classes: int,
                 device: str,
                 batch_size: int):
        self._model = LinearClassifier(n_features, n_classes).to(device)
        self._scaler = StandardScaler()
        self._device = device
        self._batch_size = batch_size

    def run_evaluation(self, train_data: np.ndarray, train_labels: np.ndarray,
                       test_data: np.ndarray, test_labels: np.ndarray,
                       epochs: int):
        # standard dataset, create dataloaders
        train_data, test_data = self._standard_dataset(train_data, test_data)
        train = TensorDataset(torch.from_numpy(train_data),
                              torch.from_numpy(train_labels).type(torch.long))
        train_loader = DataLoader(train, batch_size=self._batch_size, shuffle=False)

        test = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_labels).type(torch.long))
        test_loader = DataLoader(test, batch_size=self._batch_size, shuffle=False)

        # weight_decay = 1e-4
        lr = 1e-4

        optimizer = torch.optim.Adam(self._model.parameters(), lr)
        criterion = torch.nn.CrossEntropyLoss()

        best_acc = 0
        best_acc5 = 0
        best_epoch = -1

        # train model
        for e in trange(epochs):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self._device), batch_y.to(self._device)

                optimizer.zero_grad()
                logits = self._model(batch_x)
                loss = criterion(logits, batch_y)

                loss.backward()
                optimizer.step()
            acc, acc5 = self._eval(test_loader)

            if acc > best_acc:
                best_acc = acc
                best_acc5 = acc5
                best_epoch = e

        print(f'Best epoch: {best_epoch}')
        return best_acc, best_acc5

    def _standard_dataset(self, train_data: np.ndarray,
                          test_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self._scaler.fit(train_data)
        train_data = self._scaler.transform(train_data)
        test_data = self._scaler.transform(test_data)
        return train_data, test_data

    def _eval(self, loader: DataLoader):
        acc = torchmetrics.Accuracy().to(self._device)
        acc_top5 = torchmetrics.Accuracy(top_k=5).to(self._device)

        with torch.no_grad():
            self._model.eval()

            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self._device), batch_y.to(self._device)
                logits = self._model(batch_x)

                pred = F.softmax(logits,  dim=1)
                curr_acc = acc(pred, batch_y)
                curr_acc_top5 = acc_top5(pred, batch_y)

            acc_final = acc.compute()
            acc_top5_final = acc_top5.compute()
            self._model.train()
            return acc_final, acc_top5_final
