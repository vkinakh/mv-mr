from typing import List
import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CosineWarmupScheduler(_LRScheduler):
    """Cosine learning rate scheduler with a warmup phase.

    This scheduler adjusts the learning rate according to a cosine annealing
    schedule after a linear warmup phase.

    Args:
        optimizer (Optimizer): The PyTorch optimizer for which the learning rate will be scheduled.
        warmup_steps (int): The number of warmup steps during which the learning rate is linearly increased.
        total_steps (int): The total number of steps for the entire training.
        eta_min (float, optional): The minimum learning rate value the scheduler can reach. Default: 0.
        last_step (int, optional): The index of the last completed step. Default: -1.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        eta_min: float = 0,
        last_step: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        super(CosineWarmupScheduler, self).__init__(optimizer, last_step)

    def get_lr(self) -> List[float]:
        """Get the current learning rate.

        Returns:
            list[float]: A list of the current learning rates for each parameter group in the optimizer.
        """
        if self._step_count < self.warmup_steps:
            warmup_factor = self._step_count / self.warmup_steps
        else:
            progress = (self._step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            warmup_factor = 0.5 * (1 + torch.cos(torch.tensor(progress * math.pi)))

        return [self.eta_min + (base_lr - self.eta_min) * warmup_factor for base_lr in self.base_lrs]
