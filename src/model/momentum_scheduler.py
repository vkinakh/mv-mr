import torch.nn as nn


class MomentumScheduler(nn.Module):

    def __init__(self, start_m: float, final_m: float,
                 num_epochs: int, iterators_per_epoch: int):
        super().__init__()
        self.start_m = start_m
        self.final_m = final_m
        self._step = 0
        self.iterators_per_epoch = iterators_per_epoch
        self.num_epochs = num_epochs
        self.increment = (final_m - start_m) / (iterators_per_epoch * num_epochs * 1.25)

        self.momentum_scheduler = (self.start_m + (self.increment * i)
                                   for i in range(int(iterators_per_epoch * num_epochs * 1.25) + 1))

    def forward(self, encoder, target_encoder):
        self._step += 1
        if self._step < self.iterators_per_epoch * self.num_epochs:
            m = next(self.momentum_scheduler)
        else:
            m = self.final_m

        for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
            param_k.data.mul_(m).add_((1. - m) * param_q.detach().data)
