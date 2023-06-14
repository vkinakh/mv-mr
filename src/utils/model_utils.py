import numpy as np

import torch.nn as nn


def count_trainable_parameters(model: nn.Module) -> int:
    """Counts number of trainable parameters in the mode

    Args:
        model: model to compute number of trainable parameters

    Returns:
        int: number of trainable parameters
    """

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params
