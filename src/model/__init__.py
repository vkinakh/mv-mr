from .resnet import ResnetMultiProj
from .resnet import resnet8x4
from .hog import HOGLayer

from .cosine_annealing_warmup_restarts_scheduler import CosineAnnealingWarmupRestarts

from .self_supervised_module import SelfSupervisedModule, OnlineFineTuner
from .semi_supervised_module import SemiSupervisedModule
from .voc_linear_eval_module import VocLinearEvalModule
from .clip_self_supervised_module import CLIPSelfSupervisedModule
