# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math
from bisect import bisect_right
from functools import wraps
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
# from torch_optimizer.lr_scheduler import GradualWarmupScheduler


# class WarmupMultiStepLR():
#     def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3, warmup_epochs=500, warmup_method="linear", last_epoch=-1,):
            
#         # Define your MultiStepLR scheduler
#         scheduler_multistep = MultiStepLR(optimizer, milestones=milestones, gamma=gamma, last_epoch=last_epoch)

#         # Define your GradualWarmupScheduler
#         scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler_multistep)


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_epochs=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_epochs
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class WarmupReduceLROnPlateau(object):
    def __init__(
        self,
        optimizer,
        gamma=0.5,
        warmup_factor=1.0 / 3,
        warmup_epochs=500,
        warmup_method="linear",
        last_epoch=-1,
        patience=2,
        threshold=1e-4,
        cooldown=1,
        logger=None,
    ):
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        self.patience = patience
        self.threshold = threshold
        self.cooldown = cooldown
        self.stage_count = 0
        self.best = -1e12
        self.num_bad_epochs = 0
        self.under_cooldown = self.cooldown
        self.logger = logger

        # The following code is copied from Pytorch=1.2.0
        # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            last_epoch = 0
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch

        self.step(last_epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        warmup_factor = 1
        # during warming up
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_epochs
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        # 
        return [
            base_lr
            * warmup_factor
            * self.gamma ** self.stage_count
            for base_lr in self.base_lrs
        ]

    def step(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        # The following part is modified from ReduceLROnPlateau
        if metrics is None:
            # not conduct validation yet
            pass
        else:
            if float(metrics) > (self.best + self.threshold):
                self.best = float(metrics)
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1
            
            if self.under_cooldown > 0:
                self.under_cooldown -= 1
                self.num_bad_epochs = 0

            if self.num_bad_epochs >= self.patience:
                if self.logger is not None:
                    self.logger.info("Trigger Schedule Decay, RL has been reduced by factor {}".format(self.gamma))
                self.stage_count += 1  # this will automatically decay the learning rate
                self.under_cooldown = self.cooldown
                self.num_bad_epochs = 0


        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup followed by cosine annealing decay to eta_min.

    The de-facto standard scheduler for transformer/attention-based models
    (DETR, DAB-DETR, ViT, etc.) paired with AdamW. Deterministic, smooth,
    and requires no metric-dependent hyperparameter tuning.

        NOTE: scheduler.step() is called once per EPOCH (not per iteration).
            Set warmup_epochs to the desired number of warmup *epochs* (e.g. 3).
          T_max is taken from cfg.solver.max_epoch automatically.
    """

    def __init__(
        self,
        optimizer,
        T_max,              # total training epochs
        warmup_epochs=3,    # number of warmup epochs
        warmup_factor=0.1,  # initial LR = base_lr * warmup_factor
        eta_min=1e-7,       # floor LR at end of cosine cycle
        last_epoch=-1,
    ):
        self.T_max = T_max
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup: LR rises from base_lr*warmup_factor → base_lr.
            # Using (last_epoch + 1) ensures alpha reaches exactly 1.0 at the
            # final warmup epoch, so LR == base_lr continuously into cosine phase.
            # The old formula (last_epoch / warmup_epochs) only reached 0.8× base_lr
            # at epoch warmup_epochs-1, then cosine started at 1.0× — a ~22% spike.
            alpha = float(self.last_epoch + 1) / float(max(1, self.warmup_epochs))
            warmup_factor = self.warmup_factor * (1.0 - alpha) + alpha
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing: LR decays from base_lr → eta_min
            cosine_epochs = self.last_epoch - self.warmup_epochs
            cosine_total = max(1, self.T_max - self.warmup_epochs)
            cos_factor = 0.5 * (1.0 + math.cos(math.pi * cosine_epochs / cosine_total))
            return [
                self.eta_min + (base_lr - self.eta_min) * cos_factor
                for base_lr in self.base_lrs
            ]


class WarmupCosineAnnealingIterLR(torch.optim.lr_scheduler._LRScheduler):
    """Per-**iteration** linear warmup followed by cosine annealing.

    scheduler.step() must be called after every optimizer step (not once per
    epoch).  The math is identical to WarmupCosineAnnealingLR but the counter
    increments every iteration, giving a much smoother LR curve.

    Args:
        optimizer:     wrapped optimizer
        total_iters:   total number of training iterations (max_epoch * iters_per_epoch)
        warmup_iters:  number of warmup iterations (warmup_epochs * iters_per_epoch)
        warmup_factor: initial LR multiplier at iter 0 (e.g. 0.1 → 10% of base_lr)
        eta_min:       floor LR at the end of the cosine cycle
    """

    # Marker so the training loop can distinguish iter-based from epoch-based schedulers
    # without importing this class explicitly.
    _is_iter_based: bool = True

    def __init__(
        self,
        optimizer,
        total_iters: int,
        warmup_iters: int = 0,
        warmup_factor: float = 0.1,
        eta_min: float = 1e-7,
        last_epoch: int = -1,
    ):
        self.total_iters = total_iters
        self.warmup_iters = warmup_iters
        self.warmup_factor = warmup_factor
        self.eta_min = eta_min
        super(WarmupCosineAnnealingIterLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        it = self.last_epoch
        if it < self.warmup_iters:
            # Linear warmup: LR rises from base_lr*warmup_factor → base_lr
            alpha = float(it + 1) / float(max(1, self.warmup_iters))
            wf = self.warmup_factor * (1.0 - alpha) + alpha
            return [base_lr * wf for base_lr in self.base_lrs]
        else:
            # Cosine decay: LR falls from base_lr → eta_min
            cosine_total = max(1, self.total_iters - self.warmup_iters)
            # Clamp so LR stays at eta_min if training runs past total_iters
            cosine_iters = min(it - self.warmup_iters, cosine_total)
            cos_factor = 0.5 * (1.0 + math.cos(math.pi * cosine_iters / cosine_total))
            return [
                self.eta_min + (base_lr - self.eta_min) * cos_factor
                for base_lr in self.base_lrs
            ]