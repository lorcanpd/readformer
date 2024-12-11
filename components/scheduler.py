import math
from torch.optim.lr_scheduler import LRScheduler

class CosineAnnealingScheduler(LRScheduler):
    """
    Multi-epoch cosine annealing scheduler with restarts.

    :param optimiser:
        The optimiser to apply the scheduler to.
    :param steps_per_epoch:
        The number of steps in each epoch.
    :param peak_pct:
        The proportion of the epoch to increase the learning rate.
    :param eta_min:
        The minimum learning rate.
    :param last_epoch:
        The index of the last epoch.
    :param max_lr:
        A list or single value specifying the maximum learning rate(s) for the param groups.
    """
    def __init__(
            self, optimiser, steps_per_epoch, peak_pct=0.3, eta_min=1e-5,
            last_epoch=-1, max_lr=None, **kwargs
    ):
        self.steps_per_epoch = steps_per_epoch
        self.peak_step = int(steps_per_epoch * peak_pct)
        self.eta_min = eta_min

        # If max_lr is given as a single value, replicate it for all param groups
        if max_lr is not None:
            if isinstance(max_lr, (float, int)):
                max_lr = [max_lr] * len(optimiser.param_groups)
        else:
            # If max_lr is not provided, default to current optimizer's initial lr
            max_lr = [group['lr'] for group in optimiser.param_groups]

        # Replace self.base_lrs with self.max_lrs for clarity
        self.max_lrs = max_lr

        super(CosineAnnealingScheduler, self).__init__(optimiser, last_epoch, **kwargs)

    def get_lr(self):
        step_in_epoch = self.last_epoch % self.steps_per_epoch
        if step_in_epoch < self.peak_step:
            # Increasing phase
            return [
                self.eta_min + (max_lr - self.eta_min) *
                (math.cos(math.pi * (1 - step_in_epoch / self.peak_step)) + 1) / 2
                for max_lr in self.max_lrs
            ]
        else:
            # Decreasing phase
            remaining_steps = self.steps_per_epoch - self.peak_step
            current_step = step_in_epoch - self.peak_step
            return [
                self.eta_min + (max_lr - self.eta_min) *
                (math.cos(math.pi * (current_step / remaining_steps)) + 1) / 2
                for max_lr in self.max_lrs
            ]
