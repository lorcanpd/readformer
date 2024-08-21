import torch
from torch.optim import Optimizer

class LAMB(Optimizer):
    def __init__(
            self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0,
            adam=True, adaptive_noise=False, use_curvature=False, noise_std=0.01
    ):
        """
        LAMB optimiser with optional adaptive gradient noise.

        :param params:
            Model parameters.
        :param lr:
            Learning rate.
        :param betas:
            Coefficients used for computing running averages of gradient and
            its square.
        :param eps:
            Term added to the denominator to improve numerical stability.
        :param weight_decay:
            Weight decay (L2 penalty).
        :param adam:
            If True, update with Adam, otherwise use LAMB.
        :param adaptive_noise:
            If True, add adaptive gradient noise.
        :param use_curvature:
            If True, scale the noise based on curvature (second-moment
            statistics).
        :param noise_std:
            Standard deviation for the Gaussian noise (base noise level).
        """
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, adam=adam,
            adaptive_noise=adaptive_noise, use_curvature=use_curvature,
            noise_std=noise_std
        )
        super(LAMB, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'LAMB does not support sparse gradients, consider '
                        'SparseAdam instead'
                    )

                state = self.state[p]

                # State initialisation
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if group['use_curvature']:
                        state['grad_sq_avg'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Adaptive gradient noise
                if group['adaptive_noise']:
                    noise_std = group['noise_std']

                    if group['use_curvature']:
                        # Track the running average of squared gradients to
                        # approximate curvature
                        state['grad_sq_avg'].mul_(beta2).addcmul_(
                            grad, grad, value=1 - beta2
                        )
                        curvature = state['grad_sq_avg'].sqrt() + group['eps']
                        # Scale the noise based on curvature
                        noise_std *= curvature
                    else:
                        grad_norm = grad.norm(p=2)
                        # Scale the noise by the gradient norm
                        noise_std *= grad_norm

                    noise = torch.randn_like(grad) * noise_std

                    grad.add_(noise)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() + group['eps'])

                # Compute the ratio for LAMB
                update = exp_avg / denom

                if group['weight_decay'] != 0:
                    update.add_(p.data, alpha=group['weight_decay'])

                r1 = p.data.norm(p=2)
                r2 = update.norm(p=2)

                if r1 == 0 or r2 == 0:
                    trust_ratio = 1.0
                else:
                    trust_ratio = r1 / r2

                # Apply weight decay and update with trust ratio
                if group['adam']:
                    p.data.add_(update, alpha=-group['lr'])
                else:
                    p.data.add_(update, alpha=-group['lr'] * trust_ratio)

        return loss
