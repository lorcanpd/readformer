import torch
from torch.optim import Optimizer

class LAMB(Optimizer):
    def __init__(
            self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0,
            adam=True, adaptive_noise=False, use_curvature=False, noise_std=0.01,
            sharpness_aware=False, rho=0.05
    ):
        """
        LAMB optimiser with optional adaptive gradient noise and sharpness-aware minimization.
        Now uses decoupled weight decay as in AdamW.

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
            If True, scale the perturbation/noise based on curvature (second-moment statistics).
        :param noise_std:
            Standard deviation for the Gaussian noise (base noise level).
        :param sharpness_aware:
            If True, apply sharpness-aware minimization (SAM).
        :param rho:
            SAM perturbation factor, controlling the size of the worst-case scenario.
        """
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, adam=adam,
            adaptive_noise=adaptive_noise, use_curvature=use_curvature,
            noise_std=noise_std, sharpness_aware=sharpness_aware, rho=rho
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

                # State initialization
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

                # If sharpness-aware minimization (SAM) is enabled
                if group['sharpness_aware']:
                    # Step 1: Calculate the perturbation based on the gradient norm (not curvature)
                    grad_norm = grad.norm(p=2)
                    scale = group['rho'] / (grad_norm + group['eps'])
                    perturbation = grad * scale

                    # Apply the perturbation to the weights
                    p.add_(perturbation)

                    # Recompute the gradients with perturbed weights
                    if closure is not None:
                        with torch.enable_grad():
                            closure()

                    grad = p.grad.data

                    # Remove the perturbation
                    p.sub_(perturbation)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() + group['eps'])

                # Compute the ratio for LAMB
                update = exp_avg / denom

                # Apply decoupled weight decay as in AdamW
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                r1 = p.data.norm(p=2)
                r2 = update.norm(p=2)

                if r1 == 0 or r2 == 0:
                    trust_ratio = 1.0
                else:
                    trust_ratio = r1 / r2

                # Apply the update
                if group['adam']:
                    p.data.add_(update, alpha=-group['lr'])
                else:
                    p.data.add_(update, alpha=-group['lr'] * trust_ratio)

        return loss
