import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple

import os
import csv

import logging
import math

from torchmetrics import (
    Precision, Recall, F1Score, AUROC, AveragePrecision,
    CalibrationError
)

from components.better_device_handling import Module
from components.utils import get_effective_number

from tensordict import TensorDict


class MLMLoss(Module):
    """
    Loss function for Masked Language Modeling (MLM) with optional entropy regularization.
    """

    def __init__(self):
        super(MLMLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, target, scale_factor=1.0, entropy_reg=False, entropy_weight=0.1):
        """
        Compute the MLM loss with optional entropy regularization.

        :param logits:
            Predicted logits of shape (batch_size, seq_length, num_classes).
        :param target:
            True indices of shape (batch_size, seq_length).
        :param scale_factor:
            Scaling factor for the loss. Use this to make loss stable when using
            curriculum learning. Default is 1.0.
        :param entropy_reg:
            Whether to apply entropy regularization. Default is False.
        :param entropy_weight:
            Weight for the entropy regularization term. Default is 0.1.
        :returns:
            Scalar loss value.
        """
        # Flatten logits and target for loss computation
        logits_flat = logits.view(-1, logits.size(-1))
        target_flat = target.view(-1).long()

        # Compute softmax probabilities
        probs = F.softmax(logits_flat, dim=-1)

        # Gather probabilities of the true classes
        true_probs = probs.gather(dim=-1, index=target_flat.unsqueeze(-1)).squeeze(-1)

        # # get index of target_flat which == 15
        # target_flat == 15

        # Calculate the negative log-likelihood loss
        nll_loss = -torch.log(true_probs).mean() * scale_factor

        # Apply entropy regularization if specified
        if entropy_reg:
            # Calculate the entropy of the probability distribution
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
            # Add the entropy regularization term to the loss
            nll_loss = nll_loss + entropy_weight * entropy

        return nll_loss


def mlm_accuracy(logits, target):
    with torch.no_grad():
        # Flatten logits and target for loss computation
        logits_flat = logits.view(-1, logits.size(-1))
        target_flat = target.view(-1).long()
        pred_probs = F.softmax(logits_flat, dim=-1)
        pred_classes = torch.argmax(pred_probs, dim=-1)
        correct_predictions = (pred_classes == target_flat).float()
        accuracy = correct_predictions.mean().item()

        return accuracy


def calculate_perplexity(logits, targets):
    """
    Calculate perplexity given the logits and target labels.

    :param logits:
        Predicted logits of shape (batch_size, seq_length, vocab_size).
    :param targets:
        Ground truth labels of shape (batch_size, seq_length).
    :return:
        Perplexity score (scalar).
    """
    # Compute negative log-likelihood loss (cross-entropy)
    nll_loss = F.cross_entropy(logits, targets.long(), reduction='mean')

    # Calculate perplexity
    perplexity = torch.exp(nll_loss)

    return perplexity.item()


def beta_nll_loss(alpha, beta, y_true, eps=1e-9):
    """
    Compute the negative log likelihood (NLL) of the true label y_true under the
    predicted Beta distribution defined by parameters alpha and beta.
    """
    y_true = y_true.clamp(eps, 1 - eps)  # Avoid log(0)
    log_prob = torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)
    log_prob += (alpha - 1) * torch.log(y_true) + (beta - 1) * torch.log(1 - y_true)
    nll = -log_prob
    return nll.sum()


class BetaBernoulliLoss(nn.Module):
    def __init__(self, epsilon=1e-7, reduction=None):
        super(BetaBernoulliLoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, alpha, beta, labels, weighting=None):

        # Clamping to avoid issues with digamma at 0 or negative values
        alpha = alpha.clamp(min=self.epsilon)
        beta = beta.clamp(min=self.epsilon)

        term1 = torch.digamma(alpha) - torch.digamma(alpha + beta)
        term2 = torch.digamma(beta) - torch.digamma(alpha + beta)

        loss = - (labels * term1 + (1.0 - labels) * term2)

        if weighting is not None:
            # Apply weighting per-sample
            loss = loss * weighting

        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            if weighting is not None:
                # Weighted mean: divide by sum of weights rather than batch size
                total_weight = weighting.sum().clamp_min(self.epsilon)
                return loss.sum() / total_weight
            else:
                # Standard mean reduction
                return loss.mean()
        elif self.reduction is None:
            # No reduction, return raw vector
            return loss


# def mad_mask(x: torch.Tensor, k: float, top_k: int):
#     """
#     Return a boolean mask keeping entries that are either:
#     1. Outside the top_k largest values, or
#     2. Within top_k largest values but ≤ median + k·MAD
#     """
#     if x.numel() == 0:
#         return torch.zeros_like(x, dtype=torch.bool)
#
#     if top_k <= 0:
#         return torch.ones_like(x, dtype=torch.bool)
#
#     m = x.median()
#     mad = (x - m).abs().median()
#     threshold = m + k * mad
#
#     # Get indices of top_k largest values
#     _, top_indices = torch.topk(x, min(top_k, x.numel()))
#
#     # Create base mask keeping everything
#     mask = torch.ones_like(x, dtype=torch.bool)
#
#     # Only check MAD threshold for top_k elements
#     mask[top_indices] = x[top_indices] <= threshold
#
#     return mask


class BalancedPUQuantileLoss(nn.Module):
    """
    Balanced-PU risk with sliding-window τ-quantile screening
    (no MAD clipping), at most 5% of U dropped per batch, and
    a δτ gating so we only start filtering once τ⁺ − τ⁻ ≥ δτₘᵢₙ.

    Args
    ----
    pi_real       float       – true P(Y=1) in the wild (e.g. 1/10000)
    window_size   int         – how many recent batches to average histograms over
    base_loss     nn.Module   – per-sample Beta–Bernoulli loss (no reduction)
    kappa_neg     float       – lower percentile κ⁻ (keep κ⁻ of U as negatives)
    kappa_pos     float       – upper percentile κ⁺ (not used for U-filtering,
                               but needed to compute δτ = τ⁺ − τ⁻)
    delta_tau_min float       – minimum τ⁺−τ⁻ gap before any filtering
    """

    def __init__(
            self,
            pi_real: float,
            # window_size: int,
            base_loss: Optional[nn.Module] = None,
            # kappa_neg: float = 0.9998,
            # kappa_pos: float = 0.999999,
            # delta_tau_min: float = 0.1,
            # max_remove_prop: float = 0.1,
    ):
        super().__init__()
        if not (0.0 < pi_real < 1.0):
            raise ValueError("`pi_real` must be in (0,1)")
        # if not (0.0 < kappa_neg < kappa_pos < 1.0):
        #     raise ValueError("Require 0 < κ⁻ < κ⁺ < 1")

        self.pi_real = float(pi_real)
        # self.kn = float(kappa_neg)
        # self.kp = float(kappa_pos)
        # self.delta_tau_min = float(delta_tau_min)
        # self.window_size = int(window_size)
        # self.max_remove_prop = float(max_remove_prop)
        self.base = base_loss or BetaBernoulliLoss(reduction=None)

        # # sliding-window histogram buffers
        # self.register_buffer("hist_edges", torch.linspace(0.0, 1.0, 51))
        # self.hist_queue = deque(maxlen=self.window_size)

        # Balanced-PU constants (Su et al. IJCAI’21, Eq.9)
        self.c_pos = 0.5
        self.c_un = 0.5 / (1.0 - self.pi_real)

    # @torch.no_grad()
    # def _update_window(self, p_hat: torch.Tensor):
    #     # per-batch normalised histogram over [0,1]
    #     h = torch.histc(p_hat, bins=50, min=0.0, max=1.0)
    #     h = h / h.sum().clamp_min(1e-6)
    #     self.hist_queue.append(h)

    # @torch.no_grad()
    # def _taus_from_window(self) -> Tuple[float, float]:
    #     # average the window, compute CDF, find τ⁺ and τ⁻
    #     H = torch.stack(list(self.hist_queue), dim=0)  # (w,50)
    #     avg = H.mean(dim=0)  # (50,)
    #     cdf = torch.cumsum(avg, dim=0)
    #     idx_neg = (cdf >= self.kn).nonzero(as_tuple=True)[0][0]
    #     idx_pos = (cdf >= 1 - self.kp).nonzero(as_tuple=True)[0][0]
    #     tau_pos = float(self.hist_edges[idx_pos])
    #     tau_neg = float(self.hist_edges[idx_neg])
    #     return tau_pos, tau_neg

    def forward(
            self,
            alpha: torch.Tensor,
            beta: torch.Tensor,
            labels: torch.Tensor,
            weighting: Optional[torch.Tensor] = None,
            training: bool = True
    ) -> torch.Tensor:
        device = alpha.device
        if weighting is None:
            weighting = torch.ones_like(labels, dtype=alpha.dtype, device=device)

        # split P vs U
        P = labels == 1
        U = labels == 0

        w_P = weighting[P]
        w_U = weighting[U]

        # normalise weights
        w_P = w_P * (w_P.numel() / w_P.sum().clamp_min(1e-6))
        w_U = w_U * (w_U.numel() / w_U.sum().clamp_min(1e-6))

        # 1) Positive bag risks
        L_pp = self.base(
            alpha[P], beta[P],
            torch.ones_like(labels[P]), w_P
        )
        L_pn = self.base(
            alpha[P], beta[P],
            torch.zeros_like(labels[P]), w_P
        )

        R_P_pos = L_pp.mean() if L_pp.numel() else torch.tensor(0.0, device=device)
        R_P_neg = L_pn.mean() if L_pn.numel() else torch.tensor(0.0, device=device)

        raw_U_neg = self.base(
            alpha[U], beta[U],
            torch.zeros_like(labels[U]), w_U
        )

        R_U_neg = raw_U_neg.mean() if raw_U_neg.numel() else torch.tensor(0.0, device=device)

        # 3) Balanced-PU risk estimate
        inner = R_U_neg - self.pi_real * R_P_neg
        # rescue negative risk when inner < 0.0
        risk = torch.where(
            inner >= 0.0,
            self.c_pos * R_P_pos + self.c_un * inner,
            self.c_un * (-inner)
        )
        return risk


class BrierScoreMetric:
    def __init__(self, device='cuda', penalty_weight=0.0):
        self.device = device
        self.penalty_weight = penalty_weight
        self.reset()

    def update(self, p, labels, weights=None):
        """
        Updates the cumulative squared errors, optionally weighted per sample.

        :param p:
            Tensor of predicted probabilities, shape (batch_size,)
        :param labels:
            Tensor of ground truth labels (0 or 1), shape (batch_size,)
        :param weights:
            Optional Tensor of sample weights, shape (batch_size,)
        """
        p = p.to(self.device)
        labels = labels.to(self.device).float()
        squared_errors = (p - labels) ** 2

        if weights is not None:
            w = weights.to(self.device).float()
            self.sum_squared_error += (w * squared_errors).sum()
            self.sum_misclass_error += (w * ((p >= 0.5).float() != labels).float()).sum()
            self.sum_weights += w.sum()
        else:
            self.sum_squared_error += squared_errors.sum()
            self.count += labels.numel()

    def compute(self):
        """
        Computes the weighted Brier + misclassification score, scaled to [0,1].
        """
        if self.sum_weights > 0:
            total = self.sum_weights
            brier = self.sum_squared_error / (total + 1e-12)
            misrate = self.sum_misclass_error / (total + 1e-12)
            # convex combination: (1-λ)*Brier + λ*MisRate
            return ((1 - self.penalty_weight) * brier
                    + self.penalty_weight * misrate).item()
        else:
            # unweighted: pure Brier in [0,1]
            return (self.sum_squared_error / (self.count + 1e-12)).item()

    def reset(self):
        """
        Resets accumulators.
        """
        self.sum_squared_error = torch.tensor(0.0, device=self.device)
        self.sum_misclass_error = torch.tensor(0.0, device=self.device)
        self.count = 0
        self.sum_weights = torch.tensor(0.0, device=self.device)


class FineTuningMetrics:
    def __init__(
            self, thresholds=[0.5], num_classes=1, device='cuda',
            store_predictions=False, alpha_prior=1.0, beta_prior=1.0
    ):
        """
        Initialise metrics. Does not open CSV yet.
        Use supply_phase() to specify phase details and open CSV if needed.
        """
        self.device = device
        self.thresholds = thresholds
        self.num_thresholds = len(thresholds)
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior

        task = 'binary' if num_classes == 1 else 'multiclass'

        self.precision = {
            th: Precision(num_classes=num_classes, task=task).to(self.device)
            for th in thresholds
        }
        self.recall = {
            th: Recall(num_classes=num_classes, task=task).to(self.device)
            for th in thresholds
        }
        self.f1 = {
            th: F1Score(num_classes=num_classes, task=task).to(self.device)
            for th in thresholds
        }

        self.roc_auc = AUROC(task=task).to(self.device)
        self.pr_auc = AveragePrecision(task=task).to(self.device)
        self.brier = BrierScoreMetric(device=self.device)
        self.brier_prior = BrierScoreMetric(device=self.device)
        self.calibration_error = CalibrationError(task=task).to(self.device)
        self.calibration_error_prior = CalibrationError(task=task).to(self.device)

        self.store_predictions = store_predictions
        self.csv_writer = None
        self.csv_file = None
        self.output_dir = None
        self.fold = None
        self.phase = None
        self.epoch = None

        self.reset()

    def supply_phase(self, fold, phase, output_dir):
        """
        Supply the phase details before starting validation for that phase.
        If store_predictions is True, open the CSV file here.
        """
        self.fold = fold
        self.phase = phase
        self.output_dir = output_dir

        if self.store_predictions:
            os.makedirs(self.output_dir, exist_ok=True)
            filename = f"fold_{self.fold}_phase_{self.phase:03d}_predictions.csv"
            output_path = os.path.join(self.output_dir, filename)
            self.csv_file = open(output_path, mode='w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                'chr', 'pos', 'ref', 'alt', 'mapped_to_reverse', 'read_id',
                'alpha', 'beta', 'label'
            ])

    def update(
            self, alpha, beta, labels, chr_=None, pos=None, ref=None, alt=None,
            mapped_to_reverse=None, read_id=None
    ):
        alpha = alpha.to(self.device)
        beta = beta.to(self.device)
        labels = labels.to(self.device)
        p = alpha / (alpha + beta + 1e-12)

        # class-imbalance weights reflecting deployment prevalence
        pi_plus = self.alpha_prior / (self.alpha_prior + self.beta_prior)
        batch_size = labels.numel()
        num_pos = labels.sum().float()
        num_neg = batch_size - num_pos
        w_pos = (pi_plus / num_pos) if num_pos > 0 else 0.0
        w_neg = ((1 - pi_plus) / num_neg) if num_neg > 0 else 0.0
        weights = torch.where(labels == 1, w_pos, w_neg)

        self.roc_auc.update(p, labels)
        self.pr_auc.update(p, labels)
        self.brier.update(p, labels)
        self.brier_prior.update(p, labels, weights=weights)
        self.calibration_error.update(p, labels)
        self.calibration_error_prior.update(p, labels)

        for th in self.thresholds:
            preds = (p >= th).float()
            self.precision[th].update(preds, labels)
            self.recall[th].update(preds, labels)
            self.f1[th].update(preds, labels)

        if self.store_predictions and self.csv_writer is not None:
            batch_size = alpha.shape[0]
            alpha_np = alpha.detach().cpu().numpy()
            beta_np = beta.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()

            # Handle missing info by filling with defaults
            chr_ = chr_ if chr_ is not None else ['NA'] * batch_size
            pos = pos if pos is not None else torch.zeros(batch_size)
            if torch.is_tensor(pos):
                pos = pos.detach().cpu().numpy()
            # If pos is nested numpy array, flatten it
            pos = [item for sublist in pos for item in sublist]
            ref = ref if ref is not None else ['N'] * batch_size
            alt = alt if alt is not None else ['N'] * batch_size
            mapped_to_reverse = mapped_to_reverse if mapped_to_reverse is not None else ['NA'] * batch_size
            read_id = read_id if read_id is not None else ['NA'] * batch_size

            for i in range(batch_size):
                self.csv_writer.writerow([
                    chr_[i], pos[i], ref[i], alt[i], mapped_to_reverse[i],
                    read_id[i], alpha_np[i], beta_np[i], labels_np[i]
                ])

    def compute(self):
        metrics = {}
        for th in self.thresholds:
            metrics[f'Precision@{th}'] = self.precision[th].compute().item()
            metrics[f'Recall@{th}'] = self.recall[th].compute().item()
            metrics[f'F1-Score@{th}'] = self.f1[th].compute().item()

        metrics['ROC AUC'] = self.roc_auc.compute().item()
        metrics['PR AUC'] = self.pr_auc.compute().item()
        metrics['Brier Score'] = self.brier.compute()
        metrics['Calibration Error (ECE)'] = self.calibration_error.compute().item()
        metrics['Brier Score (With prior)'] = self.brier_prior.compute()
        metrics['Calibration Error (With prior)'] = self.calibration_error_prior.compute().item()

        return metrics

    def reset(self):
        # Reset metrics
        for th in self.thresholds:
            self.precision[th].reset()
            self.recall[th].reset()
            self.f1[th].reset()
        self.roc_auc.reset()
        self.pr_auc.reset()
        self.brier.reset()
        self.brier_prior.reset()
        self.calibration_error.reset()
        self.calibration_error_prior.reset()

        # If a CSV file is open, close it.
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None


def compute_load_balance_loss(
        gate_alpha_scores, gate_beta_scores, weights=None, eps=1e-7
):
    """
    Compute an entropy-based load-balancing loss to ensure even usage of experts.
    Higher entropy of expert usage distribution => more balanced use of experts.

    :param gate_alpha_scores: (batch_size, num_experts)
    :param gate_beta_scores: (batch_size, num_experts)
    :param weights: (batch_size,) or None
        A tensor of weights to handle class imbalance. If provided,
        this will reweight the contribution of each sample to the
        average usage.
    :param eps: small number for numerical stability

    :return: load_balance_loss (scalar)
    """

    # If weights are provided, compute weighted averages
    if weights is not None:
        # Ensure weights is a float tensor
        weights = weights.to(gate_alpha_scores.dtype)
        total_weight = weights.sum() + eps

        # Weighted expert usage
        alpha_expert_usage = (gate_alpha_scores * weights.unsqueeze(-1)).sum(dim=0) / total_weight
        beta_expert_usage = (gate_beta_scores * weights.unsqueeze(-1)).sum(dim=0) / total_weight
    else:
        # Unweighted mean
        alpha_expert_usage = gate_alpha_scores.mean(dim=0)  # (num_experts,)
        beta_expert_usage = gate_beta_scores.mean(dim=0)  # (num_experts,)

    # Define entropy function
    def entropy(p):
        return -(p * (p + eps).log()).sum()

    # We want to maximize entropy, so we minimize negative entropy
    load_balance_loss = - (entropy(alpha_expert_usage) + entropy(beta_expert_usage))
    return load_balance_loss


class ValidationWriter:
    """
    A standalone class to handle writing validation results to a CSV file.

    Attributes:
        fold (int): The current fold number.
        phase (int): The current validation phase.
        output_dir (str): Directory where the CSV file will be saved.
        csv_file (file object): The opened CSV file.
        csv_writer (csv.writer): The CSV writer object.
    """

    def __init__(self, fold, phase, output_dir):
        """
        Initialises the ValidationWriter by setting up the CSV file.

        Args:
            fold (int): The current fold number.
            phase (int): The current validation phase.
            output_dir (str): Directory where the CSV file will be saved.
        """
        self.fold = fold
        self.phase = phase
        self.output_dir = output_dir
        self.csv_file = None
        self.csv_writer = None

        # Initialise the CSV file
        self._initialise_csv()

    def _initialise_csv(self):
        """
        Creates the output directory and initialises the CSV file with headers.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        filename = f"fold_{self.fold}_phase_{self.phase:03d}_predictions.csv"
        output_path = os.path.join(self.output_dir, filename)
        self.csv_file = open(output_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'chr', 'pos', 'ref', 'alt', 'mapped_to_reverse', 'read_id',
            'alpha', 'beta', 'label'
        ])

    def write(
            self, alpha, beta, labels,
            chr_=None, pos=None, ref=None, alt=None,
            mapped_to_reverse=None, read_id=None
    ):
        """
        Writes a batch of validation results to the CSV file.

        Args:
            alpha (torch.Tensor): Tensor of alpha parameters.
            beta (torch.Tensor): Tensor of beta parameters.
            labels (torch.Tensor): Tensor of true labels.
            chr_ (list or None): List of chromosome identifiers.
            pos (torch.Tensor or list or None): Tensor or list of positions.
            ref (list or None): List of reference alleles.
            alt (list or None): List of alternate alleles.
            mapped_to_reverse (list or None): List indicating if mapped to reverse.
            read_id (list or None): List of read identifiers.
        """
        # Move tensors to CPU and convert to NumPy arrays
        alpha_np = alpha.detach().cpu().numpy()
        beta_np = beta.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        batch_size = alpha_np.shape[0]

        # Handle missing metadata by filling with default values
        chr_ = chr_ if chr_ is not None else ['NA'] * batch_size
        pos = pos if pos is not None else torch.zeros(batch_size)
        if torch.is_tensor(pos):
            pos = pos.detach().cpu().numpy()
            # Flatten pos if it's a nested array
            pos = pos.flatten().tolist()
        # pos is list of lists, flatten it
        if isinstance(pos[0], list):
            pos = [item for sublist in pos for item in sublist]

        ref = ref if ref is not None else ['N'] * batch_size
        alt = alt if alt is not None else ['N'] * batch_size
        mapped_to_reverse = mapped_to_reverse if mapped_to_reverse is not None else ['NA'] * batch_size
        read_id = read_id if read_id is not None else ['NA'] * batch_size

        # Ensure all metadata lists are of the correct length
        assert len(chr_) == batch_size, "Length of 'chr_' does not match batch size."
        assert len(pos) == batch_size, "Length of 'pos' does not match batch size."
        assert len(ref) == batch_size, "Length of 'ref' does not match batch size."
        assert len(alt) == batch_size, "Length of 'alt' does not match batch size."
        assert len(mapped_to_reverse) == batch_size, "Length of 'mapped_to_reverse' does not match batch size."
        assert len(read_id) == batch_size, "Length of 'read_id' does not match batch size."

        # Write each sample in the batch to the CSV file
        for i in range(batch_size):
            self.csv_writer.writerow([
                chr_[i],
                pos[i],
                ref[i],
                alt[i],
                mapped_to_reverse[i],
                read_id[i],
                alpha_np[i],
                beta_np[i],
                labels_np[i]
            ])

    def close(self):
        """
        Closes the CSV file if it is open.
        """
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

    def __enter__(self):
        """
        Enables usage of the class with the 'with' statement.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Ensures the CSV file is closed when exiting the 'with' block.
        """
        self.close()


def ndcg(scores: torch.Tensor, labels: torch.Tensor) -> float:
    # sort descending
    order = scores.argsort(descending=True)
    gains = labels[order].float()
    discounts = 1.0 / torch.log2(torch.arange(2, gains.size(0) + 2, device=gains.device))
    dcg = (gains * discounts).sum()
    # ideal DCG
    ideal = (gains.sort(descending=True).values * discounts).sum()
    return (dcg / ideal).item() if ideal > 0 else 0.0


def _adaptive_kde(logit_samples, eval_points, k=20, c=0.9):
    """Balloon-type KDE on R with per-point bandwidth h(x)."""
    on_mps = logit_samples.device.type == "mps"
    if on_mps:
        logit_samples = logit_samples.cpu()
        eval_points = eval_points.cpu()
    # pair-wise distances |x - x_i|
    d = (eval_points.unsqueeze(1) - logit_samples).abs()
    # kth-nearest neighbour distance for every x
    kth = d.kthvalue(min(k, d.size(1) - 1), dim=1).values
    h = c * kth.clamp_min(1e-6)

    z = (eval_points.unsqueeze(1) - logit_samples) / h.unsqueeze(1)
    kern = torch.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    # note: each row has its own bandwidth → divide by h_i
    return (kern / h.unsqueeze(1)).mean(dim=1)  # f_z(x)


def kde_overlap_adaptive(scores_pos, scores_neg,
                         grid=1024, k=20, c=0.9, device=None):
    """Continuous OVL via adaptive Gaussian KDE in logit space."""
    device = device or scores_pos.device

    # 1. logit transform
    def logit(p): return torch.logit(p.clamp(1e-9, 1 - 1e-9))

    z_pos, z_neg = logit(scores_pos), logit(scores_neg)

    # 2. evaluation grid in logit space
    z_min = torch.min(z_pos.min(), z_neg.min()) - 1.0
    z_max = torch.max(z_pos.max(), z_neg.max()) + 1.0
    zs = torch.linspace(z_min, z_max, grid, device=device)

    # 3. adaptive KDE
    f_z_pos = _adaptive_kde(z_pos, zs, k=k, c=c).to(device)
    f_z_neg = _adaptive_kde(z_neg, zs, k=k, c=c).to(device)

    # 4. back-transform densities to probability space
    ps = torch.sigmoid(zs)  # p = σ(z)
    jac = 1.0 / (ps * (1 - ps))  # |dz/dp|
    f_p_pos = f_z_pos * jac
    f_p_neg = f_z_neg * jac

    area_pos = 0.5 * ((f_p_pos[1:] + f_p_pos[:-1]) * (ps[1:] - ps[:-1])).sum()
    area_neg = 0.5 * ((f_p_neg[1:] + f_p_neg[:-1]) * (ps[1:] - ps[:-1])).sum()

    f_p_pos = f_p_pos / area_pos.clamp_min(1e-12)
    f_p_neg = f_p_neg / area_neg.clamp_min(1e-12)

    # 5. integral step size in *probability* space
    dps = ps[1:] - ps[:-1]
    # trapezoidal integration of min(f+ , f-)
    min_density = torch.minimum(f_p_pos, f_p_neg)
    ovl = torch.sum(0.5 * (min_density[1:] + min_density[:-1]) * dps)

    return ovl.item(), ps, f_p_pos, f_p_neg  # returning extras is handy


def ovl_and_bayes_error(scores, labels, pi_pos,
                        assume_logits=False, **kde_kw):
    """
    OVL (balanced) and pessimistic Bayes error using adaptive KDE.
    Extra **kde_kw are forwarded to kde_overlap_adaptive.
    """
    if assume_logits:
        scores = torch.sigmoid(scores)
    scores, labels = scores.detach().float(), labels.detach()

    pos = scores[labels.bool()]
    neg = scores[~labels.bool()]
    if pos.numel() == 0 or neg.numel() == 0:
        return float('nan'), float('nan')

    ovl_bal, ps, f_pos, f_neg = kde_overlap_adaptive(pos, neg, **kde_kw)

    pi_neg = 1.0 - pi_pos
    # Bayes error integrand in prob. space, same grid as KDE
    min_weighted = torch.minimum(pi_pos * f_pos, pi_neg * f_neg)
    # trapezoid again
    bayes = torch.sum(0.5 * (min_weighted[1:] + min_weighted[:-1]) * (ps[1:] - ps[:-1]))

    return ovl_bal, bayes.item()


@torch.no_grad()
def weighted_precision_at_k(
        scores: torch.Tensor,  # shape (N,) – higher = “more positive”
        labels: torch.Tensor,  # shape (N,) – 1 = pos, 0 = neg
        k: int | float,  # integer k  or  fraction 0<k≤1 (e.g. 0.02 = top-2 %)
        pi_pos: float,  # deployment prior P(Y=1) in the wild
):
    """
    Importance-weighted precision in the *top-k* predictions.

    A balanced validation batch (≈50/50) is re-weighted so that its
    effective class-prior matches `pi_pos`.  The estimate is then an
    unbiased prediction of the *PPV@k* you will see in production.

    Returns
    -------
    ppv_k : float   # weighted precision among the top-k by score
    """

    if scores.ndim != 1 or labels.ndim != 1:
        raise ValueError("`scores` and `labels` must be 1-D tensors.")
    if scores.numel() != labels.numel():
        raise ValueError("size mismatch between `scores` and `labels`.")
    if not (0.0 < pi_pos < 1.0):
        raise ValueError("`pi_pos` must lie strictly between 0 and 1.")

    # cast to CPU for the tiny bit of sorting work (keeps GPU free)
    scores = scores.detach().float().cpu()
    labels = labels.detach().cpu()

    n = scores.numel()
    if isinstance(k, float):  # allow a percentage
        k = int(max(1, round(k * n)))
    k = min(k, n)

    # empirical class-prior in the (balanced) validation slice
    q_pos = labels.float().mean()  # ≈ 0.5
    q_neg = 1.0 - q_pos

    # importance weights that turn validation → deployment
    w_pos = pi_pos / (q_pos + 1e-12)
    w_neg = (1.0 - pi_pos) / (q_neg + 1e-12)
    weights = torch.where(labels.bool(), w_pos, w_neg)

    # indices of the top-k scores (large → small)
    topk_idx = torch.topk(scores, k=k, largest=True, sorted=False).indices

    topk_weights = weights[topk_idx]
    topk_labels = labels[topk_idx]

    # weighted TP (numerator)  /  weighted count (denominator)
    tp_weighted = (topk_weights * topk_labels).sum()
    ppv_k = (tp_weighted / topk_weights.sum()).item()

    return ppv_k


# Unweighted precision at k which takes an input of scores, labels, and a list of ks.
def precision_at_ks(
        scores: torch.Tensor,  # shape (N,) – higher = “more positive”
        labels: torch.Tensor,  # shape (N,) – 1 = pos, 0 = neg
        ks: list[int | float],  # list of integers k or fractions 0<k≤1
):
    """
    Compute unweighted precision at multiple k values.

    Args:
        scores: Tensor of model scores.
        labels: Tensor of true labels.
        ks: List of integers or fractions representing top-k thresholds.

    Returns:
        A dictionary mapping each k to its precision value.
    """
    if not isinstance(ks, list):
        raise ValueError("`ks` must be a list of integers or floats.")

    precisions = {}
    for k in ks:
        if not (isinstance(k, int) or (isinstance(k, float) and 0 < k <= 1)):
            raise ValueError("Each k must be an integer or a fraction in (0, 1].")
        # get top_k precision
        if isinstance(k, float):
            k = int(max(1, round(k * scores.numel())))
            k = min(k, scores.numel())
        elif isinstance(k, int):
            k = min(k, scores.numel())
        topk_idx = torch.topk(scores, k=k, largest=True, sorted=False).indices
        topk_labels = labels[topk_idx]
        tp = topk_labels.sum().item()
        precision = tp / k if k > 0 else 0.0
        precisions[k] = precision

    return precisions


@torch.no_grad()
def pr_auc_at_prior(
        y_true: torch.Tensor,  # 1-D (N,)  –  1 for positives, 0 for negatives
        y_score: torch.Tensor,  # 1-D (N,)  –  higher = “more positive”
        pi: float = 1 / 10_000,  # desired class-prior in the wild
) -> float:
    """
    Compute the PR AUC at a given class prior.
    """

    # 1) empirical class prior of this *validation set*
    q_p = y_true.float().mean()  # P(Y=1 | validation)
    q_n = 1.0 - q_p

    # 2) importance weights that turn 'validation' into 'wild'
    w_pos = pi / q_p.clamp_min(1e-12)
    w_neg = (1.0 - pi) / q_n.clamp_min(1e-12)
    weights = torch.where(y_true.bool(), w_pos, w_neg)

    # 3) sort by model score, high → low
    scores, order = torch.sort(y_score, descending=True)
    y_true = y_true[order]
    weights = weights[order]

    # 4) cumulative weighted TP / FP
    # cumTP[i] = sum_{j<=i} w_pos for positives
    # cumFP[i] = sum_{j<=i} w_neg for negatives
    w_pos_masked = weights * y_true
    w_neg_masked = weights * (1.0 - y_true)
    cum_tp = torch.cumsum(w_pos_masked, dim=0)
    cum_fp = torch.cumsum(w_neg_masked, dim=0)

    # total weighted positives in the (re-weighted) data set
    tot_pos = w_pos_masked.sum().clamp_min(1e-12)

    # 5) precision & recall at each threshold
    precision = cum_tp / (cum_tp + cum_fp + 1e-12)
    recall = cum_tp / tot_pos

    # 6) prepend (recall=0, precision=1) for the usual PR-curve start
    recall = torch.cat([torch.zeros(1, device=recall.device), recall])
    precision = torch.cat([torch.ones(1, device=precision.device), precision])

    # 7) trapezoidal integration
    auc = torch.trapz(precision, recall).item()
    return auc


def run_validation(
        args, model, input_embedding, classifier, mini_val_batch, loss_fn,
        device, val_metric_obj, ref_base_embedding=None
):
    input_embedding.eval()
    model.eval()
    classifier.eval()
    if ref_base_embedding is not None: ref_base_embedding.eval()
    with torch.no_grad():
        ns = mini_val_batch['nucleotide_sequences']
        bq = mini_val_batch['base_qualities']
        ce = mini_val_batch['cigar_encoding']
        isf = mini_val_batch['is_first']
        m2r = mini_val_batch['mapped_to_reverse']
        pos = mini_val_batch['positions']
        rs = mini_val_batch['read_support']
        lbl = mini_val_batch['labels']
        ref = mini_val_batch['reference'] if not args.no_reference else None
        mutpos = mini_val_batch['mut_pos']
        model_in = input_embedding(ns, ce, bq, m2r, isf)
        out = model(model_in, pos)
        refemb = ref_base_embedding(ref).squeeze(-2) if ref is not None else None

        idx = torch.nonzero(pos == mutpos, as_tuple=True)
        # if idx[0].shape[0] != ns.size(0):
        #     keep = set(idx[0].tolist())
        #     mask = torch.tensor([i in keep for i in range(ns.size(0))], device=device)
        #     lbl, rs = lbl[mask], rs[mask]
        #     if refemb is not None:
        #         refemb = refemb[mask]
        cin = out[idx]
        if idx[0].shape[0] != ns.size(0):
            keep = set(idx[0].tolist())
            batch_idx = torch.arange(ns.size(0))
            mask = torch.tensor([i in keep for i in batch_idx], device=device)
            lbl = lbl[mask]
            rs = rs[mask]
            if refemb is not None:
                refemb = refemb[mask]
        alphas, betas = classifier(cin, refemb)
        alphas, betas = alphas.squeeze(-1), betas.squeeze(-1)
        rdw = get_effective_number(rs)

        lw = 1.0 / rdw

        # lw = lw * (lw.numel() / lw.sum().clamp_min(1e-6))

        loss = loss_fn(alphas, betas, lbl, lw, training=False)

        pt = alphas / (alphas + betas + 1e-8)

        is1 = isf.bool()[:, 0]
        is2 = ~is1
        pm = lbl == 1
        nm = lbl == 0
        fp1 = pt[pm & is1]
        fp2 = pt[pm & is2]
        fn1 = pt[nm & is1]
        fn2 = pt[nm & is2]
        dpos = (fp1.mean() - fp2.mean()).abs().item() if fp1.numel() else 0.0
        dneg = (fn1.mean() - fn2.mean()).abs().item() if fn1.numel() else 0.0
        # Balanced Accuracy
        preds = (pt > 0.5).float()
        tp = (preds * lbl).sum()
        tn = ((1 - preds) * (1 - lbl)).sum()
        fp = (preds * (1 - lbl)).sum()
        fn = ((1 - preds) * lbl).sum()
        tpr = tp / (tp + fn) if tp + fn > 0 else 0
        tnr = tn / (tn + fp) if tn + fp > 0 else 0
        balanced_acc = (tpr + tnr) / 2
        fpr = fp / (fp + tn) if fp + tn > 0 else 0
        fnr = fn / (fn + tp) if fn + tp > 0 else 0
        # get gmean - sqrt(recall * specificity)
        recall = tpr
        specificity = tnr
        gmean = torch.sqrt(recall * specificity) if (recall + specificity) > 0 else 0.0
        # get fscore sqrt(precision * recall)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        fscore = torch.sqrt(precision * recall) if (precision + recall) > 0 else 0.0

        val_metric_obj.update(
            alphas.detach(), betas.detach(), lbl.detach().to(torch.int32)
        )

        pr_with_prior = pr_auc_at_prior(
            lbl.detach().to(torch.int32),
            pt.detach(),
            pi=args.alpha_prior / args.beta_prior
        )

        # ndgc
        ndcg_score = ndcg(pt, lbl)

        # ovl and bayes error
        ovl_bal, bayes_err = ovl_and_bayes_error(
            pt, lbl,
            pi_pos=args.alpha_prior / (args.alpha_prior + args.beta_prior)
        )

        # prec_1 = weighted_precision_at_k(
        #     pt, lbl,
        #     k=1,
        #     pi_pos=args.alpha_prior / (args.alpha_prior + args.beta_prior)
        # )
        #
        # prec_2 = weighted_precision_at_k(
        #     pt, lbl,
        #     k=2,
        #     pi_pos=args.alpha_prior / (args.alpha_prior + args.beta_prior)
        # )
        #
        # prec_5 = weighted_precision_at_k(
        #     pt, lbl,
        #     k=5,
        #     pi_pos=args.alpha_prior / (args.alpha_prior + args.beta_prior)
        # )
        #
        # prec_10 = weighted_precision_at_k(
        #     pt, lbl,
        #     k=10,
        #     pi_pos=args.alpha_prior / (args.alpha_prior + args.beta_prior)
        # )
        #
        # prec_20 = weighted_precision_at_k(
        #     pt, lbl,
        #     k=20,
        #     pi_pos=args.alpha_prior / (args.alpha_prior + args.beta_prior)
        # )

        precisions = precision_at_ks(
            pt,
            lbl,
            ks=[1, 2, 5, 10, 20, 30, 50]
        )

        vm = val_metric_obj.compute()
        vm['Balanced Accuracy'] = balanced_acc
        vm['TPR'] = tpr
        vm['TNR'] = tnr
        vm['FPR'] = fpr
        vm['FNR'] = fnr
        vm['G-Mean'] = gmean
        vm['F-Score'] = fscore
        vm['PR_AUC_with_prior'] = pr_with_prior
        vm['NDCG'] = ndcg_score
        vm['Overlap Coefficient (Balanced)'] = ovl_bal
        vm['Bayes Error'] = bayes_err
        vm['Precision@1'] = precisions.get(1, 0.0)
        vm['Precision@2'] = precisions.get(2, 0.0)
        vm['Precision@5'] = precisions.get(5, 0.0)
        vm['Precision@10'] = precisions.get(10, 0.0)
        vm['Precision@20'] = precisions.get(20, 0.0)
        vm['Precision@30'] = precisions.get(30, 0.0)
        vm['Precision@50'] = precisions.get(50, 0.0)
        val_metric_obj.reset()
        return loss.item(), vm, dpos, dneg


@torch.no_grad()
def run_validation_rl(
        args,
        q_model: torch.nn.Module,
        mini_val_batch_td: TensorDict,
        lbl: torch.Tensor,
        device: torch.device,
) -> dict:
    """
    DDQN validation that returns the same metrics keys as the original run_validation:
      'ROC AUC', 'PR AUC', 'Brier Score', 'Calibration Error (ECE)',
      'Balanced Accuracy', 'TPR', 'TNR', 'FPR', 'FNR', 'G-Mean', 'F-Score',
      'PR_AUC_with_prior'
    """
    # 1) Eval mode
    q_model.eval()

    # 3) Forward pass to get Q-values
    q_vals = q_model(mini_val_batch_td)  # shape (N, 2)

    # 4) Derive a ranking score
    #    (difference between Q(pos) and Q(neg))
    scores = (q_vals[:, 1] - q_vals[:, 0])

    # 5) True labels
    n = lbl.numel()
    pos_count = lbl.sum().item()
    neg_count = n - pos_count

    # 6) ROC AUC (manual)
    sorted_scores, order = torch.sort(scores, descending=True)
    sorted_lbl = lbl[order]
    tps = torch.cumsum(sorted_lbl, dim=0)
    fps = torch.cumsum(1 - sorted_lbl, dim=0)
    tpr_curve = tps / pos_count
    fpr_curve = fps / neg_count
    roc_auc = torch.trapz(tpr_curve, fpr_curve).item()

    # 7) PR AUC (manual)
    precision_curve = tps / (tps + fps + 1e-12)
    recall_curve = tps / pos_count
    pr_auc = torch.trapz(precision_curve, recall_curve).item()

    # 8) Brier score & ECE
    probs = torch.sigmoid(scores)
    # probs = 0.5 + 0.5 * scores
    brier = torch.mean((probs - lbl) ** 2).item()

    # Brier score weighted by prior
    pi = args.alpha_prior / (args.alpha_prior + args.beta_prior)

    # 1) per‐sample squared error
    sqerr = (probs - lbl).pow(2)  # shape (N,)

    # 2) importance weights that re‐weight your balanced validation set into the wild prior
    #    here q_p = empirical P(Y=1) in your mini_val_batch
    q_p = lbl.float().mean().clamp_min(1e-12)
    q_n = 1.0 - q_p
    w_pos = pi / q_p
    w_neg = (1 - pi) / q_n
    weights = torch.where(lbl.bool(), w_pos, w_neg)  # shape (N,)

    # 3) weighted Brier
    brier_weighted = (weights * sqerr).sum() / weights.sum()

    # ECE with 10 bins
    ece = 0.0
    bins = 10
    edges = torch.linspace(0, 1, bins + 1)
    for i in range(bins):
        mask = (probs >= edges[i]) & (probs < edges[i + 1])
        if mask.any():
            p_bin = probs[mask].mean()
            y_bin = lbl[mask].mean()
            ece += (mask.float().mean() * torch.abs(p_bin - y_bin)).item()

    # 9) Confusion metrics at threshold=0
    preds = (scores > 0).float()

    tp = (preds * lbl).sum()
    tn = ((1 - preds) * (1 - lbl)).sum()
    fp = (preds * (1 - lbl)).sum()
    fn = ((1 - preds) * lbl).sum()
    tpr = tp / (tp + fn) if tp + fn > 0 else 0
    tnr = tn / (tn + fp) if tn + fp > 0 else 0
    balanced_acc = (tpr + tnr) / 2
    fpr = fp / (fp + tn) if fp + tn > 0 else 0
    fnr = fn / (fn + tp) if fn + tp > 0 else 0
    # get gmean - sqrt(recall * specificity)
    recall = tpr
    specificity = tnr
    gmean = torch.sqrt(recall * specificity) if (recall + specificity) > 0 else 0.0
    # get fscore sqrt(precision * recall)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fscore = torch.sqrt(precision * recall) if (precision + recall) > 0 else 0.0

    # 10) Importance-weighted PR AUC at deployment prior
    pr_prior = pr_auc_at_prior(
        lbl,
        scores,
        pi=pi
    )

    # ndgc
    ndcg_score = ndcg(scores, lbl)

    ovl_bal, bayes_err = ovl_and_bayes_error(
        scores, lbl,
        pi_pos=args.alpha_prior / (args.alpha_prior + args.beta_prior),
        assume_logits=True
    )

    # prec_1 = weighted_precision_at_k(
    #     scores, lbl,
    #     k=1,
    #     pi_pos=1.0
    # )
    #
    # prec_2 = weighted_precision_at_k(
    #     scores, lbl,
    #     k=2,
    #     pi_pos=1.0
    # )
    #
    # prec_5 = weighted_precision_at_k(
    #     scores, lbl,
    #     k=5,
    #     pi_pos=1.0
    # )
    #
    # prec_10 = weighted_precision_at_k(
    #     scores, lbl,
    #     k=10,
    #     pi_pos=1.0
    # )
    #
    # prec_20 = weighted_precision_at_k(
    #     scores, lbl,
    #     k=20,
    #     pi_pos=1.0
    # )

    precisions = precision_at_ks(
        scores,
        lbl,
        ks=[1, 2, 5, 10, 20, 30, 50]
    )

    # 11) Return all metrics in a dict
    return {
        "ROC AUC": roc_auc,
        "PR AUC": pr_auc,
        "Brier Score": brier,
        "Brier Score (With prior)": brier_weighted,
        "Calibration Error (ECE)": ece,
        "Balanced Accuracy": balanced_acc,
        "TPR": tpr,
        "TNR": tnr,
        "FPR": fpr,
        "FNR": fnr,
        "G-Mean": gmean,
        "F-Score": fscore,
        "PR_AUC_with_prior": pr_prior,
        "NDCG": ndcg_score,
        "Overlap Coefficient (Balanced)": ovl_bal,
        "Bayes Error": bayes_err,
        "Precision@1": precisions[1],
        "Precision@2": precisions[2],
        "Precision@5": precisions[5],
        "Precision@10": precisions[10],
        "Precision@20": precisions[20],
        "Precision@30": precisions[30],
        "Precision@50": precisions[50],
    }
