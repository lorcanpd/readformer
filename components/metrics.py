import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import csv

import logging

from torchmetrics import (
    Precision, Recall, F1Score, AUROC, AveragePrecision,
    CalibrationError
)

from components.better_device_handling import Module

from torch.distributions import Beta

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

        loss = - (labels * term1 + (1 - labels) * term2)

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


class BrierScoreMetric:
    def __init__(self, device='cuda'):
        """
        Initialises the Brier Score metric.

        Args:
            device (str): Device to perform computations on ('cuda' or 'cpu').
        """
        self.device = device
        self.reset()

    def update(self, p, labels):
        """
        Updates the cumulative squared errors.

        :param p (Tensor):
            Predicted probabilities (batch_size,).
        :param labels (Tensor):
            Ground truth labels (batch_size,).
        """
        p = p.to(self.device)
        labels = labels.to(self.device).float()

        # Compute squared errors
        squared_errors = (p - labels) ** 2

        # Accumulate sum of squared errors and count
        self.sum_squared_error += squared_errors.sum()
        self.count += labels.numel()

    def compute(self):
        """
        Computes the Brier Score.

        Returns:
            float: The Brier Score.
        """
        if self.count == 0:
            return 0.0
        return (self.sum_squared_error / self.count).item()

    def reset(self):
        """
        Resets the metric accumulators.
        """
        self.sum_squared_error = torch.tensor(0.0, device=self.device)
        self.count = 0


class FineTuningMetrics:
    def __init__(
            self, thresholds=[0.5], num_classes=1, device='cuda',
            store_predictions=False
    ):
        """
        Initialise metrics. Does not open CSV yet.
        Use supply_phase() to specify phase details and open CSV if needed.
        """
        self.device = device
        self.thresholds = thresholds
        self.num_thresholds = len(thresholds)

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
        self.calibration_error = CalibrationError(task=task).to(self.device)

        self.store_predictions = store_predictions
        self.csv_writer = None
        self.csv_file = None
        self.output_dir = None
        self.fold = None
        self.phase = None
        self.epoch = None

        self.reset()

    def supply_phase(self, fold, phase, epoch, output_dir):
        """
        Supply the phase details before starting validation for that phase.
        If store_predictions is True, open the CSV file here.
        """
        self.fold = fold
        self.phase = phase
        self.epoch = epoch
        self.output_dir = output_dir

        if self.store_predictions:
            os.makedirs(self.output_dir, exist_ok=True)
            filename = f"fold_{self.fold}_phase_{self.phase:03d}_epoch_{self.epoch:03d}_predictions.csv"
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

        self.roc_auc.update(p, labels)
        self.pr_auc.update(p, labels)
        self.brier.update(p, labels)
        self.calibration_error.update(p, labels)

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
        self.calibration_error.reset()

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
        beta_expert_usage = gate_beta_scores.mean(dim=0)    # (num_experts,)

    # Define entropy function
    def entropy(p):
        return -(p * (p + eps).log()).sum()

    # We want to maximize entropy, so we minimize negative entropy
    load_balance_loss = - (entropy(alpha_expert_usage) + entropy(beta_expert_usage))
    return load_balance_loss


