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

        alpha = alpha.clamp(min=self.epsilon)
        beta = beta.clamp(min=self.epsilon)

        term1 = torch.digamma(alpha) - torch.digamma(alpha + beta)
        term2 = torch.digamma(beta) - torch.digamma(alpha + beta)

        loss = - (labels * term1 + (1 - labels) * term2)

        if weighting is not None:
            loss = loss * weighting

        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction is None:
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
        Initializes all metrics, including the custom Brier Score, for multiple
        thresholds.

        :param thresholds:
            List of classification thresholds to compute metrics for.
        :param num_classes:
            Number of classes in the classification task. Default is 1 for
            binary classification.
        :param device:
            Device to perform computations on ('cuda' or 'cpu').
        """
        self.device = device
        self.thresholds = thresholds
        self.num_thresholds = len(thresholds)
        if num_classes == 1:
            task = 'binary'
        else:
            task = 'multiclass'
        # Initialise dictionaries to hold metrics for each threshold
        self.precision = {
            th: Precision(
                num_classes=num_classes, task=task).to(self.device)
            for th in thresholds
        }
        self.recall = {
            th: Recall(
                num_classes=num_classes, task=task).to(self.device)
            for th in thresholds
        }
        self.f1 = {
            th: F1Score(
                num_classes=num_classes, task=task).to(self.device)
            for th in thresholds
        }

        # Aggregated Metrics
        self.roc_auc = AUROC(task=task).to(self.device)
        self.pr_auc = AveragePrecision(task=task).to(self.device)

        # Probabilistic Metrics
        self.brier = BrierScoreMetric(device=self.device)

        # Calibration Metrics
        self.calibration_error = CalibrationError(task=task).to(self.device)
        self.store_predictions = store_predictions
        self.reset()

    def update(
            self, alpha, beta, labels, chr=None, pos=None, ref=None, alt=None,
            mapped_to_reverse=None, read_id=None
    ):
        """
        Updates all metrics with the latest batch for all thresholds.

        :param alpha (Tensor):
            Alpha parameter of the Beta distribution (batch_size,).
        :param beta (Tensor):
            Beta parameter of the Beta distribution (batch_size,).
        :param labels (Tensor):
            Ground truth labels (batch_size,).
        """
        # Ensure tensors are on the correct device
        alpha = alpha.to(self.device)
        beta = beta.to(self.device)
        labels = labels.to(self.device)

        # Compute predicted probabilities
        p = alpha / (alpha + beta + 1e-12)  # Add epsilon to avoid division by 0

        # Update aggregated metrics
        self.roc_auc.update(p, labels)
        self.pr_auc.update(p, labels)
        self.brier.update(p, labels)
        self.calibration_error.update(p, labels)

        # Update metrics for each threshold
        for th in self.thresholds:
            preds = (p >= th).float()
            self.precision[th].update(preds, labels)
            self.recall[th].update(preds, labels)
            self.f1[th].update(preds, labels)

        if self.store_predictions:
            self.pred_probs.extend(p.detach().cpu().numpy().tolist())
            self.labels.extend(labels.detach().cpu().numpy().tolist())

            for idx in range(len(alpha)):
                self.additional_info.append([
                    chr[idx], pos[idx].item(), ref[idx], alt[idx],
                    mapped_to_reverse[idx], read_id[idx],
                    alpha[idx].item(), beta[idx].item()
                ])

    def compute(self):
        """
        Computes and returns all metrics for each threshold and aggregated
        metrics.

        Returns:
            dict: Dictionary containing all metric values.
        """
        metrics = {}
        for th in self.thresholds:
            metrics[f'Precision@{th}'] = self.precision[th].compute().item()
            metrics[f'Recall@{th}'] = self.recall[th].compute().item()
            metrics[f'F1-Score@{th}'] = self.f1[th].compute().item()

        # Aggregated Metrics
        metrics['ROC AUC'] = self.roc_auc.compute().item()
        metrics['PR AUC'] = self.pr_auc.compute().item()
        metrics['Brier Score'] = self.brier.compute()
        metrics['Calibration Error (ECE)'] = self.calibration_error.compute().item()

        if self.store_predictions:
            metrics['Predictions'] = self.pred_probs
            metrics['Labels'] = self.labels

        return metrics

    def reset(self):
        """
        Resets all metrics.
        """
        for th in self.thresholds:
            self.precision[th].reset()
            self.recall[th].reset()
            self.f1[th].reset()
        self.roc_auc.reset()
        self.pr_auc.reset()
        self.brier.reset()
        self.calibration_error.reset()
        if self.store_predictions:
            self.pred_probs = []
            self.labels = []
            self.additional_info = []

    def write_predictions_to_csv(self, epoch, model_name, fold, output_dir):
        filename = f"{model_name}_fold_{fold}_epoch_{epoch}_predictions.csv"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        with open(output_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'chr', 'pos', 'ref', 'alt', 'mapped_to_reverse', 'read_id',
                'alpha', 'beta', 'label'
            ])
            for i, info in enumerate(self.additional_info):
                writer.writerow(info + [self.labels[i]])
        logging.info(f"Predictions written to {output_path}")

