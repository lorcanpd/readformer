import torch
import torch.nn as nn
import torch.nn.functional as F

from components.better_device_handling import Module


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
        nll_loss = -torch.log(true_probs).sum() * scale_factor

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


def class_balanced_beta_nll_loss(
        alpha, beta, y_true, weighting=None, eps=1e-12
):
    """
    Compute the Class-Balanced negative log likelihood (NLL) of the true label

    :param alpha:
        The predicted alpha parameter of the Beta distribution.
    :param beta:
        The predicted beta parameter of the Beta distribution.
    :param y_true:
        The true labels.
    :param weighting:
        The weighting to apply to the loss. This should be the product of the
        inverse effective number of samples in each class (artefact or mutation)
        and the inverse effective number of samples per unique
        mutation/artefact.
    :param eps:
        A small value to avoid log(0).
    :return:
        The Class-Balanced negative log likelihood.
    """

    y_true = y_true.clamp(eps, 1 - eps)  # Avoid log(0)
    log_prob = torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)
    log_prob += (alpha - 1) * torch.log(y_true) + (beta - 1) * torch.log(1 - y_true)
    nll = -log_prob

    # Apply class weights
    if weighting is not None:
        nll = nll * weighting

    return nll.sum()


