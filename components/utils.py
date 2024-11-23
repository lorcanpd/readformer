import random
import os

import numpy as np

import torch
import torch.nn.functional as F


class WarmupConstantScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, base_lr, last_epoch=-1):
        self.warmup = warmup
        self.base_lr = base_lr
        super(WarmupConstantScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup:
            return [
                (self.base_lr * self.last_epoch / self.warmup) for base_lr
                in self.base_lrs
            ]
        else:
            return [self.base_lr for base_lr in self.base_lrs]


# Functions to check that weights are not being updated when they shouldn't be.
def get_weights(model):
    return {name: param.clone() for name, param in model.named_parameters()}


def check_weights(initial_weights, model):
    for name, param in model.named_parameters():
        if not torch.equal(initial_weights[name], param):
            print(f"Warning: Weight {name} has been updated!")
        else:
            print(f"Weight {name} has not been updated.")


def generate_span_mask(
        batch_size, seq_length, span_size, corruption_rate, device
):
    """
    Generate probabilities for masking spans of positions, where the number of
    spans is proportional to the corruption rate.

    :param positions:
        Tensor of shape (batch_size, seq_length) representing genomic positions.
    :param span_size:
        The size of each span to mask.
    :param corruption_rate:
        Proportion of positions to be masked.
    :return:
        A tensor of shape (batch_size, seq_length) with span masking probabilities.
    """

    # Determine the number of spans based on the corruption rate and span size
    num_spans = max(1, int(seq_length * corruption_rate / span_size))

    # Randomly select span centers
    span_centers = torch.randint(
        0, seq_length - span_size + 1, (batch_size, num_spans),
        device=device
    )

    # Generate probabilities for each span, initialized to 1
    # span_probabilities = torch.ones(
    #     (batch_size, seq_length), device=positions.device)

    # Create a tensor of False values with shape (batch_size, seq_length)
    span_mask = torch.zeros(
        (batch_size, seq_length), dtype=torch.bool, device=device
    )

    # Create the range of indices to zero out spans
    span_offsets = torch.arange(span_size, device=device)  # Shape: (span_size,)
    span_indices = (span_centers.unsqueeze(-1) + span_offsets) % seq_length  # Shape: (batch_size, num_spans, span_size)

    # Use advanced indexing to zero out the spans in a fully vectorized way
    batch_indices = torch.arange(batch_size, device=device).view(-1, 1, 1)  # Shape: (batch_size, 1, 1)
    span_mask[batch_indices, span_indices] = True

    return span_mask


def generate_uniform_mask(inputs, mask_rate):
    """
    Generate a boolean mask for uniform masking of the input tensor. The mask
    shape is the same as the input tensor, with True values indicating positions
    to be masked. Each position has a probability of `mask_rate` to be masked.

    :param inputs:
        The input tensor for which to generate the mask.
    :param mask_rate:
        The probability of masking each position.

    :return:
        A boolean mask tensor with the same shape as the input tensor.
    """
    mask = torch.rand_like(inputs) < mask_rate
    return mask


def apply_masking_with_consistent_replacements(
        nucleotide_sequences, mask_token, rate=0.15, mask_rate=0.8,
        replace_rate=0.1, kernel_size=5, split=0.5
):
    """
    Apply masking and consistent replacements with a mix of span-based and uniform masking.

    :param nucleotide_sequences:
        Tensor of shape (batch_size, seq_length) representing nucleotide sequences.
    :param mask_token:
        The token used for masking.
    :param rate:
        The overall corruption rate.
    :param mask_rate:
        Proportion of corrupted tokens to replace with the mask token.
    :param replace_rate:
        Proportion of corrupted tokens to replace randomly among other tokens.
    :param kernel_size:
        The kernel size used for span masking.
    :param split:
        Proportion of sequences to mask using span masking; the remainder are
        masked uniformly.
    :return:
        Tuple containing (
            masked_sequence, mask_indices, replace_indices, keep_indices).
    """

    # true_mask_rate = rate * mask_rate
    # # true_keep_rate = rate * keep_rate
    # true_replace_rate = rate * replace_rate

    batch_size, seq_length = nucleotide_sequences.size()
    random_probs = torch.rand((batch_size, seq_length), device=nucleotide_sequences.device)

    # Split the batch into two parts based on the split argument
    split_index = int(split * batch_size)
    # span_masking_batch = positions[:split_index]
    # uniform_masking_batch = positions[split_index:].to(torch.float32)

    # Generate span-based masking for the first part of the batch
    batch_size, seq_length = random_probs[:split_index].shape
    span_mask = generate_span_mask(
        batch_size, seq_length, kernel_size, rate,
        device=nucleotide_sequences.device
    )

    # Generate uniform masking for the second part of the batch
    uniform_mask = generate_uniform_mask(
        random_probs[split_index:], rate
    )

    # Combine the span-based and uniform masking probabilities
    initial_mask = torch.cat(
        [span_mask, uniform_mask], dim=0
    )

    # Create the final masking decision
    mask_indices = initial_mask & (random_probs <= mask_rate)
    replace_indices = initial_mask & (mask_rate < random_probs) & (
            random_probs <= mask_rate + replace_rate)
    # Apply the masking
    masked_sequence = nucleotide_sequences.clone()
    masked_sequence[mask_indices] = mask_token

    if replace_indices.sum() > 0:
        bases = [0, 1, 2, 3]
        random.shuffle(bases)

        masked_sequence[replace_indices] = bases[0]

        replacement_invalid_mask = (
            masked_sequence == nucleotide_sequences) & replace_indices

        i = 1

        while replacement_invalid_mask.any():
            try:
                masked_sequence[replacement_invalid_mask] = bases[i]
            except IndexError:
                # If canonical bases are exhausted, do not replace
                break
            replacement_invalid_mask = (
                masked_sequence == nucleotide_sequences) & replace_indices

    return masked_sequence, mask_indices, replace_indices


def get_random_alternative_labels(original_labels, mlm=True):
    """
    Generate a random set of class labels that are different from the original
    input labels, selected from a given set of label options.

    :param original_labels: Tensor of shape (batch_size, seq_length) containing the original class labels.
    :param mlm: Boolean flag indicating whether to use labels for an MLM (Masked Language Model).
                If True, the label options include [0, 1, 2, 3, 15].
                If False, the label options include [0, 1, 2, 3].
    :return: Tensor of the same shape as `original_labels` containing random class labels different from the originals.
    """
    if mlm:
        labels = torch.tensor([0, 1, 2, 3, 15], device=original_labels.device)
    else:
        labels = torch.tensor([0, 1, 2, 3], device=original_labels.device)

    # Generate random indices to select labels from the predefined list
    random_indices = torch.randint(
        0, len(labels), original_labels.shape, device=original_labels.device
    )
    random_labels = labels[random_indices]

    # Ensure random labels are different from the original labels
    mismatch_mask = random_labels == original_labels
    while mismatch_mask.any():
        random_indices = torch.randint(
            0, len(labels), (mismatch_mask.sum().item(),),
            device=original_labels.device
        )
        random_labels[mismatch_mask] = labels[random_indices]
        mismatch_mask = random_labels == original_labels

    return random_labels


def load_validation_tensors(validation_dir):
    """Load validation tensors from the specified directory."""
    tensor_dict = {}
    tensor_names = [
        'positions', 'valid_positions',
        'masked_sequences', 'masked_cigar_encodings',
        'masked_base_qualities', 'masked_is_first', 'masked_mapped_to_reverse',
        'masked_indices', 'replaced_indices',
        'nucleotide_sequences', 'base_qualities'
    ]

    for name in tensor_names:
        tensor_path = os.path.join(validation_dir, f"{name}.pt")
        tensor_dict[name] = torch.load(tensor_path)
        print(f"Loaded {name} from {tensor_path}")

    return tensor_dict


def get_effective_number(n_samples):
    """
    Compute the effective number of samples for class balancing.

    :param n_samples:
        The number of samples in the class.
    :return:
        The effective number of samples.
    """

    if n_samples == 1:
        return 1.0

    beta = (n_samples - 1) / n_samples

    numerator = 1 - np.exp(n_samples * np.log(beta))
    denominator = 1 - beta

    effective_num = numerator / denominator

    return effective_num
