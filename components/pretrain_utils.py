import random

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


def create_intervals(max_sequence_length, min_length=256):
    """
    Create a list of intervals from a nucleotide threshold.

    :param max_sequence_length:
        The nucleotide threshold.
    :param min_length:
        The minimum length of the interval.
    :returns:
        A list of intervals.
    """
    num_intervals = 0
    interval_size = min_length
    while interval_size < max_sequence_length:
        interval_size *= 2
        num_intervals += 1

    intervals = [(2 ** (i+1)) * min_length for i in range(0, num_intervals)]
    # set last interval to the remainder of the nucleotide threshold
    intervals[-1] = max_sequence_length
    return intervals


def create_corruption_rates(intervals, min_rate=0.15, read_length=250, scale=0.9):
    """
    Create a list of corruption rates for each interval.

    :param intervals:
        A list of intervals.
    :param min_rate:
        The minimum corruption rate.
    :param read_length:
        The length of the reads.
    :param scale:
        The scaling factor for the corruption rates. This is used to control
        the overall corruption rate. To be tuned as not sure how aggressively
        the redundancy from the overlapping reads needs to counteract the
        corruption.
    :returns:
        A list of corruption rates.
    """
    rates = []
    # We want to ensure that there is at least one reads worth of data untouched
    # in each sample, based upon the interval (now many nucleotides in total) and
    # the read length.
    for interval in intervals:
        rates.append(max(min_rate, 1 - read_length / interval) * scale)
    return rates


def replacement_loss(predictions, replacement_mask, mixing_mask, valid_mask):
    """
    Compute the replacement loss with mixed labels.

    :param predictions:
        The predictions of the model (logits) with shape (batch_size,).
    :param replacement_mask:
        Boolean mask indicating which elements are replaced.
    :param mixing_mask:
        The mixing coefficients indicating the proportion of replacement for
        each element in the replacement_mask, with values between 0 and 1.
    :param valid_mask:
        Boolean mask indicating which elements are valid for loss computation.

    :returns:
        The computed replacement loss averaged over valid elements.
    """
    # True labels: 1 for replaced, 0 for original
    true_labels = torch.zeros_like(predictions)
    true_labels[replacement_mask] = mixing_mask

    # Compute the binary cross-entropy loss without reduction
    loss = F.binary_cross_entropy(
        predictions, true_labels,
        reduction='none'
    )

    # Apply the valid mask to filter valid elements for loss computation
    valid_loss = loss * valid_mask

    # Compute the average loss over valid elements
    average_loss = valid_loss.sum() / valid_mask.sum()

    return average_loss


def adversarial_loss(predictions, replacement_mask, mixing_mask):
    """
    Compute the adversarial loss with mixed labels.

    :param predictions:
        The predictions of the model (logits) with shape (batch_size,).
    :param replacement_mask:
        Boolean mask indicating which elements are replaced.
    :param mixing_mask:
        The mixing coefficients indicating the proportion of replacement for
        each element in the replacement_mask, with values between 0 and 1.

    :returns:
        The computed adversarial loss.
    """
    true_labels = torch.zeros_like(predictions)
    true_labels[replacement_mask] = mixing_mask

    adv_labels = 1 - true_labels

    # Adversarial loss is higher if the main model gets more predictions correct
    loss = F.binary_cross_entropy(
        predictions, adv_labels, reduction='mean'
    )
    return loss


def map_probabilities_to_positions(positions):
    """
    Generate random probabilities for unique positions in each sequence and map
    them back to the original positions.

    :param positions:
        Tensor of shape (batch_size, seq_length) representing genomic positions.
    :return:
        Tensor of the same shape as `positions`, where each value is a
        probability mapped from unique positions.
    """
    batch_size, seq_length = positions.size()

    # Flatten positions for easier processing across the entire batch
    positions_flat = positions.view(-1)

    # Generate unique positions across the batch and get the inverse indices
    unique_positions, inverse_indices = torch.unique(
        positions_flat, return_inverse=True
    )

    # Generate random probabilities for each unique position
    random_probs = torch.rand(
        unique_positions.size(0), device=positions.device
    )

    # Map probabilities back to the original positions using inverse indices
    probabilities_mapped_flat = random_probs[inverse_indices]

    # Reshape back to the original batch and sequence shape
    probabilities_mapped = probabilities_mapped_flat.view(
        batch_size, seq_length
    )

    return probabilities_mapped


def generate_random_for_unique_positions(positions):
    """
    Generate random values for each unique position in the input sequence.
    :param positions:
        Genomic positions of each element in the input sequence of shape
        (batch_size, seq_length).
    :return:
        A tuple of lists containing unique positions and random values for each
        unique position in the input sequence.
    """
    breakpoint()
    batch_size, seq_length = positions.size()
    unique_values = []
    unique_randoms = []

    for i in range(batch_size):
        seq = positions[i]
        unique_vals, inverse_indices = torch.unique(
            seq[seq != -1], return_inverse=True
        )
        lognorms = torch.distributions.LogNormal(
            0, 1
        ).sample(unique_vals.size()).to(positions.device)
        random_vals = 1 - lognorms

        unique_values.append(unique_vals)
        unique_randoms.append(random_vals)

    return unique_values, unique_randoms


def broadcast_unique_randoms_to_input(
        positions, unique_values, unique_randoms
):
    """
    Broadcast the random values to the input sequence based on the unique
    positions.

    :param positions:
        Genomic positions of each element in the input sequence of shape
        (batch_size, seq_length).
    :param unique_values:
        A list of unique positions for each input sequence.
    :param unique_randoms:
        A list of random values for each unique position in the input sequence.
    :return:
        A tensor of random values broadcast to the input sequence.
    """
    batch_size, seq_length = positions.size()
    broadcasted_randoms = torch.zeros_like(
        positions, dtype=torch.float, #device=positions.device
    )

    for i in range(batch_size):
        seq = positions[i]
        valid_mask = seq != -1
        inverse_indices = torch.searchsorted(
            unique_values[i], seq[valid_mask]
        )
        broadcasted_randoms[i][valid_mask] = unique_randoms[i][
            inverse_indices
        ]

    return broadcasted_randoms


def gaussian_kernel_1d(positions, centers, variances):
    """
    Generate a 1D Gaussian kernel over a given set of positions for multiple
    centers.

    :param positions:
        Tensor of shape (num_positions,) representing position indices.
    :param centers:
        Tensor of shape (num_gaussians,) representing the Gaussian centers.
    :param variances:
        Tensor of shape (num_gaussians,) representing the Gaussian variances.
    :return:
        Tensor of shape (num_gaussians, num_positions) representing the
        Gaussian values.
    """
    # Vectorised computation of Gaussian values for each center and variance
    positions = positions.unsqueeze(0)  # Shape: (1, num_positions)
    centers = centers.unsqueeze(1)  # Shape: (num_gaussians, 1)
    gaussians = torch.exp(
        -0.5 * ((positions - centers) ** 2) / variances.unsqueeze(1)
    )  # Shape: (num_gaussians, num_positions)
    return gaussians


def generate_gaussian_probabilities(positions, kernel_size, bias=0.3):
    """
    Generate Gaussian kernel-based probabilities for masking, considering a
    specific kernel size.

    :param positions:
        Tensor of shape (batch_size, seq_length) representing genomic positions.
    :param kernel_size:
        The kernel size used to define the variance of the Gaussians.
    :param bias:
        A small value to add to the Gaussian sum for every position to prevent
        zero probabilities.
    :return:
        A tensor of shape (batch_size, seq_length) with Gaussian probabilities
        mapped to the original position space.
    """
    batch_size, seq_length = positions.size()

    # Flatten positions for easier processing across the entire batch
    positions_flat = positions.view(-1)

    # Generate unique positions across the batch and get the inverse indices
    unique_positions, inverse_indices = torch.unique(
        positions_flat, return_inverse=True
    )
    num_unique_pos = unique_positions.size(0)

    # Generate random Gaussian centers and variances
    num_gaussians = num_unique_pos // (10 * kernel_size)
    gauss_centers = torch.randint(
        0, num_unique_pos, (num_gaussians,), device=positions.device
    )
    gauss_variances = torch.full(
        (num_gaussians,), kernel_size * 2, device=positions.device
    )  # Variance based on kernel size

    # Generate Gaussian kernels over the unique positions
    position_indices = torch.arange(
        num_unique_pos, device=positions.device
    ).float()
    gaussian_kernels = gaussian_kernel_1d(
        position_indices, gauss_centers, gauss_variances
    )  # Shape: (num_gaussians, num_unique_pos)

    # Sum all the Gaussian kernels and add the bias
    gaussian_sum = gaussian_kernels.sum(dim=0) + bias  # Shape: (num_unique_pos,)

    # Map the Gaussian probabilities back to the original position space
    gaussian_probs_mapped_flat = gaussian_sum[inverse_indices].view(
        batch_size, seq_length)


    # Generate random probabilities for each unique position
    random_probs = torch.rand(num_unique_pos, device=positions.device)

    # Map probabilities back to the original positions using inverse indices
    probabilities_mapped_flat = random_probs[inverse_indices].view(
        batch_size, seq_length
    )

    # Multiply by Gaussian probabilities
    gaussian_probs_mapped = gaussian_probs_mapped_flat * probabilities_mapped_flat

    # Normalize by each sequence's max probability
    max_probs_per_sequence = gaussian_probs_mapped.max(dim=1, keepdim=True)[0]
    gaussian_probs_mapped = 1 - (gaussian_probs_mapped / max_probs_per_sequence)

    return gaussian_probs_mapped


def apply_masking_with_consistent_replacements(
        positions, nucleotide_sequences, mask_token, kernel_size, rate=0.15,
        mask_rate=0.8, keep_rate=0.1, random_replace_rate=0.1
):
    assert mask_rate + keep_rate + random_replace_rate == 1.0, "The rates must sum to 1.0."
    # Adjust the rate due to the fact we apply two sets of random masks.
    # Otherwise, the real masking rate will be higher than the desired rate.
    rate = rate * 0.55
    batch_size, seq_length = nucleotide_sequences.size()

    # Generate the initial position-based mask
    # position_probabilities = map_probabilities_to_positions(positions)
    position_probabilities = generate_gaussian_probabilities(
        positions, kernel_size=kernel_size, bias=0.2
    )

    # Create masks based on the probabilities
    random_probs = torch.rand(
        (batch_size, seq_length), device=nucleotide_sequences.device
    )
    mask_mask = (random_probs < rate) | (position_probabilities < rate)
    random_replace_mask = (
            (random_probs >= mask_rate) &
            (random_probs < mask_rate + random_replace_rate) &
            mask_mask
    )
    mask_mask = mask_mask & ~random_replace_mask
    kept_mask = (random_probs >= mask_rate + random_replace_rate) & mask_mask

    replacements = nucleotide_sequences.clone().to(nucleotide_sequences.device)
    bases = [0, 1, 2, 3]
    random.shuffle(bases)
    i = 0
    if random_replace_mask.sum() > 0:
        replacement_base = bases[i]
        replacements[random_replace_mask] = replacement_base
        replacement_invalid_mask = (replacements == nucleotide_sequences) & random_replace_mask

        while replacement_invalid_mask.any():
            i += 1
            try:
                replacement_base = bases[i]
            except IndexError:
                # If canonical bases are exhausted, do not replace
                break
            replacements[replacement_invalid_mask] = replacement_base
            replacement_invalid_mask = (replacements == nucleotide_sequences) & random_replace_mask

        random_replace_mask = random_replace_mask & ~replacement_invalid_mask
    # Apply the masking
    replacements[mask_mask] = mask_token

    return replacements, mask_mask, random_replace_mask, kept_mask


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
            0, len(labels), (mismatch_mask.sum().item(), ),
            device=original_labels.device
        )
        random_labels[mismatch_mask] = labels[random_indices]
        mismatch_mask = random_labels == original_labels

    return random_labels

#
# # test
# import torch
#
# batch_size = 8
# seq_length = 20
# kernel_size = 3
# nucleotide_sequences = torch.randint(0, 4, (batch_size, seq_length))
# positions = torch.tensor([
#     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
#     [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
#     [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
#     [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
#     [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
#     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
#     [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
#     [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
# ])
#
# # Repeat multiple times to see true masking rate
# total_rates = []
# replacement_rates = []
# masked_rates = []
# for _ in range(10000):
#     replacements, masked, replaced, kept = apply_masking_with_consistent_replacements(
#         positions, nucleotide_sequences, 4, kernel_size,
#         rate=0.2, mask_rate=0.5, keep_rate=0.25, random_replace_rate=0.25
#     )
#
#     total_rates.append((replaced | masked).sum() / (batch_size * seq_length))
#     replacement_rates.append(replaced.sum() / (batch_size * seq_length))
#     masked_rates.append(masked.sum() / (batch_size * seq_length))
#
#
# # plot histogram containing all three rates with different colors
# import matplotlib.pyplot as plt
#
# plt.hist(total_rates, bins=20, alpha=0.5, label='Total Rate')
# plt.hist(replacement_rates, bins=20, alpha=0.5, label='Replacement Rate')
# plt.hist(masked_rates, bins=20, alpha=0.5, label='Masked Rate')
# plt.legend()
# plt.show()

def mlm_accuracy(logits, target):
    with torch.no_grad():
        # Flatten logits and target for loss computation
        logits_flat = logits.view(-1, logits.size(-1))
        target_flat = target.view(-1).long()
        pred_probs = F.softmax(logits_flat, dim=-1)
        pred_classes = torch.argmax(pred_probs, dim=-1)
        correct_predictions = (pred_classes == target_flat).float()
        accuracy = correct_predictions.mean().item()
        # print(f"Batch Accuracy: {accuracy * 100:.2f}%")
        # print(f"Loss: {loss.item()}")
        return accuracy
