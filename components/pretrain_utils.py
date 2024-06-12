import torch
import torch.nn.functional as F


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
        ).sample((unique_vals.size(0),))
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
    broadcasted_randoms = torch.zeros_like(positions, dtype=torch.float)

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


def get_replacement_mask(positions, rate=0.15):
    """
    Generate a mask for replacing elements in the input sequence based on a
    corruption rate. This is done in such a manner that a large proportion of
    positons will see few replacements, while a small proportion of positions
    will see many replacements, at a given genomic position. This is intended
    to replicate the patterns exhibited by mutations and sequencing artefacts.

    :param positions:
        Genomic positions of each element in the input sequence of shape
        (batch_size, seq_length).
    :param rate:
        The corruption rate for replacing elements in the input sequence.
    :return:
        A boolean mask indicating which elements to replace in the input
        sequence.
    """
    unique_values, unique_randoms = generate_random_for_unique_positions(
        positions
    )
    broadcasted_randoms = broadcast_unique_randoms_to_input(
        positions, unique_values, unique_randoms
    )
    valid_mask = positions != -1
    # Generate random numbers for each valid position in the sequence
    element_limit = 1 / broadcasted_randoms
    element_randoms = torch.rand(
        positions.shape, device=positions.device, dtype=torch.float32
    ) * (1 - broadcasted_randoms) + broadcasted_randoms
    product = element_randoms * broadcasted_randoms * element_limit
    # Create a replacement mask based on the threshold
    replace_mask = product < rate
    replace_mask = replace_mask & valid_mask

    return replace_mask

