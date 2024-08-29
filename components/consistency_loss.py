import torch
import torch.nn.functional as F


def calculate_consistency_loss(output, num_repeats):
    """
    Calculate the consistency loss for repeated sequences in parallel.

    :param output: Tensor of shape (batch_size, seq_length, num_classes).
    :param num_repeats: Number of repeats for contrastive learning.
    :return: Consistency loss.
    """
    batch_size, seq_length, num_classes = output.size()
    # Reshape to (num_repeats, batch_size // num_repeats, seq_length, num_classes)
    repeated_outputs = output.view(num_repeats, batch_size // num_repeats, seq_length, num_classes)

    # Normalize the outputs to get unit vectors
    normalized_outputs = F.normalize(repeated_outputs, p=2, dim=-1)
    # Shape: (num_repeats, batch_size // num_repeats, seq_length, num_classes)

    # Compute pairwise cosine similarities across the repeats

    normalized_outputs = normalized_outputs.unbind(0)
    similarities = torch.einsum('bld,bld->bl', normalized_outputs[0], normalized_outputs[1])
    # similarities = torch.einsum('rikl,rjkl->rijk', normalized_outputs, normalized_outputs)

    # We only want the upper triangle (excluding the diagonal) for each batch
    # mask = torch.triu(torch.ones(num_repeats, num_repeats), diagonal=1).bool()

    # Apply mask to get only unique pairs
    # masked_similarities = similarities[mask].view(-1, seq_length)

    # Compute the loss as 1 - mean cosine similarity
    consistency_loss = (1 - similarities).mean()

    return consistency_loss