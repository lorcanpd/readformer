import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_mapping_and_base_quality_histogram(final_dict, log_scale=False):
    """
    Plot the distribution of mapping quality and base quality scores.

    :param final_dict:
        A dictionary containing mapping and base quality scores for reads.
    :param log_scale:
        A boolean indicating whether to use a logarithmic scale for the y-axis.
    :returns:
        None
    """
    # initialise the lattice plot
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    # Plot the distribution of mapping quality scores
    mapping_quality = [
        values['mapping_quality'] for values in final_dict.values()
    ]
    axs[0].hist(mapping_quality, bins=40, edgecolor='black')
    axs[0].set_xlabel('Mapping Quality')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Distribution of Mapping Quality Scores')

    # Plot the distribution of base quality scores, flattening the list of lists
    base_quality = [
        q for sublist in final_dict.values() for q
        in sublist['adjusted_base_qualities']
    ]
    axs[1].hist(base_quality, bins=40, edgecolor='black')
    axs[1].set_xlabel('Base Quality')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Distribution of Base Quality Scores')

    # Optionally, set a log scale for the y-axis if the data is skewed
    if log_scale:
        axs[0].set_yscale('log')
        axs[1].set_yscale('log')

    plt.tight_layout()
    plt.show()


def plot_nucleotide_coverage(read_dict, position) -> None:
    """
    Plot the per-nucleotide coverage depth around a given genomic position.

    :param read_dict:
        The dictionary of fetched reads.
    :param position:
        The genomic position around which to plot the coverage.

    :return None:

    """
    # Initialize variables to find the range of positions covered by fetched
    # reads
    min_start = float('inf')
    max_end = -float('inf')

    # Determine the minimum start and maximum end positions from fetched reads
    for read in read_dict.values():
        read_start = read["reference_start"]
        read_end = read_start + len(read["query_sequence"])
        min_start = min(min_start, read_start)
        max_end = max(max_end, read_end)

    # Initialize the coverage array for the range of positions covered
    coverage = np.zeros(max_end - min_start + 1)

    # Populate the coverage array with counts for each position
    for read in read_dict.values():
        read_start = read["reference_start"]
        read_end = read_start + len(read["query_sequence"])
        for pos in range(read_start, read_end):
            if min_start <= pos <= max_end:
                coverage[pos - min_start] += 1

    # Plotting
    # Convert position to relative position from 'min_start'
    # relative_position = position - min_start

    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(min_start, max_end + 1) - position,
            coverage, width=1.0, edgecolor='black', linewidth=0.5)
    plt.xlabel('Position relative to specified genomic position')
    plt.ylabel('Coverage Depth')
    plt.title('Per-Nucleotide Coverage Depth around Genomic Position')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.show()


def plot_replacement_statistics(
        original_sequences, corrupted_sequences, positions
):
    """
    Plot statistics on the replacement of nucleotides in sequences.

    :param original_sequences:
        A tensor of the original nucleotide sequences.
    :param corrupted_sequences:
        A tensor of the corrupted nucleotide sequences.
    :param positions:
        A tensor of the positions in the sequences.
    :returns:
        None
    """
    batch_size, seq_length = original_sequences.size()

    replacement_counts = []
    proportion_replaced = []
    for i in range(batch_size):
        seq_positions = positions[i][positions[i] != -1]
        unique_positions = torch.unique(seq_positions)
        for pos in unique_positions:
            if pos == -1:
                continue
            original_count = torch.sum(
                (positions[i] == pos) &
                (original_sequences[i] == corrupted_sequences[i])
            ).item()
            replaced_count = torch.sum(
                (positions[i] == pos) &
                (original_sequences[i] != corrupted_sequences[i])
            ).item()
            proportion = replaced_count / (original_count + replaced_count)
            if replaced_count > 0:
                replacement_counts.append(replaced_count)
                proportion_replaced.append(proportion)

    # Calculate the proportion of replaced bases per position

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(proportion_replaced, range=(0, 1), bins=100, edgecolor='black')
    plt.xlabel('Proportion of Bases Replaced at a Position')
    plt.ylabel('Number of Positions')
    plt.title('Distribution of Base Replacements per Position')
    # plt.yscale('log')  # Use a logarithmic scale to emphasize the long tail
    plt.show()


def plot_nucleotide_replacement_histogram(
        original_sequences, corrupted_sequences, positions
):
    """
    Plot a histogram of nucleotide replacements across sequences.

    :param original_sequences:
        A tensor of the original nucleotide sequences.
    :param corrupted_sequences:
        A tensor of the corrupted nucleotide sequences.
    :param positions:
        A tensor of the positions in the sequences.
    :returns:
        None
    """
    batch_size, seq_length = original_sequences.size()
    fig, axes = plt.subplots(batch_size, 1, sharex=True)

    if batch_size == 1:
        axes = [axes]

    for i in range(batch_size):
        seq_positions = positions[i][positions[i] != -1]
        min_pos = seq_positions.min().item()
        max_pos = seq_positions.max().item()
        all_positions = np.arange(min_pos, max_pos + 1)
        original_counts = np.zeros(max_pos - min_pos + 1, dtype=int)
        replaced_counts = np.zeros(max_pos - min_pos + 1, dtype=int)

        for j, pos in enumerate(all_positions):
            original_counts[j] = torch.sum(
                (positions[i] == pos) & (
                        original_sequences[i] == corrupted_sequences[i])
            ).item()
            replaced_counts[j] = torch.sum(
                (positions[i] == pos) & (
                        original_sequences[i] != corrupted_sequences[i])
            ).item()

        # Create the data for plotting
        df = pd.DataFrame({
            'Position': all_positions,
            'Original': original_counts,
            'Replaced': replaced_counts
        })

        # Plot stacked bar chart
        df.set_index('Position').plot(
            kind='bar', stacked=True, ax=axes[i], color=['blue', 'red']
        )
        axes[i].set_title(f"Sequence {i + 1}")
        axes[i].set_xlabel('Position')
        axes[i].set_ylabel('Count')
        axes[i].legend(title='Type', labels=['Original', 'Replaced'])

    plt.tight_layout()
    plt.show()


def seq_to_str(sequence):
    mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'R', 5: 'Y', 6: 'S', 7: 'W',
               8: 'K', 9: 'M', 10: 'B', 11: 'D', 12: 'H', 13: 'V', 14: 'N', 15: '', 16: 'X'}
    return ''.join([mapping[nuc] if nuc in mapping else '' for nuc in sequence])

def seq_to_str_with_replacement(original, replaced, mask_token_mask, random_mask):
    """
    Convert nucleotide sequence to string with masked positions and random replacements.

    :param original: The original nucleotide sequence.
    :param replaced: The sequence with random replacements.
    :param mask_token_mask: Boolean mask for masked positions.
    :param random_mask: Boolean mask for random replacements.
    :return: A string with masked and replaced nucleotides.
    """
    mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'R', 5: 'Y', 6: 'S', 7: 'W',
               8: 'K', 9: 'M', 10: 'B', 11: 'D', 12: 'H', 13: 'V', 14: 'N', 15: '', 16: 'X'}
    seq = []
    for i, nuc in enumerate(original):
        if i >= len(mask_token_mask) or i >= len(random_mask):
            seq.append(mapping[nuc] if nuc in mapping else '')
            continue
        if mask_token_mask[i]:
            seq.append('X')
        elif random_mask[i]:
            seq.append('acgt'[replaced[i]] if replaced[i] in range(4) else '')  # Only considering 'A', 'C', 'G', 'T' for replacements
        else:
            seq.append(mapping[nuc] if nuc in mapping else '')
    return ''.join(seq)

def plot_aligned_sequences(original_sequences, masked_sequences, replaced_sequences, mask_token_mask, random_mask, positions,
                           title="Aligned Sequence Masking and Replacement"):
    """
    Plot the aligned original, masked, and randomly replaced nucleotide sequences.

    :param original_sequences: List of original nucleotide sequences.
    :param masked_sequences: List of sequences with mask tokens.
    :param replaced_sequences: List of sequences with random replacements.
    :param mask_token_mask: Boolean mask for masked positions.
    :param random_mask: Boolean mask for random replacements.
    :param positions: List of positions in the sequences used to align
    overlapping reads.
    :param title: The title of the plot.
    """
    fig, ax = plt.subplots(figsize=(20, 6))
    y_pos = np.arange(len(original_sequences))

    aligned_originals = []
    aligned_masked = []
    aligned_replaced = []

    for i in range(len(original_sequences)):
        max_pos = positions[i].max().item()
        min_pos = positions[i].min().item()

        aligned_length = max_pos - min_pos + 1

        aligned_original = [-1] * aligned_length
        aligned_masked = [-1] * aligned_length
        aligned_replaced = [-1] * aligned_length

        for j in range(len(original_sequences[i])):
            pos = positions[i][j].item() - min_pos
            if pos >= 0 and pos < aligned_length:
                aligned_original[pos] = original_sequences[i][j].item()
                if mask_token_mask[i][j]:
                    aligned_masked[pos] = 16
                elif random_mask[i][j]:
                    aligned_masked[pos] = replaced_sequences[i][j].item()
                else:
                    aligned_masked[pos] = original_sequences[i][j].item()
                aligned_replaced[pos] = replaced_sequences[i][j].item() if random_mask[i][j] else original_sequences[i][j].item()

        aligned_originals.append(seq_to_str(aligned_original))
        aligned_masked.append(seq_to_str_with_replacement(aligned_original, aligned_replaced, mask_token_mask[i], random_mask[i]))

    for i, (original, masked) in enumerate(zip(aligned_originals, aligned_masked)):
        ax.text(0.5, y_pos[i], original, ha='center', va='center',
                fontdict={'fontsize': 12, 'fontfamily': 'monospace'})
        ax.text(0.5, y_pos[i] - 0.2, masked, ha='center',
                va='center', fontdict={'fontsize': 12, 'fontfamily': 'monospace', 'color': 'blue'})

    ax.set_xlim(0, 1)
    ax.set_ylim(-len(original_sequences), 1)
    ax.set_xticks([])
    ax.set_yticks(y_pos - 0.1)
    ax.set_yticklabels(['Seq {}'.format(i + 1) for i in range(len(original_sequences))])
    ax.set_title(title)
    plt.show()

#
# original_sequences = [
#     torch.tensor([0, 3, 0, 2, 0, 3, 0, 2]),
#     torch.tensor([3, 0, 3, 0, 2, 0, 2, 1]),
#     torch.tensor([0, 3, 0, 2, 0, 3, 0, 0])
# ]
#
# masked_sequences = [
#     torch.tensor([0, 16, 0, 2, 0, 3, 0, 2]),
#     torch.tensor([3, 0, 3, 0, 2, 0, 2, 1]),
#     torch.tensor([0, 3, 0, 2, 0, 3, 0, 0])
# ]
#
# replaced_sequences = [
#     torch.tensor([0, 3, 0, 2, 1, 3, 0, 2]),
#     torch.tensor([3, 0, 3, 0, 2, 0, 2, 1]),
#     torch.tensor([0, 3, 0, 2, 0, 3, 0, 0])
# ]
#
# # masked_boolean is a tensor of where the masked positions = 16
#
# masked_boolean = [
#     torch.tensor([0, 1, 0, 0, 0, 0, 0, 0]),
#     torch.tensor([0, 0, 0, 0, 0, 0, 0, 0]),
#     torch.tensor([0, 0, 0, 0, 0, 0, 0, 0])
# ]
#
# # random_boolean is a tensor of where the replaced sequences do not match original sequences
#
# random_boolean = [
#     torch.tensor([0, 0, 0, 0, 1, 0, 0, 0]),
#     torch.tensor([0, 0, 0, 0, 0, 0, 0, 0]),
#     torch.tensor([0, 0, 0, 0, 0, 0, 0, 0])
# ]
#
# plot_aligned_sequences(original_sequences[0:2], masked_sequences[0:2], replaced_sequences[0:2], masked_boolean[0:2], random_boolean[0:2])