import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_mapping_and_base_quality_histogram(final_dict, log_scale=False):
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


def plot_replacement_statistics(original_sequences, corrupted_sequences, positions):
    batch_size, seq_length = original_sequences.size()

    replacement_counts = []
    proportion_replaced = []
    for i in range(batch_size):
        seq_positions = positions[i][positions[i] != -1]
        unique_positions = torch.unique(seq_positions)
        for pos in unique_positions:
            if pos == -1:
                continue
            original_count = torch.sum((positions[i] == pos) & (original_sequences[i] == corrupted_sequences[i])).item()
            replaced_count = torch.sum((positions[i] == pos) & (original_sequences[i] != corrupted_sequences[i])).item()
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
        df.set_index('Position').plot(kind='bar', stacked=True, ax=axes[i], color=['blue', 'red'])
        axes[i].set_title(f"Sequence {i + 1}")
        axes[i].set_xlabel('Position')
        axes[i].set_ylabel('Count')
        axes[i].legend(title='Type', labels=['Original', 'Replaced'])

    plt.tight_layout()
    plt.show()
