
import os
import random
import pandas as pd
import tensorflow as tf
from extract_reads import (
    extract_reads_from_position_onward, sample_positions,  get_read_info
)


def process_read_entry(read_entry):
    # Extract relevant fields from the dictionary
    query_sequence = read_entry['query_sequence']

    # Convert the query sequence to indices (A, C, G, T, etc. mapped to 0, 1, 2, 3, ..., respectively)
    # This requires a mapping from nucleotide to index. Here's a simple approach:
    nucleotide_to_index_map = {
        'A': 1, 'C': 2, 'G': 3, 'T': 4, 'R': 5, 'Y': 6, 'S': 7, 'W': 8, 'K': 9,
        'M': 10, 'B': 11, 'D': 12, 'H': 13, 'V': 14, 'N': 15
    }
    indices = [nucleotide_to_index_map.get(nuc, 0) for nuc in query_sequence]

    # Convert indices list to a tensor
    indices_tensor = tf.constant(indices, dtype=tf.int32)

    # Apply the embedding layer
    embedded_sequence = nucleotide_embedding_layer(indices_tensor)

    return embedded_sequence


def replicate_binary_flag_vector(
        binary_flag_vector, nucleotide_threshold, sequence_length
)-> tf.Tensor:
    """
    Replicates a binary flag vector for each nucleotide position in the sequence.

    :param binary_flag_vector:
        The binary flag vector for a read.
    :param nucleotide_threshold:
        The nucleotide threshold or maximum sequence length for padding.
    :return:
        A replicated binary flag vector of shape (sequence_length, 12).
    """
    # Expand dimensions to allow replication across the sequence length
    # Shape: (1, 12)
    expanded_vector = tf.expand_dims(binary_flag_vector, 0)
    # Replicate the binary flag vector for each nucleotide position
    # Shape: (sequence_length, 12)
    replicated_vector = tf.tile(
        expanded_vector, [sequence_length, 1]
    )

    # If the actual sequence is shorter than the nucleotide_threshold, pad the
    # remainder
    if sequence_length < nucleotide_threshold:
        # Pad sequences to match the threshold
        padding = [[0, nucleotide_threshold - sequence_length], [0, 0]]
        replicated_vector = tf.pad(
            replicated_vector, padding, "CONSTANT", constant_values=0
        )

    return replicated_vector


def bam_read_generator(file_paths, metadata_path, nucleotide_threshold, batch_size):
    """
    Generator function to yield batches of processed reads from BAM files, with
    dynamic sampling of file paths, chromosomes, and positions.

    :param file_paths:
        List of paths to BAM files or a directory path containing BAM files.
    :param metadata_path:
        Path to the metadata file containing information about the samples.
    :param nucleotide_threshold:
        Nucleotide coverage threshold.
    :param batch_size:
        Number of samples per batch.
    """
    if os.path.isdir(file_paths):  # If file_paths is a directory
        file_paths = [
            os.path.join(file_paths, f) for f in os.listdir(file_paths)
            if f.endswith('.bam')
        ]

    # TODO define metadata file when I have samples.
    # Load metadata
    metadata = pd.read_csv(metadata_path)

    while True:  # Loop forever so the generator never terminates
        batch_nucleotide_sequences = []
        batch_base_qualities = []
        batch_read_qualities = []
        batch_cigar_match = []
        batch_cigar_insertion = []
        batch_bitwise_flags = []
        batch_positions = []

        for _ in range(batch_size):
            file_path = random.choice(file_paths)
            # look up sex from metadata
            sex = metadata[
                    metadata['file_path'] == file_path.split('/')[-1]
                ]['sex'].values[0]

            chromosome, position = sample_positions(1, sex)

            read_dict = extract_reads_from_position_onward(
                file_path, chromosome, str(position), nucleotide_threshold
            )
            read_info = get_read_info(read_dict)

            for read in read_info.values():
                sequence_length = len(read['query_sequence'])
                batch_nucleotide_sequences.append(
                    read['query_sequence']
                )
                batch_base_qualities.append(
                    tf.cast(read['adjusted_base_qualities'], tf.int32)
                )
                batch_read_qualities.append(
                    tf.fill(
                        [sequence_length],
                        read['mapping_quality']
                    )
                )
                batch_cigar_match.append(
                    tf.cast(read['cigar_match_vector'], tf.int32)
                )
                batch_cigar_insertion.append(
                    tf.cast(read['cigar_insertion_vector'], tf.int32)
                )
                batch_bitwise_flags.append(
                    replicate_binary_flag_vector(
                        read['binary_flag_vector'],
                        nucleotide_threshold,
                        sequence_length
                    )
                )
                batch_positions.append(read['positions'])

            batch_nucleotide_sequences = tf.keras.preprocessing.sequence.pad_sequences(
                batch_nucleotide_sequences, maxlen=nucleotide_threshold,
                padding='post', truncating='post', dtype='str', value=''
            )
            batch_base_qualities = tf.keras.preprocessing.sequence.pad_sequences(
                batch_base_qualities, maxlen=nucleotide_threshold,
                padding='post', truncating='post'
            )
            batch_read_qualities = tf.keras.preprocessing.sequence.pad_sequences(
                batch_read_qualities, maxlen=nucleotide_threshold,
                padding='post', truncating='post'
            )
            batch_cigar_match = tf.keras.preprocessing.sequence.pad_sequences(
                batch_cigar_match, maxlen=nucleotide_threshold,
                padding='post', truncating='post'
            )
            batch_cigar_insertion = tf.keras.preprocessing.sequence.pad_sequences(
                batch_cigar_insertion, maxlen=nucleotide_threshold,
                padding='post', truncating='post'
            )
            batch_bitwise_flags = tf.keras.preprocessing.sequence.pad_sequences(
                batch_bitwise_flags, maxlen=nucleotide_threshold,
                padding='post', truncating='post'
            )
            batch_positions = tf.keras.preprocessing.sequence.pad_sequences(
                batch_positions, maxlen=nucleotide_threshold,
                padding='post', truncating='post', value=-1
            )

        yield (
            batch_nucleotide_sequences, batch_base_qualities,
            batch_read_qualities, batch_cigar_match, batch_cigar_insertion,
            batch_bitwise_flags, batch_positions
        )


def create_tf_dataset(file_paths, metadata_path, nucleotide_threshold, batch_size):
    """
    Create a TensorFlow dataset from the BAM read generator.

    :param file_paths:
        List of paths to BAM files, or a directory path containing BAM files.
    :param metadata_path:
        Path to the metadata file containing information about the samples.
    :param nucleotide_threshold:
        Nucleotide coverage threshold.
    :param batch_size:
        Number of samples per batch.
    """

    output_types = (
        tf.string, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32
    )
    # Define the output shapes, all outputs are [batch size, nucleotide_threshold]
    # except for the bitwise flags with shape [batch size, nucleotide_threshold, 12]
    output_shapes = (
        [batch_size, nucleotide_threshold], [batch_size, nucleotide_threshold],
        [batch_size, nucleotide_threshold], [batch_size, nucleotide_threshold],
        [batch_size, nucleotide_threshold],
        [batch_size, nucleotide_threshold, 12],
        [batch_size, nucleotide_threshold]
    )

    # Create a dataset from the generator
    dataset = tf.data.Dataset.from_generator(
        lambda: bam_read_generator(
            file_paths, metadata_path, nucleotide_threshold, batch_size
        ),
        output_types=output_types,
        output_shapes=output_shapes
    )

    return dataset
