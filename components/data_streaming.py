
import os
# import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from components.extract_reads import (
    extract_reads_from_position_onward, sample_positions, get_read_info
)


def replicate_binary_flag_vector_list(
        binary_flag_vector, sequence_length
):
    """
    Replicates a binary flag vector for each nucleotide position in the sequence
    using lists.

    :param binary_flag_vector:
        The binary flag vector for a read.
    :param sequence_length:
        The actual sequence length of the read.
    :return:
        A replicated binary flag vector of shape (sequence_length, vector_size).
    """
    # Initialize the replicated list with zeros
    replicated_vector = [
        [0] * len(binary_flag_vector) for _ in range(sequence_length)
    ]

    # Fill the replicated list with the binary_flag_vector values up to the
    # sequence length
    for i in range(sequence_length):
        replicated_vector[i] = binary_flag_vector.copy()

    return replicated_vector


# TODO: Write data streaming code for specific mutation positions.
class BAMReadDataset(Dataset):
    """
    A PyTorch Dataset to manage randomly selecting a genomic position and
    sampling reads from BAM files.
    """
    def __init__(
            self, file_paths, metadata_path, nucleotide_threshold,
            max_sequence_length, selected_positions=False, min_quality=0
    ):
        """
        Initialise the dataset with file paths and metadata.

        :param file_paths:
            List of paths to BAM files or a directory containing BAM files.
        :param metadata_path:
            Path to the metadata file containing sample information.
        :param nucleotide_threshold:
            Nucleotide coverage threshold. This is the maximum sequence length
            to be generated.
        :param max_sequence_length:
            The sequence length dimension the model expects. Sequences will be
            padded to this length.
        """

        self.nucleotide_threshold = nucleotide_threshold
        self.max_sequence_length = max_sequence_length
        self.metadata = pd.read_csv(metadata_path)
        self.selected_positions = selected_positions
        self.min_quality = min_quality
        if os.path.isdir(file_paths):
            basenames = set(self.metadata['file_path'].apply(os.path.basename))
            self.file_paths = [
                os.path.join(file_paths, f) for f in os.listdir(file_paths)
                if (f.endswith('.bam') and f in basenames)
            ]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # To ensure circular sampling, we reset the index to 0 if it exceeds
        # the number of file paths
        file_path = self.file_paths[idx % len(self.file_paths)]
        sex = self.metadata[
            self.metadata['file_path'] == file_path.split('/')[-1]
            ]['sex'].values[0]
        retries = 0
        max_retries = 10

        while retries < max_retries:
            positions = sample_positions(1, sex)
            read_dict = extract_reads_from_position_onward(
                file_path, 'chr' + positions[0][0], positions[0][1],
                self.nucleotide_threshold, self.min_quality
            )
            if read_dict:
                read_info = get_read_info(read_dict)
                break
            retries += 1
        else:
            # Return None if no valid read is found after all retries
            return None

        nucleotide_sequences = []
        base_qualities = []
        read_qualities = []
        cigar_match = []
        cigar_insertion = []
        bitwise_flags = []
        positions = []
        total_sequence_length = 0

        # Parse read data
        for read in read_info.values():
            sequence_length = len(read['query_sequence'])
            nucleotide_sequences += list(read['query_sequence'])
            base_qualities += read['base_qualities']
            read_qualities += [read['mapping_quality']] * sequence_length
            cigar_match += read['cigar_match_vector']
            cigar_insertion += read['cigar_insertion_vector']
            bitwise_flags += replicate_binary_flag_vector_list(
                read['binary_flag_vector'],
                sequence_length
            )
            positions += read['positions']
            total_sequence_length += sequence_length

        # Pad sequences to the threshold length
        padding_length = self.max_sequence_length - total_sequence_length
        nucleotide_sequences += [''] * (padding_length)
        base_qualities += [0.0] * (padding_length)
        read_qualities += [0.0] * (padding_length)
        cigar_match += [0.0] * (padding_length)
        cigar_insertion += [0.0] * (padding_length)
        bitwise_flags += [[0.0]*12] * (padding_length)
        positions += [-1] * (padding_length)

        return {
            'nucleotide_sequences': nucleotide_sequences,
            'base_qualities': base_qualities,
            'read_qualities': read_qualities,
            'cigar_match': cigar_match,
            'cigar_insertion': cigar_insertion,
            'bitwise_flags': bitwise_flags,
            'positions': positions
        }


class InfiniteSampler(Sampler):
    def __init__(self, data_source, shuffle=True):
        super().__init__(data_source)
        self.data_source = data_source
        self.shuffle = shuffle

    def __iter__(self):
        idx = 0
        while True:
            yield idx % len(self.data_source)
            idx += 1


def create_data_loader(
        file_paths, metadata_path, nucleotide_threshold, max_sequence_length,
        batch_size, min_quality, shuffle=True, num_workers=4, prefetch_factor=2
):
    """
    Create a DataLoader for batch processing of BAM file reads.

    :param file_paths:
        Paths to the BAM files or directory containing them.
    :param metadata_path:
        Path to the metadata file.
    :param nucleotide_threshold:
        Max length of sequences to process.
    :param max_sequence_length:
        Max sequence length to pad to.
    :param batch_size:
        Number of samples per batch.
    :param shuffle:
        Whether to shuffle the data.
    :return:
        DataLoader instance.
    """
    dataset = BAMReadDataset(
        file_paths, metadata_path, nucleotide_threshold, max_sequence_length,
        min_quality=min_quality
    )
    sampler = InfiniteSampler(dataset, shuffle)
    return DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn, sampler=sampler,
        num_workers=num_workers, prefetch_factor=prefetch_factor
    )


nucleotide_to_index = {
    'A': 0, 'C': 1, 'G': 2, 'T': 3, 'R': 4, 'Y': 5, 'S': 6, 'W': 7,
    'K': 8, 'M': 9, 'B': 10, 'D': 11, 'H': 12, 'V': 13, 'N': 14, '': 15
}


def encode_nucleotides(sequence):
    """
    Encode a nucleotide sequence into a list of integer indices.

    :param sequence:
        A string representing the nucleotide sequence.
    :returns:
        A list of integer indices corresponding to the nucleotide sequence.
    """
    # Handle case where base is not in the lookup table
    return [nucleotide_to_index.get(nuc, 15) for nuc in sequence]


def collate_fn(batch):
    """
    Collate function to process and batch the data samples.

    :param batch:
        A list of dictionaries where each dictionary represents a data sample.
    :returns:
        A dictionary of batched tensors, or None if the batch is empty after
        filtering.
    """
    # Filter out None samples
    batch = [x for x in batch if x is not None]

    # Handle edge case where batch might be empty after filtering
    if not batch:
        return None

    # Creating batch tensors for each key in the dictionary
    batched_data = {}

    # Iterate over the keys in a sample's dictionary
    for key in batch[0]:
        if key == 'nucleotide_sequences':
            # Encode nucleotide sequences and convert them to tensor
            encoded_sequences = [encode_nucleotides(b[key]) for b in batch]
            batched_data[key] = torch.tensor(
                encoded_sequences, dtype=torch.int32
            )
        else:
            # Assume other keys are already appropriate for conversion to tensor
            batched_data[key] = torch.tensor(
                [b[key] for b in batch],
                dtype=torch.float32 if isinstance(batch[0][key][0], float)
                else torch.int32
            )

    return batched_data

