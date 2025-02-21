
import os
# import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, get_worker_info
from components.extract_reads import (
    extract_random_read_from_position, sample_positions, get_read_info
)
import multiprocessing as mp
import logging


# def replicate_binary_flag_vector_list(
#         binary_flag_vector, sequence_length
# ):
#     """
#     Replicates a binary flag vector for each nucleotide position in the sequence
#     using lists.
#
#     :param binary_flag_vector:
#         The binary flag vector for a read.
#     :param sequence_length:
#         The actual sequence length of the read.
#     :return:
#         A replicated binary flag vector of shape (sequence_length, vector_size).
#     """
#     # Initialize the replicated list with zeros
#     replicated_vector = [
#         [0] * len(binary_flag_vector) for _ in range(sequence_length)
#     ]
#
#     # Fill the replicated list with the binary_flag_vector values up to the
#     # sequence length
#     for i in range(sequence_length):
#         replicated_vector[i] = binary_flag_vector.copy()
#
#     return replicated_vector


# TODO: Write data streaming code for specific mutation positions.
class BAMReadDataset(Dataset):
    """
    A PyTorch Dataset to manage randomly selecting a genomic position and
    sampling reads from BAM files.
    """

    def __init__(
            self, file_paths, metadata_path, nucleotide_threshold,
            max_sequence_length, base_quality_pad_idx,
            cigar_pad_idx, position_pad_idx, is_first_pad_idx,
            mapped_to_reverse_pad_idx,
            selected_positions=False, min_quality=0
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
        self.base_quality_pad_idx = base_quality_pad_idx
        self.cigar_pad_idx = cigar_pad_idx
        self.position_pad_idx = position_pad_idx
        self.is_first_pad_idx = is_first_pad_idx
        self.mapped_to_reverse_pad_idx = mapped_to_reverse_pad_idx
        self.selected_positions = selected_positions
        self.min_quality = min_quality
        if os.path.isdir(file_paths):
            basenames = set(self.metadata['file_path'].apply(os.path.basename))
            self.file_paths = [
                os.path.join(file_paths, f) for f in os.listdir(file_paths)
                if (f.endswith('.bam') and f in basenames)
            ]
        logging.info("BAMReadDataset initialised successfully")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx % len(self.file_paths)]
        sex = self.metadata[
            self.metadata['file_path'] == file_path]['sex'].values[0]
        retries = 0
        max_retries = 10

        while retries < max_retries:
            positions = sample_positions(1, sex)
            try:
                read_dict = extract_random_read_from_position(
                    file_path, 'chr' + positions[0][0], positions[0][1],
                    self.min_quality
                )
            except Exception as e:
                logging.error(f"Error extracting reads: {e}")
                retries += 1
                continue

            if read_dict:
                try:
                    read_info = get_read_info(read_dict)
                    break
                except TypeError as e:
                    logging.error(f"Error getting read info: {e}")
                    retries += 1
                    continue
            retries += 1

        if retries == max_retries:
            logging.warning("Max retries reached, returning None")
            return None

        nucleotide_sequences = []
        base_qualities = []
        cigar_encoding = []
        positions = []
        is_first_flags = []
        mapped_to_reverse_flags = []
        total_sequence_length = 0

        # Parse read data
        for read in read_info.values():
            is_first = read['orientation_flags'].get('is_first_in_pair', False)
            mapped_to_reverse = read['orientation_flags'].get('is_reverse', False)

            if mapped_to_reverse:
                # Get the reverse complement of the sequence and reverse the
                # order of the nucleotides, base qualities, cigar encoding, and
                # positions
                nucleotide_sequences += list(
                    get_reverse_complement(read['query_sequence']))[::-1]
                base_qualities += read['base_qualities'][::-1]
                cigar_encoding += read['cigar_encoding'][::-1]
                positions += read['positions'][::-1]
            else:
                nucleotide_sequences += list(read['query_sequence'])
                base_qualities += read['base_qualities']
                cigar_encoding += read['cigar_encoding']
                positions += read['positions']

            sequence_length = len(read['query_sequence'])

            # Append flags per nucleotide
            is_first_flags += [0 if is_first else 1] * sequence_length
            mapped_to_reverse_flags += [1 if mapped_to_reverse else 0] * sequence_length

            total_sequence_length += sequence_length

        # Trim sequences to the nucleotide threshold
        if total_sequence_length > self.nucleotide_threshold:
            nucleotide_sequences = nucleotide_sequences[:self.nucleotide_threshold]
            base_qualities = base_qualities[:self.nucleotide_threshold]
            cigar_encoding = cigar_encoding[:self.nucleotide_threshold]
            positions = positions[:self.nucleotide_threshold]
            is_first_flags = is_first_flags[:self.nucleotide_threshold]
            mapped_to_reverse_flags = mapped_to_reverse_flags[:self.nucleotide_threshold]

        # Pad sequences to the max sequence length
        padding_length = self.max_sequence_length - len(nucleotide_sequences)
        nucleotide_sequences += [''] * padding_length
        base_qualities += [self.base_quality_pad_idx] * padding_length
        cigar_encoding += [self.cigar_pad_idx] * padding_length
        positions += [self.position_pad_idx] * padding_length
        is_first_flags += [self.is_first_pad_idx] * padding_length
        mapped_to_reverse_flags += [self.mapped_to_reverse_pad_idx] * padding_length

        logging.debug(
            f"Returning batch, number of nucleotide sequences: {len(nucleotide_sequences)}"
        )

        return {
            'nucleotide_sequences': nucleotide_sequences,
            'base_qualities': base_qualities,
            'cigar_encoding': cigar_encoding,
            'positions': positions,
            'is_first': is_first_flags,
            'mapped_to_reverse': mapped_to_reverse_flags
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


def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    if worker_info is not None:
        logging.info(f"Initialising worker {worker_info.id}/{worker_info.num_workers}")


def create_data_loader(
        file_paths, metadata_path, nucleotide_threshold, max_sequence_length,
        batch_size, min_quality, base_quality_pad_idx, cigar_pad_idx,
        position_pad_idx, is_first_pad_idx, mapped_to_reverse_pad_idx,
        shuffle=True, num_workers=0, prefetch_factor=None,

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
    logging.info("Creating BAMReadDataset...")
    dataset = BAMReadDataset(
        file_paths, metadata_path, nucleotide_threshold, max_sequence_length,
        base_quality_pad_idx, cigar_pad_idx, position_pad_idx, is_first_pad_idx,
        mapped_to_reverse_pad_idx,
        min_quality=min_quality
    )

    if len(dataset) > 0:
        logging.info("Dataset created successfully")
    else:
        logging.error("Dataset is empty")
        return None

    logging.info("Creating InfiniteSampler...")
    sampler = InfiniteSampler(dataset, shuffle)
    logging.info("InfiniteSampler created successfully")

    if torch.cuda.is_available() and num_workers != 0:
        multiprocessing_context = mp.get_context('spawn')
        logging.info("Using multiprocessing context: 'spawn'")
    else:
        multiprocessing_context = None
        logging.info("Multiprocessing context is None")

    if num_workers == 0:
        prefetch_factor = None

    logging.info("Creating DataLoader...")
    data_loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn, sampler=sampler,
        num_workers=num_workers, prefetch_factor=prefetch_factor,
        # This should allow two DataLoader instances to share the same workers.
        # Useful for having a separate DataLoader for validation data.
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        multiprocessing_context=multiprocessing_context,
        persistent_workers=(multiprocessing_context is not None)
    )

    logging.info("DataLoader created successfully")

    return data_loader


nucleotide_to_index = {
    'A': 0, 'C': 1, 'G': 2, 'T': 3, 'R': 4, 'Y': 5, 'S': 6, 'W': 7,
    'K': 8, 'M': 9, 'B': 10, 'D': 11, 'H': 12, 'V': 13, 'N': 14, '': 15
}

reverse_complement = {
    'A': 'T',  # Adenine pairs with Thymine
    'C': 'G',  # Cytosine pairs with Guanine
    'G': 'C',  # Guanine pairs with Cytosine
    'T': 'A',  # Thymine pairs with Adenine
    'R': 'Y',  # Purine (A or G) pairs with Pyrimidine (T or C)
    'Y': 'R',  # Pyrimidine (T or C) pairs with Purine (A or G)
    'S': 'S',  # Strong interaction (C or G), reverse complement is itself
    'W': 'W',  # Weak interaction (A or T), reverse complement is itself
    'K': 'M',  # Keto (G or T) pairs with Amino (A or C)
    'M': 'K',  # Amino (A or C) pairs with Keto (G or T)
    'B': 'V',  # Not A (C or G or T) pairs with Not T (A or C or G)
    'D': 'H',  # Not C (A or G or T) pairs with Not G (A or T or C)
    'H': 'D',  # Not G (A or C or T) pairs with Not C (A or G or T)
    'V': 'B',  # Not T (A or C or G) pairs with Not A (C or G or T)
    'N': 'N',  # Any base (A, C, G, T), reverse complement is itself
    '': ''     # Empty character, remains unchanged
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


def get_reverse_complement(sequence):
    """
    Get the reverse complement of a nucleotide sequence.

    :param sequence:
        A string representing the nucleotide sequence.
    :returns:
        The reverse complement of the input sequence.
    """
    return ''.join(reverse_complement[nuc] for nuc in sequence)


class CustomBatch:
    def __init__(self, batch_data):
        self.batch_data = batch_data

    def pin_memory(self):
        for key in self.batch_data:
            if isinstance(self.batch_data[key], torch.Tensor):
                self.batch_data[key] = self.batch_data[key].pin_memory()
        return self

    def __getitem__(self, item):
        return self.batch_data[item]

    def keys(self):
        return self.batch_data.keys()

    def items(self):
        return self.batch_data.items()


def collate_fn(batch):
    """
    Collate function to process and batch the data samples.
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
            # Assume other keys are appropriate for conversion to tensor
            sequence = [b[key] for b in batch]
            dtype = torch.int32 if isinstance(sequence[0][0], int) else torch.float32
            batched_data[key] = torch.tensor(sequence, dtype=dtype)
    batch = CustomBatch(batched_data)

    return batch


