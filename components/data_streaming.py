
import os
# import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, get_worker_info
from components.extract_reads import (
    extract_reads_from_position_onward, sample_positions, get_read_info
)
import multiprocessing as mp
import logging

logging.basicConfig(level=logging.DEBUG)


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
                read_dict = extract_reads_from_position_onward(
                    file_path, 'chr' + positions[0][0], positions[0][1],
                    self.nucleotide_threshold, self.min_quality
                )
            except Exception as e:
                logging.error(f"Error extracting reads: {e}")
                retries += 1
                continue
            if read_dict:
                read_info = get_read_info(read_dict)
                break
            retries += 1
        else:
            logging.warning("Max retries reached, returning None")
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
        nucleotide_sequences += [''] * padding_length
        base_qualities += [0.0] * padding_length
        read_qualities += [0.0] * padding_length
        cigar_match += [0.0] * padding_length
        cigar_insertion += [0.0] * padding_length
        bitwise_flags += [[0.0]*12] * padding_length
        positions += [-1] * padding_length

        logging.debug(
            f"Returning batch, number of nucleotide sequences: {len(nucleotide_sequences)}"
        )
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


def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    if worker_info is not None:
        logging.info(f"Initialising worker {worker_info.id}/{worker_info.num_workers}")

        # Pin each worker to a specific CPU core if supported
        try:
            core_id = worker_id % os.cpu_count()
            if hasattr(os, 'sched_setaffinity'):
                os.sched_setaffinity(0, {core_id})
                logging.info(f"Worker {worker_info.id} is pinned to CPU core {core_id}")
            else:
                logging.warning("CPU affinity setting not supported on this OS.")
        except Exception as e:
            logging.error(f"Error setting CPU affinity: {e}")

        # Ensure CUDA is initialized in each worker
        try:
            if torch.cuda.is_available():
                device_id = worker_info.id % torch.cuda.device_count()
                torch.cuda.set_device(device_id)
                logging.info(f"Worker {worker_info.id} is using device {device_id}")
        except Exception as e:
            logging.error(f"Error setting CUDA device: {e}")
    else:
        logging.info("Initialising main process")


def create_data_loader(
        file_paths, metadata_path, nucleotide_threshold, max_sequence_length,
        batch_size, min_quality, shuffle=True, num_workers=0, prefetch_factor=None
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
    print("Creating BAMReadDataset...")
    dataset = BAMReadDataset(
        file_paths, metadata_path, nucleotide_threshold, max_sequence_length,
        min_quality=min_quality
    )

    if len(dataset) > 0:
        print("Dataset created successfully")
    else:
        print("Dataset is empty")
        return None

    print("Creating InfiniteSampler...")
    sampler = InfiniteSampler(dataset, shuffle)
    print("InfiniteSampler created successfully")

    if torch.cuda.is_available() and num_workers is None:
        multiprocessing_context = mp.get_context('spawn')
        print("Using multiprocessing context: 'spawn'")
    else:
        multiprocessing_context = None
        print("Multiprocessing context is None")

    print("Creating DataLoader...")
    data_loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn, sampler=sampler,
        num_workers=num_workers, prefetch_factor=prefetch_factor,
        pin_memory=(multiprocessing_context is not None),
        worker_init_fn=worker_init_fn,
        multiprocessing_context=multiprocessing_context,
        persistent_workers=(multiprocessing_context is not None)
    )

    print("DataLoader created successfully")

    return data_loader


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

#
# def collate_fn(batch):
#     """
#     Collate function to process and batch the data samples.
#
#     :param batch:
#         A list of dictionaries where each dictionary represents a data sample.
#     :returns:
#         A dictionary of batched tensors, or None if the batch is empty after
#         filtering.
#     """
#     # Filter out None samples
#     batch = [x for x in batch if x is not None]
#
#     # Handle edge case where batch might be empty after filtering
#     if not batch:
#         return None
#
#     # Creating batch tensors for each key in the dictionary
#     batched_data = {}
#
#     # Iterate over the keys in a sample's dictionary
#     for key in batch[0]:
#         if key == 'nucleotide_sequences':
#             # Encode nucleotide sequences and convert them to tensor
#             encoded_sequences = [encode_nucleotides(b[key]) for b in batch]
#             batched_data[key] = torch.tensor(
#                 encoded_sequences, dtype=torch.int32
#             )
#         else:
#             # Assume other keys are already appropriate for conversion to tensor
#             batched_data[key] = torch.tensor(
#                 [b[key] for b in batch],
#                 dtype=torch.float32 if isinstance(batch[0][key][0], float)
#                 else torch.int32
#             )
#
#     return batched_data


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

    batch = CustomBatch(batched_data)

    return batch
