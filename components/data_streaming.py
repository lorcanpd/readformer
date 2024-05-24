
import os
import random
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
            selected_positions=False
    ):
        """
        Initialise the dataset with file paths and metadata.

        :param file_paths:
            List of paths to BAM files or a directory containing BAM files.
        :param metadata_path:
            Path to the metadata file containing sample information.
        :param nucleotide_threshold:
            Nucleotide coverage threshold.
        """

        self.nucleotide_threshold = nucleotide_threshold
        self.metadata = pd.read_csv(metadata_path)
        self.selected_positions = selected_positions
        if os.path.isdir(file_paths):
            self.file_paths = [
                os.path.join(file_paths, f) for f in os.listdir(file_paths)
                if (f.endswith('.bam')
                    and f in self.metadata['file_path'].values)
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
                self.nucleotide_threshold
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
        padding_length = self.nucleotide_threshold - total_sequence_length
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
        self.data_source = data_source
        self.shuffle = shuffle

    def __iter__(self):
        idx = 0
        while True:
            yield idx % len(self.data_source)
            idx += 1

def create_data_loader(
        file_paths, metadata_path, nucleotide_threshold, batch_size,
        shuffle=True
):
    """
    Create a DataLoader for batch processing of BAM file reads.

    :param file_paths:
        Paths to the BAM files or directory containing them.
    :param metadata_path:
        Path to the metadata file.
    :param nucleotide_threshold:
        Max length of sequences to process.
    :param batch_size:
        Number of samples per batch.
    :param shuffle:
        Whether to shuffle the data.
    :return:
        DataLoader instance.
    """
    dataset = BAMReadDataset(file_paths, metadata_path, nucleotide_threshold)
    sampler = InfiniteSampler(dataset, shuffle)
    return DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn, sampler=sampler
    )



nucleotide_to_index = {
    '': -1, 'A': 0, 'C': 1, 'G': 2, 'T': 3, 'R': 4, 'Y': 5, 'S': 6, 'W': 7,
    'K': 8, 'M': 9, 'B': 10, 'D': 11, 'H': 12, 'V': 13, 'N': 14
}

def encode_nucleotides(sequence):
    return [nucleotide_to_index[nuc] for nuc in sequence]


def collate_fn(batch):
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


# Example usage
data_loader = create_data_loader(
    file_paths='TEST_DATA', metadata_path='TEST_DATA/test_metadata.csv',
    nucleotide_threshold=1024*8, batch_size=4, shuffle=True
)

# Iterate through data
for batch in data_loader:
    print(batch)  # Process your batch here
    break












## OLD TENSORFLOW IMPLEMENTATION ###
#
# import os
# import random
# import pandas as pd
# import tensorflow as tf
# from components.extract_reads import (
#     extract_reads_from_position_onward, sample_positions,  get_read_info
# )
#
#
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
#
#
# def random_bam_read_generator(
#         file_paths, metadata_path, nucleotide_threshold, batch_size,
#         max_retries=10
# ):
#     """
#     Generator function to yield batches of processed reads from BAM files, with
#     dynamic sampling of file paths, chromosomes, and positions.
#
#     :param file_paths:
#         List of paths to BAM files or a directory path containing BAM files.
#     :param metadata_path:
#         Path to the metadata file containing information about the samples.
#     :param nucleotide_threshold:
#         Nucleotide coverage threshold.
#     :param batch_size:
#         Number of samples per batch.
#     """
#     if os.path.isdir(file_paths):  # If file_paths is a directory
#         file_paths = [
#             os.path.join(file_paths, f) for f in os.listdir(file_paths)
#             if f.endswith('.bam')
#         ]
#
#     # TODO define metadata file when I have samples.
#     # Load metadata
#     metadata = pd.read_csv(metadata_path)
#
#     while True:  # Loop forever so the generator never terminates
#         batch_nucleotide_sequences = []
#         batch_base_qualities = []
#         batch_read_qualities = []
#         batch_cigar_match = []
#         batch_cigar_insertion = []
#         batch_bitwise_flags = []
#         batch_positions = []
#
#         for _ in range(batch_size):
#             file_path = random.choice(file_paths)
#             # look up sex from metadata
#             sex = metadata[
#                     metadata['file_path'] == file_path.split('/')[-1]
#                 ]['sex'].values[0]
#
#             retries = 0
#             while retries < max_retries:
#                 positions = sample_positions(1, sex)
#                 read_dict = extract_reads_from_position_onward(
#                     file_path, 'chr' + positions[0][0], positions[0][1],
#                     nucleotide_threshold
#                 )
#                 if not read_dict:  # If read_dict is empty
#                     retries += 1
#                 else:
#                     break
#
#             read_info = get_read_info(read_dict)
#
#             nucleotide_sequences = []
#             base_qualities = []
#             read_qualities = []
#             cigar_match = []
#             cigar_insertion = []
#             bitwise_flags = []
#             positions = []
#
#             for read in read_info.values():
#                 sequence_length = len(read['query_sequence'])
#                 nucleotide_sequences += list(read['query_sequence'])
#                 base_qualities += read['base_qualities']
#                 read_qualities += [read['mapping_quality']] * sequence_length
#                 cigar_match += read['cigar_match_vector']
#                 cigar_insertion += read['cigar_insertion_vector']
#                 bitwise_flags += replicate_binary_flag_vector_list(
#                         read['binary_flag_vector'],
#                         sequence_length
#                     )
#                 positions += read['positions']
#
#             batch_nucleotide_sequences.append(nucleotide_sequences)
#             batch_base_qualities.append(base_qualities)
#             batch_read_qualities.append(read_qualities)
#             batch_cigar_match.append(cigar_match)
#             batch_cigar_insertion.append(cigar_insertion)
#             batch_bitwise_flags.append(bitwise_flags)
#             batch_positions.append(positions)
#
#         batch_nucleotide_sequences = (
#             tf.keras.preprocessing.sequence.pad_sequences(
#             batch_nucleotide_sequences, maxlen=nucleotide_threshold,
#             padding='post', truncating='post', dtype='str', value=''
#         ))
#         batch_base_qualities = tf.keras.preprocessing.sequence.pad_sequences(
#             batch_base_qualities, maxlen=nucleotide_threshold,
#             padding='post', truncating='post', value=0.0, dtype='float32'
#         )
#         batch_read_qualities = tf.keras.preprocessing.sequence.pad_sequences(
#             batch_read_qualities, maxlen=nucleotide_threshold,
#             padding='post', truncating='post', value=0.0, dtype='float32'
#         )
#         batch_cigar_match = tf.keras.preprocessing.sequence.pad_sequences(
#             batch_cigar_match, maxlen=nucleotide_threshold,
#             padding='post', truncating='post', value=0.0, dtype='float32'
#         )
#         batch_cigar_insertion = tf.keras.preprocessing.sequence.pad_sequences(
#             batch_cigar_insertion, maxlen=nucleotide_threshold,
#             padding='post', truncating='post', value=0.0, dtype='float32'
#         )
#         batch_bitwise_flags = tf.keras.preprocessing.sequence.pad_sequences(
#             batch_bitwise_flags, maxlen=nucleotide_threshold,
#             padding='post', truncating='post', value=0.0, dtype='float32'
#         )
#         batch_positions = tf.keras.preprocessing.sequence.pad_sequences(
#             batch_positions, maxlen=nucleotide_threshold,
#             padding='post', truncating='post', value=-1
#         )
#
#         yield (
#             batch_nucleotide_sequences, batch_base_qualities,
#             batch_read_qualities, batch_cigar_match, batch_cigar_insertion,
#             batch_bitwise_flags, batch_positions
#         )
#
# # TODO: Develop a generator function for known mutations.
# def selected_bam_read_generator(
#         data_dir, vcf_path, nucleotide_threshold, batch_size,
#         max_retries=10
# ):
#     """
#     Generator function to yield batches of processed reads from BAM files, with
#     dynamic sampling of file paths, chromosomes, and positions.
#
#     :param file_paths:
#         List of paths to BAM files or a directory path containing BAM files.
#     :param metadata_path:
#         Path to the metadata file containing information about the samples.
#     :param nucleotide_threshold:
#         Nucleotide coverage threshold.
#     :param batch_size:
#         Number of samples per batch.
#     """
#     if os.path.isdir(data_dir):
#         file_paths = [
#             os.path.join(data_dir, f) for f in os.listdir(data_dir)
#             if f.endswith('.bam')
#         ]
#     else:
#         print('Data directory does not exist.')
#         raise NotADirectoryError
#
#
#     mutations = pd.read_csv(vcf_path)
#     # Only keep case_id, sample_id, Chr, Position, Ref and Alt columns.
#     mutations = mutations[
#         ['case_id', 'sample_id', 'Chr', 'Position', 'Ref', 'Alt']
#     ]
#
#     while True:  # Loop forever so the generator never terminates
#         batch_nucleotide_sequences = []
#         batch_base_qualities = []
#         batch_read_qualities = []
#         batch_cigar_match = []
#         batch_cigar_insertion = []
#         batch_bitwise_flags = []
#         batch_positions = []
#
#         # Select batch_size number of rows from the mutations dataframe
#         selected_mutations = mutations.sample(n=batch_size)
#
#         for index, row in selected_mutations.iterrows():
#             # TODO:
#             file_path = os.path.join(
#                 data_dir, row['case_id'], row['sample_id'] + '.bam'
#             )
#
#         for _ in range(batch_size):
#
#
#             retries = 0
#             while retries < max_retries:
#                 positions = sample_positions(1, sex)
#                 read_dict = extract_reads_from_position_onward(
#                     file_path, 'chr' + positions[0][0], positions[0][1],
#                     nucleotide_threshold
#                 )
#                 if not read_dict:  # If read_dict is empty
#                     retries += 1
#                 else:
#                     break
#
#             read_info = get_read_info(read_dict)
#
#             nucleotide_sequences = []
#             base_qualities = []
#             read_qualities = []
#             cigar_match = []
#             cigar_insertion = []
#             bitwise_flags = []
#             positions = []
#
#             for read in read_info.values():
#                 sequence_length = len(read['query_sequence'])
#                 nucleotide_sequences += list(read['query_sequence'])
#                 base_qualities += read['base_qualities']
#                 read_qualities += [read['mapping_quality']] * sequence_length
#                 cigar_match += read['cigar_match_vector']
#                 cigar_insertion += read['cigar_insertion_vector']
#                 bitwise_flags += replicate_binary_flag_vector_list(
#                         read['binary_flag_vector'],
#                         sequence_length
#                     )
#                 positions += read['positions']
#
#             batch_nucleotide_sequences.append(nucleotide_sequences)
#             batch_base_qualities.append(base_qualities)
#             batch_read_qualities.append(read_qualities)
#             batch_cigar_match.append(cigar_match)
#             batch_cigar_insertion.append(cigar_insertion)
#             batch_bitwise_flags.append(bitwise_flags)
#             batch_positions.append(positions)
#
#         batch_nucleotide_sequences = (
#             tf.keras.preprocessing.sequence.pad_sequences(
#             batch_nucleotide_sequences, maxlen=nucleotide_threshold,
#             padding='post', truncating='post', dtype='str', value=''
#         ))
#         batch_base_qualities = tf.keras.preprocessing.sequence.pad_sequences(
#             batch_base_qualities, maxlen=nucleotide_threshold,
#             padding='post', truncating='post', value=0.0, dtype='float32'
#         )
#         batch_read_qualities = tf.keras.preprocessing.sequence.pad_sequences(
#             batch_read_qualities, maxlen=nucleotide_threshold,
#             padding='post', truncating='post', value=0.0, dtype='float32'
#         )
#         batch_cigar_match = tf.keras.preprocessing.sequence.pad_sequences(
#             batch_cigar_match, maxlen=nucleotide_threshold,
#             padding='post', truncating='post', value=0.0, dtype='float32'
#         )
#         batch_cigar_insertion = tf.keras.preprocessing.sequence.pad_sequences(
#             batch_cigar_insertion, maxlen=nucleotide_threshold,
#             padding='post', truncating='post', value=0.0, dtype='float32'
#         )
#         batch_bitwise_flags = tf.keras.preprocessing.sequence.pad_sequences(
#             batch_bitwise_flags, maxlen=nucleotide_threshold,
#             padding='post', truncating='post', value=0.0, dtype='float32'
#         )
#         batch_positions = tf.keras.preprocessing.sequence.pad_sequences(
#             batch_positions, maxlen=nucleotide_threshold,
#             padding='post', truncating='post', value=-1
#         )
#
#         yield (
#             batch_nucleotide_sequences, batch_base_qualities,
#             batch_read_qualities, batch_cigar_match, batch_cigar_insertion,
#             batch_bitwise_flags, batch_positions
#         )
#
#
#
# def create_tf_dataset(
#         file_paths, metadata_path, nucleotide_threshold, batch_size
# ):
#     """
#     Create a TensorFlow dataset from the BAM read generator.
#
#     :param file_paths:
#         List of paths to BAM files, or a directory path containing BAM files.
#     :param metadata_path:
#         Path to the metadata file containing information about the samples.
#     :param nucleotide_threshold:
#         Nucleotide coverage threshold.
#     :param batch_size:
#         Number of samples per batch.
#     """
#
#     output_signature = (
#         tf.TensorSpec(
#             shape=(batch_size, nucleotide_threshold), dtype=tf.string,
#             name='nucleotide_sequences'
#         ),
#         tf.TensorSpec(
#             shape=(batch_size, nucleotide_threshold), dtype=tf.float32,
#             name='base_qualities'
#         ),
#         tf.TensorSpec(
#             shape=(batch_size, nucleotide_threshold), dtype=tf.float32,
#             name='read_qualities'
#         ),
#         tf.TensorSpec(
#             shape=(batch_size, nucleotide_threshold), dtype=tf.float32,
#             name='cigar_match'
#         ),
#         tf.TensorSpec(
#             shape=(batch_size, nucleotide_threshold), dtype=tf.float32,
#             name='cigar_insertion'
#         ),
#         tf.TensorSpec(
#             shape=(batch_size, nucleotide_threshold, 12), dtype=tf.float32,
#             name='bitwise_flags'
#         ),
#         tf.TensorSpec(
#             shape=(batch_size, nucleotide_threshold), dtype=tf.int32,
#             name='positions'
#         ),
#     )
#
#     dataset = tf.data.Dataset.from_generator(
#         lambda: random_bam_read_generator(
#             file_paths, metadata_path, nucleotide_threshold, batch_size
#         ),
#         output_signature=output_signature
#     )
#
#     return dataset
#
