import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from components.extract_reads import extract_read_by_id, get_read_info
from components.pretrain_data_streaming import (
    get_reverse_complement, encode_nucleotides, CustomBatch
)
import multiprocessing as mp
import logging


class PredictionDataset(Dataset):
    def __init__(
            self, csv_path, bam_path,
            base_quality_pad_idx,
            cigar_pad_idx, position_pad_idx, is_first_pad_idx,
            mapped_to_reverse_pad_idx,
            max_read_length=151, **kwargs
    ):
        self.data = pd.read_csv(csv_path)
        self.data = self.data.drop_duplicates(
            subset=['chrom', 'pos', 'ref', 'alt', 'read_id', 'mutation_type']
        ).reset_index(drop=True)
        self.bam_path = bam_path
        self.max_read_length = max_read_length
        self.base_quality_pad_idx = base_quality_pad_idx
        self.cigar_pad_idx = cigar_pad_idx
        self.position_pad_idx = position_pad_idx
        self.is_first_pad_idx = is_first_pad_idx
        self.mapped_to_reverse_pad_idx = mapped_to_reverse_pad_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        position = row['pos']
        chr_ = str(row['chrom'])
        read_id = row['read_id']
        mutation_type = row['mutation_type']
        extracted_read = extract_read_by_id(
            bam_file_path=self.bam_path,
            chromosome=chr_,
            position=position,
            read_id=read_id
        )

        if extracted_read is None:
            return None

        read_info = get_read_info(extracted_read)

        nucleotide_sequences = []
        base_qualities = []
        cigar_encoding = []
        # sequenced_from = []
        positions = []
        is_first_flags = []
        mapped_to_reverse_flags = []
        reference = []
        total_sequence_length = 0

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
                reference += get_reverse_complement(row['ref'])
            else:
                nucleotide_sequences += list(read['query_sequence'])
                base_qualities += read['base_qualities']
                cigar_encoding += read['cigar_encoding']
                positions += read['positions']
                reference += row['ref']

            sequence_length = len(read['query_sequence'])

            # Append flags per nucleotide
            is_first_flags += [0 if is_first else 1] * sequence_length
            mapped_to_reverse_flags += [1 if mapped_to_reverse else 0] * sequence_length

            total_sequence_length += sequence_length

            # Trim sequences to the nucleotide threshold
        if total_sequence_length > self.max_read_length:
            nucleotide_sequences = nucleotide_sequences[:self.max_read_length]
            base_qualities = base_qualities[:self.max_read_length]
            cigar_encoding = cigar_encoding[:self.max_read_length]
            positions = positions[:self.max_read_length]
            is_first_flags = is_first_flags[:self.max_read_length]
            mapped_to_reverse_flags = mapped_to_reverse_flags[:self.max_read_length]

            # Pad sequences to the max sequence length
        padding_length = self.max_read_length - len(nucleotide_sequences)
        nucleotide_sequences += [''] * padding_length
        base_qualities += [self.base_quality_pad_idx] * padding_length
        cigar_encoding += [self.cigar_pad_idx] * padding_length
        positions += [self.position_pad_idx] * padding_length
        is_first_flags += [self.is_first_pad_idx] * padding_length
        mapped_to_reverse_flags += [self.mapped_to_reverse_pad_idx] * padding_length

        return {
            # Sequences
            'nucleotide_sequences': nucleotide_sequences,
            'base_qualities': base_qualities,
            'cigar_encoding': cigar_encoding,
            'positions': positions,
            'is_first': is_first_flags,
            'mapped_to_reverse': mapped_to_reverse_flags,
            # Single values
            'mut_pos': position,
            'reference': reference,
            'is_reverse': mapped_to_reverse,
            # Added for output
            'chr': chr_,
            'read_id': read_id,
            'ref': row['ref'],
            'alt': row['alt'],
            'mutation_type': mutation_type
        }


def collate_fn(batch):

    batch = [item for item in batch if item is not None]

    if not batch:
        return None

    nucleotide_sequences = [
        encode_nucleotides(item['nucleotide_sequences']) for item in batch]
    base_qualities = [item['base_qualities'] for item in batch]
    cigar_encoding = [item['cigar_encoding'] for item in batch]
    positions = [item['positions'] for item in batch]
    is_first_flags = [item['is_first'] for item in batch]
    mapped_to_reverse_flags = [item['mapped_to_reverse'] for item in batch]
    # read_support = torch.tensor(
    #     [item['read_support'] for item in batch], dtype=torch.float32)
    # num_in_class = torch.tensor(
    #     [item['num_in_class'] for item in batch], dtype=torch.float32)
    # labels = torch.tensor(
    #     [item['label'] for item in batch], dtype=torch.float32)
    is_reverse = [item['is_reverse'] for item in batch]
    reference = [encode_nucleotides(item['reference']) for item in batch]
    mut_pos = [item['mut_pos'] for item in batch]
    chr_ = [item['chr'] for item in batch]
    read_id = [item['read_id'] for item in batch]
    ref = [item['ref'] for item in batch]
    alt = [item['alt'] for item in batch]
    mutation_type = [item['mutation_type'] for item in batch]

    # nucleotide_sequences = encode_nucleotides(nucleotide_sequences)
    nucleotide_sequences = torch.tensor(nucleotide_sequences, dtype=torch.int32)
    base_qualities = torch.tensor(base_qualities, dtype=torch.int32)
    cigar_encoding = torch.tensor(cigar_encoding, dtype=torch.int32)
    positions = torch.tensor(positions, dtype=torch.int32)
    is_first_flags = torch.tensor(is_first_flags, dtype=torch.int32)
    mapped_to_reverse_flags = torch.tensor(
        mapped_to_reverse_flags, dtype=torch.int32)
    reference = torch.tensor(reference, dtype=torch.int32)
    mut_pos = torch.tensor(mut_pos, dtype=torch.int32)

    batched_data = {
        # For model input:
        # Sequences
        'nucleotide_sequences': nucleotide_sequences,
        'base_qualities': base_qualities,
        'cigar_encoding': cigar_encoding,
        'positions': positions,
        'is_first': is_first_flags,
        'mapped_to_reverse': mapped_to_reverse_flags,
        # Single values
        'is_reverse': is_reverse,
        'reference': reference,
        'mut_pos': mut_pos,
        # For output (plus mut_pos):
        'chr': chr_,
        'read_id': read_id,
        'ref': ref,
        'alt': alt,
        'mutation_type': mutation_type
    }

    batch = CustomBatch(batched_data)

    return batch


def create_prediction_dataloader(
        csv_path, bam_path, batch_size,
        base_quality_pad_idx, cigar_pad_idx,
        position_pad_idx, is_first_pad_idx, mapped_to_reverse_pad_idx,
        max_read_length=100, shuffle=True, num_workers=0, prefetch_factor=None,
        collate_fn=collate_fn
):
    logging.info("Creating fine-tuning data loader.")
    dataset = PredictionDataset(
        csv_path, bam_path,
        base_quality_pad_idx, cigar_pad_idx,
        position_pad_idx, is_first_pad_idx, mapped_to_reverse_pad_idx,
        max_read_length=max_read_length
    )

    if len(dataset) > 0:
        logging.info("Dataset created successfully.")
    else:
        logging.info("Dataset is empty.")
        return None

    if torch.cuda.is_available() and num_workers != 0:
        multiprocessing_context = mp.get_context('spawn')
        logging.info("Using multiprocessing context: 'spawn'")
    else:
        multiprocessing_context = None
        logging.info("Multiprocessing context is None")

    if num_workers == 0:
        prefetch_factor = None

    logging.info("Creating DataLoader...")
    dataloader = DataLoader(
        dataset,
        shuffle=shuffle,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=(multiprocessing_context is not None),
        persistent_workers=(multiprocessing_context is not None),
        multiprocessing_context=multiprocessing_context
    )

    logging.info("DataLoader created successfully.")

    return dataloader


