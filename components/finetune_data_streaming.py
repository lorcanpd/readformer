import pandas as pd
import random
import math
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from components.extract_reads import extract_read_by_id, get_read_info
from components.pretrain_data_streaming import (
    get_reverse_complement, encode_nucleotides, CustomBatch
)
import multiprocessing as mp
import logging
import numpy as np
from collections import deque, defaultdict
from typing import Iterator, List


class BalancedBatchSampler(Sampler[List[int]]):
    """
    Yield balanced (pos,neg) pairs until `pos_pool` empties.
    """

    # ---------------------------------------------------------------------
    def __init__(self,
                 df: pd.DataFrame,
                 batch_size: int,
                 positive_label: float = 1.0,
                 prop: float = 0.10):

        if batch_size % 2:
            raise ValueError("`batch_size` must be even (pairs of indices required)")

        super().__init__(df)

        # ---------- batch-composition constants --------------------------
        self.pairs_per_batch = batch_size // 2
        self.n_mut = int(round(self.pairs_per_batch * prop))   # mutation-led pairs
        self.n_art = self.n_mut                                # artefact-led pairs
        self.n_rand = self.pairs_per_batch - self.n_mut - self.n_art

        # ---------- build index arrays -----------------------------------
        # pos_mask = df['label'].to_numpy() == positive_label
        pos_mask = (df.data['label'].to_numpy() == positive_label)
        self.pos_idx = np.flatnonzero(pos_mask)
        self.neg_idx = np.flatnonzero(~pos_mask)

        if not len(self.pos_idx) or not len(self.neg_idx):
            raise ValueError("Need at least one positive **and** one negative row")

        # vectorised key arrays for fast equality checks
        keys = df.data[['REF', 'mapped_to_reverse', 'mutation_type']].to_numpy()
        self.pos_keys = keys[self.pos_idx]
        self.neg_keys = keys[self.neg_idx]

        self.idx2key = {
            idx: tuple(k) for idx, k in
            zip(
                np.concatenate([self.pos_idx, self.neg_idx]),
                np.concatenate([self.pos_keys, self.neg_keys])
            )
        }

        # bucket builders (called each epoch)
        def _buckets(idxs, key_arr):
            d = defaultdict(deque)
            for idx, k in zip(idxs, map(tuple, key_arr)):
                d[k].append(idx)
            return d
        self._build_pos_buckets = lambda: _buckets(self.pos_idx, self.pos_keys)
        self._build_neg_buckets = lambda: _buckets(self.neg_idx, self.neg_keys)

        self.num_batches = math.ceil(len(self.pos_idx) / self.pairs_per_batch)

    # ---------------------------------------------------------------------
    def __len__(self) -> int:
        # only used by progress-bars – not critical to logic
        return self.num_batches

    # ---------------------------------------------------------------------
    def __iter__(self) -> Iterator[List[int]]:
        """
        One epoch:
          * shuffle both pools once;
          * keep pairing until `pos_pool` is empty;
          * yield *smaller* final batch if that is all we can form.
        """
        rng = np.random.default_rng()

        # fresh, shuffled copies
        pos_pool = deque(rng.permutation(self.pos_idx))
        neg_pool = deque(rng.permutation(self.neg_idx))
        pos_buckets = self._build_pos_buckets()
        neg_buckets = self._build_neg_buckets()

        while pos_pool:                                  # ← sole stopping rule
            batch: List[int] = []
            for _ in range(self.n_mut):
                if not pos_pool:
                    break
                p = pos_pool.popleft()
                k = self.idx2key[int(p)]

                # ----------- ensure the bucket head is unused -------------------- #
                while neg_buckets[k] and neg_buckets[k][0] not in neg_pool:
                    neg_buckets[k].popleft()

                if neg_buckets[k]:
                    n = neg_buckets[k].popleft()
                    neg_pool.remove(n)
                elif neg_pool:
                    n = neg_pool.popleft()
                else:
                    break
                batch += [p, n]

            # ---------- artefact-led pairs -------------------------------------- #
            for _ in range(self.n_art):
                if not pos_pool or not neg_pool:
                    break
                n = neg_pool.popleft()
                k = self.idx2key[int(n)]

                # ----------- ensure the bucket head is unused -------------------- #
                while pos_buckets[k] and pos_buckets[k][0] not in pos_pool:
                    pos_buckets[k].popleft()

                if pos_buckets[k]:
                    p = pos_buckets[k].popleft()
                    pos_pool.remove(p)
                else:
                    p = pos_pool.popleft()
                batch += [n, p]

            # ----- random pairs ------------------------------------------
            k = min(self.n_rand, len(pos_pool), len(neg_pool))
            if k:
                batch += rng.choice(pos_pool, k, replace=False).tolist()
                batch += rng.choice(neg_pool, k, replace=False).tolist()

            # Keep batch length even (balanced). If we happened to pull an
            # extra index because one pool ran out mid-loop, just drop it.
            if len(batch) % 2:
                batch.pop()

            # No pairs ⇒ epoch finished
            if len(batch) < 2:
                break

            yield rng.permutation(batch).tolist()


# class PriorityBuffer:
#     def __init__(self, capacity=100_000, alpha=0.6):
#         self.capacity = capacity
#         self.alpha = alpha
#         self.data, self.prior = [], []
#
#     def add(self, idx, td_err):
#         p = (abs(td_err) + 1e-6) ** self.alpha
#         if idx in self.data:
#             i = self.data.index(idx)
#             self.prior[i] = p
#         elif len(self.data) < self.capacity:
#             self.data.append(idx)
#             self.prior.append(p)
#         else:
#             i = np.argmin(self.prior)
#             self.data[i] = idx
#             self.prior[i] = p
#
#     def sample(self, k):
#         prob = np.array(self.prior) / np.sum(self.prior)
#         return list(np.random.choice(self.data, k, p=prob, replace=False))

#
# class HybridSampler(Sampler):
#     """N = batch; keep n_balanced balanced samples; n_per from PER."""
#
#     def __init__(self, balanced_sampler, per_buffer, n_balanced, n_per):
#         super().__init__()
#         self.bal = balanced_sampler
#         self.per_buf = per_buffer
#         self.n_bal = n_balanced
#         self.n_per = n_per
#
#     def __iter__(self):
#         bal_iter = iter(self.bal)
#         while True:
#             try:
#                 fresh = next(bal_iter)[:self.n_bal]
#             except StopIteration:
#                 break
#             if len(self.per_buf.data) >= self.n_per:
#                 per = self.per_buf.sample(self.n_per)
#             else:
#                 per = fresh[:self.n_per]
#             yield fresh + per
#
#     def __len__(self):
#         return len(self.bal)


class FineTuningDataset(Dataset):
    def __init__(
            self, csv_path, artefact_bam_path, mutation_bam_path,
            base_quality_pad_idx,
            cigar_pad_idx, position_pad_idx, is_first_pad_idx,
            mapped_to_reverse_pad_idx, # batch_size,
            max_read_length=151,  **kwargs
    ):
        self.data = pd.read_csv(csv_path)
        self.artefact_bam_path = artefact_bam_path
        self.mutation_bam_path = mutation_bam_path
        self.max_read_length = max_read_length
        self.base_quality_pad_idx = base_quality_pad_idx
        self.cigar_pad_idx = cigar_pad_idx
        self.position_pad_idx = position_pad_idx
        self.is_first_pad_idx = is_first_pad_idx
        self.mapped_to_reverse_pad_idx = mapped_to_reverse_pad_idx
        # self.batch_size = batch_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        bam_path = self.mutation_bam_path if row['label'] == 1.0 else self.artefact_bam_path
        position = row['POS']
        chr_ = str(row['CHROM'])
        read_id = row['READ_ID']
        extracted_read = extract_read_by_id(
            bam_file_path=bam_path,
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
                reference += get_reverse_complement(row['REF'])
            else:
                nucleotide_sequences += list(read['query_sequence'])
                base_qualities += read['base_qualities']
                cigar_encoding += read['cigar_encoding']
                positions += read['positions']
                reference += row['REF']

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
            'is_reverse': row['mapped_to_reverse'],
            'read_support': row['read_support'],
            'num_in_class': row['num_in_class'],
            'label': row['label'],
            # 'trinucleotide_context': row['trinucleotide_context'],
            # Added for output
            'chr': chr_,
            'read_id': read_id,
            'ref': row['REF'],
            'alt': row['ALT']
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
    read_support = torch.tensor(
        [item['read_support'] for item in batch], dtype=torch.float32)
    num_in_class = torch.tensor(
        [item['num_in_class'] for item in batch], dtype=torch.float32)
    labels = torch.tensor(
        [item['label'] for item in batch], dtype=torch.float32)
    is_reverse = [item['is_reverse'] for item in batch]
    reference = [encode_nucleotides(item['reference']) for item in batch]
    mut_pos = [item['mut_pos'] for item in batch]
    # trinucleotide_context = [
    #     encode_trinucleotide_context(item['trinucleotide_context']) for item in batch
    # ]


    chr_ = [item['chr'] for item in batch]
    read_id = [item['read_id'] for item in batch]
    ref = [item['ref'] for item in batch]
    alt = [item['alt'] for item in batch]

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
        'nucleotide_sequences': nucleotide_sequences,
        'base_qualities': base_qualities,
        'cigar_encoding': cigar_encoding,
        'positions': positions,
        'is_first': is_first_flags,
        'mapped_to_reverse': mapped_to_reverse_flags,
        'is_reverse': is_reverse,
        'read_support': read_support,
        'num_in_class': num_in_class,
        'labels': labels,
        'reference': reference,
        'mut_pos': mut_pos,
        'chr': chr_,
        'read_id': read_id,
        'ref': ref,
        'alt': alt,
        # 'trinucleotide_context': trinucleotide_context
    }

    batch = CustomBatch(batched_data)

    return batch


def create_finetuning_dataloader(
        csv_path, artefact_bam_path, mutation_bam_path, batch_size,
        base_quality_pad_idx, cigar_pad_idx,
        position_pad_idx, is_first_pad_idx, mapped_to_reverse_pad_idx,
        max_read_length=151, shuffle=True, num_workers=0, prefetch_factor=None,
        collate_fn=collate_fn, balanced=False
):
    logging.info("Creating fine-tuning data loader.")
    dataset = FineTuningDataset(
        csv_path, artefact_bam_path, mutation_bam_path,
        base_quality_pad_idx, cigar_pad_idx,
        position_pad_idx, is_first_pad_idx, mapped_to_reverse_pad_idx,
        batch_size=batch_size, max_read_length=max_read_length
    )

    if len(dataset) > 0:
        logging.info("Dataset created successfully.")
    else:
        logging.info("Dataset is empty.")
        return None

    # logging.info("Creating EpochSampler.")
    # sampler = EpochSampler(dataset, num_epochs=num_epochs, shuffle=shuffle)
    # logging.info("EpochSampler created.")

    if torch.cuda.is_available() and num_workers != 0:
        multiprocessing_context = mp.get_context('spawn')
        logging.info("Using multiprocessing context: 'spawn'")
    else:
        multiprocessing_context = None
        logging.info("Multiprocessing context is None")

    if num_workers == 0:
        prefetch_factor = None

    sampler = BalancedBatchSampler(
        dataset, batch_size=batch_size, positive_label=1.0
    )


    logging.info("Creating DataLoader...")
    if balanced:
        logging.info("Using balanced batches.")
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            # shuffle=shuffle,
            collate_fn=collate_fn,
            # batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=(multiprocessing_context is not None),
            persistent_workers=(multiprocessing_context is not None),
            multiprocessing_context=multiprocessing_context
        )
    else:
        logging.info("Using normal batches.")
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


