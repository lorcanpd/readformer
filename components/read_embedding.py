import torch
import torch.nn as nn
# import torch.nn.functional as F

from components.better_device_handling import Module


class NucleotideLookup:
    def __init__(self):
        """
        Initialises the nucleotide lookup table.
        """
        self.nucleotide_to_index = {
            'A': 0, 'C': 1, 'G': 2, 'T': 3, 'R': 4, 'Y': 5, 'S': 6, 'W': 7,
            'K': 8, 'M': 9, 'B': 10, 'D': 11, 'H': 12, 'V': 13, 'N': 14, '': 15
        }
        self.index_to_nucleotide = {
            v: k for k, v in self.nucleotide_to_index.items()
        }

    def nucleotide_to_index(self, nucleotides):
        """
        Converts a batch of nucleotide sequences to their corresponding indices.

        :param nucleotides:
            A list of nucleotide sequences (strings).
        :return:
            A tensor of corresponding nucleotide indices.
        """
        batch_size = len(nucleotides)
        seq_length = len(nucleotides[0])
        indices = torch.full((batch_size, seq_length), -1, dtype=torch.long)

        for i, sequence in enumerate(nucleotides):
            for j, nucleotide in enumerate(sequence):
                indices[i, j] = self.nucleotide_to_index.get(nucleotide, -1)

        return indices

    def index_to_nucleotide(self, indices):
        """
        Converts a batch of nucleotide indices to their corresponding nucleotide
        sequences.

        :param indices:
            A tensor of nucleotide indices.
        :return:
            A list of corresponding nucleotide sequences (strings).
        """
        nucleotides = []
        for index_sequence in indices:
            nucleotide_sequence = ''.join(
                self.index_to_nucleotide.get(index.item(), '')
                for index in index_sequence
            )
            nucleotides.append(nucleotide_sequence)

        return nucleotides


class NucleotideEmbeddingLayer(Module):
    def __init__(self, embedding_dim, mlm_mode=False):
        """
        Initialises the nucleotide embedding layer.

        :param embedding_dim:
            The dimensionality of the embedding space.
        """
        super(NucleotideEmbeddingLayer, self).__init__()

        self.embedding_dim = embedding_dim
        self.mlm_mode = mlm_mode
        num_nucleotides = 16
        self.mask_index = num_nucleotides if mlm_mode else None
        self.padding_idx = 15
        self.embedding = nn.Embedding(
            num_embeddings=num_nucleotides + (1 if mlm_mode else 0),
            embedding_dim=embedding_dim,
            padding_idx=self.padding_idx,
        )

    def forward(self, inputs):
        """
        Maps the input nucleotide sequences to their embeddings.

        :param inputs:
            A batch of sequences as torch tensors with nucleotide indices.
        :return:
            The corresponding nucleotide embeddings.
        """
        embeddings = self.embedding(inputs)
        # Mask the padding indices with zero vectors
        mask = (inputs != self.padding_idx).unsqueeze(-1).float()
        embeddings = embeddings * mask
        return embeddings


class CigarEmbeddingLayer(Module):
    def __init__(self, embedding_dim):
        """
        Initialises the CIGAR embedding layer.

        :param embedding_dim:
            The dimensionality of the embedding space.
        """
        super(CigarEmbeddingLayer, self).__init__()
        # There are 9 cigar operations, including -1 for padding
        num_cigar_ops = 5
        self.embedding = nn.Embedding(
            num_embeddings=num_cigar_ops,
            embedding_dim=embedding_dim,
            padding_idx=-1,
        )

    def forward(self, inputs):
        """
        Maps the input CIGAR operations to their embeddings.

        :param inputs:
            A batch of sequences as torch tensors with CIGAR operation indices.
        :return:
            The corresponding CIGAR operation embeddings.
        """
        embeddings = self.embedding(inputs)
        return embeddings


class BaseQualityEmbeddingLayer(Module):
    """
    Initialises the base quality embedding layer.

    :param embedding_dim:
        The dimensionality of the embedding space.

    """
    def __init__(self, embedding_dim):
        super(BaseQualityEmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=45,
            embedding_dim=embedding_dim,
            padding_idx=-1
        )

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        return embeddings


class SequencingDirectionEmbeddingLayer(Module):
    def __init__(self, embedding_dim):
        """
        Initialises the sequencing direction embedding layer.

        :param embedding_dim:
            The dimensionality of the embedding space.
        """
        super(SequencingDirectionEmbeddingLayer, self).__init__()
        num_directions = 2
        self.embedding = nn.Embedding(
            num_embeddings=num_directions,
            embedding_dim=embedding_dim,
            padding_idx=-1
        )

    def forward(self, inputs):
        """
        Maps the input sequencing directions to their embeddings.

        :param inputs:
            A batch of sequences as torch tensors with sequencing direction indices.
        :return:
            The corresponding sequencing direction embeddings.
        """
        embeddings = self.embedding(inputs)
        return embeddings


class ReadReversalEmbeddingLayer(Module):
    def __init__(self, embedding_dim):
        """
        Initialises the read reversal embedding layer. Which indicates whether
        the read has been reversed so that the 5' end is at the start of the
        read.

        :param embedding_dim:
            The dimensionality of the embedding space.
        """
        super(ReadReversalEmbeddingLayer, self).__init__()
        num_reversals = 2
        self.embedding = nn.Embedding(
            num_embeddings=num_reversals,
            embedding_dim=embedding_dim,
            padding_idx=-1
        )

    def forward(self, inputs):
        """
        Maps the input read reversals to their embeddings.

        :param inputs:
            A batch of sequences as torch tensors with read reversal indices.
        :return:
            The corresponding read reversal embeddings.
        """
        embeddings = self.embedding(inputs)
        return embeddings
