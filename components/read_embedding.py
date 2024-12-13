import torch
import torch.nn as nn
import torch.nn.functional as F

from components.better_device_handling import Module
from components.base_model import init_weights

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
        # # Mask the padding indices with zero vectors
        # mask = (inputs != self.padding_idx).unsqueeze(-1).float()
        # embeddings = embeddings * mask
        return embeddings


class CigarEmbeddingLayer(Module):
    def __init__(self, embedding_dim):
        """
        Initialises the CIGAR embedding layer.

        :param embedding_dim:
            The dimensionality of the embedding space.
        """
        super(CigarEmbeddingLayer, self).__init__()

        num_cigar_ops = 5
        self.mask_index = num_cigar_ops + 1
        self.padding_idx = num_cigar_ops
        self.embedding = nn.Embedding(
            num_embeddings=num_cigar_ops + 2,
            embedding_dim=embedding_dim,
            padding_idx=self.padding_idx,
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


# class BaseQualityEmbeddingLayer(Module):
#     """
#     Initialises the base quality embedding layer.
#
#     :param embedding_dim:
#         The dimensionality of the embedding space.
#
#     """
#     def __init__(self, embedding_dim, max_quality=40):
#         super(BaseQualityEmbeddingLayer, self).__init__()
#         index_of_highest_quality = max_quality + 1
#         self.embedding = nn.Embedding(
#             num_embeddings=index_of_highest_quality + 1,
#             embedding_dim=embedding_dim,
#             padding_idx=-1
#         )
#         self.mask_index = index_of_highest_quality + 1
#         self.max_quality = max_quality
#
#     def forward(self, inputs):
#         inputs = inputs.clamp(min=0, max=self.max_quality)
#         embeddings = self.embedding(inputs)
#         return embeddings


class BaseQualityEmbeddingLayer(Module):
    """
    Initialises the base quality embedding layer with separate masking and
    padding indices.

    :param embedding_dim:
        The dimensionality of the embedding space.
    :param max_quality:
        The maximum quality score.
    """
    def __init__(self, embedding_dim, max_quality=40):
        super(BaseQualityEmbeddingLayer, self).__init__()
        self.max_quality = max_quality

        # Define separate indices for mask and padding
        self.mask_idx = self.max_quality + 1      # e.g., 41 if max_quality=40
        self.padding_idx = self.max_quality + 2   # e.g., 42 if max_quality=40

        # Total number of embeddings:
        # - Quality scores: 0 to max_quality (inclusive) => (max_quality + 1)
        # - Mask index
        # - Padding index
        self.num_embeddings = self.max_quality + 3  # 0 to 42 inclusive for max_quality=40

        # Initialize the embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=self.padding_idx  # Ensure padding_idx is within [0, num_embeddings - 1]
        )

    def forward(self, inputs, to_be_masked=None, to_be_padded=None):
        """
        Maps the input base quality scores to their embeddings.

        :param inputs:
            A batch of base quality scores as torch tensors of shape
            (batch_size, seq_length).
        :param to_be_masked:
            A boolean tensor of the same shape as inputs indicating which
            elements should be masked.
        :param to_be_padded:
            A boolean tensor of the same shape as inputs indicating which
            elements should be padded.
        :return:
            The corresponding base quality embeddings of shape
            (batch_size, seq_length, embedding_dim).
        """
        # Clamp inputs to ensure they are within [0, max_quality]
        inputs = inputs.clamp(min=0, max=self.max_quality)

        # set those to be masked to mask index
        if to_be_masked is not None:
            inputs[to_be_masked] = self.mask_idx

        # set those to be padded to padding index
        if to_be_padded is not None:
            inputs[to_be_padded] = self.padding_idx

        # Pass inputs through the embedding layer
        embeddings = self.embedding(inputs)

        return embeddings


# class StrandEmbeddingLayer(Module):
#     def __init__(self, embedding_dim):
#         """
#         Initialises the reverse complement embedding layer. This layer is used
#         to indicate whether a read was originally mapped to the reverse strand
#         and then reverse complemented.
#
#         :param embedding_dim:
#             The dimensionality of the embedding space.
#         """
#         super(StrandEmbeddingLayer, self).__init__()
#         num_reversals = 2
#         self.embedding = nn.Embedding(
#             num_embeddings=num_reversals+1,
#             embedding_dim=embedding_dim,
#             padding_idx=-1
#         )
#         self.mask_index = num_reversals
#
#     def forward(self, inputs):
#         """
#         Maps the input flag to their embeddings.
#
#         :param inputs:
#             A batch of sequences as torch tensors with read reversal indices.
#         :return:
#             The corresponding read reversal embeddings.
#         """
#         embeddings = self.embedding(inputs)
#         return embeddings


class StrandEmbeddingLayer(Module):
    def __init__(self, embedding_dim):
        """
        Initializes the strand embedding layer.

        This layer is used to indicate whether a read was originally mapped to
        the reverse strand and then reverse complemented.

        :param embedding_dim:
            The dimensionality of the embedding space.
        """
        super(StrandEmbeddingLayer, self).__init__()

        # Number of strand reversals
        num_reversals = 2  # Typically: 0 - forward, 1 - reverse

        # Define separate indices for padding and masking
        self.padding_idx = num_reversals  # e.g., 2
        self.mask_index = num_reversals + 1  # e.g., 3

        # Total number of embeddings:
        # - Strand reversals: 0 to 1 (inclusive)
        # - Padding index: 2
        # - Masking index: 3
        self.num_embeddings = num_reversals + 2  # 0 to 3 inclusive

        # Initialise the embedding layer with separate padding_idx
        self.embedding = nn.Embedding(
            num_embeddings=self.num_embeddings,  # 4 embeddings: 0, 1, 2, 3
            embedding_dim=embedding_dim,
            padding_idx=self.padding_idx,  # Set to 2
        )

    def forward(self, inputs):
        """
        Maps the input strand flags to their embeddings.

        :param inputs:
            A batch of strand reversal indices as torch tensors with shape (batch_size, seq_length).
            Expected values:
                - 0: Forward strand
                - 1: Reverse strand
                - 2: Padding
                - 3: Masking (if applicable during pre-training)
        :return:
            The corresponding strand embeddings of shape (batch_size, seq_length, embedding_dim).
        """
        # Validate input indices to prevent out-of-range errors
        if torch.any(inputs < 0) or torch.any(inputs >= self.num_embeddings):
            min_idx = inputs.min().item()
            max_idx = inputs.max().item()
            raise ValueError(
                f"Input indices out of range in StrandEmbeddingLayer: min={min_idx}, "
                f"max={max_idx}, allowed range=0 to {self.num_embeddings - 1}"
            )

        # Pass inputs through the embedding layer
        embeddings = self.embedding(inputs)

        return embeddings


# class MatePairEmbeddingLayer(Module):
#     def __init__(self, embedding_dim):
#         """
#         Initialises the mate pair embedding layer. This layer is used to indicate
#         whether a read is the first or second read of the pair.
#
#         :param embedding_dim:
#             The dimensionality of the embedding space.
#         """
#         super(MatePairEmbeddingLayer, self).__init__()
#         num_mate_pairs = 2
#         self.embedding = nn.Embedding(
#             num_embeddings=num_mate_pairs+1,
#             embedding_dim=embedding_dim,
#             padding_idx=-1
#         )
#         self.mask_index = num_mate_pairs
#
#     def forward(self, inputs):
#         """
#         Maps the input flag to their embeddings.
#
#         :param inputs:
#             A batch of sequences as torch tensors with mate pair indices.
#         :return:
#             The corresponding mate pair embeddings.
#         """
#         embeddings = self.embedding(inputs)
#         return embeddings

class MatePairEmbeddingLayer(Module):
    def __init__(self, embedding_dim):
        """
        Initializes the mate pair embedding layer.

        This layer is used to indicate whether a read is the first or second read of the pair.

        :param embedding_dim:
            The dimensionality of the embedding space.
        """
        super(MatePairEmbeddingLayer, self).__init__()

        # Number of mate pair types
        num_mate_pairs = 2  # Typically: 0 - first read, 1 - second read

        # Define separate indices for padding and masking
        self.padding_idx = num_mate_pairs  # e.g., 2
        self.mask_index = num_mate_pairs + 1  # e.g., 3

        # Total number of embeddings:
        # - Mate pairs: 0 to 1 (inclusive)
        # - Padding index: 2
        # - Masking index: 3
        self.num_embeddings = num_mate_pairs + 2  # 0 to 3 inclusive

        # Initialize the embedding layer with separate padding_idx
        self.embedding = nn.Embedding(
            num_embeddings=self.num_embeddings,  # 4 embeddings: 0, 1, 2, 3
            embedding_dim=embedding_dim,
            padding_idx=self.padding_idx,  # Set to 2
        )

    def forward(self, inputs):
        """
        Maps the input mate pair flags to their embeddings.

        :param inputs:
            A batch of mate pair indices as torch tensors with shape
            (batch_size, seq_length).
            Expected values:
                - 0: First read of the pair
                - 1: Second read of the pair
                - 2: Padding
                - 3: Masking (if applicable during pre-training)
        :return:
            The corresponding mate pair embeddings of shape
            (batch_size, seq_length, embedding_dim).
        """

        # Pass inputs through the embedding layer
        embeddings = self.embedding(inputs)

        # Zero out padded indices
        # Create a mask where padding_idx positions are False, others are True
        padding_mask = (inputs != self.padding_idx).unsqueeze(-1).float()  # Shape: (batch_size, seq_length, 1)

        # Apply the mask to embeddings
        embeddings = embeddings * padding_mask  # Zero out embeddings where padding_idx is present

        return embeddings


class InputEmbeddingLayer(Module):
    def __init__(self, embedding_dim, max_quality=40):
        """
        Initializes the combined input embedding layer, which includes
        all individual embedding layers and a gating mechanism based on SwiGLU.

        :param embedding_dim:
            The dimensionality of the embedding space.
        :param max_quality:
            The maximum quality score for base quality embeddings.
        """
        super(InputEmbeddingLayer, self).__init__()

        # Initialise individual embedding layers
        self.nucleotide_embeddings = NucleotideEmbeddingLayer(
            embedding_dim=embedding_dim,
            mlm_mode=True
        ).apply(init_weights)

        self.cigar_embeddings = CigarEmbeddingLayer(
            embedding_dim=embedding_dim // 4
        ).apply(init_weights)

        self.base_quality_embeddings = BaseQualityEmbeddingLayer(
            embedding_dim=embedding_dim // 4,
            max_quality=max_quality
        ).apply(init_weights)

        self.strand_embeddings = StrandEmbeddingLayer(
            embedding_dim=embedding_dim // 4
        ).apply(init_weights)

        self.mate_pair_embeddings = MatePairEmbeddingLayer(
            embedding_dim=embedding_dim // 4
        ).apply(init_weights)

        # Define gate and feature projections
        # Assuming metric_emb is concatenated embeddings from Cigar, BaseQuality, Strand, MatePair
        # Each of these has embedding_dim//4, so concatenated metric_emb has embedding_dim
        self.gate_projection = nn.Linear(embedding_dim, embedding_dim)
        self.gate_projection.apply(init_weights)
        self.feature_projection = nn.Linear(embedding_dim, embedding_dim)
        self.feature_projection.apply(init_weights)
        self.silu = nn.SiLU()

    def forward(
        self,
        nucleotide_sequences,
        cigar_encodings,
        base_qualities,
        strand_flags,
        mate_pair_flags,
        to_be_masked=None,
        to_be_padded=None
    ):
        """
        Combines all embedding layers with a gating mechanism to produce the
        final model input.

        :param nucleotide_sequences:
            Tensor of shape (batch_size, seq_length) with nucleotide indices.
        :param cigar_encodings:
            Tensor of shape (batch_size, seq_length) with CIGAR operation indices.
        :param base_qualities:
            Tensor of shape (batch_size, seq_length) with base quality scores.
        :param strand_flags:
            Tensor of shape (batch_size, seq_length) with strand indices.
        :param mate_pair_flags:
            Tensor of shape (batch_size, seq_length) with mate pair indices.
        :param to_be_masked:
            Boolean tensor of shape (batch_size, seq_length) indicating which
            elements to mask.
        :param to_be_padded:
            Boolean tensor of shape (batch_size, seq_length) indicating which
            elements to pad.
        :return:
            Tensor of shape (batch_size, seq_length, embedding_dim) as model
            input.
        """

        masked_nucleotide_emb = self.nucleotide_embeddings(nucleotide_sequences)

        masked_base_quality_emb = self.base_quality_embeddings(
            base_qualities,
            to_be_masked=to_be_masked,
            to_be_padded=to_be_padded
        )

        # Get other embeddings without masking/padding
        masked_cigar_emb = self.cigar_embeddings(cigar_encodings)
        masked_strand_emb = self.strand_embeddings(strand_flags)
        masked_mate_pair_emb = self.mate_pair_embeddings(mate_pair_flags)

        # Concatenate metric embeddings
        metric_emb = torch.cat(
            [
                masked_cigar_emb,
                masked_base_quality_emb,
                masked_strand_emb,
                masked_mate_pair_emb
            ],
            dim=-1
        )

        # Project nucleotide embeddings
        nuc_proj = self.feature_projection(masked_nucleotide_emb)

        metric_swish = self.silu(self.feature_projection(metric_emb))

        # Apply the gating mechanism with dropout and residual connection
        model_input = F.dropout(
            nuc_proj * metric_swish, p=0.1, training=self.training
        ) + masked_nucleotide_emb

        return model_input
