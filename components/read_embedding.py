import torch
import torch.nn as nn


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


class MetricEmbedding(nn.Module):
    def __init__(self, embedding_dim, num_metrics, name=None):
        """
        Initialises the metric embedding layer.

        :param embedding_dim:
            The dimensionality of the embedding space used for the nucleotide
            representations.
        :param name:
            The name of the layer.
        """
        super(MetricEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_metrics = num_metrics

        if name is not None:
            self.name = name + "_metric_embedding"
        else:
            self.name = "metric_embedding"

        self.embedding_matrix = nn.Parameter(
            nn.init.kaiming_normal_(torch.empty(num_metrics, embedding_dim))
        )

    def forward(self, inputs):
        return torch.matmul(inputs, self.embedding_matrix)


class NucleotideEmbeddingLayer(nn.Module):
    def __init__(self, embedding_dim):
        """
        Initialises the nucleotide embedding layer.

        :param embedding_dim:
            The dimensionality of the embedding space.
        """
        super(NucleotideEmbeddingLayer, self).__init__()
        self.embedding_dim = embedding_dim
        num_nucleotides = 16  # Total number of unique nucleotide representations
        self.padding_idx = num_nucleotides - 1
        self.embedding = nn.Embedding(
            num_embeddings=num_nucleotides,  # Include padding
            embedding_dim=embedding_dim,
            padding_idx=self.padding_idx
        )
        nn.init.kaiming_normal_(self.embedding.weight)

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

# import tensorflow as tf
# # import numpy as np
# # from extract_reads import extract_reads_around_position, extract_reads_from_position_onward, get_read_info
#
#
# class NucleotideEmbeddingLayer(tf.keras.layers.Layer):
#     def __init__(self, embedding_dim, **kwargs):
#         """
#         Initialises the nucleotide embedding layer.
#
#         :param embedding_dim:
#             The dimensionality of the embedding space.
#         """
#         super(NucleotideEmbeddingLayer, self).__init__(**kwargs)
#         self.embedding_dim = embedding_dim
#
#         # Define the nucleotide mapping as a TensorFlow lookup table
#         keys = tf.constant(
#             [
#                 # Empty string '' for padding.
#                 '', 'A', 'C', 'G', 'T', 'R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D',
#                 'H', 'V', 'N'
#             ], dtype=tf.string
#         )
#         values = tf.range(start=0, limit=len(keys), dtype=tf.int32)
#
#         self.nucleotide_to_index_table = tf.lookup.StaticHashTable(
#             tf.lookup.KeyValueTensorInitializer(keys, values),
#             default_value=len(keys) - 1  # Default value for padding
#         )
#
#         self.embedding = tf.keras.layers.Embedding(
#             input_dim=len(keys)-1,  # Exclude padding
#             output_dim=self.embedding_dim,
#             embeddings_initializer="he_normal",
#             mask_zero=True,  # Enables padding handling
#             trainable=True,
#             name="nucleotide_embeddings"
#         )
#
#     def call(self, inputs):
#         """
#         Maps the input nucleotide sequences to their embeddings.
#
#         :param inputs:
#             A batch of sequences as tf.string tensors.
#         :return:
#             The corresponding nucleotide embeddings.
#         """
#         input_indices = self.nucleotide_to_index_table.lookup(inputs)
#         embeddings = self.embedding(input_indices)
#         return embeddings
#
#     def compute_output_shape(self, input_shape):
#         return input_shape + (self.embedding_dim,)
#
#     def get_config(self):
#         config = super(NucleotideEmbeddingLayer, self).get_config()
#         config.update({
#             'embedding_dim': self.embedding_dim
#         })
#         return config
#
#
# class MetricEmbedding(tf.keras.layers.Layer):
#     """
#     Custom layer for metric embeddings. CURRENTLY, IT IS IN ITS SIMPLEST FORM.
#     """
#
#     def __init__(self, embedding_dim, name=None, **kwargs):
#         """
#         Initialises the metric embedding layer.
#
#         :param embedding_dim:
#             The dimensionality of the embedding space used for the nucleotide
#             representations.
#         :param kwargs:
#             Additional keyword arguments for the layer.
#         """
#         super(MetricEmbedding, self).__init__(**kwargs)
#         self.embedding_dim = embedding_dim
#         if name is not None:
#             self.name = name + "_metric_embedding"
#         else:
#             self.name = "metric_embedding"
#
#     def build(self, input_shape):
#         self.embedding_matrix = self.add_weight(
#             shape=(input_shape[-1], self.embedding_dim),
#             initializer="he_normal",
#             trainable=True,
#             name=self.name + "_matrix"
#         )
#         super(MetricEmbedding, self).build(input_shape)
#
#     def call(self, inputs):
#         return tf.matmul(inputs, self.embedding_matrix)
#
