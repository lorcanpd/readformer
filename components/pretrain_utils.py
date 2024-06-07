from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# Nucleotide index look up

class NucleotidePerturbation:
    def __init__(self, rate):
        self.rate = rate
        nucleotide_to_index = {
            '': -1, 'A': 0, 'C': 1, 'G': 2, 'T': 3, 'R': 4, 'Y': 5, 'S': 6,
            'W': 7, 'K': 8, 'M': 9, 'B': 10, 'D': 11, 'H': 12, 'V': 13, 'N': 14
        }
        self.valid_bases = torch.tensor(
            [nucleotide_to_index[base] for base in ['A', 'C', 'G', 'T']],
            dtype=torch.int32
        )

    def generate_random_for_unique_positions(self, positions):
        batch_size, seq_length = positions.size()
        unique_values = []
        unique_randoms = []

        for i in range(batch_size):
            seq = positions[i]
            unique_vals, inverse_indices = torch.unique(
                seq[seq != -1], return_inverse=True
            )
            lognorms = torch.distributions.LogNormal(
                0, 1
            ).sample((unique_vals.size(0),))
            random_vals = 1 - lognorms

            unique_values.append(unique_vals)
            unique_randoms.append(random_vals)

        return unique_values, unique_randoms

    def broadcast_unique_randoms_to_input(
            self, positions, unique_values, unique_randoms
    ):
        batch_size, seq_length = positions.size()
        broadcasted_randoms = torch.zeros_like(positions, dtype=torch.float)

        for i in range(batch_size):
            seq = positions[i]
            valid_mask = seq != -1
            inverse_indices = torch.searchsorted(
                unique_values[i], seq[valid_mask]
            )
            broadcasted_randoms[i][valid_mask] = unique_randoms[i][
                inverse_indices
            ]

        return broadcasted_randoms

    def replace_nucleotides_with_random_bases(
            self, nucleotide_sequences, positions, broadcasted_randoms
    ):
        batch_size, seq_length = nucleotide_sequences.size()
        corrupted_sequences = nucleotide_sequences.clone()

        for i in range(batch_size):
            seq = corrupted_sequences[i]
            pos_seq = positions[i]
            valid_mask = pos_seq != -1

            # Generate random numbers for each valid position in the sequence
            element_limit = 1 / broadcasted_randoms[i]
            element_randoms = (
                torch.rand(
                    seq_length, device=seq.device
                ) * (1 - broadcasted_randoms[i]) + broadcasted_randoms[i]
            ) * element_limit
            product = element_randoms * broadcasted_randoms[i]

            # Calculate the expected number of replacements
            total_bases = torch.sum(valid_mask).item()
            expected_replacements = int(self.rate * total_bases)

            # Sort the product and determine the threshold for the expected
            # replacements
            sorted_product, indices = torch.sort(product)
            threshold = sorted_product[expected_replacements]

            # Create a replacement mask based on the threshold
            replace_mask = product < threshold
            replace_mask = replace_mask & valid_mask

            # Apply replacements, ensuring elements at the same position get
            # the same random base
            unique_positions_to_replace = torch.unique(pos_seq[replace_mask])
            for pos in unique_positions_to_replace:
                if pos == -1:
                    continue
                pos_mask = (pos_seq == pos) & replace_mask
                random_base = self.valid_bases[torch.randint(0, 4, (1,))]
                seq[pos_mask] = random_base

        return corrupted_sequences

    def perturb_sequences(self, nucleotide_sequences, positions):
        unique_values, unique_randoms = (
            self.generate_random_for_unique_positions(positions)
        )
        broadcasted_randoms = self.broadcast_unique_randoms_to_input(
            positions, unique_values, unique_randoms
        )
        return self.replace_nucleotides_with_random_bases(
            nucleotide_sequences, positions, broadcasted_randoms
        )


# def replacement_loss(
#         predictions, original_sequences, corrupted_sequences, padding_idx=15
# ):
#     # Create a tensor signifying which elements have been replaced
#     replacement_mask = (original_sequences != corrupted_sequences).float()
#     valid_mask = (original_sequences != padding_idx).float()
#
#     # True labels: 1 for replaced, 0 for original
#     true_labels = replacement_mask
#
#     # Calculate class weights based on the imbalance
#     num_replaced = replacement_mask.sum()
#     num_original = valid_mask.sum() - num_replaced
#     total_elements = valid_mask.sum()
#
#     weight_replaced = total_elements / (2 * num_replaced)
#     weight_original = total_elements / (2 * num_original)
#
#     weights = true_labels * weight_replaced + (1 - true_labels) * weight_original
#
#     # Apply valid mask to ignore padding elements
#     predictions = predictions * valid_mask
#
#     # Compute the binary cross-entropy loss with class weights
#     loss = F.binary_cross_entropy(predictions, true_labels, weight=weights, reduction='none')
#
#     # Compute the average loss over valid elements
#     average_loss = loss.sum() / valid_mask.sum()
#
#     return average_loss

def replacement_loss(replacement_mask, predictions):
    # True labels: 1 for replaced, 0 for original
    true_labels = replacement_mask.float()

    # Calculate class weights based on the imbalance
    num_replaced = replacement_mask.sum()
    num_original = (~replacement_mask).sum()
    total_elements = replacement_mask.numel()

    weight_replaced = total_elements / (2 * num_replaced)
    weight_original = total_elements / (2 * num_original)

    weights = true_labels * weight_replaced + (1 - true_labels) * weight_original

    # Compute the binary cross-entropy loss with class weights
    loss = F.binary_cross_entropy(predictions, true_labels, weight=weights, reduction='none')

    # Compute the average loss over valid elements
    average_loss = loss.sum() / total_elements

    return average_loss

import torch.nn.functional as F

def adversarial_loss(predictions, replacement_mask):
    # True labels: 1 for replaced, 0 for original
    true_labels = torch.zeros_like(predictions)
    true_labels[replacement_mask] = 1

    # Calculate correct predictions mask
    correct_mask = (predictions.round() == true_labels).float() * replacement_mask

    adv_labels = 1 - true_labels

    # Adversarial loss is higher if the main model gets more predictions correct
    loss = F.binary_cross_entropy(predictions, adv_labels, weight=correct_mask + 1)
    return loss


def generate_random_for_unique_positions(positions):
    batch_size, seq_length = positions.size()
    unique_values = []
    unique_randoms = []

    for i in range(batch_size):
        seq = positions[i]
        unique_vals, inverse_indices = torch.unique(
            seq[seq != -1], return_inverse=True
        )
        lognorms = torch.distributions.LogNormal(
            0, 1
        ).sample((unique_vals.size(0),))
        random_vals = 1 - lognorms

        unique_values.append(unique_vals)
        unique_randoms.append(random_vals)

    return unique_values, unique_randoms

def broadcast_unique_randoms_to_input(
        positions, unique_values, unique_randoms
):
    batch_size, seq_length = positions.size()
    broadcasted_randoms = torch.zeros_like(positions, dtype=torch.float)

    for i in range(batch_size):
        seq = positions[i]
        valid_mask = seq != -1
        inverse_indices = torch.searchsorted(
            unique_values[i], seq[valid_mask]
        )
        broadcasted_randoms[i][valid_mask] = unique_randoms[i][
            inverse_indices
        ]

    return broadcasted_randoms


def get_replacement_mask(positions, rate=0.15):
    unique_values, unique_randoms = generate_random_for_unique_positions(
        positions
    )
    broadcasted_randoms = broadcast_unique_randoms_to_input(
        positions, unique_values, unique_randoms
    )
    valid_mask = positions != -1
    # Generate random numbers for each valid position in the sequence
    element_limit = 1 / broadcasted_randoms
    element_randoms = torch.rand(
        positions.shape, device=positions.device, dtype=torch.float32
    ) * (1 - broadcasted_randoms) + broadcasted_randoms
    product = element_randoms * broadcasted_randoms * element_limit
    # Create a replacement mask based on the threshold
    replace_mask = product < rate
    replace_mask = replace_mask & valid_mask

    return replace_mask


# # TEST SCRIPTS.
# from components.data_streaming import create_data_loader
# from components.plots import (
#     plot_nucleotide_replacement_histogram, plot_replacement_statistics
# )
#
# data_loader = create_data_loader(
#     file_paths='TEST_DATA', metadata_path='TEST_DATA/test_metadata.csv',
#     nucleotide_threshold=1024*8, batch_size=4, shuffle=True
# )
#
# # Iterate through data
# for batch in data_loader:
#     print(batch)  # Process your batch here
#     break
#
# # Generate random numbers for unique positions and broadcast them
# rate = 0.05
# # position_rate = 0.001
# positions = batch['positions']
# nucleotide_sequences = batch['nucleotide_sequences']
#
# sequence_perturbation = NucleotidePerturbation(rate)
# corrupted_sequences = sequence_perturbation.perturb_sequences(
#     nucleotide_sequences.clone(), positions
# )
#
# # Proportion of all bases replaced
# total_replaced = torch.sum(nucleotide_sequences != corrupted_sequences).item()
# total_bases = nucleotide_sequences.numel()
# proportion_replaced = total_replaced / total_bases
# print(f"Proportion of bases replaced: {proportion_replaced:.2f}")
#
# plot_replacement_statistics(nucleotide_sequences, corrupted_sequences, positions)
#
#
# print("Original Sequences:\n", nucleotide_sequences)
# print("Positions:\n", positions)
# print("Corrupted Sequences:\n", corrupted_sequences)
#
#
# plot_nucleotide_replacement_histogram(nucleotide_sequences, corrupted_sequences, positions)


### OLD TENSOFRLOW IMPLEMENTATION
# import tensorflow as tf
#
#
# class ElectraDiscriminator(tf.keras.Model):
#     def __init__(self, embedding_dim):
#         super(ElectraDiscriminator, self).__init__()
#         init = tf.keras.initializers.HeNormal()
#         self.W1 = tf.Variable(
#             init(shape=[embedding_dim, embedding_dim*2], dtype=tf.float32),
#             trainable=True,
#             name='W1'
#         )
#         self.b1 = tf.Variable(
#             tf.zeros(shape=[embedding_dim*2], dtype=tf.float32),
#             trainable=True,
#             name='b1'
#         )
#         self.LayerNorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#         self.W2 = tf.Variable(
#             init(shape=[embedding_dim*2, 64], dtype=tf.float32),
#             trainable=True,
#             name='W2'
#         )
#         self.b2 = tf.Variable(
#             tf.zeros(shape=[64], dtype=tf.float32),
#             trainable=True,
#             name='b2'
#         )
#         self.LayerNorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#         self.W3 = tf.Variable(
#             init(shape=[64, 1], dtype=tf.float32),
#             trainable=True,
#             name='W3'
#         )
#         self.b3 = tf.Variable(
#             tf.zeros(shape=[1], dtype=tf.float32),
#             trainable=True,
#             name='b3'
#         )
#
#     def call(self, inputs, training=False):
#         """
#         Classify whether an embedding in a batch of sequences of embeddings is replaced.
#
#         :param inputs:
#             A batch of sequences of embeddings with shape
#             (batch_size, sequence_length, embedding_dim)
#         :param training:
#             Whether the model is in training mode.
#         :return:
#             A batch of sequence of predictions with shape
#             (batch_size, sequence_length, 1). Each value is a probability
#             indicating the likelihood that the corresponding token was replaced.
#         """
#         x = tf.tensordot(inputs, self.W1, axes=1) + self.b1
#         x = self.LayerNorm1(x)
#         x = tf.nn.gelu(x)
#
#         if training:
#             x = tf.nn.dropout(x, rate=0.1)
#
#         x = tf.tensordot(x, self.W2, axes=1) + self.b2
#         x = self.LayerNorm2(x)
#         x = tf.nn.gelu(x)
#
#         if training:
#             x = tf.nn.dropout(x, rate=0.1)
#
#         x = tf.tensordot(x, self.W3, axes=1) + self.b3
#         x = tf.nn.sigmoid(x)
#
#         return x
#
#     def get_config(self):
#         return {
#             'embedding_dim': self.embedding_dim
#         }
#
#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)
#
