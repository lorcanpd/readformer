import torch

# Nucleotide index look up
nucleotide_to_index = {
    '': -1, 'A': 0, 'C': 1, 'G': 2, 'T': 3, 'R': 4, 'Y': 5, 'S': 6, 'W': 7,
    'K': 8, 'M': 9, 'B': 10, 'D': 11, 'H': 12, 'V': 13, 'N': 14
}
index_to_nucleotide = {v: k for k, v in nucleotide_to_index.items()}
valid_bases = torch.tensor([nucleotide_to_index[base] for base in ['A', 'C', 'G', 'T']], dtype=torch.int32)

def generate_random_for_unique_positions(positions):
    batch_size, seq_length = positions.size()
    unique_values = []
    unique_randoms = []

    for i in range(batch_size):
        seq = positions[i]
        unique_vals, inverse_indices = torch.unique(seq[seq != -1], return_inverse=True)
        # random_vals = torch.rand_like(unique_vals.float())
        lognorms = torch.distributions.log_normal.LogNormal(0, 1).sample((unique_vals.size(0),))
        # clip max value to 0.9999
        random_vals = 1 - lognorms


        unique_values.append(unique_vals)
        unique_randoms.append(random_vals)

    return unique_values, unique_randoms

def broadcast_unique_randoms_to_input(positions, unique_values, unique_randoms):
    batch_size, seq_length = positions.size()
    broadcasted_randoms = torch.zeros_like(positions, dtype=torch.float)

    for i in range(batch_size):
        seq = positions[i]
        valid_mask = seq != -1
        inverse_indices = torch.searchsorted(unique_values[i], seq[valid_mask])
        broadcasted_randoms[i][valid_mask] = unique_randoms[i][inverse_indices]

    return broadcasted_randoms

# TODO revisit to make sure it works as intended.
def replace_nucleotides_with_random_bases(nucleotide_sequences, positions, broadcasted_randoms, rate):
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

        # Create a replacement mask based on the threshold
        # replace_mask = product < rate

        # Calculate the expected number of replacements
        total_bases = torch.sum(valid_mask).item()
        expected_replacements = int(rate * total_bases)

        # Sort the product and determine the threshold for the expected replacements
        sorted_product, indices = torch.sort(product)
        threshold = sorted_product[expected_replacements]

        # Create a replacement mask based on the threshold
        replace_mask = product < threshold
        replace_mask = replace_mask & valid_mask

        # Apply replacements, ensuring elements at the same position get the same random base
        unique_positions_to_replace = torch.unique(pos_seq[replace_mask])
        for pos in unique_positions_to_replace:
            if pos == -1:
                continue
            pos_mask = (pos_seq == pos) & replace_mask
            random_base = valid_bases[torch.randint(0, 4, (1,))]
            seq[pos_mask] = random_base

    return corrupted_sequences

def plot_replacement_statistics(original_sequences, corrupted_sequences, positions):
    import matplotlib.pyplot as plt
    batch_size, seq_length = original_sequences.size()

    replacement_counts = []
    proportion_replaced = []
    for i in range(batch_size):
        seq_positions = positions[i][positions[i] != -1]
        unique_positions = torch.unique(seq_positions)
        for pos in unique_positions:
            if pos == -1:
                continue
            original_count = torch.sum((positions[i] == pos) & (original_sequences[i] == corrupted_sequences[i])).item()
            replaced_count = torch.sum((positions[i] == pos) & (original_sequences[i] != corrupted_sequences[i])).item()
            proportion = replaced_count / (original_count + replaced_count)
            if replaced_count > 0:
                replacement_counts.append(replaced_count)
                proportion_replaced.append(proportion)

    # Calculate the proportion of replaced bases per position

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(proportion_replaced, range=(0, 1), bins=100, edgecolor='black')
    plt.xlabel('Proportion of Bases Replaced at a Position')
    plt.ylabel('Number of Positions')
    plt.title('Distribution of Base Replacements per Position')
    # plt.yscale('log')  # Use a logarithmic scale to emphasize the long tail
    plt.show()


def plot_nucleotide_histogram(original_sequences, corrupted_sequences, positions):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np

    batch_size, seq_length = original_sequences.size()
    fig, axes = plt.subplots(batch_size, 1, sharex=True)

    if batch_size == 1:
        axes = [axes]

    for i in range(batch_size):
        seq_positions = positions[i][positions[i] != -1]
        min_pos = seq_positions.min().item()
        max_pos = seq_positions.max().item()
        all_positions = np.arange(min_pos, max_pos + 1)
        original_counts = np.zeros(max_pos - min_pos + 1, dtype=int)
        replaced_counts = np.zeros(max_pos - min_pos + 1, dtype=int)

        for j, pos in enumerate(all_positions):
            original_counts[j] = torch.sum(
                (positions[i] == pos) & (
                        original_sequences[i] == corrupted_sequences[i])
            ).item()
            replaced_counts[j] = torch.sum(
                (positions[i] == pos) & (
                        original_sequences[i] != corrupted_sequences[i])
            ).item()

        # Create the data for plotting
        df = pd.DataFrame({
            'Position': all_positions,
            'Original': original_counts,
            'Replaced': replaced_counts
        })

        # Plot stacked bar chart
        df.set_index('Position').plot(kind='bar', stacked=True, ax=axes[i], color=['blue', 'red'])
        axes[i].set_title(f"Sequence {i + 1}")
        axes[i].set_xlabel('Position')
        axes[i].set_ylabel('Count')
        axes[i].legend(title='Type', labels=['Original', 'Replaced'])

    plt.tight_layout()
    plt.show()


# TEST SCRIPTS.
from components.data_streaming import create_data_loader

data_loader = create_data_loader(
    file_paths='TEST_DATA', metadata_path='TEST_DATA/test_metadata.csv',
    nucleotide_threshold=1024*8, batch_size=4, shuffle=True
)

# Iterate through data
for batch in data_loader:
    print(batch)  # Process your batch here
    break

# Generate random numbers for unique positions and broadcast them
rate = 0.15
# position_rate = 0.001
positions = batch['positions']
nucleotide_sequences = batch['nucleotide_sequences']

# TODO revisit this.
unique_values, unique_randoms = generate_random_for_unique_positions(positions)
broadcasted_randoms = broadcast_unique_randoms_to_input(positions, unique_values, unique_randoms)

# Replace nucleotides with random bases
corrupted_sequences = replace_nucleotides_with_random_bases(
    nucleotide_sequences.clone(), positions, broadcasted_randoms, rate
)

# Proportion of all bases replaced
total_replaced = torch.sum(nucleotide_sequences != corrupted_sequences).item()
total_bases = nucleotide_sequences.numel()
proportion_replaced = total_replaced / total_bases
print(f"Proportion of bases replaced: {proportion_replaced:.2f}")

plot_replacement_statistics(nucleotide_sequences, corrupted_sequences, positions)


print("Original Sequences:\n", nucleotide_sequences)
print("Positions:\n", positions)
print("Corrupted Sequences:\n", corrupted_sequences)


plot_nucleotide_histogram(nucleotide_sequences, corrupted_sequences, positions)





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
