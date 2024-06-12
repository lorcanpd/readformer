
### IGNORE - OLD TENSORFLOW CODE - MOVED TO PYTORCH

import tensorflow as tf
from self_attention import EncoderBlock
from read_embedding import NucleotideEmbeddingLayer, MetricEmbedding


def extract_unique_positions(tensor):
    """
    Extracts unique elements from each sequence in the batch, along with
    indices.

    :param tensor:
        A 2D tensor of shape [batch_size, sequence_length].
    :return:
        A tuple of a RaggedTensor and a tensor, where the first element contains
        the unique elements of each sequence, and the second element contains
        the indices of the original elements.
    """

    def get_unique(sequence):
        unique_values, indices = tf.unique(sequence)
        return unique_values, indices

    # Handling unique values
    unique_positions = tf.map_fn(
        lambda seq: get_unique(seq)[0],
        tensor,
        fn_output_signature=tf.RaggedTensorSpec(shape=[None], dtype=tensor.dtype)
    )

    # Handling indices - note: indices are not naturally ragged.
    indices = tf.map_fn(
        lambda seq: get_unique(seq)[1],
        tensor,
        fn_output_signature=tf.TensorSpec(shape=[None], dtype=tf.int32)
    )

    return unique_positions, indices


def create_random_uniform_ragged(unique_positions, rate):
    """
    Creates a ragged tensor of random uniform values with the same shape as
    another ragged tensor.

    :param unique_positions:
        A RaggedTensor of shape [batch_size, None].
    :param rate:
        The minimum value for the random uniform distribution.
    :return:
        A RaggedTensor of random uniform values with the same shape as
        unique_positions.
    """
    # Flatten the unique_positions ragged tensor to a 1D tensor
    flat_values = unique_positions.flat_values

    # Generate random uniform values for the flat_values
    random_flat_values = tf.random.uniform(
        shape=tf.shape(flat_values), minval=rate, maxval=1.0, dtype=tf.float32
    )

    # Use the original row_splits to construct the new ragged tensor with random values
    random_uniform_ragged = tf.RaggedTensor.from_row_splits(
        values=random_flat_values, row_splits=unique_positions.row_splits
    )

    return random_uniform_ragged


def broadcast_sample_rates(indices, sample_rates):
    """
    Broadcasts sample rates to their original positions based on indices.

    :param unique_positions:
        A RaggedTensor containing the unique elements of each sequence.
    :param indices:
        A RaggedTensor containing the indices of the original elements.
    :param sample_rates:
        A RaggedTensor of sample rates corresponding to unique_positions.
    :return:
        A RaggedTensor of broadcasted sample rates with the same shape as
        indices.
    """
    # Flatten the unique positions and their corresponding sample rates
    flat_sample_rates = sample_rates.flat_values

    # Since indices are aligned with unique_positions, use them to broadcast sample rates
    broadcasted_sample_rates = tf.gather(flat_sample_rates, indices.flat_values)

    # Reconstruct the ragged structure
    broadcasted_sample_rates_ragged = tf.RaggedTensor.from_row_splits(
        values=broadcasted_sample_rates, row_splits=indices.row_splits
    )

    return broadcasted_sample_rates_ragged



def sample_positions(positions, rate=0.1):
    """
    Samples positions and nucleotides based on a given rate.

    :param positions:

    :param rate:

    :return:

    """
    unique_positions, indices = extract_unique_positions(positions)
    position_sample_rate = create_random_uniform_ragged(unique_positions, rate)

    def map_sequence(args):
        seq_indices, seq_sample_rates = args
        return tf.gather(seq_sample_rates, seq_indices)

    position_sample_rate = tf.map_fn(
        map_sequence, (indices, position_sample_rate),
        fn_output_signature=tf.float32
    )
    nucleotide_sample_rate = rate / tf.random.uniform(
        shape=tf.shape(positions), minval=rate, maxval=1.0
    )

    # Element-wise multiplication of the two sample rates
    sample_rates = position_sample_rate * nucleotide_sample_rate

    return sample_rates



class NucleotideGenerator(tf.keras.layers.Layer):

    def __init__(self, embedding_dim, num_heads, ff_dim, num_layers, **kwargs):
        super(NucleotideGenerator, self).__init__(**kwargs)

        keys = tf.constant(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            dtype=tf.int32
        )
        values = tf.constant(
            [
                'A', 'C', 'G', 'T', 'R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D',
                'H', 'V', 'N'
            ], dtype=tf.string
        )
        self.replacement_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value='N'
        )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers

        self.nucleotide_embedding_layer = NucleotideEmbeddingLayer(embedding_dim)
        self.metric_encoding_layer = MetricEmbedding(embedding_dim)

        self.encoder_blocks = [
            EncoderBlock(embedding_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ]

        self.output = tf.keras.layers.Dense(len(keys), activation='softmax')


    def call(self, inputs, training=True):

        nucleotides, positions, metrics = inputs

        nucleotide_embedding = self.nucleotide_embedding_layer(nucleotides)
        metric_encoding = self.metric_encoding_layer(metrics)
        embeddings = nucleotide_embedding + metric_encoding

        for encoder_block in self.encoder_blocks:
            embeddings = encoder_block([embeddings, positions])

        probabilities = self.output(embeddings)

        return probabilities

    # @staticmethod
    # def find_median_position(tensor):
    #     # Ensure tensor is float32 for NaN handling
    #     tensor = tf.cast(tensor, tf.float32)
    #
    #     # Function to process each sequence, excluding -1 values and finding the median
    #     def process_sequence(sequence):
    #         # Filter out -1 values directly within each sequence
    #         filtered_sequence = tf.boolean_mask(sequence, sequence != -1.0)
    #         # Sort the filtered sequence
    #         sorted_sequence = tf.sort(filtered_sequence)
    #         # Calculate the median index
    #         num_valid = tf.size(sorted_sequence)
    #         median_index = (num_valid - 1) // 2
    #         # Fetch the median value based on the median index
    #         median_value = sorted_sequence[median_index]
    #         return median_value
    #
    #     # Apply the function to each sequence in the batch
    #     median_positions = tf.map_fn(process_sequence, tensor, fn_output_signature=tf.float32)
    #
    #     return tf.cast(median_positions, tf.int32)

    def generate_replacements(self, inputs, prop_replaced=0.1):

        # get media positions of each sequence in the batch
        nucleotides, positions, metrics = inputs

        # median

        self.call(inputs, training=True)

        # Generate a mask for the tokens to replace
        mask = tf.random.uniform(tf.shape(nucleotides), maxval=1.0) < prop_replaced

        # Replace the tokens
        replacements = tf.random.categorical(probabilities, 1)[:, :, 0]
        replacements = tf.where(mask, replacements, nucleotides)

        return replacements


