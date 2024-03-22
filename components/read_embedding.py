
import tensorflow as tf
import numpy as np
from extract_reads import extract_reads_around_position, extract_reads_from_position_onward, get_read_info


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
#         # Define the nucleotide mapping
#         self.nucleotide_to_index = {
#             'A': 0, 'C': 1, 'G': 2, 'T': 3, 'R': 4, 'Y': 5, 'S': 6, 'W': 7,
#             'K': 8, 'M': 9, 'B': 10, 'D': 11, 'H': 12, 'V': 13, 'N': 14
#         }
#         self.num_nucleotides = len(self.nucleotide_to_index)
#         self.embedding = self.add_weight(
#             shape=(self.num_nucleotides, self.embedding_dim),
#             initializer="he_normal",
#             trainable=True,
#             name="nucleotide_embeddings"
#         )
#
#
#     def build(self, input_shape):
#         """
#         Creates the embedding matrix for nucleotides.
#         """
#         super(NucleotideEmbeddingLayer, self).build(input_shape)
#
#     def call(self, inputs):
#         """
#         Maps the input nucleotide indices to their embeddings.
#
#         :param inputs:
#             A batch of sequences with nucleotide indices.
#
#         :return:
#             The corresponding nucleotide embeddings.
#         """
#         # Convert nucleotide characters to indices
#         input_indices = tf.map_fn(
#             lambda x: tf.map_fn(
#                 lambda y: self.nucleotide_to_index[y.numpy().decode('utf-8')],
#                 x, dtype=tf.int32
#             ),
#             inputs, dtype=tf.int32
#         )
#         # Lookup embeddings
#         return tf.nn.embedding_lookup(self.embedding, input_indices)
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
# import tensorflow as tf

class NucleotideEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, **kwargs):
        """
        Initializes the nucleotide embedding layer.

        :param embedding_dim: The dimensionality of the embedding space.
        """
        super(NucleotideEmbeddingLayer, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim

        # Define the nucleotide mapping as a TensorFlow lookup table
        keys = tf.constant(
            [
                'A', 'C', 'G', 'T', 'R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D', 'H',
                'V', 'N'
            ]
        )
        values = tf.range(start=0, limit=len(keys), dtype=tf.int64)
        self.nucleotide_to_index_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=-1
        )

    def build(self, input_shape):
        """
        Creates the embedding matrix for nucleotides.
        """
        self.embedding = self.add_weight(
            shape=(15, self.embedding_dim),
            initializer="he_normal",
            trainable=True,
            name="nucleotide_embeddings"
        )
        super(NucleotideEmbeddingLayer, self).build(input_shape)

    def call(self, inputs):
        """
        Maps the input nucleotide sequences to their embeddings.

        :param inputs:
            A batch of sequences as tf.string tensors.
        :return:
            The corresponding nucleotide embeddings.
        """
        # Convert nucleotide characters to indices using the lookup table
        flat_inputs = tf.strings.unicode_split(inputs, 'UTF-8')
        input_indices = self.nucleotide_to_index_table.lookup(flat_inputs)

        # Lookup embeddings and reshape to match the original input shape plus embedding dimension
        embeddings = tf.nn.embedding_lookup(self.embedding, input_indices)
        return embeddings

    def compute_output_shape(self, input_shape):
        return input_shape + (self.embedding_dim,)

    def get_config(self):
        config = super(NucleotideEmbeddingLayer, self).get_config()
        config.update({
            'embedding_dim': self.embedding_dim
        })
        return config


class MetricEmbedding(tf.keras.layers.Layer):
    """
    Custom layer for metric embeddings. CURRENTLY, IT IS IN ITS SIMPLEST FORM.
    """

    def __init__(self, embedding_dim, **kwargs):
        """
        Initialises the metric embedding layer.

        :param embedding_dim:
            The dimensionality of the embedding space used for the nucleotide
            representations.
        :param kwargs:
            Additional keyword arguments for the layer.
        """
        super(MetricEmbedding, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        self.embedding_matrix = self.add_weight(
            shape=(input_shape[-1], self.embedding_dim),
            initializer="he_normal",
            trainable=True,
            name="metric_embeddings"
        )
        super(MetricEmbedding, self).build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.embedding_matrix)



# Example usage
bam_file_path = 'TEST_DATA/HG00326.chrom11.ILLUMINA.bwa.FIN.low_coverage.20120522.bam'
chromosome = '11'
# Random position on chromosome 11
position = np.random.randint(1, 135086622)
nucleotide_threshold = 1024

# read_dict = extract_reads_around_position(
#     bam_file_path, chromosome, position, nucleotide_threshold
# )
read_dict = extract_reads_from_position_onward(
    bam_file_path, chromosome, position, nucleotide_threshold
)
read_info = get_read_info(read_dict)

# plot_nucleotide_coverage(read_dict, position)
# plot_mapping_and_base_quality_histogram(read_info, log_scale=True)


# Create embedding layer
# embedding_dim = 4
# nucleotide_embedding = NucleotideEmbeddingLayer(embedding_dim)
#
# # Create metric embedding layer
# metric_embedding = MetricEmbedding(embedding_dim)

# Unpack the read dictionary into a single list of tuples containing the
# nucleotide base, base quality, read quality and binary flags.

read_id = []
nucleotide_bases = []

base_qualities = []
read_qualities = []
cigar_match = []
cigar_insertion = []
bitwise_flags = []

positions = []
for i, read in enumerate(read_info.values()):
    read_id.extend([i] * len(read['adjusted_base_qualities']))
    nucleotide_bases.extend([*read['query_sequence']])

    base_qualities.extend(read['adjusted_base_qualities'])
    read_qualities.extend([read['mapping_quality']] * len(read['adjusted_base_qualities']))
    cigar_match.extend([read['cigar_match_vector']] * len(read['adjusted_base_qualities']))
    cigar_insertion.extend([read['cigar_insertion_vector']] * len(read['adjusted_base_qualities']))
    bitwise_flags.extend([read['binary_flag_vector']] * len(read['adjusted_base_qualities']))
    positions.extend([read['positions']])


