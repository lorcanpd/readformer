
import tensorflow as tf
# import numpy as np
# from extract_reads import extract_reads_around_position, extract_reads_from_position_onward, get_read_info


class NucleotideEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, **kwargs):
        """
        Initialises the nucleotide embedding layer.

        :param embedding_dim:
            The dimensionality of the embedding space.
        """
        super(NucleotideEmbeddingLayer, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim

        # Define the nucleotide mapping as a TensorFlow lookup table
        keys = tf.constant(
            [
                # Empty string '' for padding.
                '', 'A', 'C', 'G', 'T', 'R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D',
                'H', 'V', 'N'
            ], dtype=tf.string
        )
        values = tf.range(start=0, limit=len(keys), dtype=tf.int32)

        self.nucleotide_to_index_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=len(keys) - 1  # Default value for padding
        )

        self.embedding = tf.keras.layers.Embedding(
            input_dim=len(keys)-1,  # Exclude padding
            output_dim=self.embedding_dim,
            embeddings_initializer="he_normal",
            mask_zero=True,  # Enables padding handling
            trainable=True,
            name="nucleotide_embeddings"
        )

    def call(self, inputs):
        """
        Maps the input nucleotide sequences to their embeddings.

        :param inputs:
            A batch of sequences as tf.string tensors.
        :return:
            The corresponding nucleotide embeddings.
        """
        input_indices = self.nucleotide_to_index_table.lookup(inputs)
        embeddings = self.embedding(input_indices)
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

    def __init__(self, embedding_dim, name=None, **kwargs):
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
        if name is not None:
            self.name = name + "_metric_embedding"
        else:
            self.name = "metric_embedding"

    def build(self, input_shape):
        self.embedding_matrix = self.add_weight(
            shape=(input_shape[-1], self.embedding_dim),
            initializer="he_normal",
            trainable=True,
            name=self.name + "_matrix"
        )
        super(MetricEmbedding, self).build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.embedding_matrix)

