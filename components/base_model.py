import torch.nn as nn

# TODO fix the import statements
from components.hyena import HyenaBlock, FeedForward
from components.self_attention import MultiHeadSelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, heads, hyena=False, kernel_size=3):
        super(TransformerBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.layer_norm2 = nn.LayerNorm(emb_dim)
        self.feed_forward = FeedForward(
            emb_dim, hidden_dim=emb_dim * 2, activation="mish"
        )
        self.hyena = hyena
        if hyena:
            self.self_attention = HyenaBlock(
                emb_dim, n_order=heads, kernel_size=kernel_size
            )
        else:
            self.self_attention = MultiHeadSelfAttention(
                emb_dim, num_heads=heads
            )

    def forward(self, x, positions):
        # x: input tensor of shape (batch_size, seq_length, emb_dim)

        # Apply layer normalisation before self-attention
        normed_x = self.layer_norm1(x)
        self_attention_output = self.self_attention(normed_x, positions)
        x = x + self_attention_output

        # Apply layer normalisation before feedforward network
        normed_x = self.layer_norm2(x)
        ff_output = self.feed_forward(normed_x)
        x = x + ff_output

        return x

# # Example usage:
# import torch
# batch_size = 2
# seq_length = 8
# emb_dim = 256
#
# # Example input tensor (e.g., embedding representation of the input sequence)
# inputs = torch.randn(batch_size, seq_length, emb_dim)
# positions = torch.tensor([[1, 2, 3, 4, 5, 1, 2, 3], [2, 3, 4, 5, 6, 4, 5, 6]])
#
# # Instantiate the transformer block
# hyena_block = TransformerBlock(
#     emb_dim, heads=16, hyena=True, kernel_size=3
# )
#
# self_attention_block = TransformerBlock(
#     emb_dim, heads=16
# )
#
# # Forward pass through the transformer block
# output = self_attention_block(hyena_block(inputs, positions), positions)
#
#
# print("Output shape:", output.shape)


# OLD TENSOFRLOW IMPLEMENTATION.
# import tensorflow as tf
# from .hyena import HyenaBlock
#
# class RowWiseFeedForwardLayer(tf.keras.layers.Layer):
#     def __init__(self, d_model, dff, **kwargs):
#         """
#         Initialises the feed-forward layer.
#
#         :param d_model:
#             The dimensionality of the embedding vectors.
#         :param dff:
#             The dimensionality of the feed-forward layer.
#         """
#         super(RowWiseFeedForwardLayer, self).__init__(**kwargs)
#         self.d_model = d_model
#         self.dff = dff
#
#         init = tf.keras.initializers.he_uniform()
#         self.W1 = tf.Variable(
#             initial_value=init(
#                 shape=[self.d_model, self.dff],
#                 dtype='float32'
#             ),
#             name='W1',
#             trainable=True
#         )
#         self.b1 = tf.Variable(
#             initial_value=tf.zeros([self.dff]),
#             name='b1',
#             trainable=True
#         )
#         self.W2 = tf.Variable(
#             initial_value=init(
#                 shape=[self.dff, self.d_model],
#                 dtype='float32'
#             ),
#             name='W2',
#             trainable=True
#         )
#         self.b2 = tf.Variable(
#             initial_value=tf.zeros([self.d_model]),
#             name='b2',
#             trainable=True
#         )
#
#     def call(self, inputs, training=False):
#         """
#         Forward pass for the feed-forward layer.
#
#         :param inputs:
#             The input tensor.
#         :return:
#             The output tensor.
#         """
#         x = tf.tensordot(inputs, self.W1, axes=1) + self.b1
#         x = tf.nn.gelu(x)
#         if training:
#             x = tf.nn.dropout(x, rate=0.1)
#
#         x = tf.tensordot(x, self.W2, axes=1) + self.b2
#
#         return x
#
#     def get_config(self):
#         config = super(RowWiseFeedForwardLayer, self).get_config()
#         config.update({
#             'd_model': self.d_model,
#             'dff': self.dff
#         })
#         return config
#
#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)
#
#     def compute_output_shape(self, input_shape):
#         return input_shape
#
#
# class EncoderBlock(tf.keras.layers.Layer):
#
#     def __init__(self, d_model, num_heads, dff, **kwargs):
#         """
#         Initialises the encoder block.
#
#         :param d_model:
#             The dimensionality of the embedding vectors.
#         :param num_heads:
#             The number of attention heads.
#         :param dff:
#             The dimensionality of the feed-forward layer.
#         """
#         super(EncoderBlock, self).__init__(**kwargs)
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.dff = dff
#
#         self.multi_head_self_attention = MultiHeadRotarySelfAttentionLayer(
#             d_model=d_model, num_heads=num_heads
#         )
#         self.feed_forward = RowWiseFeedForwardLayer(
#             d_model=d_model, dff=dff
#         )
#
#         self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#         self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#
#     def call(self, inputs, training=False):
#         """
#         Forward pass for the encoder block. The input tensor is passed through
#         the multi-head self-attention layer and the feed-forward layer.
#
#         :param inputs:
#             A list containing the input tensor and the genomic position tensor.
#         :param training:
#             A boolean indicating whether the model is in training mode.
#         :return:
#             The output tensor.
#         """
#         x = self.multi_head_self_attention(inputs, training=training)
#         x = self.layernorm1(inputs[0] + x)
#
#         y = self.feed_forward(x, training=training)
#         y = self.layernorm2(x + y)
#
#         return y
#
#     def get_config(self):
#         config = super(EncoderBlock, self).get_config()
#         config.update({
#             'd_model': self.d_model,
#             'num_heads': self.num_heads,
#             'dff': self.dff
#         })
#         return config
#
#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)
#
#     def compute_output_shape(self, input_shape):
#         return input_shape
