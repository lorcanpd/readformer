import torch
import torch.nn as nn
import torch.nn.functional as F
# TODO: Sort out the imports.
from components.rotary_encoding import (
    compute_theta_vector, compute_rotation_angles, apply_dimensionwise_rotation
)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert emb_dim % num_heads == 0, \
            "Embedding dimension must be divisible by the number of heads"

        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.emb_dim = emb_dim

        self.query = nn.Linear(emb_dim, emb_dim)
        self.key = nn.Linear(emb_dim, emb_dim)
        self.value = nn.Linear(emb_dim, emb_dim)
        self.out = nn.Linear(emb_dim, emb_dim)

        self.theta_vector = compute_theta_vector(self.head_dim)

    def forward(self, x, positions):
        batch_size, seq_length, emb_dim = x.size()
        # Linear projections
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim)
        # Compute rotation matrices using positional information
        rotation_matrices = compute_rotation_angles(
            positions, self.head_dim, self.theta_vector
        )
        # Expand rotation matrices to match the number of heads
        rotation_matrices_expanded = rotation_matrices.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1, 1, 1
        )
        Q_rotated = apply_dimensionwise_rotation(
            Q.view(
                batch_size, seq_length, self.num_heads, self.head_dim
            ).permute(0, 2, 1, 3).contiguous().view(
                -1, seq_length, self.head_dim
            ),
            rotation_matrices_expanded.view(
                -1, seq_length, self.head_dim // 2, 2, 2
            )
        ).view(batch_size, self.num_heads, seq_length, self.head_dim)
        K_rotated = apply_dimensionwise_rotation(
            K.view(
                batch_size, seq_length, self.num_heads, self.head_dim
            ).permute(0, 2, 1, 3).contiguous().view(
                -1, seq_length, self.head_dim
            ),
            rotation_matrices_expanded.view(
                -1, seq_length, self.head_dim // 2, 2, 2
            )
        ).view(batch_size, self.num_heads, seq_length, self.head_dim)

        V = V.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        # Compute attention scores
        attention_scores = torch.matmul(
            Q_rotated, K_rotated.transpose(-1, -2)
        ) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_probs = F.softmax(attention_scores, dim=-1)
        # Compute the context
        context = torch.matmul(attention_probs, V)
        # Concatenate heads and put through final linear layer
        context = context.permute(0, 2, 1, 3).contiguous().view(
            batch_size, seq_length, self.emb_dim
        )
        output = self.out(context)

        return output

# # Example usage:
# batch_size = 2
# seq_length = 8
# emb_dim = 64
# num_heads = 8
#
# # Example input tensor (e.g., embedding representation of the input sequence)
# inputs = torch.randn(batch_size, seq_length, emb_dim)
# positions = torch.arange(seq_length).unsqueeze(0).repeat(batch_size, 1)  # Example positions
#
# # Instantiate the multi-head self-attention module
# mhsa = MultiHeadSelfAttention(emb_dim, num_heads)
#
# # Forward pass through the multi-head self-attention module
# output = mhsa(inputs, positions)
#
# print("Output shape:", output.shape)
#


# OLD TENSORFLOW IMPLEMENTATION.
# import tensorflow as tf
# from rotary_encoding import compute_theta_vector, compute_rotation_angles, apply_dimensionwise_rotation
#
# # TODO: ADD EFFICIENT FINEUNING PARAMTERES TO EACH COMPONENT THAT CAN BE
# #  INDEPENDENTLY FROZEN OR UNFROZEN. SAME FOR THE MAIN MODEL PARAMS.
# class MultiHeadRotarySelfAttentionLayer(tf.keras.layers.Layer):
#     def __init__(self, d_model, num_heads, **kwargs):
#         """
#         Initialises the self-attention layer.
#
#         :param d_model:
#             The dimensionality of the embedding vectors.
#         :param num_heads:
#             The number of attention heads.
#         """
#         super(MultiHeadRotarySelfAttentionLayer, self).__init__(**kwargs)
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.dim_per_head = d_model // num_heads
#
#         init = tf.keras.initializers.he_uniform()
#         # Define the weight matrices for Q, K, V, and the output
#         self.Wq = tf.Variable(
#             initial_value=init(
#                 shape=[self.d_model, self.num_heads * self.dim_per_head],
#                 dtype='float32'
#             ),
#             name='Wq',
#             trainable=True
#         )
#         self.bq = tf.Variable(
#             initial_value=tf.zeros([self.num_heads * self.dim_per_head]),
#             name='bq',
#             trainable=True
#         )
#         self.Wk = tf.Variable(
#             initial_value=init(
#                 shape=[self.d_model, self.num_heads * self.dim_per_head],
#                 dtype='float32'
#             ),
#             name='Wk',
#             trainable=True
#         )
#         self.bk = tf.Variable(
#             initial_value=tf.zeros([self.num_heads * self.dim_per_head]),
#             name='bk',
#             trainable=True
#         )
#         self.Wv = tf.Variable(
#             initial_value=init(
#                 shape=[self.d_model, self.num_heads * self.dim_per_head],
#                 dtype='float32'
#             ),
#             name='Wv',
#             trainable=True
#         )
#         self.bv = tf.Variable(
#             initial_value=tf.zeros([self.num_heads * self.dim_per_head]),
#             name='bv',
#             trainable=True
#         )
#         self.Wo = tf.Variable(
#             initial_value=init(
#                 shape=[self.num_heads * self.dim_per_head, self.d_model],
#                 dtype='float32'
#             ),
#             name='Wo',
#             trainable=True
#         )
#
#         self.theta = compute_theta_vector(d_model)
#
    # def call(self, inputs, training=False):
    #     """
    #     Approximate the self-attention mechanism using Fast Attention Via
    #     Orthogonal Random Features (FAVOR+). Query and key vectors are rotated
    #     using their genomic position to encode relative nucleotide positions.
    #
    #     :param inputs:
    #         A list containing the input tensor and the genomic position tensor.
    #     :return:
    #         The self-attention output tensor.
    #     """
    #     Q = tf.tensordot(inputs[0], self.Wq, axes=1) + self.bq
    #     K = tf.tensordot(inputs[0], self.Wk, axes=1) + self.bk
    #     V = tf.tensordot(inputs[0], self.Wv, axes=1) + self.bv
    #
    #     if training:
    #         Q = tf.nn.dropout(Q, rate=0.1)
    #         K = tf.nn.dropout(K, rate=0.1)
    #         V = tf.nn.dropout(V, rate=0.1)
    #
    #     # Create rotation matrices for Q and K using the genomic position tensor
    #     rotation_matrices = compute_rotation_angles(
    #         inputs[1], self.d_model, self.theta
    #     )
    #     # Expand rotation_matrices to match the reshaped structure for Q and K
    #     rotation_matrices_expanded = tf.expand_dims(
    #         rotation_matrices,
    #         axis=1  # Add a head dimension
    #     )
    #     rotation_matrices_expanded = tf.repeat(
    #         rotation_matrices_expanded,
    #         repeats=self.num_heads,
    #         axis=1
    #     )
    #     # Apply dimension-wise rotation to Q and K
    #     Q = apply_dimensionwise_rotation(Q, rotation_matrices_expanded)
    #     K = apply_dimensionwise_rotation(K, rotation_matrices_expanded)
    #
    #     # Split the heads
    #     Q_ = tf.reshape(
    #         Q, [-1, tf.shape(Q)[1], self.num_heads, self.dim_per_head]
    #     )
    #     Q_ = tf.transpose(Q_, [0, 2, 1, 3])
    #     K_ = tf.reshape(
    #         K, [-1, tf.shape(K)[1], self.num_heads, self.dim_per_head]
    #     )
    #     K_ = tf.transpose(
    #         K_, [0, 2, 1, 3]
    #     )
    #     V_ = tf.reshape(
    #         V, [-1, tf.shape(V)[1], self.num_heads, self.dim_per_head]
    #     )
    #     V_ = tf.transpose(
    #         V_, [0, 2, 1, 3]
    #     )

#         # Approximate the attention scores using Fast Attention Via Orthogonal
#         # Random Features (FAVOR+) technique
#         # Compute the gaussian random projection matrix for the reshaped Q and K
#         # tensors
#         random_features = tf.random.normal(
#             # TODO determine the best dimensionality for the higher dimensional
#             #  random feature space.
#             shape=[self.dim_per_head, 2 * self.dim_per_head],
#             mean=0.0, stddev=1.0
#         )
#         Q_projected = tf.einsum('bhld,df->bhlf', Q_, random_features)
#         K_projected = tf.einsum('bhld,df->bhlf', K_, random_features)
#
#         Q_proj_cos = tf.cos(Q_projected)
#         Q_proj_sin = tf.sin(Q_projected)
#         K_proj_cos = tf.cos(K_projected)
#         K_proj_sin = tf.sin(K_projected)
#
#         # cos(q') . cos(k')^T + sin(q') . sin(k')^T
#         attention_scores = tf.add(
#             tf.einsum('bhlf,bhlf->bhll', Q_proj_cos, K_proj_cos),
#             tf.einsum('bhlf,bhlf->bhll', Q_proj_sin, K_proj_sin)
#         )
#
#         if training:
#             attention_scores = tf.nn.dropout(attention_scores, rate=0.1)
#
#         # Multiply attention scores by the reshaped V tensor
#         Z = tf.matmul(attention_scores, V_)
#
#         if training:
#             Z = tf.nn.dropout(Z, rate=0.1)
#
#         # Concatenate the heads
#         Z = tf.transpose(Z, [0, 2, 1, 3])
#         Z = tf.reshape(Z, [-1, tf.shape(Z)[1], self.d_model])
#
#         # Apply the output weight matrix
#         output = tf.tensordot(Z, self.Wo, axes=1)
#
#         return output
#
#     def get_config(self):
#
#         config = super(MultiHeadRotarySelfAttentionLayer, self).get_config()
#         config.update({
#             'd_model': self.d_model,
#             'num_heads': self.num_heads
#         })
#         return config
#
#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)
#
#     def compute_output_shape(self, input_shape):
#         return input_shape[0] + (self.d_model,)
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
#
#
# class Readformer(tf.keras.Model):
#
#     def __init__(self, config):
#         super(Readformer, self).__init__()
#
#         self.d_model = config['d_model']
#         self.num_heads = config['num_heads']
#         self.dff = config['dff']
#         self.num_layers = config['num_layers']
#
#         self.encoder_blocks = [
#             EncoderBlock(
#                 d_model=self.d_model,
#                 num_heads=self.num_heads,
#                 dff=self.dff
#             ) for _ in range(self.num_layers)
#         ]
#
#
#     def call(self, inputs, training=False):
#         """
#         Forward pass for the Readformer model.
#
#         :param inputs:
#             A list containing the input nucleotide embedding tensor and the
#             genomic position tensor.
#         :param training:
#             A boolean indicating whether the model is in training mode.
#
#         :return:
#             The output tensor.
#         """
#
#         x, positions = inputs
#         for i in range(self.num_layers):
#             x = self.encoder_blocks[i]([x, positions], training=training)
#
#         return x
#
#
#     def get_config(self):
#         config = {
#             'd_model': self.d_model,
#             'num_heads': self.num_heads,
#             'dff': self.dff,
#             'num_layers': self.num_layers
#         }
#         return config
#
#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)
#
#     def compute_output_shape(self, input_shape):
#         return input_shape[0] + (self.d_model,)
#
