import torch
import torch.nn as nn
import torch.nn.functional as F
# TODO: Sort out the imports.
from components.rotary_encoding import (
    compute_theta_vector, compute_rotation_angles, apply_dimensionwise_rotation
)
from components.hyena import FeedForward
from components.better_device_handling import Module

# class RoPEMultiHeadSelfAttention(Module):
#     """
#     Multi-Head Self-Attention layer with rotary positional encoding.
#
#     :param emb_dim:
#         The dimensionality of the embedding space.
#     :param num_heads:
#         The number of attention heads.
#
#     :param inputs:
#         x: Input tensor of shape (batch_size, seq_length, emb_dim).
#         positions: Positional information tensor of shape
#         (batch_size, seq_length).
#
#     :returns:
#         Output tensor after applying multi-head self-attention.
#     """
#     def __init__(self, emb_dim, num_heads):
#         super(RoPEMultiHeadSelfAttention, self).__init__()
#         assert emb_dim % num_heads == 0, \
#             "Embedding dimension must be divisible by the number of heads"
#
#         self.num_heads = num_heads
#         self.head_dim = emb_dim // num_heads
#         self.emb_dim = emb_dim
#
#         self.query = nn.Linear(emb_dim, emb_dim)
#         self.key = nn.Linear(emb_dim, emb_dim)
#         self.value = nn.Linear(emb_dim, emb_dim)
#         self.out = nn.Linear(emb_dim, emb_dim)
#
#         # Initialise scaling vectors
#         self.query_scale = nn.Parameter(torch.ones(1, 1, emb_dim))
#         self.key_scale = nn.Parameter(torch.ones(1, 1, emb_dim))
#         self.value_scale = nn.Parameter(torch.ones(1, 1, emb_dim))
#         self.out_scale = nn.Parameter(torch.ones(1, 1, emb_dim))
#
#
#         # For rotary encoding
#         self.theta_vector = compute_theta_vector(self.head_dim)
#
#     def init_scaling_vectors(self):
#         nn.init.ones_(self.query_scale)
#         nn.init.ones_(self.key_scale)
#         nn.init.ones_(self.value_scale)
#         nn.init.ones_(self.out_scale)
#
#     def freeze_scaling_vectors(self):
#         self.query_scale.requires_grad = False
#         self.key_scale.requires_grad = False
#         self.value_scale.requires_grad = False
#         self.out_scale.requires_grad = False
#
#     def unfreeze_scaling_vectors(self):
#         self.query_scale.requires_grad = True
#         self.key_scale.requires_grad = True
#         self.value_scale.requires_grad = True
#         self.out_scale.requires_grad = True
#
#     def forward(self, x, positions):
#         batch_size, seq_length, emb_dim = x.size()
#         # Linear projections
#         Q = self.query(x) * self.query_scale
#         K = self.key(x) * self.key_scale
#         V = self.value(x) * self.value_scale
#         # Reshape for multi-head attention
#         Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim)
#         K = K.view(batch_size, seq_length, self.num_heads, self.head_dim)
#         V = V.view(batch_size, seq_length, self.num_heads, self.head_dim)
#         # Compute rotation matrices using positional information
#         rotation_matrices = compute_rotation_angles(
#             positions, self.head_dim, self.theta_vector
#         )
#         # Expand rotation matrices to match the number of heads
#         rotation_matrices_expanded = rotation_matrices.unsqueeze(1).repeat(
#             1, self.num_heads, 1, 1, 1, 1
#         )
#         Q_rotated = apply_dimensionwise_rotation(
#             Q.view(
#                 batch_size, seq_length, self.num_heads, self.head_dim
#             ).permute(0, 2, 1, 3).contiguous().view(
#                 -1, seq_length, self.head_dim
#             ),
#             rotation_matrices_expanded.view(
#                 -1, seq_length, self.head_dim // 2, 2, 2
#             )
#         ).view(batch_size, self.num_heads, seq_length, self.head_dim)
#         K_rotated = apply_dimensionwise_rotation(
#             K.view(
#                 batch_size, seq_length, self.num_heads, self.head_dim
#             ).permute(0, 2, 1, 3).contiguous().view(
#                 -1, seq_length, self.head_dim
#             ),
#             rotation_matrices_expanded.view(
#                 -1, seq_length, self.head_dim // 2, 2, 2
#             )
#         ).view(batch_size, self.num_heads, seq_length, self.head_dim)
#
#         V = V.view(
#             batch_size, seq_length, self.num_heads, self.head_dim
#         ).permute(0, 2, 1, 3)
#         # Compute attention scores
#         attention_scores = torch.matmul(
#             Q_rotated, K_rotated.transpose(-1, -2)
#         ) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
#         attention_probs = F.softmax(attention_scores, dim=-1)
#         # Compute the context
#         context = torch.matmul(attention_probs, V)
#         # Concatenate heads and put through final linear layer
#         context = context.permute(0, 2, 1, 3).contiguous().view(
#             batch_size, seq_length, self.emb_dim
#         )
#         output = self.out(context) * self.out_scale
#
#         return output


class RoPEMultiHeadSelfAttention(Module):
    """
    Multi-Head Self-Attention layer with rotary positional encoding.

    :param emb_dim:
        The dimensionality of the embedding space.
    :param num_heads:
        The number of attention heads.

    :param inputs:
        x: Input tensor of shape (batch_size, seq_length, emb_dim).
        positions: Positional information tensor of shape
        (batch_size, seq_length).

    :returns:
        Output tensor after applying multi-head self-attention.
    """
    def __init__(self, emb_dim, num_heads, lamda_init=0.5):
        super(RoPEMultiHeadSelfAttention, self).__init__()
        assert emb_dim % num_heads == 0, \
            "Embedding dimension must be divisible by the number of heads"

        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.emb_dim = emb_dim

        self.query = nn.Linear(emb_dim, emb_dim*2)
        self.key = nn.Linear(emb_dim, emb_dim*2)
        self.value = nn.Linear(emb_dim, emb_dim)
        self.out = nn.Linear(emb_dim, emb_dim)

        # Initialise scaling vectors
        self.query_scale = nn.Parameter(torch.ones(1, 1, emb_dim*2))
        self.key_scale = nn.Parameter(torch.ones(1, 1, emb_dim*2))
        self.value_scale = nn.Parameter(torch.ones(1, 1, emb_dim))
        self.out_scale = nn.Parameter(torch.ones(1, 1, emb_dim))

        self.lambda_param = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.fixed_multiplier = 1.0 - lamda_init
        # For rotary encoding
        self.theta_vector = compute_theta_vector(self.head_dim)

    def init_scaling_vectors(self):
        nn.init.ones_(self.query_scale)
        nn.init.ones_(self.key_scale)
        nn.init.ones_(self.value_scale)
        nn.init.ones_(self.out_scale)

    def freeze_scaling_vectors(self):
        self.query_scale.requires_grad = False
        self.key_scale.requires_grad = False
        self.value_scale.requires_grad = False
        self.out_scale.requires_grad = False

    def unfreeze_scaling_vectors(self):
        self.query_scale.requires_grad = True
        self.key_scale.requires_grad = True
        self.value_scale.requires_grad = True
        self.out_scale.requires_grad = True

    def forward(self, x):
        batch_size, seq_length, emb_dim = x.size()

        # Create positions 0 to seq_length for each sequence in the batch
        positions = torch.arange(seq_length, device=x.device).repeat(batch_size, 1)

        # Linear projections
        Q = self.query(x) * self.query_scale
        K = self.key(x) * self.key_scale
        V = self.value(x) * self.value_scale

        Q1, Q2 = torch.chunk(Q, 2, dim=-1)
        K1, K2 = torch.chunk(K, 2, dim=-1)

        # Reshape for multi-head attention
        Q1 = Q1.view(batch_size, seq_length, self.num_heads, self.head_dim)
        Q2 = Q2.view(batch_size, seq_length, self.num_heads, self.head_dim)
        K1 = K1.view(batch_size, seq_length, self.num_heads, self.head_dim)
        K2 = K2.view(batch_size, seq_length, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim)
        # Compute rotation matrices using positional information
        rotation_matrices = compute_rotation_angles(
            positions, self.head_dim, self.theta_vector
        )
        # Expand rotation matrices to match the number of heads
        rotation_matrices_expanded = rotation_matrices.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1, 1, 1
        )
        Q1_rotated = apply_dimensionwise_rotation(
            Q1.view(
                batch_size, seq_length, self.num_heads, self.head_dim
            ).permute(0, 2, 1, 3).contiguous().view(
                -1, seq_length, self.head_dim
            ),
            rotation_matrices_expanded.view(
                -1, seq_length, self.head_dim // 2, 2, 2
            )
        ).view(batch_size, self.num_heads, seq_length, self.head_dim)
        Q2_rotated = apply_dimensionwise_rotation(
            Q2.view(
                batch_size, seq_length, self.num_heads, self.head_dim
            ).permute(0, 2, 1, 3).contiguous().view(
                -1, seq_length, self.head_dim
            ),
            rotation_matrices_expanded.view(
                -1, seq_length, self.head_dim // 2, 2, 2
            )
        ).view(batch_size, self.num_heads, seq_length, self.head_dim)
        K1_rotated = apply_dimensionwise_rotation(
            K1.view(
                batch_size, seq_length, self.num_heads, self.head_dim
            ).permute(0, 2, 1, 3).contiguous().view(
                -1, seq_length, self.head_dim
            ),
            rotation_matrices_expanded.view(
                -1, seq_length, self.head_dim // 2, 2, 2
            )
        ).view(batch_size, self.num_heads, seq_length, self.head_dim)
        K2_rotated = apply_dimensionwise_rotation(
            K2.view(
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
        attention_scores_1 = torch.matmul(
            Q1_rotated, K1_rotated.transpose(-1, -2)
        ) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_probs_1 = F.softmax(attention_scores_1, dim=-1)
        attention_scores_2 = torch.matmul(
            Q2_rotated, K2_rotated.transpose(-1, -2)
        ) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_probs_2 = F.softmax(attention_scores_2, dim=-1)
        lambda_value = torch.sigmoid(self.lambda_param)

        differential_attention = attention_probs_1 - lambda_value * attention_probs_2
        # Compute the context
        context = torch.matmul(differential_attention, V)

        context = context.view(batch_size * self.num_heads, seq_length, self.head_dim)
        context = nn.functional.layer_norm(context, normalized_shape=(self.head_dim,))
        context = context.view(batch_size, self.num_heads, seq_length, self.head_dim)

        context = context * self.fixed_multiplier

        # Concatenate heads and put through final linear layer
        context = context.permute(0, 2, 1, 3).contiguous().view(
            batch_size, seq_length, self.emb_dim
        )
        output = self.out(context) * self.out_scale

        return output


class MultiHeadSelfAttention(Module):
    """
    Multi-Head Self-Attention layer without rotary positional encoding.

    :param emb_dim:
        The dimensionality of the embedding space.
    :param num_heads:
        The number of attention heads.

    :param inputs:
        x: Input tensor of shape (batch_size, seq_length, emb_dim).

    :returns:
        Output tensor after applying multi-head self-attention.
    """
    def __init__(self, emb_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert emb_dim % num_heads == 0, \
            "Embedding dimension must be divisible by the number of heads"

        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.emb_dim = emb_dim

        self.query = nn.Linear(emb_dim, emb_dim, bias=False)
        self.key = nn.Linear(emb_dim, emb_dim, bias=False)
        self.value = nn.Linear(emb_dim, emb_dim, bias=False)
        self.out = nn.Linear(emb_dim, emb_dim, bias=False)

        # Initialise scaling vectors
        self.query_scale = nn.Parameter(torch.ones(1, 1, emb_dim))
        self.key_scale = nn.Parameter(torch.ones(1, 1, emb_dim))
        self.value_scale = nn.Parameter(torch.ones(1, 1, emb_dim))
        self.out_scale = nn.Parameter(torch.ones(1, 1, emb_dim))


    def init_scaling_vectors(self):
        nn.init.ones_(self.query_scale)
        nn.init.ones_(self.key_scale)
        nn.init.ones_(self.value_scale)
        nn.init.ones_(self.out_scale)

    def freeze_scaling_vectors(self):
        self.query_scale.requires_grad = False
        self.key_scale.requires_grad = False
        self.value_scale.requires_grad = False
        self.out_scale.requires_grad = False

    def unfreeze_scaling_vectors(self):
        self.query_scale.requires_grad = True
        self.key_scale.requires_grad = True
        self.value_scale.requires_grad = True
        self.out_scale.requires_grad = True

    def forward(self, x):
        batch_size, seq_length, emb_dim = x.size()
        # Linear projections
        Q = self.query(x) * self.query_scale
        K = self.key(x) * self.key_scale
        V = self.value(x) * self.value_scale
        # Reshape for multi-head attention
        Q = Q.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        K = K.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        V = V.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)

        # Compute attention scores
        attention_scores = torch.matmul(
            Q, K.transpose(-1, -2)
        ) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_probs = F.softmax(attention_scores, dim=-1)
        # Compute the context
        context = torch.matmul(attention_probs, V)
        # Concatenate heads and put through final linear layer
        context = context.permute(0, 2, 1, 3).contiguous().view(
            batch_size, seq_length, self.emb_dim
        )
        output = self.out(context) * self.out_scale

        return output


class TransformerBlock(Module):
    """
    Transformer block that can switch between self-attention and Hyena block.

    :param emb_dim:
        Dimension of the input embeddings.
    :param heads:
        Number of heads for self-attention or global filters for the Hyena
        block.
    """
    def __init__(self, emb_dim, heads):
        super(TransformerBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(emb_dim, eps=1e-7)
        nn.init.ones_(self.layer_norm1.weight)
        nn.init.zeros_(self.layer_norm1.bias)
        self.layer_norm2 = nn.LayerNorm(emb_dim, eps=1e-7)
        nn.init.ones_(self.layer_norm2.weight)
        nn.init.zeros_(self.layer_norm2.bias)
        self.feed_forward = FeedForward(
            emb_dim, hidden_dim=emb_dim, activation="mish"
        )
        self.self_attention = RoPEMultiHeadSelfAttention(
            emb_dim, num_heads=heads
        )

    def forward(self, embeddings, positions):
        """
        Perform the forward pass of the transformer block.

        :param embeddings:
            Input tensor of shape (batch_size, seq_length, emb_dim).
        :param positions:
            Position tensor of shape (batch_size, seq_length).
        :returns:
            Output tensor after applying self-attention and feedforward network.
        """
        # Apply layer normalisation before self-attention
        self_attention_input = self.layer_norm1(embeddings)
        self_attention_out = self.self_attention(
            self_attention_input, positions
        ) + embeddings
        ffn_input = self.layer_norm2(self_attention_out)
        ffn_output = self.feed_forward(ffn_input)
        # Residual connection
        output = ffn_output + self_attention_out

        return output
