import torch
import torch.nn as nn
import torch.nn.functional as F
# TODO: Sort out the imports.
from components.rotary_encoding import (
    compute_theta_vector, compute_rotation_angles, apply_dimensionwise_rotation
)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention layer with positional encoding.

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

        # Initialise scaling vectors
        self.query_scale = nn.Parameter(torch.ones(1, 1, emb_dim))
        self.key_scale = nn.Parameter(torch.ones(1, 1, emb_dim))
        self.value_scale = nn.Parameter(torch.ones(1, 1, emb_dim))
        self.out_scale = nn.Parameter(torch.ones(1, 1, emb_dim))


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

    def forward(self, x, positions):
        batch_size, seq_length, emb_dim = x.size()
        # Linear projections
        Q = self.query(x) * self.query_scale
        K = self.key(x) * self.key_scale
        V = self.value(x) * self.value_scale
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
        output = self.out(context) * self.out_scale

        return output
