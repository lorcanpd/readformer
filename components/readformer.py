import torch
import torch.nn as nn
import torch.nn.functional as F

from components.hyena import (
    HyenaProjection, HyenaFilter, FFTLongConv, FeedForward,
    sinusoidal_positional_encoding
)
# from components.rotary_encoding import (
#     compute_theta_vector, compute_rotation_angles, apply_dimensionwise_rotation
# )
from components.self_attention import MultiHeadSelfAttention

from components.better_device_handling import Module


# TODO: Create some sort of scaling vector system for rapid fine-tuning. Perhaps
# allow full finetuning for the Hyena-based components and partial finetuning
# for the self-attention components and feedforward.

def adjust_positions(positions):
    dtype = positions.dtype
    adjusted_positions = positions.clone().to(torch.float32)
    adjusted_positions[adjusted_positions == -1] = torch.inf
    # Calculate the minimum value ignoring -1 (now inf)
    min_positions = adjusted_positions.min(dim=-1, keepdim=True)[0]
    # Subtract the minimum values from the original positions, -1 stays
    # unaffected
    adjusted_positions = positions - min_positions
    # Reset -1 values to their original state
    adjusted_positions[positions == -1] = -1

    return adjusted_positions.to(dtype)


def split_into_reads(embeddings, positions):
    batch_size, seq_length, emb_dim = embeddings.shape
    # Calculate position differences to determine the boundaries of segments
    position_differences = torch.diff(
        positions, dim=1,
        append=torch.full((batch_size, 1), -1).to(embeddings.device)
    ).to(embeddings.device)

    segment_starts = torch.cat(
        tensors=(
            torch.full(
                size=(batch_size, 1), fill_value=True, device=embeddings.device
            ),
            position_differences != 1
        ),
        dim=1
    )[..., :-1]
    segment_ends = (position_differences != 1).to(embeddings.device)

    # Gather all segment indices
    segmented_inputs = []
    segmented_positions = []
    segment_starts_indices = []
    segment_ends_indices = []
    batch_indices = []
    max_seg_length = 0
    for b in range(batch_size):
        starts = torch.where(segment_starts[b])[0]
        ends = torch.where(segment_ends[b])[0]
        for start, end in zip(starts, ends):
            if start != end:
                segmented_inputs.append(embeddings[b, start:end + 1])
                segmented_positions.append(positions[b, start:end + 1])
                segment_starts_indices.append(start)
                batch_indices.append(b)
                max_seg_length = max(max_seg_length, end + 1 - start)

    # Pad each segment to the maximum segment length
    padded_inputs = [
        F.pad(seg, (0, 0, 0, max_seg_length - seg.shape[0]))
        for seg in segmented_inputs
    ]
    reshaped_positions = [
        F.pad(pos, (0, max_seg_length - pos.shape[0]), value=-1)
        for pos in segmented_positions
    ]

    return (torch.stack(padded_inputs).to(embeddings.device),
            torch.stack(reshaped_positions).to(embeddings.device),
            torch.stack(segment_starts_indices).to(embeddings.device),
            torch.tensor(batch_indices).to(embeddings.device))


def reassemble_sequences(original_shape, read_tensor, positions, segment_starts, batch_indices):
    # batch_size, seq_length, emb_dim = read_tensor.shape
    output = torch.zeros(original_shape, device=read_tensor.device)

    for processed_read, read_positions, start_idx, batch_idx in zip(
            read_tensor, positions, segment_starts, batch_indices
    ):
        read_vectors = processed_read[read_positions != -1]
        # read_positions = read_positions[read_positions != -1]

        output[batch_idx, start_idx:start_idx + len(read_vectors)] = read_vectors

    return output


def reshape_by_position_and_track(inputs, positions):
    batch_size, seq_length, emb_dim = inputs.shape
    max_embeddings_count = 0
    index_map = {}

    combos = {}

    out_tensor_index = 0
    # Collect embeddings by their positions and sequence
    for b in range(batch_size):
        position_counter = {}
        for i in range(seq_length):
            pos = positions[b, i].item()
            if pos != -1:
                if pos not in position_counter:
                    position_counter[pos] = 0
                else:
                    position_counter[pos] += 1

                if (b, pos) not in combos:
                    combos[(b, pos)] = out_tensor_index
                    out_tensor_index += 1
                key = (b, i)
                value = (combos[(b, pos)], position_counter[pos])
                index_map[key] = value
        max_embeddings_count = max(
            max_embeddings_count, max(position_counter.values()) + 1
        )

    all_positions_tensor = torch.zeros(
        (len(combos), max_embeddings_count, emb_dim), device=inputs.device
    )

    for key, value in index_map.items():
        old_row, old_idx = key
        new_row, new_idx = value
        all_positions_tensor[new_row, new_idx] = inputs[old_row, old_idx]

    return all_positions_tensor, index_map


class RotaryHyenaFilter(Module):
    """
    Wraps HyenaFilter but instead of providing normal positional encodings,
    it provides position-wise rotated embeddings as positional encodings.

    :param emb_dim:
        Dimension of the input embeddings.
    :param n_order:
        The number of long convolution filters to generate.
    """

    def __init__(self, emb_dim, n_order, num_heads=4):
        super(RotaryHyenaFilter, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.filter_generator = HyenaFilter(emb_dim, n_order, num_heads)
        # Range from 0 to 256-1
        self.positions = torch.arange(0, 256).to(torch.float32)
        # self.theta_vector = compute_theta_vector(emb_dim)
        self.t = nn.Parameter(
            sinusoidal_positional_encoding(self.positions, self.emb_dim),
            requires_grad=False
        )

    def forward(self):
        """
        Perform the forward pass to compute the filters.

        :return:
            List of filters, each of shape (batch_size, emb_dim, seq_length).
        """

        filters = self.filter_generator(self.t)

        return filters


class ReadwiseHyena(Module):
    """
    A custom, position-aware Hyena block combining projection, filter, and
    FFT-based long convolution with multi-head support.

    :param emb_dim:
        Dimension of the input embeddings.
    :param n_order:
        Number of orders for the block.
    :param kernel_size:
        Size of the convolution kernel.
    :param num_heads:
        Number of heads for multi-head processing.
    """

    def __init__(self, emb_dim, n_order, kernel_size, num_heads=4, nonlinearity=None):
        super(ReadwiseHyena, self).__init__()
        self.n_order = n_order
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.projection = HyenaProjection(emb_dim, n_order, kernel_size, num_heads)
        self.filter = RotaryHyenaFilter(emb_dim, n_order, num_heads)
        self.fft_long_conv = FFTLongConv()
        self.output_projection = nn.Parameter(
            nn.init.kaiming_uniform_(torch.randn(emb_dim, emb_dim))
        )
        self.output_bias = nn.Parameter(torch.zeros(emb_dim))
        self.B = nn.Parameter(
            nn.init.kaiming_uniform_(torch.randn((n_order, 1, self.head_dim, 1)))
        )
        self.dropout = nn.Dropout(0.1)

        if nonlinearity == 'gelu':
            self.activation = F.gelu
        elif nonlinearity == 'relu':
            self.activation = F.relu
        elif nonlinearity == 'swish':
            self.activation = lambda x: x * torch.sigmoid(x)
        elif nonlinearity == 'mish':
            self.activation = lambda x: x * torch.tanh(F.softplus(x))
        elif nonlinearity == 'elu':
            self.activation = F.elu
        elif nonlinearity == 'leaky_relu':
            self.activation = F.leaky_relu
        elif nonlinearity == 'sine':
            self.activation = torch.sin
        elif nonlinearity is None:
            self.activation = None

    def forward(self, embeddings, positions):
        """
        Perform the forward pass of the Hyena block.

        :param embeddings:
            Input tensor of shape (batch_size, seq_length, emb_dim).
        :param positions:
            Position tensor of shape (batch_size, seq_length).
        :return:
            Output tensor of shape (batch_size, seq_length, emb_dim).
        """

        original_shape = embeddings.shape
        if original_shape[1] > 256:
            # Split the input embeddings into reads
            (
                read_embeddings, read_positions, start_indices, batch_indices
            ) = split_into_reads(embeddings, positions)
        else:
            read_embeddings = embeddings
            read_positions = positions

        *x, v = self.projection(read_embeddings, read_positions)
        # Initial residual connection
        # v = v + embeddings.view(batch_size, embeddings.shape[1], self.num_heads, self.head_dim).transpose(2, 3)

        filters = self.filter()

        for i, x_i in enumerate(reversed(x[1:])):
            h_i = filters[i]
            v = self.dropout(v * x_i)
            v = self.fft_long_conv(v, h_i, self.B[i], read_positions)

        v = v * x[0]

        v = v.permute(0, 3, 1, 2)  # (batch_size, seq_len, num_heads, head_dim)
        v = v.reshape(v.shape[0], v.shape[1], v.shape[2] * v.shape[3])  # Combine heads: (batch_size, seq_len, emb_dim)

        if self.activation is not None:
            v = self.activation(v)

        hyena_out = v.matmul(self.output_projection) + self.output_bias

        if original_shape[1] > 256:
            hyena_out = reassemble_sequences(
                original_shape, hyena_out, read_positions, start_indices,
                batch_indices
            )

        return hyena_out


class PositionwiseSelfAttention(Module):

    def __init__(self, emb_dim, num_heads):
        super(PositionwiseSelfAttention, self).__init__()
        self.self_attention = MultiHeadSelfAttention(emb_dim, num_heads)
        self.W0 = nn.Parameter(
            nn.init.kaiming_uniform_(torch.randn(emb_dim, emb_dim))
        )

    def forward(self, embeddings, positions):
        embs_by_position, index_map = reshape_by_position_and_track(
            embeddings, positions
        )

        self_attention_out = self.self_attention(embs_by_position)
        self_attention_out = self_attention_out.matmul(self.W0)

        # Reshape the self-attention output to the original shape using the
        # position_index_map
        output = torch.zeros(embeddings.shape, device=embeddings.device)

        for key, index in index_map.items():
            old_row, old_idx = key
            new_row, new_idx = index
            output[old_row, old_idx] = self_attention_out[new_row, new_idx]

        return output


class ReadformerBlock(nn.Module):

    def __init__(
            self, emb_dim, n_order, kernel_size, num_heads, num_hyena=1,
            dropout=0.1
    ):
        super(ReadformerBlock, self).__init__()
        self.layer_norms_hyena = nn.ModuleList(
            [nn.LayerNorm(emb_dim) for _ in range(num_hyena)]
        )
        self.hyenas = nn.ModuleList(
            [
                ReadwiseHyena(emb_dim, n_order, kernel_size, num_heads)
                for _ in range(num_hyena)
            ]
        )
        self.layer_norm_2 = nn.LayerNorm(emb_dim)
        self.feed_forward_1 = FeedForward(
            emb_dim, hidden_dim=emb_dim, activation="mish"
        )
        self.layer_norm_3 = nn.LayerNorm(emb_dim)
        self.self_attention = PositionwiseSelfAttention(emb_dim, num_heads)
        self.layer_norm_4 = nn.LayerNorm(emb_dim)
        self.feed_forward_2 = FeedForward(
            emb_dim, hidden_dim=emb_dim, activation="mish"
        )
        self.dropout = nn.Dropout(dropout)
        self.num_hyena = num_hyena

        self.use_self_attention = True

    def forward(self, embeddings, positions):
        hyena_out = embeddings

        # Process multiple Hyena layers sequentially
        for i in range(self.num_hyena):
            hyena_input = self.layer_norms_hyena[i](hyena_out)
            hyena_out = self.dropout(self.hyenas[i](hyena_input, positions)) + hyena_out

        # Apply first feed-forward layer
        ffn_1_input = self.layer_norm_2(hyena_out)
        ffn_1_out = self.dropout(self.feed_forward_1(ffn_1_input)) + hyena_out

        if self.use_self_attention:
            # Apply self-attention
            self_attention_input = self.layer_norm_3(ffn_1_out)
            self_attention_out = self.dropout(
                self.self_attention(self_attention_input, positions)
            ) + ffn_1_out

            # Apply second feed-forward layer
            ffn_2_input = self.layer_norm_4(self_attention_out)
            output = self.dropout(self.feed_forward_2(ffn_2_input)) + self_attention_out
        else:
            output = ffn_1_out

        return output

    def set_use_self_attention(self, use_self_attention):
        self.use_self_attention = use_self_attention

    def freeze_hyena(self):
        for hyena in self.hyenas:
            for param in hyena.parameters():
                param.requires_grad = False

        for layer_norm in self.layer_norms_hyena:
            for param in layer_norm.parameters():
                param.requires_grad = False

        for param in self.feed_forward_1.parameters():
            param.requires_grad = False

        for param in self.layer_norm_2.parameters():
            param.requires_grad = False

    def unfreeze_hyena(self):
        for hyena in self.hyenas:
            for param in hyena.parameters():
                param.requires_grad = True

        for layer_norm in self.layer_norms_hyena:
            for param in layer_norm.parameters():
                param.requires_grad = True

        for param in self.feed_forward_1.parameters():
            param.requires_grad = True

        for param in self.layer_norm_2.parameters():
            param.requires_grad = True

