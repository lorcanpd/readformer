import torch
import torch.nn as nn
import torch.nn.functional as F

from components.hyena import (
    HyenaProjection, HyenaFilter, FFTLongConv, FeedForward
)
from components.rotary_encoding import (
    compute_theta_vector, compute_rotation_angles, apply_dimensionwise_rotation
)
from components.self_attention import MultiHeadSelfAttention

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
        append=torch.full((batch_size, 1), -1)#, device=embeddings.device
    )#.to(embeddings.device)

    segment_starts = torch.cat(
        tensors=(
            torch.full(
                size=(batch_size, 1), fill_value=True, #device=embeddings.device
            ),
            position_differences != 1
        ),
        dim=1
    )[..., :-1]
    segment_ends = (position_differences != 1)#.to(embeddings.device)

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
                segmented_inputs.append(embeddings[b, start:end+1])
                segmented_positions.append(positions[b, start:end+1])
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

    return (torch.stack(padded_inputs),#.to(embeddings.device),
            torch.stack(reshaped_positions),#.to(embeddings.device),
            torch.stack(segment_starts_indices),#.to(embeddings.device),
            torch.tensor(batch_indices))#.to(embeddings.device))


def reassemble_sequences(original_shape, read_tensor, positions, segment_starts, batch_indices):
    # batch_size, seq_length, emb_dim = read_tensor.shape
    output = torch.zeros(original_shape)#, device=read_tensor.device)

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
        (len(combos), max_embeddings_count, emb_dim), #device=inputs.device
    )

    for key, value in index_map.items():
        old_row, old_idx = key
        new_row, new_idx = value
        all_positions_tensor[new_row, new_idx] = inputs[old_row, old_idx]

    return all_positions_tensor, index_map


class RotaryHyenaFilter(nn.Module):
    """
    Wraps HyenaFilter but instead of providing normal positional encodings,
    it provides position-wise rotated embeddings as positional encodings.

    :param emb_dim:
        Dimension of the input embeddings.
    :param n_order:
        The numb ser of long convolution filters to generate.
    """

    def __init__(self, emb_dim, n_order):
        super(RotaryHyenaFilter, self).__init__()
        self.emb_dim = emb_dim
        self.filter_generator = HyenaFilter(emb_dim, n_order)
        self.theta_vector = compute_theta_vector(emb_dim)

    def forward(self, embeddings, positions):
        """
        Perform the forward pass to compute the filters.

        :param embeddings:
            Input embeddings of shape (batch_size, seq_length, emb_dim).
        :param positions:
            Position tensor of shape (batch_size, seq_length).
        :return:
            List of filters, each of shape (batch_size, emb_dim, seq_length).
        """
        # device = embeddings.device
        # self.theta_vector = self.theta_vector.to(device)
        adjusted_positions = adjust_positions(positions)
        rotation_matrices = compute_rotation_angles(
            adjusted_positions, self.emb_dim, self.theta_vector
        )
        t = apply_dimensionwise_rotation(embeddings, rotation_matrices)
        filters = self.filter_generator(t)#, adjusted_positions)

        return filters


class ReadwiseHyena(nn.Module):
    """
    A custom, position-aware Hyena block combining projection, filter, and
    FFT-based long convolution. Can be used as a direct replacement to
    multi-head self-attention in a transformer. It is intended to be used in
    applications of multiple overlapping sequences such as aligned DNA
    sequencing reads.

    :param emb_dim:
        Dimension of the input embeddings.
    :param n_order:
        Number of orders for the block.
    :param kernel_size:
        Size of the convolution kernel.
    """
    def __init__(self, emb_dim, n_order, kernel_size):
        super(ReadwiseHyena, self).__init__()
        self.n_order = n_order
        self.embedding_dim = emb_dim
        self.kernel_size = kernel_size
        self.projection = HyenaProjection(emb_dim, n_order, kernel_size)
        # self.filter = HyenaFilter(emb_dim, n_order)
        self.filter = RotaryHyenaFilter(emb_dim, n_order)
        self.fft_long_conv = FFTLongConv()
        self.output_projection = nn.Parameter(
            nn.init.kaiming_uniform_(torch.randn(emb_dim, emb_dim))
        )
        self.B = nn.Parameter(
            nn.init.kaiming_uniform_(torch.randn((n_order, 1, emb_dim, 1)))
        )

    def forward(self, embeddings, positions):
        """
        Perform the forward pass of the Hyena block.

        :param embeddings:
            Input embeddings of shape (batch_size, seq_length, emb_dim).
        :param positions:
            Position tensor of shape (batch_size, seq_length).
        :return:
            Output tensor of shape (batch_size, seq_length, emb_dim).
        """

        # device = embeddings.device
        original_shape = embeddings.shape
        # self.B = self.B.to(device)
        # self.output_projection = self.output_projection.to(device)

        # Split the input embeddings into reads
        (
            read_embeddings, read_positions, start_indices, batch_indices
        ) = split_into_reads(embeddings, positions)

        # padding = where read_positions are -1 for an element
        not_padded = read_positions != -1
        *x, v = self.projection(read_embeddings, read_positions)
        filters = self.filter(read_embeddings, read_positions)


        for i, x_i in enumerate(x):
            # Multiply filter by not_padded to zero out the padded positions.
            h_i = filters[i]
            v = x_i * (v * self.B[i] + self.fft_long_conv(v, h_i, read_positions))

        # Transpose v to shape (batch_size, seq_len, emb_dim)
        v = v.transpose(2, 1)

        hyena_out = v.matmul(self.output_projection)

        hyena_out = reassemble_sequences(
            original_shape, hyena_out, read_positions, start_indices,
            batch_indices
        )
        return hyena_out


class PositionwiseSelfAttention(nn.Module):

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
        output = torch.zeros(embeddings.shape)#, device=embeddings.device)

        for key, index in index_map.items():
            old_row, old_idx = key
            new_row, new_idx = index
            output[old_row, old_idx] = self_attention_out[new_row, new_idx]

        return output


class ReadformerBlock(nn.Module):

    def __init__(self, emb_dim, n_order, kernel_size):
        super(ReadformerBlock, self).__init__()
        self.layer_norm_1 = nn.LayerNorm(emb_dim)
        self.hyena = ReadwiseHyena(emb_dim, n_order, kernel_size)
        self.layer_norm_2 = nn.LayerNorm(emb_dim)
        self.self_attention = PositionwiseSelfAttention(emb_dim, 8)
        self.layer_norm_3 = nn.LayerNorm(emb_dim)
        self.feed_forward = FeedForward(
            emb_dim, hidden_dim=emb_dim, activation="mish"
        )

    def forward(self, embeddings, positions):
        # Apply layer normalisation
        hyena_input = self.layer_norm_1(embeddings)
        # Update nucleotide embeddings using intra-read information and add
        # residual connection.
        hyena_out = self.hyena(hyena_input, positions) + embeddings
        # Apply layer normalisation.
        self_attention_input = self.layer_norm_2(hyena_out)
        # Update nucleotide embeddings using inter-read information
        # position-wise.
        self_attention_out = self.self_attention(
            self_attention_input, positions
        ) + hyena_out
        # Apply layer normalisation.
        ffn_input = self.layer_norm_3(self_attention_out)
        # Apply feed-forward layer and add residual connection.
        output = self.feed_forward(ffn_input) + self_attention_out

        return output
