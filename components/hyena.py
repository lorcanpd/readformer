import torch
import torch.nn as nn
import torch.nn.functional as F

from components.rotary_encoding import (
    compute_theta_vector, compute_rotation_angles, apply_dimensionwise_rotation
)
from components.self_attention import MultiHeadSelfAttention

from torch.nn.utils.rnn import pad_sequence


# class ParallelFFTLongConv(nn.Module):
#     def __init__(self):
#         super(ParallelFFTLongConv, self).__init__()
#
#     def forward(self, inputs, filters, positions):
#         batch_size, seq_length, emb_dim = inputs.shape
#         outputs = torch.zeros_like(inputs)
#
#         position_differences = torch.diff(
#             positions, dim=1,
#             append=torch.full((batch_size, 1), -1, device=inputs.device)
#         )
#
#         # Create a mask for the start of each segment
#         segment_starts = F.pad(position_differences != 1, (1, 0), "constant", 1)
#         # Create a mask for the end of each segment
#         segment_ends = F.pad(position_differences != 1, (0, 1), "constant", 1)
#
#         # Gather all segment indices
#         segmented_inputs = []
#         segmented_filters = []
#         max_seg_length = 0
#         for b in range(batch_size):
#             starts = torch.where(segment_starts[b])[0]
#             ends = torch.where(segment_ends[b])[0]
#             segments = [inputs[b, start:end] for start, end in zip(starts, ends) if start != end]
#             filter_segs = [filters[b, start:end] for start, end in zip(starts, ends) if start != end]
#             max_seg_length = max(max_seg_length, *[s.shape[0] for s in segments])
#             segmented_inputs.extend(segments)
#             segmented_filters.extend(filter_segs)
#
#         # Pad each segment to the maximum segment length
#         padded_inputs = [F.pad(seg, (0, 0, 0, max_seg_length - seg.shape[0])) for seg in segmented_inputs]
#         padded_filters = [F.pad(flt, (0, 0, 0, max_seg_length - flt.shape[0])) for flt in segmented_filters]
#
#         # Stack segments and apply FFT
#         stacked_inputs = torch.stack(padded_inputs)
#         stacked_filters = torch.stack(padded_filters)
#         padded_length = 2 ** max_seg_length.bit_length()  # Using power of 2 for efficient FFT
#
#         inputs_fft = torch.fft.rfft(stacked_inputs, n=padded_length, dim=2, norm='forward')
#         filters_fft = torch.fft.rfft(stacked_filters, n=padded_length, dim=2, norm='forward')
#
#         # Perform element-wise multiplication in the frequency domain and inverse FFT
#         result_fft = inputs_fft * filters_fft
#         conv_results = torch.fft.irfft(result_fft, n=padded_length, dim=2, norm='forward')
#
#         # Map results back to the output tensor
#         segment_idx = 0
#         breakpoint()
#         for b in range(batch_size):
#             starts = torch.where(segment_starts[b])[0]
#             ends = torch.where(segment_ends[b])[0]
#             for start, end in zip(starts, ends):
#                 length = end - start
#                 outputs[b, start:end] = conv_results[segment_idx, :length]
#                 segment_idx += 1
#
#         return outputs



# class IndependentDepthwiseSeparableConv1D(nn.Module):
#     def __init__(self, kernel_size, in_channels, out_channels, groups, stride=1):
#         super(IndependentDepthwiseSeparableConv1D, self).__init__()
#         self.stride = stride
#         self.kernel_size = kernel_size
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.groups = groups
#
#         # Depthwise convolution
#         self.depthwise = nn.Conv1d(
#             in_channels, out_channels, kernel_size,
#             stride=stride, groups=groups, padding=kernel_size//2
#         )
#
#     def forward(self, inputs, positions):
#         """
#         Apply convolutions with splitting and independent processing.
#
#         :param inputs: Input tensor of shape (batch_size, seq_length, in_channels).
#         :param positions: Position tensor indicating where to split the sequence.
#         :return: Output tensor after processing.
#         """
#         batch_size, seq_length, emb_size = inputs.shape
#
#         # Compute differences to identify gaps
#         position_differences = torch.diff(
#             positions, dim=1,
#             append=torch.full((batch_size, 1), -1, device=inputs.device)
#         )
#         # Identify boundaries where the absolute differences between adjacent
#         # positions are not equal to 1 and are finite.
#         boundaries = (position_differences != 1)
#         not_boundaries = (position_differences == 0)  # Padded positions
#         final_boundaries = boundaries & ~not_boundaries
#
#         # Create new tensor with padding inserted at the boundaries
#         padded_inputs = []
#         for i in range(batch_size):
#             # Split input into segments without crossing boundaries
#             segments = []
#             last_index = 0
#             for idx in torch.where(final_boundaries[i])[0] + 1:
#                 segments.append(inputs[i, last_index:idx])
#                 # Add padding segments
#                 if idx < seq_length:
#                     segments.append(
#                         torch.zeros((self.kernel_size // 2, emb_size),
#                                     device=inputs.device)
#                     )
#                 last_index = idx
#             # Add the last segment
#             segments.append(inputs[i, last_index:])
#             padded_inputs.append(torch.cat(segments, dim=-2))
#         # Pad the sequences to the same length as come may contain more segments
#         # than others.
#         padded_inputs = pad_sequence(
#             padded_inputs, batch_first=True, padding_value=0
#         ).to(inputs.device)
#         # Concatenate all batch segments and apply depthwise convolution
#         # padded_inputs = torch.stack(padded_inputs).to(inputs.device)
#         conv_output = self.depthwise(
#             padded_inputs.transpose(1, 2)
#         ).transpose(1, 2)
#         # Remove the indices that were added as padding. This is done by
#         # removing the indices of vectors that sum to zero in the padded
#         # inputs from the conv_output. Then reshape the output to the original
#         # shape.
#
#         try:
#             # conv_output = conv_output[torch.sum(padded_inputs, dim=-1) != 0].view(
#             #     batch_size, -1, self.out_channels
#             # )
#             # identify vectors of zeros in the padded inputs
#             # zero_vectors = torch.all(padded_inputs != 0, dim=-1)
#             conv_output = conv_output[
#                 torch.all(padded_inputs != 0, dim=-1)
#             ].view(
#                 batch_size, -1, self.out_channels
#             )
#         except RuntimeError:
#             breakpoint()
#
#
#         # pad to the original sequence length
#         conv_output = F.pad(
#             conv_output, (0, 0, 0, seq_length - conv_output.size(-2))
#         )
#
#         return conv_output

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
        append=torch.full((batch_size, 1), -1, device=embeddings.device)
    ).to(embeddings.device)

    # Create a mask for the start of each segment. Do this by adding a true to
    # start of position differences != 1 tensor. Remove the final element to
    # prevent out of bounds indexing.
    # segment_starts = torch.cat((torch.full((batch_size, 1), True, device=embeddings.device) ,position_differences != 1), dim=1)
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
                segmented_inputs.append(embeddings[b, start:end+1])
                segmented_positions.append(positions[b, start:end+1])
                segment_starts_indices.append(start)
                # segment_ends_indices.append(end)
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


def reassamble_sequences(original_shape, read_tensor, positions, segment_starts, batch_indices):
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


class HyenaProjection(nn.Module):
    """
    Layer for projecting input embeddings into multiple groups.

    :param emb_dim:
        Dimension of the input embeddings.
    :param n_order:
        Number of orders for the projection.
    :param kernel_size:
        Size of the convolution kernel.
    """
    def __init__(self, emb_dim, n_order, kernel_size):
        super(HyenaProjection, self).__init__()
        self.groups = n_order + 1
        self.emb_dim = emb_dim
        self.W = nn.Parameter(
            nn.init.kaiming_uniform_(torch.randn(emb_dim, self.groups * emb_dim))
        )
        # self.custom_conv = CustomMaskedConv1D(
        #     kernel_size, self.groups * emb_dim, self.groups * emb_dim,
        #     self.groups
        # )
        self.conv = nn.Conv1d(
            self.groups * emb_dim, self.groups * emb_dim, kernel_size,
            groups=self.groups, padding=kernel_size//2
        )

    def forward(self, inputs, positions):
        """
        Perform the forward pass of the Hyena projection.

        :param inputs:
            Input tensor of shape (batch_size, seq_length, emb_dim).
        :param positions:
            Position tensor of shape (batch_size, seq_length).
        :return:
            List of n_order projected tensors, each of shape (batch_size,
            emb_dim, seq_length).
        """
        device = inputs.device
        self.W = self.W.to(device)
        x = inputs
        x = torch.matmul(x, self.W)
        z = self.conv(x.transpose(1,2))
        # Set the values at the padded positions to zero
        z = z * (positions != -1).unsqueeze(-2).to(torch.float32)
        # Reshape z to batch_size x D x N x L
        z = z.reshape(z.size(0), self.emb_dim, self.groups, -1)
        # Unstack the groups for separate processing
        z = z.unbind(dim=-2)

        return z




class FeedForward(nn.Module):
    """
    Feed-forward layer with configurable activation function and dimensions.

    :param emb_dim:
        Dimension of the input embeddings.
    :param hidden_dim:
        Dimension of the hidden layer. If None, defaults to emb_dim.
    :param n_order:
        Number of orders for the output. Default is 1.
    :param activation:
        Activation function to use. Default is "gelu".
    """
    def __init__(self, emb_dim, hidden_dim=None, n_order=1, activation="gelu"):
        super(FeedForward, self).__init__()
        self.emb_dim = emb_dim
        self.n_order = n_order
        self.hidden_dim = hidden_dim if hidden_dim is not None else emb_dim
        self.W1 = nn.Parameter(
            nn.init.kaiming_uniform_(torch.randn(emb_dim, self.hidden_dim))
        )
        self.b1 = nn.Parameter(torch.zeros(self.hidden_dim))
        self.W2 = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.randn(self.hidden_dim, n_order * emb_dim)
            )
        )
        self.b2 = nn.Parameter(torch.zeros(n_order * emb_dim))

        # Initialise scaling vector
        self.ff_scale = nn.Parameter(torch.ones(1, 1, n_order * emb_dim))

        self.layer_norm = nn.LayerNorm(self.hidden_dim)

        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'swish':
            self.activation = lambda x: x * torch.sigmoid(x)
        elif activation == 'mish':
            self.activation = lambda x: x * torch.tanh(F.softplus(x))
        elif activation == 'elu':
            self.activation = F.elu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def init_scaling_vector(self):
        nn.init.ones_(self.ff_scale)

    def freeze_scaling_vector(self):
        self.ff_scale.requires_grad = False

    def unfreeze_scaling_vector(self):
        self.ff_scale.requires_grad = True

    def forward(self, inputs):
        """
        Perform the forward pass of the feed-forward layer.

        :param inputs:
            Input tensor of shape (batch_size, seq_length, emb_dim).
        :return:
            Output tensor after applying the feed-forward layer.
        """
        device = inputs.device
        self.W1 = self.W1.to(device)
        self.b1 = self.b1.to(device)
        self.W2 = self.W2.to(device)
        self.b2 = self.b2.to(device)
        self.ff_scale = self.ff_scale.to(device)

        x = torch.matmul(inputs, self.W1) + self.b1
        x = self.layer_norm(x)
        x = self.activation(x)
        x = torch.matmul(x, self.W2) + self.b2
        x = x * self.ff_scale
        return x


class HyenaFilter(nn.Module):
    """
    Layer to learn filters for global convolution using FFT.

    :param emb_dim:
        Dimension of the input embeddings.
    :param n_order:
        Number of orders for the filter.
    """
    def __init__(self, emb_dim, n_order):
        super(HyenaFilter, self).__init__()
        self.emb_dim = emb_dim
        self.n_order = n_order
        self.ffn = FeedForward(emb_dim, n_order=n_order)
        self.theta_vector = compute_theta_vector(emb_dim)
        self.epsilon = 1e-8  # Small value to avoid division by zero

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
        device = embeddings.device
        self.theta_vector = self.theta_vector.to(device)
        # make each sequence in the positions batch be normalised so that the
        # lowest position is 0 in each sequence. Must ignore -1 values.
        adjusted_positions = adjust_positions(positions)
        rotation_matrices = compute_rotation_angles(
            adjusted_positions, self.emb_dim, self.theta_vector
        )
        t = apply_dimensionwise_rotation(embeddings, rotation_matrices)
        h_hat = self.ffn(t)
        # reshape h_hat from batch_size x L x ND to batch_size x L x N x D
        h_hat = h_hat.view(
            h_hat.size(0), h_hat.size(1), self.n_order, self.emb_dim
        )

        # reshape to batch_size x N x D x L
        h_hat = h_hat.permute(0, 2, 3, 1)
        # Normalise h_hat along the channel dimension D.
        h_hat = h_hat / (h_hat.norm(p=1, dim=-2, keepdim=True) + self.epsilon)
        # split h_hat into h1, h2, ..., hN
        h = h_hat.unbind(dim=-3)

        return h


class FFTLongConv(nn.Module):
    """
    FFT-based long convolution layer.

    """
    def __init__(self):
        super(FFTLongConv, self).__init__()

    def forward(self, inputs, filters, positions):
        """
        Perform the forward pass of the FFT-based long convolution.

        :param inputs:
            Input tensor of shape (batch_size, emb_dim, seq_length).
        :param filters:
            Filters tensor of shape (batch_size, emb_dim, seq_length).
        :param positions:
            Position tensor of shape (batch_size, seq_length).
        :return:
            Convolution result tensor of shape (batch_size, emb_dim, seq_length).
        """
        L = inputs.size(-1)
        padded_length = 2 * L  # Double the length for FFT
        # Pad the inputs and filters
        inputs = F.pad(inputs, (0, padded_length - L))
        filters = F.pad(filters, (0, padded_length - L))
        # Perform FFT
        inputs = torch.fft.rfft(
            inputs, n=padded_length, dim=-1, norm='forward'
        )
        filters = torch.fft.rfft(
            filters, n=padded_length, dim=-1, norm='forward'
        )
        # Element-wise multiplication in the frequency domain
        inputs.mul_(filters)
        # Inverse FFT to get the convolution result
        result = torch.fft.irfft(
            inputs, n=padded_length, dim=-1, norm='forward'
        )
        # Remove padding
        result = result[..., :L]
        # Zero out the padded positions. These positions do not represent
        # nucleotides and should not contribute to the convolution result.
        return result * (positions != -1).unsqueeze(-2).to(torch.float32)


class ReadformerBlock(nn.Module):
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
        super(ReadformerBlock, self).__init__()
        self.n_order = n_order
        self.embedding_dim = emb_dim
        self.kernel_size = kernel_size
        self.projection = HyenaProjection(emb_dim, n_order, kernel_size)
        self.filter = HyenaFilter(emb_dim, n_order)
        self.fft_long_conv = FFTLongConv()
        self.output_projection = nn.Parameter(
            nn.init.kaiming_uniform_(torch.randn(emb_dim, emb_dim))
        )
        self.B = nn.Parameter(
            nn.init.kaiming_uniform_(torch.randn((n_order, 1, emb_dim, 1)))
        )
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.self_attention = MultiHeadSelfAttention(emb_dim, 8)
        self.W0 = nn.Parameter(
            nn.init.kaiming_uniform_(torch.randn(emb_dim, emb_dim))
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

        device = embeddings.device
        original_shape = embeddings.shape
        self.B = self.B.to(device)
        self.output_projection = self.output_projection.to(device)

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

        hyena_out = reassamble_sequences(
            original_shape, hyena_out, read_positions, start_indices,
            batch_indices
        )

        # Apply residual connection
        hyena_out = self.layer_norm(hyena_out + embeddings)

        # Use position information to reshape the hyena_out into "sequences"
        # containing the embeddings of nucleotides at the same position in
        # different reads in the same sequence.

        embs_by_position, index_map = reshape_by_position_and_track(
            hyena_out, positions
        )

        self_attention_out = self.self_attention(embs_by_position)
        self_attention_out = self_attention_out.matmul(self.W0)

        # Reshape the self-attention output to the original shape using the
        # position_index_map
        output = torch.zeros(original_shape, device=hyena_out.device)

        for key, index in index_map.items():
            old_row, old_idx = key
            new_row, new_idx = index
            output[old_row, old_idx] = self_attention_out[new_row, new_idx]

        # Residual connection
        return output + hyena_out


# Todo FIX device crap so can run on GPU.

# test
# emb_dim = 16
# n_order = 2
# kernel_size = 3
# seq_length = 10
# batch_size = 2
#
# inputs = torch.randn(batch_size, seq_length, emb_dim)
#
# positions = torch.tensor([
#     [0, 1, 2, 3, 1, 2, 3, 4, 5, 6],
#     [1, 2, 3, 1, 2, 3, 4, 5, -1, -1]
# ])
#
# # Mask the inputs with 0s where the positions are -1
# inputs = inputs * (positions != -1).unsqueeze(-1).to(torch.float32)
#
# readformer = ReadformerBlock(emb_dim, n_order, kernel_size)
#
# output = readformer(inputs, positions)
