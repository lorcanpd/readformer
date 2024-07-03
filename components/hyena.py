import torch
import torch.nn as nn
import torch.nn.functional as F

from components.rotary_encoding import (
    compute_theta_vector, compute_rotation_angles, apply_dimensionwise_rotation
)
from torch.nn.utils.rnn import pad_sequence


#
# class CustomMaskedConv1D(nn.Module):
#     """
#     Custom masked 1D convolution layer that prevents aggregating signals from
#     bases on different reads.
#
#     :param kernel_size:
#         Size of the convolution kernel.
#     :param in_channels:
#         Number of input channels.
#     :param out_channels:
#         Number of output channels.
#     :param groups:
#         Number of groups for the grouped convolution.
#     :param stride:
#         Stride of the convolution. Default is 1.
#     """
#     def __init__(
#             self, kernel_size, in_channels, out_channels, groups, stride=1
#     ):
#         super(CustomMaskedConv1D, self).__init__()
#         self.groups = groups
#         self.stride = stride
#         # self.padding = padding
#         self.kernel_size = kernel_size
#         self.in_channels_per_group = in_channels // groups
#         self.out_channels_per_group = out_channels // groups
#         self.kernel = nn.Parameter(
#             nn.init.kaiming_uniform_(
#                 torch.randn(
#                     1, 1, groups,
#                     self.out_channels_per_group,
#                     self.in_channels_per_group,
#                     kernel_size
#                 )
#             )
#         )
#
#     def forward(self, inputs, positions):
#         """
#         Perform the forward pass of the position-aware masked convolution.
#
#         :param inputs:
#             Input tensor of shape (batch_size, seq_length, in_channels).
#         :param positions:
#             Position tensor of shape (batch_size, seq_length).
#         :return:
#             Output tensor after masked convolution.
#         """
#         self.kernel = self.kernel.to(inputs.device)
#
#         batch_size, seq_length, _ = inputs.size()
#         kernel_center = self.kernel_size // 2
#
#         # Efficient padding integrated with unfold
#         input_patches = F.pad(
#             inputs, (0, 0, kernel_center, kernel_center), mode='reflect'
#         ).unfold(1, self.kernel_size, self.stride).contiguous()
#         position_patches = F.pad(
#             positions, (kernel_center, kernel_center), value=-1
#         ).unfold(1, self.kernel_size, self.stride)
#
#         # Calculate the expected positions and create the mask
#         expected_positions = position_patches[:, :, kernel_center].unsqueeze(
#             2) + torch.arange(
#             -kernel_center, kernel_center + 1, device=inputs.device
#         )
#         mask = (position_patches == expected_positions).float().unsqueeze(-2)
#
#         # Apply the mask using in-place operations
#         input_patches.mul_(mask)
#
#         # Use view to reshape input patches to reduce copying data
#         grouped_input_patches = input_patches.view(
#             batch_size, seq_length, self.groups, self.in_channels_per_group,
#             self.kernel_size
#         )
#
#         # Elementwise multiplication using broadcasting
#         conv_result = grouped_input_patches.unsqueeze(-3) * self.kernel
#
#         # Sum over the last two dimensions
#         output = conv_result.sum(dim=(-1, -2))
#
#         return output


class IndependentDepthwiseSeparableConv1D(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, groups, stride=1):
        super(IndependentDepthwiseSeparableConv1D, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        # Depthwise convolution
        self.depthwise = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, groups=groups, padding=kernel_size//2
        )

    def forward(self, inputs, positions):
        """
        Apply convolutions with splitting and independent processing.

        :param inputs: Input tensor of shape (batch_size, seq_length, in_channels).
        :param positions: Position tensor indicating where to split the sequence.
        :return: Output tensor after processing.
        """
        batch_size, seq_length, emb_size = inputs.shape

        # Compute differences to identify gaps
        position_differences = torch.diff(
            positions, dim=1,
            prepend=torch.full(
                (batch_size, 1), float('nan'), device=inputs.device
            )
        )
        boundaries = (position_differences != 1) & torch.isfinite(
            position_differences
        )

        # Create new tensor with padding inserted at the boundaries
        padded_inputs = []
        for i in range(batch_size):
            # Split input into segments without crossing boundaries
            segments = []
            last_index = 0
            for idx in torch.where(boundaries[i])[0] + 1:
                segments.append(inputs[i, last_index:idx])
                # Add padding segments
                segments.append(
                    torch.zeros(
                        (self.kernel_size - 1, emb_size), device=inputs.device
                    )
                )
                last_index = idx
            # Add the last segment
            segments.append(inputs[i, last_index:])
            padded_inputs.append(torch.cat(segments, dim=0))
        # Pad the sequences to the same length as come may contain more segments
        # than others.
        padded_inputs = pad_sequence(
            padded_inputs, batch_first=True
        ).to(inputs.device)
        # Concatenate all batch segments and apply depthwise convolution
        # padded_inputs = torch.stack(padded_inputs).to(inputs.device)
        conv_output = self.depthwise(
            padded_inputs.transpose(1, 2)
        ).transpose(1, 2)
        # Remove the indices that were added as padding. This is done by
        # removing the indices of vectors that sum to zero in the padded
        # inputs from the conv_output. Then reshape the output to the original
        # shape.
        conv_output = conv_output[torch.sum(padded_inputs, dim=-1) != 0].view(
            batch_size, seq_length, self.out_channels
        )

        return conv_output


# emb_dim = 2
# n_order = 2
# kernel_size = 3
# seq_length = 10
# batch_size = 2
#
# in_channels = emb_dim * (n_order + 1)
#
# inputs = torch.randn(batch_size, seq_length, in_channels)
#
# positions = torch.tensor([
#     [0, 1, 2, 3, 1, 2, 3, 1, 2, 3],
#     [1, 2, 3, 1, 2, 3, 4, 5, 6, 7]
# ])
#
# test = IndependentDepthwiseSeparableConv1D(kernel_size, in_channels, in_channels, in_channels)
#
# output = test(inputs, positions)

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
        self.custom_conv = IndependentDepthwiseSeparableConv1D(
            kernel_size, self.groups * emb_dim, self.groups * emb_dim,
            self.groups * emb_dim
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
        z = self.custom_conv(x, positions)
        # Reshape z to batch_size x D x N x L
        z = z.reshape(z.size(0), self.emb_dim, self.groups, -1)
        # Unstack the groups for separate processing
        z = z.unbind(dim=-2)

        return z


# test the hyena projection
# emb_dim = 2
# n_order = 2
# kernel_size = 3
# seq_length = 10
# batch_size = 2
#
# inputs = torch.randn(batch_size, seq_length, emb_dim)
#
# positions = torch.tensor([
#     [0, 1, 2, 3, 1, 2, 3, 4, 5, 6],
#     [1, 2, 3, 1, 2, 3, 4, 5, 6, 7]
# ])
#
# test = HyenaProjection(emb_dim, n_order, kernel_size)
#
# output = test(inputs, positions)


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

        rotation_matrices = compute_rotation_angles(
            positions, self.emb_dim, self.theta_vector
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

    def forward(self, inputs, filters):
        """
        Perform the forward pass of the FFT-based long convolution.

        :param inputs:
            Input tensor of shape (batch_size, seq_length).
        :param filters:
            Filters tensor of shape (batch_size, seq_length).
        :return:
            Convolution result tensor of shape (batch_size, seq_length).
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
        return result[..., :L]



class HyenaBlock(nn.Module):
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
        super(HyenaBlock, self).__init__()
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
        self.B = self.B.to(device)
        self.output_projection = self.output_projection.to(device)

        *x, v = self.projection(embeddings, positions)
        filters = self.filter(embeddings, positions)

        for i, x_i in enumerate(x):
            h_i = filters[i]
            v = x_i.mul_(v.mul_(self.B[i]).add_(self.fft_long_conv(v, h_i)))

        # Transpose v to shape (batch_size, seq_len, emb_dim)
        v = v.transpose(2, 1)
        output = v.matmul(self.output_projection)

        return output

