import torch
import torch.nn as nn
import torch.nn.functional as F

from components.rotary_encoding import (
    compute_theta_vector, compute_rotation_angles, apply_dimensionwise_rotation
)


class CustomMaskedConv1D(nn.Module):
    """
    Custom masked 1D convolution layer that prevents aggregating signals from
    bases on different reads.

    :param kernel_size:
        Size of the convolution kernel.
    :param in_channels:
        Number of input channels.
    :param out_channels:
        Number of output channels.
    :param groups:
        Number of groups for the grouped convolution.
    :param stride:
        Stride of the convolution. Default is 1.
    """
    def __init__(
            self, kernel_size, in_channels, out_channels, groups, stride=1
    ):
        super(CustomMaskedConv1D, self).__init__()
        self.groups = groups
        self.stride = stride
        # self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels_per_group = in_channels // groups
        self.out_channels_per_group = out_channels // groups
        self.kernel = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.randn(
                    1, 1, groups,
                    self.out_channels_per_group,
                    self.in_channels_per_group,
                    kernel_size
                )
            )
        )

    def forward(self, inputs, positions):
        """
        Perform the forward pass of the position-aware masked convolution.

        :param inputs:
            Input tensor of shape (batch_size, seq_length, in_channels).
        :param positions:
            Position tensor of shape (batch_size, seq_length).
        :return:
            Output tensor after masked convolution.
        """
        device = inputs.device
        self.kernel = self.kernel.to(device)

        batch_size, seq_length, _ = inputs.size()
        kernel_center = self.kernel_size // 2
        # Pad inputs and positions to handle the borders
        pad_size = kernel_center
        padded_inputs = F.pad(inputs, (0, 0, pad_size, pad_size))
        padded_positions = F.pad(positions, (pad_size, pad_size), value=-1)
        # Extract and reshape patches
        input_patches = padded_inputs.unfold(
            1, self.kernel_size, self.stride
        ).contiguous()
        position_patches = padded_positions.unfold(
            1, self.kernel_size, self.stride
        )
        # Create a mask to prevent short convolution aggregating signals from
        # bases on different reads.
        expected_positions = (
                position_patches[:, :, kernel_center].unsqueeze(2) +
                torch.arange(-kernel_center, kernel_center + 1)
        )
        mask = (position_patches == expected_positions).float().unsqueeze(-2)
        # Apply the mask
        masked_input_patches = input_patches * mask
        # Reshape the masked input patches to group the channels
        grouped_patches = masked_input_patches.reshape(
            batch_size, seq_length, self.groups, self.in_channels_per_group,
            self.kernel_size
        )
        # Perform an elementwise multiplication with each of the
        # num_out_channels kernels
        # output shape = (batch_size, seq_length, groups,
        # out_channels_per_group, in_channels_per_group, kernel_size)
        conv_result = (grouped_patches.unsqueeze(-3) * self.kernel)
        # Sum over the kernel size dimension and the in_channels_per_group
        # dimension
        # output shape = (batch_size, seq_length, groups, out_channels_per_group)
        output = conv_result.sum(dim=(-1, -2))

        return output


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
        self.custom_conv = CustomMaskedConv1D(
            kernel_size, self.groups * emb_dim, self.groups * emb_dim,
            self.groups
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
        inputs_padded = F.pad(inputs, (0, padded_length - L))
        filters_padded = F.pad(filters, (0, padded_length - L))
        # Perform FFT
        inputs_fft = torch.fft.rfft(
            inputs_padded, n=padded_length, dim=-1, norm='forward'
        )
        filters_fft = torch.fft.rfft(
            filters_padded, n=padded_length, dim=-1, norm='forward'
        )
        # Element-wise multiplication in the frequency domain
        product_fft = inputs_fft * filters_fft
        # Inverse FFT to get the convolution result
        result = torch.fft.irfft(
            product_fft, n=padded_length, dim=-1, norm='forward'
        )
        # Remove padding
        result_real = result[..., :L]

        return result_real


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
            v = x_i * (v * self.B[i] + self.fft_long_conv(v, h_i))

        # Transpose v to shape (batch_size, seq_len, emb_dim)
        v = v.transpose(2, 1)
        output = v.matmul(self.output_projection)

        return output

