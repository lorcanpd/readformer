import torch
import torch.nn as nn
import torch.nn.functional as F

from components.better_device_handling import Module


class HyenaProjection(Module):
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
        # device = inputs.device
        # self.W = self.W.to(device)
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


class FeedForward(Module):
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
        elif activation == 'sine':
            self.activation = torch.sin
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
        # device = inputs.device
        # self.W1 = self.W1.to(device)
        # self.b1 = self.b1.to(device)
        # self.W2 = self.W2.to(device)
        # self.b2 = self.b2.to(device)
        # self.ff_scale = self.ff_scale.to(device)

        x = torch.matmul(inputs, self.W1) + self.b1
        x = self.layer_norm(x)
        x = self.activation(x)
        x = torch.matmul(x, self.W2) + self.b2
        x = x * self.ff_scale
        return x


class HyenaFilter(Module):
    """
    Learns filters based on positionally transformed embeddings.

    :param emb_dim: Dimension of the input embeddings.
    :param n_order: Number of orders for the filter.
    """

    def __init__(self, emb_dim, n_order, max_seq_length=256):
        super(HyenaFilter, self).__init__()
        self.emb_dim = emb_dim
        self.n_order = n_order
        self.max_seq_length = max_seq_length
        self.ffn = FeedForward(emb_dim, n_order=n_order, activation='sine')
        self.epsilon = 1e-8  # Small value to avoid division by zero
        # self.seq_length = max_seq_length
        # Initialise the Gaussian window parameters
        self.mu = nn.Parameter(torch.rand(n_order, 1, 1))
        self.sigma = nn.Parameter(
            torch.full((n_order, 1, 1), 10.0)
        )

    def forward(self, positional_encodings):#, positions=None):
        """
        Perform the forward pass to compute the filters.

        :param positional_encodings:
            Positional encodings tensor of shape (batch_size, seq_length,
            emb_dim).
        :return:
            List of n_order filters, each of shape (batch_size, emb_dim,
            seq_length).
        """
        h_hat = self.ffn(positional_encodings)
        # Reshape h_hat from batch_size x L x ND to batch_size x L x N x D
        h_hat = h_hat.view(
            h_hat.size(0), h_hat.size(1), self.n_order, self.emb_dim
        )
        # Reshape to batch_size x N x D x L
        h_hat = h_hat.permute(0, 2, 3, 1)
        # Normalize h_hat along the channel dimension D
        h_hat = h_hat / (h_hat.norm(p=1, dim=-2, keepdim=True) + self.epsilon)

        # # # Apply each orders' Gaussian window to their respective filters
        seq_length = h_hat.size(-1)

        # if positions is None:
        # positions = torch.arange(
        #     # self.seq_length, device=h_hat.device
        #     seq_length
        # ).float().view(1, 1, 1, -1)
        positions = torch.arange(
            seq_length
        ).float().view(1, 1, -1).to(positional_encodings.device)

        # gaussian_windows = torch.exp(
        #     -0.5 * (
        #             (
        #                     positions - self.mu * seq_length
        #             ).permute(2, 1, 0, 3) / self.sigma
        #     ) ** 2)#.to(h_hat.device)

        gaussian_windows = torch.exp(
            -0.5 * (
                    (positions - self.mu * self.max_seq_length) / self.sigma
            ) ** 2)
        # Bias prevents zero values outside the window.

        h_hat = h_hat * (gaussian_windows + 0.01)

        # Split h_hat into h1, h2, ..., hN
        filters = h_hat.unbind(dim=-3)

        return filters


class FFTLongConv(Module):
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
