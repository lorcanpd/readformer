import torch
import torch.nn as nn
import torch.nn.functional as F

from components.better_device_handling import Module


class HyenaProjection(Module):
    """
    Layer for projecting input embeddings into multiple heads and groups.

    :param emb_dim:
        Dimension of the input embeddings.
    :param n_order:
        Number of orders for the projection.
    :param kernel_size:
        Size of the convolution kernel.
    :param num_heads:
        Number of heads for multi-head projection.
    """

    def __init__(self, emb_dim, n_order, kernel_size, num_heads):
        super(HyenaProjection, self).__init__()
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.groups = n_order + 1
        self.emb_dim = emb_dim
        self.W = nn.Parameter(
            nn.init.kaiming_uniform_(torch.randn(
                self.num_heads, self.head_dim, self.groups * self.head_dim
            ))
        )
        self.conv = nn.Conv1d(
            self.groups * self.head_dim * num_heads,
            self.groups * self.head_dim * num_heads,
            kernel_size,
            groups=self.groups * num_heads,
            padding=kernel_size // 2
        )

    def forward(self, inputs, positions):
        """
        Perform the forward pass of the Hyena projection.

        :param inputs:
            Input tensor of shape (batch_size, seq_length, emb_dim).
        :param positions:
            Position tensor of shape (batch_size, seq_length).
        :return:
            List of n_order projected tensors, each of shape
            (batch_size, num_heads, emb_dim/head_dim, seq_length).
        """
        batch_size, seq_len, emb_dim = inputs.shape
        # Reshape input to split into heads:
        # (batch_size, seq_len, num_heads, head_dim)
        x = inputs.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # Reshape for matrix multiplication and apply projection weights
        x = torch.einsum('bsgd,gdh->bsgh', x, self.W)
        # Reshape for convolution
        x = x.reshape(
            batch_size, seq_len, self.num_heads * self.groups * self.head_dim
        )
        # (batch_size, num_heads*groups*head_dim, seq_len)
        x = x.transpose(1, 2)
        # Apply the convolution
        z = self.conv(x)
        # Set the values at the padded positions to zero
        z = z * (positions != -1).unsqueeze(-2).to(torch.float32)
        # Reshape back to split heads and groups:
        # (batch_size, num_heads, groups, head_dim, seq_len)
        z = z.view(batch_size, self.num_heads, self.groups, self.head_dim, seq_len)
        # Unstack the groups for separate processing
        # List of length `n_order`, each of shape
        # (batch_size, num_heads, head_dim, seq_len)
        z = z.unbind(dim=-3)

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
        x = torch.matmul(inputs, self.W1) + self.b1
        x = self.layer_norm(x)
        x = self.activation(x)
        x = torch.matmul(x, self.W2) + self.b2
        x = x * self.ff_scale
        return x


def sinusoidal_positional_encoding(positions, emb_dim, max_seq_length=256):
    """
    Create sinusoidal positional encodings from a tensor of positions.

    Args:
    - positions (torch.Tensor): Tensor of shape (batch_size, seq_len) with integer positions.
    - emb_dim (int): The dimension of the positional encoding.

    Returns:
    - torch.Tensor: Sinusoidal positional encodings of shape (batch_size, seq_len, emb_dim).
    """
    seq_len = positions.shape[0]
    # Create a tensor for the positional encodings
    position_encodings = torch.zeros((seq_len, emb_dim), device=positions.device)

    # Compute the different angles for the sinusoidal encodings
    position = positions.unsqueeze(-1)  # (batch_size, seq_len, 1)
    div_term = torch.exp(
        torch.arange(0, emb_dim, 2, device=positions.device, dtype=torch.float32) *
        -(torch.log(torch.tensor(max_seq_length * 2.0)) / emb_dim)
    )

    # Apply sine to even indices
    position_encodings[:, 0::2] = torch.sin(position * div_term)
    # Apply cosine to odd indices
    position_encodings[:, 1::2] = torch.cos(position * div_term)

    return position_encodings


class HyenaFilter(Module):
    """
    Learns filters based on positionally transformed embeddings with support
    for multiple heads.

    :param emb_dim: Dimension of the input embeddings.
    :param n_order: Number of orders for the filter.
    :param num_heads: Number of heads for multi-head support.
    :param max_seq_length: Maximum sequence length for positional encodings.
    :param k_gaussians: Number of Gaussian mixtures for windowing.
    :param bias: Small bias to avoid division by zero.
    """

    def __init__(
            self, emb_dim, n_order, num_heads, max_seq_length=256,
            k_gaussians=10, bias=1e-6
    ):
        super(HyenaFilter, self).__init__()
        self.emb_dim = emb_dim
        self.n_order = n_order
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.max_seq_length = max_seq_length
        self.ffn = FeedForward(self.emb_dim, n_order=n_order, activation='sine')
        self.group_weight = nn.Parameter(
            torch.randn(n_order, num_heads, self.head_dim, self.head_dim)
        )
        self.group_bias = nn.Parameter(
            torch.zeros(n_order, num_heads, self.head_dim)
        )
        self.K = k_gaussians

        self.epsilon = 1e-8  # Small value to avoid division by zero

        # Initialize the Gaussian mixture parameters
        # mu and sigma will have dimensions (n_order, num_heads, K, 1)
        self.mu = nn.Parameter(torch.rand(n_order, num_heads, self.K, 1))
        self.sigma = nn.Parameter(
            torch.rand(n_order, num_heads, self.K, 1) * 20.0
        )
        self.weights = nn.Parameter(torch.rand(n_order, num_heads, self.K, 1))
        self.bias = bias

    def forward(self, positional_encodings):
        """
        Perform the forward pass to compute the filters.

        :param positional_encodings:
            Positional encodings tensor of shape
            (batch_size, seq_length, emb_dim).
        :return:
            List of n_order filters, each of shape
            (batch_size, num_heads, head_dim, seq_length).
        """
        # Reshape positional encodings for multi-head processing
        seq_length, emb_dim = positional_encodings.shape
        # positional_encodings = positional_encodings.view(1, seq_length, self.num_heads, self.head_dim)
        # Apply the feed-forward network with sine activation
        h_hat = torch.sin(self.ffn(positional_encodings)).view(
            seq_length, self.n_order, self.num_heads, self.head_dim
        )

        # Apply group weights and biases
        h_hat = torch.einsum(
            'lohd,ohdj->lohj', h_hat, self.group_weight
        ) + self.group_bias

        # Reorder to shape (n_order, num_heads, head_dim, seq_length)
        h_hat = h_hat.permute(1, 2, 3, 0)

        # Normalize h_hat along the channel dimension
        h_hat = h_hat / (h_hat.norm(p=1, dim=-2, keepdim=True) + self.epsilon)

        # Apply each order's Gaussian window to their respective filters
        seq_length = h_hat.size(-1)
        positions = torch.arange(seq_length).float().view(1, 1, 1, -1).to(
            positional_encodings.device
        )
        gaussian_windows = torch.exp(
            -0.5 * (
                    (positions - self.mu * self.max_seq_length) / self.sigma
            ) ** 2
        )
        weighted_gaussian_windows = (gaussian_windows * self.weights).sum(dim=-2)
        # Apply windowing and bias to filters
        h_hat = h_hat * (weighted_gaussian_windows + self.bias).unsqueeze(-2)

        # Split h_hat into h1, h2, ..., hN, where each has shape
        # (batch_size, num_heads, head_dim, seq_length)
        return h_hat.unbind(dim=0)


class FFTLongConv(Module):
    """
    FFT-based long convolution layer with multi-head support.
    """

    def __init__(self):
        super(FFTLongConv, self).__init__()

    def forward(self, inputs, filters, bias, positions):
        """
        Perform the forward pass of the FFT-based long convolution.

        :param inputs:
            Input tensor of shape (batch_size, num_heads, head_dim, seq_length).
        :param filters:
            Filters tensor of shape (num_heads, head_dim, seq_length).
        :param bias:
            Bias tensor of shape (n_order, 1, head_dim, 1).
        :param positions:
            Position tensor of shape (batch_size, seq_length).
        :return:
            Convolution result tensor of shape (batch_size, num_heads, head_dim, seq_length).
        """
        batch_size, num_heads, head_dim, seq_length = inputs.shape

        # Ensure filters have the correct shape
        # (num_heads, 1, head_dim, seq_length)
        filters = filters[..., :seq_length].unsqueeze(0)
        # Prepare inputs for FFT
        padded_length = 2 * seq_length  # Double the length for FFT
        padded_inputs = F.pad(inputs, (0, padded_length - seq_length))
        filters = F.pad(filters, (0, padded_length - seq_length))

        # Perform FFT
        padded_inputs = torch.fft.rfft(
            padded_inputs, n=padded_length, dim=-1, norm='forward'
        )
        filters = torch.fft.rfft(
            filters, n=padded_length, dim=-1, norm='forward'
        )

        # Element-wise multiplication in the frequency domain
        padded_inputs.mul_(filters)

        # Inverse FFT to get the convolution result
        result = torch.fft.irfft(
            padded_inputs, n=padded_length, dim=-1, norm='forward'
        )
        # Remove padding
        result = result[..., :seq_length] + inputs * bias

        # Zero out the padded positions. These positions do not represent
        # nucleotides and should not contribute to the convolution result.
        result = result * (positions != -1).unsqueeze(1).unsqueeze(1).to(
            torch.float32
        )

        return result
