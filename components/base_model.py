import torch.nn as nn

# TODO fix the import statements
from components.hyena import HyenaBlock, FeedForward
from components.self_attention import MultiHeadSelfAttention


class TransformerBlock(nn.Module):
    """
    Transformer block that can switch between self-attention and Hyena block.

    :param emb_dim:
        Dimension of the input embeddings.
    :param heads:
        Number of heads for self-attention or global filters for the Hyena
        block.
    :param hyena:
        Whether to use the Hyena block instead of self-attention. Default is
        False.
    :param kernel_size:
        Size of the convolution kernel for the Hyena block. Default is 3.
    """
    def __init__(self, emb_dim, heads, hyena=False, kernel_size=3):
        super(TransformerBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(emb_dim, eps=1e-7)
        nn.init.ones_(self.layer_norm1.weight)
        nn.init.zeros_(self.layer_norm1.bias)
        self.layer_norm2 = nn.LayerNorm(emb_dim, eps=1e-7)
        nn.init.ones_(self.layer_norm2.weight)
        nn.init.zeros_(self.layer_norm2.bias)
        self.feed_forward = FeedForward(
            emb_dim, hidden_dim=emb_dim * 2, activation="mish"
        )
        self.hyena = hyena
        if hyena:
            self.self_attention = HyenaBlock(
                emb_dim, n_order=heads, kernel_size=kernel_size
            )
        else:
            self.self_attention = MultiHeadSelfAttention(
                emb_dim, num_heads=heads
            )

    def forward(self, x, positions):
        """
        Perform the forward pass of the transformer block.

        :param x:
            Input tensor of shape (batch_size, seq_length, emb_dim).
        :param positions:
            Position tensor of shape (batch_size, seq_length).
        :returns:
            Output tensor after applying self-attention/Hyena block and
            feedforward network.
        """
        # Apply layer normalisation before self-attention
        normed_x = self.layer_norm1(x)
        self_attention_output = self.self_attention(normed_x, positions)
        # Residual connection
        x = x + self_attention_output
        # Apply layer normalization before feedforward network
        normed_x = self.layer_norm2(x)
        ff_output = self.feed_forward(normed_x)
        # Residual connection
        x = x + ff_output

        return x


class Model(nn.Module):
    """
    Transformer model consisting of multiple transformer blocks.

    :param emb_dim:
        Dimension of the input embeddings.
    :param heads:
        Number of heads or global convolution filters for each transformer
        block.
    :param num_layers:
        Number of transformer blocks.
    :param hyena:
        Whether to use the Hyena block instead of self-attention. Default is
        False.
    :param kernel_size:
        Size of the convolution kernel for the Hyena block. Default is 3.
    """
    def __init__(self, emb_dim, heads, num_layers, hyena=False, kernel_size=3):
        super(Model, self).__init__()
        self.emb_dim = emb_dim
        self.heads = heads
        self.num_layers = num_layers
        self.hyena = hyena
        self.kernel_size = kernel_size

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    emb_dim, heads, hyena=hyena, kernel_size=kernel_size
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, positions):
        """
        Perform the forward pass of the transformer model.

        :param x:
            Input tensor of shape (batch_size, seq_length, emb_dim).
        :param positions:
            Position tensor of shape (batch_size, seq_length).
        :returns:
            Output tensor after passing through all transformer blocks.
        """
        for layer in self.layers:
            x = layer(x, positions)

        return x
