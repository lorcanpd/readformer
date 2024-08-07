import torch.nn as nn

# TODO fix the import statements
from components.readformer import ReadformerBlock
from components.self_attention import TransformerBlock


from components.better_device_handling import Module


def init_weights(m):
    """
    Initialise the weights of any model.

    :param m:
    :return:
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.Parameter):
        if m.dim() > 1:
            nn.init.kaiming_uniform_(m)
        else:
            nn.init.zeros_(m)


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
    :param readformer:
        Whether to use the Hyena block instead of self-attention. Default is
        True.
    :param kernel_size:
        Size of the convolution kernel for the Hyena local CNN. Default is 3.
    """
    def __init__(
            self, emb_dim, num_layers, readformer=True, kernel_size=3, heads=8,
            n_order=4
    ):
        super(Model, self).__init__()
        self.emb_dim = emb_dim
        self.heads = heads
        self.num_layers = num_layers
        self.kernel_size = kernel_size

        if readformer:
            self.layers = nn.ModuleList(
                [
                    ReadformerBlock(
                        emb_dim, n_order, kernel_size
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    TransformerBlock(
                        emb_dim, heads
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




# test
#
# import torch
#
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
# readformer = Model(
#     emb_dim, num_layers=2, readformer=True, kernel_size=3, heads=8,
#     n_order=8
# )
#
# output = readformer(inputs, positions)
