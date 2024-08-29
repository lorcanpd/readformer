import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        """
        Root Mean Square Layer Normalisation (RMSNorm).

        :param dim:
            The dimension of the input embeddings.
        :param eps:
            A small value to avoid division by zero.
        """
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))  # Learnable scaling parameter

    def forward(self, x):
        # Compute the RMS (Root Mean Square) normalization
        rms = x.norm(2, dim=-1, keepdim=True) / (x.size(-1) ** 0.5)
        x_normalised = x / (rms + self.eps)
        return self.scale * x_normalised