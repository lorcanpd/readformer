import torch
import torch.nn as nn
import torch.nn.functional as F

from components.better_device_handling import Module


class MLMClassifier(Module):
    """
    Classification layer for Masked Language Modeling (MLM).

    :param emb_dim:
        Dimension of the input embeddings.
    :param num_classes:
        Number of possible outcome indices (nucleotide classes).
    """

    def __init__(self, emb_dim, num_classes):
        super(MLMClassifier, self).__init__()
        self.linear = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        """
        Perform the forward pass of the classification layer.

        :param x:
            Input tensor of shape (batch_size, seq_length, embedding_dim).
        :returns:
            Output tensor of shape (batch_size, seq_length, num_classes)
            containing the logits for each class.
        """
        logits = self.linear(x)
        return logits


class BetaDistributionClassifier(nn.Module):
    """
    A classification layer that learns the alpha and beta parameters of
    a Beta distribution.
    """
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.1):
        super(BetaDistributionClassifier, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_alpha = nn.Linear(hidden_dim, 1)
        self.fc_beta = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        alpha = F.softplus(self.fc_alpha(x)) + 1e-12
        beta = F.softplus(self.fc_beta(x)) + 1e-12
        return alpha, beta
