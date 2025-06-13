import torch
import torch.nn as nn

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
    def __init__(self, input_dim, hidden_dim=None, using_reference_embedding=False):
        """
        Initialises a simple beta distribution classifier with separate feedforward
        networks for alpha and beta.

        Args:
            input_dim (int): Dimension of the input features.
            hidden_dim (int, optional): Hidden dimension for the feedforward networks.
                Defaults to input_dim if not provided.
            using_reference_embedding (bool): If True, the reference embedding is concatenated
                with the input (doubling the input dimension).
        """
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.using_reference_embedding = using_reference_embedding
        combined_dim = input_dim * 2 if using_reference_embedding else input_dim

        # Feedforward network for predicting the alpha parameter.
        self.alpha_fc = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensures a positive output.
        )

        # Feedforward network for predicting the beta parameter.
        self.beta_fc = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensures a positive output.
        )

    def forward(self, inputs, reference_embedding=None):
        """
        Forward pass that computes the alpha and beta parameters.

        Args:
            inputs (Tensor): Input tensor of shape (batch_size, input_dim).
            reference_embedding (Tensor, optional): Tensor of shape (batch_size, input_dim)
                that is concatenated with inputs if using_reference_embedding is True.

        Returns:
            Tuple[Tensor, Tensor]: alpha and beta each of shape (batch_size, 1).
        """
        if self.using_reference_embedding and reference_embedding is not None:
            combined = torch.cat([inputs, reference_embedding], dim=-1)
        else:
            combined = inputs

        alpha = self.alpha_fc(combined)
        beta = self.beta_fc(combined)
        return alpha, beta

