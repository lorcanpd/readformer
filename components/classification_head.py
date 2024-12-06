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


# class BetaDistributionClassifier(nn.Module):
#     """
#     A classification layer that learns the alpha and beta parameters of
#     a Beta distribution. It takes as input the embeddings of the model output
#     corresponding to the read being classified and the embedding of the
#     reference base. It uses the reference base embedding to modulate the
#     model output embedding before passing it through a feedforward network
#     to predict the alpha and beta parameters of the Beta distribution.
#     """
#     def __init__(self, input_dim, hidden_dim=None, dropout_rate=0.1):
#         super(BetaDistributionClassifier, self).__init__()
#
#         if hidden_dim is None:
#             hidden_dim = input_dim
#
#         self.layer_norm = nn.LayerNorm(hidden_dim)
#
#         self.mod_layer_norm = nn.LayerNorm(hidden_dim)
#
#         self.model_output_proj = nn.Linear(input_dim, hidden_dim)
#         self.silu_1 = nn.SiLU()
#         self.reference_base_proj = nn.Linear(input_dim, hidden_dim)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.alpha_gate = nn.Linear(hidden_dim, hidden_dim)
#         self.alpha_proj = nn.Linear(hidden_dim, hidden_dim)
#         self.alpha_silu_1 = nn.SiLU()
#         self.fc_alpha_1 = nn.Linear(hidden_dim, hidden_dim)
#         self.alpha_silu_2 = nn.SiLU()
#         self.fc_alpha_2 = nn.Linear(hidden_dim, 1)
#         self.alpha_softplus = nn.Softplus()
#
#         self.beta_gate = nn.Linear(hidden_dim, hidden_dim)
#         self.beta_proj = nn.Linear(hidden_dim, hidden_dim)
#         self.beta_silu_1 = nn.SiLU()
#         self.fc_beta_1 = nn.Linear(hidden_dim, hidden_dim)
#         self.beta_silu_2 = nn.SiLU()
#         self.fc_beta_2 = nn.Linear(hidden_dim, 1)
#         self.beta_softplus = nn.Softplus()
#
#     def forward(self, inputs, reference_base_embedding):
#         normed_inputs = self.layer_norm(inputs)
#
#         projected_inputs = self.model_output_proj(normed_inputs)
#         reference_modulation = self.silu_1(self.reference_base_proj(reference_base_embedding))
#
#         # Modulate the model output embedding
#         input_modulation = self.dropout(projected_inputs * reference_modulation)
#
#         # Residual where modulated input is added to the original input
#         modulated_inputs = inputs + input_modulation
#
#         # Pre layer norm for alpha and beta
#         normed_inputs = self.mod_layer_norm(modulated_inputs)
#
#         # Alpha SwiGLU
#         alpha_mod = self.alpha_gate(normed_inputs)
#         alpha_mod = self.alpha_silu_1(alpha_mod)
#         alpha = normed_inputs * alpha_mod
#         alpha = self.fc_alpha_1(alpha)
#         alpha = self.alpha_silu_2(alpha)
#         alpha = self.fc_alpha_2(alpha)
#         alpha = self.alpha_softplus(alpha)
#
#         # Beta SwiGLU
#         beta_mod = self.beta_gate(normed_inputs)
#         beta_mod = self.beta_silu_1(beta_mod)
#         beta = normed_inputs * beta_mod
#         beta = self.fc_beta_1(beta)
#         beta = self.beta_silu_2(beta)
#         beta = self.fc_beta_2(beta)
#         beta = self.beta_softplus(beta)
#
#         return alpha, beta

# class BetaDistributionClassifier(nn.Module):
#     def __init__(self, input_dim, hidden_dim=None, dropout_rate=0.1):
#         super(BetaDistributionClassifier, self).__init__()
#
#         if hidden_dim is None:
#             hidden_dim = input_dim
#
#         # Normalize inputs before projection
#         self.layer_norm = nn.LayerNorm(input_dim)
#
#         # Linear projection for the main input
#         self.input_proj = nn.Linear(input_dim, hidden_dim)
#
#         # Dropout for regularization
#         self.dropout = nn.Dropout(dropout_rate)
#
#         # Separate MLP heads for alpha and beta
#         self.alpha_net = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, 1),
#             nn.Softplus()  # ensures positivity
#         )
#
#         self.beta_net = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, 1),
#             nn.Softplus()  # ensures positivity
#         )
#
#     def forward(self, inputs, reference_base_embedding):
#         # Normalize inputs
#         normed_inputs = self.layer_norm(inputs)
#
#         # Project the main inputs
#         projected_inputs = self.input_proj(normed_inputs)
#
#         # Modulate the inputs with the reference base embedding
#         modulated_inputs = projected_inputs * reference_base_embedding
#         modulated_inputs = self.dropout(modulated_inputs)
#
#         # Produce alpha and beta parameters
#         alpha = self.alpha_net(modulated_inputs)
#         beta = self.beta_net(modulated_inputs)
#
#         return alpha, beta


class BetaDistributionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, dropout_rate=0.1):
        super(BetaDistributionClassifier, self).__init__()

        if hidden_dim is None:
            hidden_dim = input_dim

        # Normalize the inputs
        self.layer_norm = nn.LayerNorm(input_dim)

        # Non-linear projection for the main input embedding
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate)
        )

        # Separate transformations of the reference embedding for alpha and beta
        self.alpha_ref_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.beta_ref_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Separate linear heads for alpha and beta
        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # ensures positivity of alpha
        )

        self.beta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # ensures positivity of beta
        )

    def forward(self, inputs, reference_base_embedding):
        # Normalise inputs
        normed_inputs = self.layer_norm(inputs)

        # Non-linear transform of main input
        base_rep = self.input_proj(normed_inputs)

        # Apply separate non-linear transformations to the reference embedding
        alpha_ref = self.alpha_ref_transform(reference_base_embedding)
        beta_ref = self.beta_ref_transform(reference_base_embedding)

        # Modulate base_rep with the reference transformations
        alpha_modulated = base_rep * alpha_ref
        beta_modulated = base_rep * beta_ref

        # Compute alpha and beta from their respective modulated inputs
        alpha = self.alpha_head(alpha_modulated)
        beta = self.beta_head(beta_modulated)

        return alpha, beta









