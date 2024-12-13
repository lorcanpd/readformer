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
    def __init__(
        self,
        input_dim,
        hidden_dim=None,
        dropout_rate=0.1,
        num_experts=8,
        expert_hidden_dim=None,
        top_k=3
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = input_dim

        if expert_hidden_dim is None:
            expert_hidden_dim = hidden_dim // 2

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_dim = hidden_dim
        self.expert_hidden_dim = expert_hidden_dim

        # We'll concatenate the input and reference embedding
        combined_dim = input_dim * 2

        # Separate gating networks for alpha and beta
        self.gate_alpha = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_experts)
        )

        self.gate_beta = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_experts)
        )

        # Shared expert pool
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(combined_dim, expert_hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(expert_hidden_dim, expert_hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_rate)
            ) for _ in range(num_experts)
        ])

        # Separate heads for alpha and beta
        self.alpha_head = nn.Sequential(
            nn.Linear(expert_hidden_dim, expert_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(expert_hidden_dim, 1),
            nn.Softplus()
        )

        self.beta_head = nn.Sequential(
            nn.Linear(expert_hidden_dim, expert_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(expert_hidden_dim, 1),
            nn.Softplus()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs, reference_base_embedding):
        """
        inputs: (batch_size, input_dim)
        reference_base_embedding: (batch_size, input_dim)

        returns:
            alpha, beta each of shape (batch_size, 1)
            gate_alpha_scores, gate_beta_scores of shape (batch_size, num_experts)
        """
        # Concatenate input and reference base embeddings
        combined = torch.cat(
            [inputs, reference_base_embedding], dim=-1)

        # Compute gating distributions
        gate_alpha_scores = F.softmax(self.gate_alpha(combined), dim=-1)
        gate_beta_scores = F.softmax(self.gate_beta(combined), dim=-1)

        # Top-k selection for alpha
        topk_alpha_values, topk_alpha_indices = torch.topk(
            gate_alpha_scores, self.top_k, dim=-1)
        # Top-k selection for beta
        topk_beta_values, topk_beta_indices = torch.topk(
            gate_beta_scores, self.top_k, dim=-1)

        # Compute all experts once
        expert_outputs = [expert(combined) for expert in self.experts]
        # (num_experts, batch_size, expert_hidden_dim) ->
        # (batch_size, num_experts, expert_hidden_dim)
        expert_outputs = torch.stack(expert_outputs, dim=0).permute(1, 0, 2)

        # Gather top-k experts for alpha
        selected_alpha_experts = torch.gather(
            expert_outputs,
            1,
            topk_alpha_indices.unsqueeze(-1).expand(
                -1, -1, self.expert_hidden_dim)
        )
        aggregated_alpha = torch.sum(
            selected_alpha_experts * topk_alpha_values.unsqueeze(-1), dim=1)

        # Gather top-k experts for beta
        selected_beta_experts = torch.gather(
            expert_outputs,
            1,
            topk_beta_indices.unsqueeze(-1).expand(
                -1, -1, self.expert_hidden_dim)
        )
        aggregated_beta = torch.sum(
            selected_beta_experts * topk_beta_values.unsqueeze(-1), dim=1)

        # Compute final alpha and beta parameters
        alpha = self.alpha_head(aggregated_alpha)
        beta = self.beta_head(aggregated_beta)

        # Return gating scores so we can compute load-balance loss externally
        return alpha, beta, gate_alpha_scores, gate_beta_scores









