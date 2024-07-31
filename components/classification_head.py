import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm as lin_layer_sn

def spectral_norm(weight, n_power_iterations=1):
    """
    Compute the spectral normalisation of a weight tensor.

    :param weight:
        The weight tensor to be normalized.
    :param n_power_iterations:
        Number of power iterations to perform. Default is 1.
    :returns: Spectrally normalised weight tensor.
    """
    # Ensure weight is a 2D tensor for spectral norm calculation
    weight_mat = weight.view(weight.size(0), -1)

    u = nn.Parameter(
        torch.randn(weight_mat.size(0), 1), requires_grad=False
    )#.to(weight.device)
    v = nn.Parameter(
        torch.randn(weight_mat.size(1), 1), requires_grad=False
    )#.to(weight.device)

    for _ in range(n_power_iterations):
        v = torch.matmul(weight_mat.t(), u)
        v = v / (torch.norm(v) + 1e-12)
        u = torch.matmul(weight_mat, v)
        u = u / (torch.norm(u) + 1e-12)

    spectral_norm_value = torch.matmul(u.t(), torch.matmul(weight_mat, v))
    return weight / spectral_norm_value


class TransformerBinaryClassifier(nn.Module):
    """
    Binary classifier that transforms the vector outputs of a transformer model
    into binary predictions.

    :param embedding_dim:
        Dimension of the input embeddings.
    :param hidden_dim:
        Dimension of the hidden layer.
    :param dropout_rate:
        Dropout rate for regularisation. Default is 0.1.
    """
    def __init__(self, embedding_dim, hidden_dim, dropout_rate=0.1):
        super(TransformerBinaryClassifier, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        # Layer Normalization
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        # Fully connected layers
        self.w1 = nn.Parameter(torch.randn(embedding_dim, hidden_dim))
        self.b1 = nn.Parameter(
            nn.init.zeros_(torch.randn(hidden_dim))
        )
        self.w2 = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.b2 = nn.Parameter(
            nn.init.zeros_(torch.randn(hidden_dim))
        )
        self.w3 = nn.Parameter(
            nn.init.xavier_uniform_(torch.randn(hidden_dim, 1))
        )
        self.b3 = nn.Parameter(
            nn.init.zeros_(torch.randn(1))
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Residual connections
        self.residual_transform = nn.Parameter(
            nn.init.xavier_uniform_(torch.randn(embedding_dim, hidden_dim))
        )
        self.residual_bias = nn.Parameter(
            nn.init.zeros_(torch.randn(hidden_dim))
        )

    def forward(self, x):
        """
        Perform the forward pass of the binary classifier.

        :param x:
            Input tensor of shape (batch_size, seq_length, embedding_dim).
        :returns:
            Output tensor after applying the classification layers and sigmoid
            activation.
        """
        # w1 = spectral_norm(self.w1)
        # w2 = spectral_norm(self.w2)
        x = self.layer_norm1(x)
        # First fully connected layer with ReLU activation
        residual = torch.matmul(x, self.residual_transform) + self.residual_bias
        x = F.relu(torch.matmul(x, self.w1) + self.b1)
        # Dropout for regularization
        x = self.dropout(x)
        # Second fully connected layer with ReLU activation
        x = F.relu(torch.matmul(x, self.w2) + self.b2)
        # Adding residual connection
        x = x + residual
        # Second layer normalization
        x = self.layer_norm2(x)
        # Dropout for regularization
        x = self.dropout(x)
        # Final fully connected layer
        x = torch.matmul(x, self.w3) + self.b3
        # Sigmoid activation for binary classification
        output = torch.sigmoid(x).squeeze(-1).clamp(0, 1)
        return output


class AdversarialLayer(nn.Module):
    """
    Adversarial layer for generating adversarial parameters to perturb the
    original read sequences. For example, it learns to produce replacement
    nucleotides with realistic base quality values and other read metrics.

    :param emb_dim:
        Dimension of the input embeddings.
    """
    def __init__(self, emb_dim):
        super(AdversarialLayer, self).__init__()
        self.transform = lin_layer_sn(nn.Linear(emb_dim, emb_dim))
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(emb_dim)
        self.activation = nn.ReLU()

        # Fully connected layers
        self.fc_base_quality = nn.Linear(emb_dim, 1)
        self.relu = nn.ReLU()
        self.fc_binary_metrics = nn.Linear(emb_dim, 14)

        # Nucleotide prediction
        # 15 possible nucleotide indices excluding the padding index
        self.num_nucleotides = 15
        self.fc_nucleotide = nn.Linear(emb_dim, self.num_nucleotides)
        self.softmax = nn.Softmax(dim=-1)

    def _initialise_weights(self):
        """
        Initialise the weights of the adversarial layer so that the base quality
        predictions are initially in a realistic range.
        """

        # nn.init.constant_(self.fc_base_quality.weight, 3)
        nn.init.normal_(self.fc_base_quality.weight, 2, 0.1)
        nn.init.constant_(self.fc_base_quality.bias, 0)

    def forward(
            self, x, original_nucleotide_indices, nucleotide_embeddings,
            temperature=1.0
    ):
        """
        Perform the forward pass of the adversarial layer.

        :param x:
            Input tensor of shape (batch_size, seq_length, emb_dim).
        :param original_nucleotide_indices:
            Tensor containing the original nucleotide indices.
        :param nucleotide_embeddings:
            Embedding tensor for nucleotides.
        :param temperature:
            Temperature parameter for Gumbel-Softmax. Default is 1.0.
        :returns:
            Adversarial nucleotide embeddings, base quality values, and binary
            vector embeddings.
        """
        # Apply linear transformation
        x = self.transform(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Generate adversarial base quality in logarithmic space and apply STE
        log_base_quality = self.fc_base_quality(x)
        log_base_quality_ste = (log_base_quality).float().round() + (
                log_base_quality - log_base_quality.detach()
        )
        base_quality_value = 10 ** (-log_base_quality_ste.squeeze(-1) / 10)

        # Generate realistic binary flag and cigar vector using STE
        binary_vector_logits = self.fc_binary_metrics(x)
        binary_vector_prob = torch.sigmoid(binary_vector_logits)
        binary_vector_emb = (binary_vector_prob > 0.5).float() + (
                    binary_vector_prob - binary_vector_prob.detach())  # STE
        # Generate nucleotide predictions
        nucleotide_logits = self.fc_nucleotide(x)
        # nucleotide_probs = self.softmax(nucleotide_logits)

        num_replaced = x.shape[0]
        mask = torch.zeros((num_replaced, self.num_nucleotides))#.to(x.device)
        valid_indices = (
                original_nucleotide_indices < self.num_nucleotides
        ).nonzero(as_tuple=True)[0]
        masked_indices = original_nucleotide_indices[valid_indices].unsqueeze(1)
        mask = torch.zeros_like(nucleotide_logits)
        mask[valid_indices] = mask[valid_indices].scatter(
            1, masked_indices.to(torch.int64),
            float('-inf')
        )

        masked_logits = nucleotide_logits + mask
        gumbel_softmax_sample = F.gumbel_softmax(
            masked_logits, tau=temperature, hard=True
        )

        adv_nucleotide_embeddings = torch.matmul(
            gumbel_softmax_sample, nucleotide_embeddings[0:self.num_nucleotides]
        )

        return adv_nucleotide_embeddings, base_quality_value, binary_vector_emb


# Under development
class MLMClassifier(nn.Module):
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


class MLMLoss(nn.Module):
    """
    Loss function for Masked Language Modeling (MLM).

    """

    def __init__(self):
        super(MLMLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        """
        Compute the MLM loss.

        :param logits:
            Predicted logits of shape (batch_size, seq_length, num_classes).
        :param target:
            True indices of shape (batch_size, seq_length).
        :returns:
            Scalar loss value.
        """
        # Flatten logits and target for loss computation
        logits_flat = logits.view(-1, logits.size(-1))
        target_flat = target.view(-1).long()
        # loss = self.loss_fn(logits_flat, target_flat)

        probs = F.softmax(logits_flat, dim=-1)

        # Gather probabilities of the true classes
        true_probs = probs.gather(
            dim=-1, index=target_flat.unsqueeze(-1)
        ).squeeze(-1)

        # Calculate the negative log-likelihood loss
        nll_loss = -torch.log(true_probs).mean()

        # with torch.no_grad():
        #     pred_probs = F.softmax(logits_flat, dim=-1)
        #     pred_classes = torch.argmax(pred_probs, dim=-1)
        #     correct_predictions = (pred_classes == target_flat).float()
        #     accuracy = correct_predictions.mean().item()
        #     print(f"Batch Accuracy: {accuracy * 100:.2f}%")
        #     # print(f"Loss: {loss.item()}")

        return nll_loss

