import torch
import torch.nn as nn
from components.data_streaming import create_data_loader
from components.base_model import Model
from components.read_embedding import NucleotideEmbeddingLayer, MetricEmbedding
from components.classification_head import TransformerBinaryClassifier, AdversarialLayer
from components.pretrain_utils import replacement_loss, get_replacement_mask, adversarial_loss

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

batch_size = 2
emb_dim = 256
nucleotide_threshold = 1024
# corruption_rate = 0.15

# Good model.
nucleotide_embeddings = NucleotideEmbeddingLayer(emb_dim)
float_metric_embeddings = MetricEmbedding(
    emb_dim // 2, name='float', num_metrics=2
)
binary_metric_embeddings = MetricEmbedding(
    emb_dim // 2, name='binary', num_metrics=14
)

readformer = Model(
    emb_dim=emb_dim, heads=8, num_layers=3, hyena=True, kernel_size=3
)
readformer.apply(init_weights)

classifier = TransformerBinaryClassifier(
    embedding_dim=emb_dim, hidden_dim=emb_dim // 2, dropout_rate=0.1
)
classifier.apply(init_weights)

optimiser = torch.optim.Adam(
    list(readformer.parameters()) + list(classifier.parameters()), lr=1e-3,
)

# Adversarial model
evil_nucleotide_embeddings = NucleotideEmbeddingLayer(emb_dim)
evil_float_metric_embeddings = MetricEmbedding(
    emb_dim // 2, name='float', num_metrics=2
)
evil_binary_metric_embeddings = MetricEmbedding(
    emb_dim // 2, name='binary', num_metrics=14
)

evil_readformer = Model(
    emb_dim=emb_dim, heads=8, num_layers=3, hyena=False
)
evil_readformer.apply(init_weights)
adv_layer = AdversarialLayer(emb_dim)
adv_layer.apply(init_weights)

evil_optimiser = torch.optim.Adam(
    list(evil_readformer.parameters()) + list(adv_layer.parameters()), lr=1e-3,
)

# Data loader
data_loader = create_data_loader(
    file_paths='TEST_DATA', metadata_path='TEST_DATA/test_metadata.csv',
    nucleotide_threshold=nucleotide_threshold, batch_size=batch_size,
    shuffle=True
)

i = 0

# Iterate through data
for batch in data_loader:
    nucleotide_sequences = batch['nucleotide_sequences']
    base_qualities = batch['base_qualities']
    read_qualities = batch['read_qualities']
    cigar_match = batch['cigar_match']
    cigar_insertion = batch['cigar_insertion']
    bitwise_flags = batch['bitwise_flags']
    positions = batch['positions']

    # Identify the positions to corrupt
    replacement_mask = get_replacement_mask(
        positions, rate=(torch.rand(1) * 0.3 + 0.05).float()
        # corruption_rate
    )

    # Adversarial model forward pass
    evil_nucleotide_input = evil_nucleotide_embeddings(nucleotide_sequences)
    evil_float_metrics = torch.stack(
        [base_qualities, read_qualities], dim=-1
    )
    evil_binary_metrics = torch.stack(
        [cigar_match, cigar_insertion], dim=-1
    )
    evil_binary_metrics = torch.cat(
        [bitwise_flags, evil_binary_metrics], dim=-1
    ).float()

    evil_metric_emb = torch.cat(
        [
            evil_float_metric_embeddings(evil_float_metrics),
            evil_binary_metric_embeddings(evil_binary_metrics)
        ],
        dim=-1
    )

    evil_model_input = evil_nucleotide_input + evil_metric_emb
    evil_output = evil_readformer(evil_model_input, positions)
    nuc_replacements, adv_base_qual, adv_binary_vec = adv_layer(
        evil_output[replacement_mask], nucleotide_sequences[replacement_mask]
    )

    nucleotide_sequences[replacement_mask] = nuc_replacements
    # Generate the real inputs
    nucleotide_emb = nucleotide_embeddings(nucleotide_sequences)

    # Replace embeddings with adversarial embeddings without in-place operation
    nucleotide_emb_replaced = nucleotide_emb.clone().detach()

    # Replace the base quality values with adversarial values
    base_qualities_replaced = base_qualities.clone().detach()
    base_qualities_replaced[replacement_mask] = adv_base_qual.clone().detach().squeeze(-1)

    # Get the binary metric embeddings
    float_metrics = torch.stack(
        [base_qualities_replaced, read_qualities], dim=-1
    ).detach()

    binary_vec = torch.cat(
        [
            bitwise_flags.detach(),
            torch.stack([cigar_match, cigar_insertion], dim=-1)
        ],
        dim=-1
    ).float().detach()
    binary_vec[replacement_mask] = adv_binary_vec.clone().detach()
    binary_metric_emb = binary_metric_embeddings(binary_vec)

    metric_emb = torch.cat(
        [
            float_metric_embeddings(float_metrics),
            binary_metric_emb
        ],
        dim=-1
    )

    model_input = nucleotide_emb_replaced + metric_emb
    output = readformer(model_input, positions)
    output = classifier(output)

    # Compute adversarial loss using the main model's output
    adv_loss = adversarial_loss(output, replacement_mask)
    evil_optimiser.zero_grad()
    adv_loss.backward(retain_graph=True)
    # torch.nn.utils.clip_grad_norm_(evil_readformer.parameters(), max_norm=1)
    evil_optimiser.step()

    # Main model loss and optimization
    loss = replacement_loss(replacement_mask, output)
    optimiser.zero_grad()
    loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(readformer.parameters(), max_norm=1)
    optimiser.step()

    print(f"Loss at iteration {i}: {loss.item()}")
    print(f"Adversarial loss at iteration {i}: {adv_loss.item()}")
    i += 1
