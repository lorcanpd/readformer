import torch
import torch.nn as nn
from components.data_streaming import create_data_loader
from components.base_model import Model
from components.read_embedding import (
    NucleotideEmbeddingLayer, MetricEmbedding)
from components.classification_head import (
    TransformerBinaryClassifier, AdversarialLayer)
from components.pretrain_utils import (
    replacement_loss, get_replacement_mask, adversarial_loss)


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


# Functions to check that weights are not being updated when they shouldn't be.
def get_weights(model):
    return {name: param.clone() for name, param in model.named_parameters()}


def check_weights(initial_weights, model):
    for name, param in model.named_parameters():
        if not torch.equal(initial_weights[name], param):
            print(f"Warning: Weight {name} has been updated!")
        else:
            print(f"Weight {name} has not been updated.")


batch_size = 4
emb_dim = 128
nucleotide_threshold = 1024
l1_lambda = 0
adv_iter = 5
main_iter = 5
corruption_rate = 0.15
proportion_mixed_labels = 0.33

# Good model.
nucleotide_embeddings = NucleotideEmbeddingLayer(emb_dim)
float_metric_embeddings = MetricEmbedding(
    emb_dim // 2, name='float', num_metrics=2
)
binary_metric_embeddings = MetricEmbedding(
    emb_dim // 2, name='binary', num_metrics=14
)

readformer = Model(
    emb_dim=emb_dim, heads=8, num_layers=4, hyena=True, kernel_size=3
)
readformer.apply(init_weights)

classifier = TransformerBinaryClassifier(
    embedding_dim=emb_dim, hidden_dim=emb_dim // 2, dropout_rate=0.1
)
classifier.apply(init_weights)

main_params = (
    list(nucleotide_embeddings.parameters()) +
    list(float_metric_embeddings.parameters()) +
    list(binary_metric_embeddings.parameters()) +
    list(readformer.parameters()) +
    list(classifier.parameters())
)
optimiser = torch.optim.Adam(
    main_params, lr=1e-3,
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
    emb_dim=emb_dim, heads=8, num_layers=4, hyena=False
)
evil_readformer.apply(init_weights)
adv_layer = AdversarialLayer(emb_dim)
adv_layer.apply(init_weights)
adv_layer._initialise_weights()

adv_params = (
        list(evil_nucleotide_embeddings.parameters()) +
        list(evil_float_metric_embeddings.parameters()) +
        list(evil_binary_metric_embeddings.parameters()) +
        list(evil_readformer.parameters()) +
        list(adv_layer.parameters())
)
evil_optimiser = torch.optim.Adam(
    adv_params, lr=5e-4,
)

# Data loader
data_loader = create_data_loader(
    file_paths='TEST_DATA', metadata_path='TEST_DATA/test_metadata.csv',
    nucleotide_threshold=nucleotide_threshold, batch_size=batch_size,
    shuffle=True
)

i = 0
train_adv = False
counter = 0

# Iterate through data
for batch in data_loader:

    nucleotide_sequences = batch['nucleotide_sequences']
    valid_mask = nucleotide_sequences != 15
    base_qualities = batch['base_qualities']
    read_qualities = batch['read_qualities']
    cigar_match = batch['cigar_match']
    cigar_insertion = batch['cigar_insertion']
    bitwise_flags = batch['bitwise_flags']
    positions = batch['positions']

    # Identify the positions to corrupt
    replacement_mask = get_replacement_mask(
        positions, rate=corruption_rate
        # corruption_rate
    )

    if train_adv:
        for param in main_params:
            param.requires_grad = False
        for param in adv_params:
            param.requires_grad = True
        # initial_weights = get_weights(nucleotide_embeddings)
        label_mixing_mask = torch.ones(torch.sum(replacement_mask))
    else:
        for param in main_params:
            param.requires_grad = True
        for param in adv_params:
            param.requires_grad = False
        # initial_weights = get_weights(evil_nucleotide_embeddings)
        label_mixing_mask = torch.ones(torch.sum(replacement_mask))
        # Randomly select proportion_mixed_labels of the label_mixing_mask to
        # be a value between 0 and 1.
        label_mixing_mask[
            torch.randperm(label_mixing_mask.size(0))[:int(
                proportion_mixed_labels * label_mixing_mask.size(0))
            ]
        ] = torch.rand(int(proportion_mixed_labels * label_mixing_mask.size(0)))

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

    adv_nucleotide_embeddings, adv_base_qual, adv_binary_vec = adv_layer(
        evil_output[replacement_mask], nucleotide_sequences[replacement_mask],
        nucleotide_embeddings.embedding.weight.clone(),
    )
    # Generate the real inputs
    nucleotide_emb = nucleotide_embeddings(nucleotide_sequences)
    # TODO LABEL MIXING

    nucleotide_emb[replacement_mask] = (
            nucleotide_emb[replacement_mask] *
            (1 - label_mixing_mask).unsqueeze(-1) +
            adv_nucleotide_embeddings.clone() * label_mixing_mask.unsqueeze(-1)
    )
    # nucleotide_emb[replacement_mask] = adv_nucleotide_embeddings.clone()
    # Replace the base quality values with adversarial values
    base_qualities_replaced = base_qualities.clone()
    # TODO LABEL MIXING
    base_qualities_replaced[replacement_mask] = (
            base_qualities_replaced[replacement_mask] * (1 - label_mixing_mask) +
            adv_base_qual.clone() * label_mixing_mask
    )
    # base_qualities_replaced[replacement_mask] = adv_base_qual.clone(
    # ).squeeze(-1)
    # Get the binary metric embeddings
    float_metrics = torch.stack(
        [base_qualities_replaced, read_qualities], dim=-1
    ).detach()
    binary_vec = torch.cat(
        [
            bitwise_flags,
            torch.stack([cigar_match, cigar_insertion], dim=-1)
        ],
        dim=-1
    ).float().detach()
    # TODO LABEL MIXING
    binary_vec[replacement_mask] = (
            binary_vec[replacement_mask] * (1 - label_mixing_mask).unsqueeze(-1) +
            adv_binary_vec.clone() * label_mixing_mask.unsqueeze(-1)
    )
    # binary_vec[replacement_mask] = adv_binary_vec.clone()
    binary_metric_emb = binary_metric_embeddings(binary_vec)
    metric_emb = torch.cat(
        [
            float_metric_embeddings(float_metrics),
            binary_metric_emb
        ],
        dim=-1
    )

    model_input = nucleotide_emb + metric_emb
    output = readformer(model_input, positions)
    output = classifier(output)

    if train_adv:
        # Compute adversarial loss using the main model's output
        adv_loss = adversarial_loss(
            output, replacement_mask, label_mixing_mask
            # l1 normalisation to keep the binary metrics sparse.
        ) + l1_lambda * torch.norm(adv_layer.fc_binary_metrics.weight, 1)
        evil_optimiser.zero_grad()
        # adv_loss.backward(retain_graph=True)
        adv_loss.backward()
        evil_optimiser.step()
        print(f"Adversarial loss at iteration {i}: {adv_loss.item()}")
        # check_weights(initial_weights, nucleotide_embeddings)
        counter += 1

        if counter == adv_iter:
            counter = 0
            train_adv = False
    else:

        # Main model loss and optimisation
        loss = replacement_loss(
            output, replacement_mask, label_mixing_mask, valid_mask
        )
        optimiser.zero_grad()
        # loss.backward(retain_graph=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(readformer.parameters(), max_norm=1)
        optimiser.step()
        print(f"Loss at iteration {i}: {loss.item()}")
        # check_weights(initial_weights, evil_nucleotide_embeddings)
        counter += 1

        if counter == main_iter:
            counter = 0
            train_adv = True
    i += 1
