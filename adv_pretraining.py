import torch
# import torch.nn as nn
from components.data_streaming import create_data_loader
from components.base_model import Model, init_weights
from components.read_embedding import (
    NucleotideEmbeddingLayer, MetricEmbedding)
from components.classification_head import (
    TransformerBinaryClassifier, AdversarialLayer)
from components.pretrain_utils import (
    replacement_loss, get_replacement_mask, adversarial_loss, create_intervals,
    WarmupConstantScheduler, get_weights, check_weights
)

import wandb

# For local machine.
device = torch.device("mps") if torch.backends.mps.is_available() \
    else torch.device("cpu")
print(f"Using device: {device}")

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)


# Training parameters.
metadata_path = 'GIAB_BAM/pretraining_metadata.csv'
data_dir = 'GIAB_BAM/illumina_2x250bps'

min_read_quality = 30
batch_size = 4
emb_dim = 128
max_sequence_length = 1024
l1_lambda = 0
adv_iter = 5
main_iter = 5
warm_up_epochs = 5
epochs_at_interval = 2
iters_in_epoch = 50
corruption_rate = 0.15
proportion_mixed_labels = 0.33
main_lr = 1e-3
adv_lr = 5e-4

wandb.init(project="adversarial-pretraining", config={
    "batch_size": batch_size,
    "emb_dim": emb_dim,
    "max_sequence_length": max_sequence_length,
    "l1_lambda": l1_lambda,
    "adv_iter": adv_iter,
    "main_iter": main_iter,
    "epochs_at_interval": epochs_at_interval,
    "iters_in_epoch": iters_in_epoch,
    "corruption_rate": corruption_rate,
    "proportion_mixed_labels": proportion_mixed_labels,
    "learning_rate_main": main_lr,
    "learning_rate_adv": adv_lr,
})

config = wandb.config

# Good model.
nucleotide_embeddings = NucleotideEmbeddingLayer(emb_dim).apply(init_weights)
float_metric_embeddings = MetricEmbedding(
    emb_dim // 2, name='float', num_metrics=2
).apply(init_weights)
binary_metric_embeddings = MetricEmbedding(
    emb_dim // 2, name='binary', num_metrics=14
).apply(init_weights)

readformer = Model(
    emb_dim=emb_dim, heads=8, num_layers=4, hyena=True, kernel_size=3
).apply(init_weights).apply(init_weights)

classifier = TransformerBinaryClassifier(
    embedding_dim=emb_dim, hidden_dim=emb_dim // 2, dropout_rate=0.1
).apply(init_weights)

main_params = (
    list(nucleotide_embeddings.parameters()) +
    list(float_metric_embeddings.parameters()) +
    list(binary_metric_embeddings.parameters()) +
    list(readformer.parameters()) +
    list(classifier.parameters())
)
optimiser = torch.optim.Adam(
    main_params, lr=main_lr,
)
scheduler = WarmupConstantScheduler(
    optimizer=optimiser, warmup=iters_in_epoch * warm_up_epochs, base_lr=main_lr
)

# Adversarial model
evil_nucleotide_embeddings = NucleotideEmbeddingLayer(emb_dim).apply(
    init_weights
)
evil_float_metric_embeddings = MetricEmbedding(
    emb_dim // 2, name='float', num_metrics=2
).apply(init_weights)
evil_binary_metric_embeddings = MetricEmbedding(
    emb_dim // 2, name='binary', num_metrics=14
).apply(init_weights)

evil_readformer = Model(
    emb_dim=emb_dim, heads=8, num_layers=3, hyena=False
).apply(init_weights)
adv_layer = AdversarialLayer(emb_dim).apply(init_weights)
# Initialise the base quality weights to produce values in a reasonable range.
adv_layer._initialise_weights()

adv_params = (
        list(evil_nucleotide_embeddings.parameters()) +
        list(evil_float_metric_embeddings.parameters()) +
        list(evil_binary_metric_embeddings.parameters()) +
        list(evil_readformer.parameters()) +
        list(adv_layer.parameters())
)

evil_optimiser = torch.optim.Adam(
    adv_params, lr=adv_lr,
)
evil_scheduler = WarmupConstantScheduler(
    optimizer=evil_optimiser, warmup=iters_in_epoch * warm_up_epochs,
    base_lr=adv_lr
)

# Get nucleotide intervals up to the nucleotide threshold
intervals = create_intervals(max_sequence_length, 256)

data_loaders = []
for i, interval in enumerate(intervals):
    data_loader = create_data_loader(
        file_paths=data_dir,
        metadata_path=metadata_path,
        nucleotide_threshold=interval,
        max_sequence_length=max_sequence_length,
        batch_size=batch_size,
        min_quality=min_read_quality,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2
    )
    data_loaders.append(data_loader)

i = 0
j = 0
train_adv = False
counter = 0
epoch = 0
epoch_main_losses = []
epoch_adv_losses = []

# Iterate through data
for batch in data_loaders[j]:

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

    nucleotide_emb[replacement_mask] = (
            nucleotide_emb[replacement_mask] *
            (1 - label_mixing_mask).unsqueeze(-1) +
            adv_nucleotide_embeddings.clone() * label_mixing_mask.unsqueeze(-1)
    )
    # Replace the base quality values with adversarial values
    base_qualities_replaced = base_qualities.clone()

    base_qualities_replaced[replacement_mask] = (
        base_qualities_replaced[replacement_mask] * (1 - label_mixing_mask) +
        adv_base_qual.clone() * label_mixing_mask
    )
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
    binary_vec[replacement_mask] = (
            binary_vec[replacement_mask] * (1 - label_mixing_mask).unsqueeze(-1) +
            adv_binary_vec.clone() * label_mixing_mask.unsqueeze(-1)
    )
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
        adv_loss.backward()
        torch.nn.utils.clip_grad_norm_(evil_readformer.parameters(), max_norm=1)
        evil_optimiser.step()
        print(f"Adversarial loss at iteration {i}: {adv_loss.item()}")
        # check_weights(initial_weights, nucleotide_embeddings)
        wandb.log(
            {
                "adv_loss": adv_loss.item(), "iteration": i,
                "adv_lr": evil_scheduler.get_last_lr()
            }
        )
        epoch_adv_losses.append(adv_loss.item())
        counter += 1
        evil_scheduler.step()

        if counter == adv_iter:
            counter = 0
            train_adv = False
    else:

        # Main model loss and optimisation
        loss = replacement_loss(
            output, replacement_mask, label_mixing_mask, valid_mask
        )
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(readformer.parameters(), max_norm=1)
        optimiser.step()
        print(f"Loss at iteration {i}: {loss.item()}")
        wandb.log(
            {
                "main_loss": loss.item(), "iteration": i,
                "main_lr": scheduler.get_last_lr()
            }
        )
        epoch_main_losses.append(loss.item())
        counter += 1
        scheduler.step()

        if counter == main_iter:
            counter = 0
            train_adv = True
    i += 1

    if i % iters_in_epoch == 0:
        mean_main_loss = sum(epoch_main_losses) / len(epoch_main_losses)
        mean_adv_loss = sum(epoch_adv_losses) / len(epoch_adv_losses)
        wandb.log(
            {
                "mean_main_loss": mean_main_loss,
                "mean_adv_loss": mean_adv_loss,
                "epoch": epoch
            }
        )
        torch.save({
            'epoch': epoch,
            'model_state_dict': readformer.state_dict(),
            'classifier_state_dict': classifier.state_dict(),
            'nucleotide_embeddings_state_dict':
                nucleotide_embeddings.state_dict(),
            'float_metric_embeddings_state_dict':
                float_metric_embeddings.state_dict(),
            'binary_metric_embeddings_state_dict':
                binary_metric_embeddings.state_dict(),
            'optimizer_state_dict': optimiser.state_dict(),
            'evil_model_state_dict': evil_readformer.state_dict(),
            'adv_layer_state_dict': adv_layer.state_dict(),
            'evil_nucleotide_embeddings_state_dict':
                evil_nucleotide_embeddings.state_dict(),
            'evil_optimizer_state_dict':
                evil_optimiser.state_dict(),
            'evil_float_metric_embeddings_state_dict':
                evil_float_metric_embeddings.state_dict(),
            'evil_binary_metric_embeddings_state_dict':
                evil_binary_metric_embeddings.state_dict(),
            'mean_main_loss': mean_main_loss,
            'mean_adv_loss': mean_adv_loss
        }, f'checkpoint_{epoch}.pth')
        wandb.save(f'checkpoint_{epoch}.pth')

        print(
            f"Epoch {epoch}: "
            f"Mean Main Loss: {mean_main_loss}, "
            f"Mean Adv Loss: {mean_adv_loss}"
        )
        if j < len(intervals) - 1 and epoch % epochs_at_interval == 0:
            j += 1

        epoch += 1
        epoch_main_losses = []
        epoch_adv_losses = []

    # # For testing.
    # if i == 20:
    #     break

wandb.finish()
