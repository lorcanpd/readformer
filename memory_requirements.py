import torch
import components.data_streaming
import components.base_model
import components.read_embedding
import components.classification_head
import components.pretrain_utils
import components.data_streaming


"""
Script for profiling the memory requirements of the model and data loader. The 
intention is to provide an estimate of the memory requirements for training the
model on a given device.
"""


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def calculate_data_loader_memory(data_loader, device):
    # Get a single batch from the data loader
    batch = next(iter(data_loader))

    # Move batch to the specified device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)

    # Calculate the memory of a single batch
    batch_memory = sum(
        tensor.numel() * tensor.element_size() for tensor in batch.values() if isinstance(tensor, torch.Tensor))
    batch_memory_MB = batch_memory / (1024 ** 2)  # Convert to MB

    # Calculate memory for prefetching extra batches
    prefetch_factor = data_loader.prefetch_factor
    num_prefetch_batches = prefetch_factor * data_loader.num_workers
    prefetch_memory_MB = num_prefetch_batches * batch_memory_MB

    total_data_memory_MB = batch_memory_MB + prefetch_memory_MB

    return batch_memory_MB, prefetch_memory_MB, total_data_memory_MB


def calculate_memory(
        nucleotide_embeddings, float_metric_embeddings, binary_metric_embeddings,
        readformer, classifier,
        batch_size, sequence_length, emb_dim,
        main_optimizer,
        device
):
    # Move models to device
    nucleotide_embeddings.to(device)
    float_metric_embeddings.to(device)
    binary_metric_embeddings.to(device)
    readformer.to(device)
    classifier.to(device)

    # Calculate the number of parameters
    num_params = (
            count_parameters(nucleotide_embeddings) +
            count_parameters(float_metric_embeddings) +
            count_parameters(binary_metric_embeddings) +
            count_parameters(readformer) +
            count_parameters(classifier)
    )
    param_memory = num_params * 4 / (1024 ** 2)  # Memory in MB

    # Estimate activations memory (including backward pass)
    activations_memory = (
            batch_size * sequence_length * emb_dim * 4 * 2 / (1024 ** 2)
    )  # *2 for forward and backward pass

    # Optimizer memory
    optimizer_memory = 0
    for param_group in main_optimizer.param_groups:
        for param in param_group['params']:
            optimizer_memory += param.numel() * 4 * 2 / (1024 ** 2)  # *2 for optimizer states

    total_memory = param_memory + activations_memory + optimizer_memory

    if device.type == 'cuda':
        # Measure GPU memory usage including backward pass
        torch.cuda.reset_peak_memory_stats(device)
        dummy_input = torch.randint(
            0, 15, (batch_size, sequence_length)
        ).to(device)
        dummy_positions = torch.randint(
            0, sequence_length, (batch_size, sequence_length)
        ).to(device)

        nucleotide_emb = nucleotide_embeddings(dummy_input)
        float_metrics = torch.zeros(
            (batch_size, sequence_length, 2), device=device
        )
        binary_metrics = torch.zeros(
            (batch_size, sequence_length, 14), device=device
        )

        float_metric_emb = float_metric_embeddings(float_metrics)
        binary_metric_emb = binary_metric_embeddings(binary_metrics)

        model_input = nucleotide_emb + float_metric_emb + binary_metric_emb
        output = readformer(model_input, dummy_positions)
        output = classifier(output)

        # Simulate backward pass
        output.sum().backward()

        # Get peak memory usage
        gpu_memory_usage = torch.cuda.max_memory_allocated(
            device) / (1024 ** 2)  # in MB

        total_memory += gpu_memory_usage

    return total_memory


def main():
    # Training parameters.
    batch_size = 512
    emb_dim = 512
    max_sequence_length = 16384

    num_layers = 12
    hyena = False
    if hyena:
        heads = 2
    else:
        heads = 16
    kernel_size = 13

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model components
    nucleotide_embeddings = components.read_embedding.NucleotideEmbeddingLayer(
        emb_dim
    ).apply(
        components.base_model.init_weights
    )
    float_metric_embeddings = components.read_embedding.MetricEmbedding(
        emb_dim // 2, name='float',
        num_metrics=2
    ).apply(
        components.base_model.init_weights
    )
    binary_metric_embeddings = components.read_embedding.MetricEmbedding(
        emb_dim // 2, name='binary',
        num_metrics=14
    ).apply(
        components.base_model.init_weights
    )

    readformer = components.base_model.Model(
        emb_dim=emb_dim, heads=heads, num_layers=num_layers, hyena=hyena,
        kernel_size=kernel_size
    ).apply(
        components.base_model.init_weights
    )
    classifier = components.classification_head.TransformerBinaryClassifier(
        embedding_dim=emb_dim,
        hidden_dim=emb_dim // 2,
        dropout_rate=0.1
    ).apply(
        components.base_model.init_weights
    )

    main_params = (
            list(nucleotide_embeddings.parameters()) +
            list(float_metric_embeddings.parameters()) +
            list(binary_metric_embeddings.parameters()) +
            list(readformer.parameters()) +
            list(classifier.parameters())
    )
    main_optimizer = torch.optim.Adam(main_params, lr=1e-3)

    total_memory = calculate_memory(
        nucleotide_embeddings, float_metric_embeddings,
        binary_metric_embeddings,
        readformer, classifier,
        batch_size, max_sequence_length,
        emb_dim,
        main_optimizer,
        device
    )

    print(f"Estimated Total Memory Requirement: {total_memory:.2f} MB")

    # Space taken up by a saved model
    num_params = sum(
        count_parameters(x) for x in
        [nucleotide_embeddings, float_metric_embeddings, binary_metric_embeddings, readformer, classifier]
    )
    size_in_bytes = num_params * 4  # 4 bytes for each float32 parameter
    # size_in_mb = size_in_bytes / (1024 ** 2)
    size_in_gb = size_in_bytes / (1024 ** 3)
    print(f"Total number of parameters: {num_params}")
    print(f"Saved model will take up: {size_in_gb:.2f} Gb")

    # Data loading parameters
    data_dir = '/lustre/scratch126/casm/team274sb/lp23/readformer/data/pretrain_symlinks'
    metadata_path = '/lustre/scratch126/casm/team274sb/lp23/readformer/data/pretrain_metadata.csv'
    # data_dir = 'GIAB_BAM/illumina_2x250bps'
    # metadata_path = 'GIAB_BAM/pretraining_metadata.csv'

    nucleotide_threshold = 16384  # 16384 = depth of 64x coverage
    min_quality = 25
    shuffle = True
    num_workers = 4
    prefetch_factor = 3

    # Create data loader
    data_loader = components.data_streaming.create_data_loader(
        data_dir, metadata_path, nucleotide_threshold, max_sequence_length,
        batch_size, min_quality, shuffle, num_workers, prefetch_factor
    )

    # Calculate data loader memory
    batch_memory_MB, prefetch_memory_MB, total_data_memory_MB = calculate_data_loader_memory(
        data_loader, device
    )
    print(f"Batch Memory Requirement: {batch_memory_MB:.2f} MB")
    print(f"Prefetch Memory Requirement: {prefetch_memory_MB:.2f} MB")
    print(f"Total Data Loader Memory Requirement: {total_data_memory_MB:.2f} MB")


if __name__ == "__main__":
    main()
