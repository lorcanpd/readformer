import torch
import argparse
import components.data_streaming
import components.base_model
import components.read_embedding
import components.classification_head
import components.pretrain_utils

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
        metrics_emb = torch.cat([float_metric_emb, binary_metric_emb], dim=-1)
        model_input = nucleotide_emb + metrics_emb
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
    parser = argparse.ArgumentParser(description="Profile memory requirements for model and data loader.")
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--emb_dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--max_sequence_length', type=int, default=8192, help='Maximum sequence length')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of layers in the model')
    parser.add_argument('--hyena', action='store_true', help='Use hyena model configuration')
    parser.add_argument('--heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--kernel_size', type=int, default=13, help='Kernel size for convolutional layers')
    parser.add_argument('--data_dir', type=str, default='GIAB_BAM/illumina_2x250bps', help='Directory for input data')
    parser.add_argument('--metadata_path', type=str, default='GIAB_BAM/pretraining_metadata.csv', help='Path to metadata file')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for data loading')
    parser.add_argument('--prefetch_factor', type=int, default=3, help='Prefetch factor for data loading')
    parser.add_argument('--min_quality', type=int, default=25, help='Minimum quality for data filtering')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle data during loading')

    args = parser.parse_args()

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model components
    nucleotide_embeddings = components.read_embedding.NucleotideEmbeddingLayer(
        args.emb_dim
    ).apply(
        components.base_model.init_weights
    )
    float_metric_embeddings = components.read_embedding.MetricEmbedding(
        args.emb_dim // 2, name='float',
        num_metrics=2
    ).apply(
        components.base_model.init_weights
    )
    binary_metric_embeddings = components.read_embedding.MetricEmbedding(
        args.emb_dim // 2, name='binary',
        num_metrics=14
    ).apply(
        components.base_model.init_weights
    )

    readformer = components.base_model.Model(
        emb_dim=args.emb_dim, heads=args.heads, num_layers=args.num_layers, hyena=args.hyena,
        kernel_size=args.kernel_size
    ).apply(
        components.base_model.init_weights
    )
    classifier = components.classification_head.TransformerBinaryClassifier(
        embedding_dim=args.emb_dim,
        hidden_dim=args.emb_dim // 2,
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
        args.batch_size, args.max_sequence_length,
        args.emb_dim,
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
    size_in_gb = size_in_bytes / (1024 ** 3)
    print(f"Total number of parameters: {num_params}")
    print(f"Saved model will take up: {size_in_gb:.2f} Gb")

    # Create data loader
    data_loader = components.data_streaming.create_data_loader(
        args.data_dir, args.metadata_path, args.max_sequence_length, args.max_sequence_length,
        args.batch_size, args.min_quality, args.shuffle, args.num_workers, args.prefetch_factor
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
