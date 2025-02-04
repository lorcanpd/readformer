import torch
import numpy as np
# import torch.nn as nn
from components.data_streaming import create_data_loader
from components.base_model import Model, init_weights
from components.read_embedding import (
    NucleotideEmbeddingLayer, MetricEmbedding)
from components.classification_head import (
    # TransformerBinaryClassifier,
    MLMClassifier, MLMLoss
)
from components.utils import (
    # replacement_loss,
    # get_replacement_mask,
    # adversarial_loss,
    apply_masking_with_consistent_replacements,
    create_intervals,
    get_random_alternative_labels,
    # create_corruption_rates,
    # WarmupConstantScheduler,
    mlm_accuracy
    # get_weights, check_weights
)

import wandb
import argparse
import os
import sys
from contextlib import contextmanager
import multiprocessing as mp
import logging


def get_allocated_cpus():
    cpus = int(os.getenv('LSB_DJOB_NUMPROC', '1'))
    logging.info(f"Allocated CPUs: {cpus}")
    return cpus


def check_cuda_availability():
    if not torch.cuda.is_available():
        logging.info("CUDA is not available.")
        return False

    num_devices = torch.cuda.device_count()
    logging.info(f"Number of CUDA devices available: {num_devices}")

    for device_id in range(num_devices):
        device = torch.device(f"cuda:{device_id}")
        properties = torch.cuda.get_device_properties(device)
        logging.info(f"Device {device_id}: {properties.name}")
        logging.info(f"  Total memory: {properties.total_memory / 1e9} GB")
        logging.info(f"  Multiprocessors: {properties.multi_processor_count}")
        logging.info(f"  Compute Capability: {properties.major}.{properties.minor}")

        # Try to allocate a small tensor on the device to check if it is free
        try:
            torch.tensor([1.0], device=device)
            logging.info(f"Device {device_id} is available and functional.")
        except RuntimeError as e:
            logging.error(f"Device {device_id} is not available: {e}")
            return False

    return True


def get_args():
    parser = argparse.ArgumentParser(
        description="Set parameters for the model and data loading."
    )

    # Adding arguments
    parser.add_argument(
        '--metadata_path', type=str,
        default='/lustre/scratch126/casm/team274sb/lp23/readformer/data/pretrain_symlinks',
        help='Path to the metadata file.'
    )
    parser.add_argument(
        '--data_dir', type=str,
        default='/lustre/scratch126/casm/team274sb/lp23/readformer/data/pretrain_metadata.csv',
        help='Directory containing the data.'
    )
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads.')
    parser.add_argument(
        '--n_order', type=int, default=4,
        help='Number of times hyena convolutions are applied in layers.'
    )
    parser.add_argument('--num_layers', type=int, default=12,
                        help='Number of layers in the model.')
    parser.add_argument('--min_read_quality', type=int, default=30,
                        help='Minimum read quality.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--emb_dim', type=int, default=1024,
                        help='Embedding dimension.')
    parser.add_argument('--max_sequence_length', type=int,
                        default=1024,
                        help='Maximum sequence length.')
    parser.add_argument('--l1_lambda', type=float, default=0,
                        help='L1 regularization lambda.')
    parser.add_argument('--warm_up_epochs', type=int, default=5,
                        help='Number of warm-up epochs.')
    parser.add_argument('--epochs_at_interval', type=int,
                        default=2, help='Number of epochs at interval.')
    parser.add_argument('--iters_in_epoch', type=int, default=50,
                        help='Number of iterations in an epoch.')
    parser.add_argument('--corruption_rate', type=float,
                        default=0.15,
                        help='Rate at which bases are selected for masking/replacement.')
    parser.add_argument(
        '--proportion_random', type=float, default=0.1,
        help='Proportion of corrupted labels to be assigned random nucleotides.'
    )
    parser.add_argument('--main_lr', type=float, default=1e-3,
                        help='Learning rate for the main optimizer.')
    parser.add_argument('--wandb', action='store_true',
                        help='Whether to use wandb for logging.')
    parser.add_argument(
        '--readformer', action='store_true',
        help='Use readformer model configuration'
    )
    parser.add_argument('--kernel_size', type=int, default=15,
                        help='Kernel size for the Hyena block.')
    # parser.add_argument(
    #     '--corruption_scale', type=float, default=0.9,
    #     help='Scale for corruption rates.'
    # )
    parser.add_argument('--name', type=str, default='readformer',
                        help='Name with which to save the model.')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save the model.')
    parser.add_argument(
        '--load_latest_checkpoint', type=bool, default=False,
        help='Whether to load the latest checkpoint.'
    )
    parser.add_argument(
        '--wandb_api_path', type=str, default='.wandb_api',
        help='Path to the wandb api key file.'
    )
    parser.add_argument(
        '--logging', type=str, default='INFO',
        help='Logging level.'
    )
    parser.add_argument(
        '--profiling', action='store_true',
        help='Enable profiling.'
    )
    parser.add_argument(
        '--mixing_alpha', type=float, default=0.2,
        help='Alpha parameter for sequence label mixing.'
    )

    args = parser.parse_args()
    return args


def load_checkpoint(
        model_dir, model_name, model, classifier, nucleotide_embeddings,
        # float_metric_embeddings, binary_metric_embeddings,
        metric_embeddings,
        optimiser
):
    checkpoint_path = os.path.join(model_dir, model_name)
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        nucleotide_embeddings.load_state_dict(checkpoint['nucleotide_embeddings_state_dict'])
        metric_embeddings.load_state_dict(checkpoint['metric_embeddings_state_dict'])
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        mean_loss = checkpoint['mean_loss']
        i = checkpoint['i']
        j = checkpoint['j']
        logging.info(f"Loaded checkpoint '{checkpoint_path}' (epoch {epoch}, mean loss {mean_loss})")
        return epoch, mean_loss, i, j
    else:
        logging.error(f"No checkpoint found at '{checkpoint_path}'")
        return None, None


def batches_are_identical(batch1, batch2):
    if batch1 is None or batch2 is None:
        return False
    if len(batch1) != len(batch2):
        return False
    for key in batch1:
        if not torch.equal(batch1[key], batch2[key]):
            return False
    return True

@contextmanager
def device_context(device):
    if device.type == 'cuda':
        with torch.cuda.device(device):
            yield
    else:
        yield


@contextmanager
def conditional_profiler(condition, use_cuda=True):
    """
    A context manager that profiles the code block if the condition is met.

    :param condition:
        A condition to determine whether to profile the code block.
    :param use_cuda:
        Whether to use CUDA for profiling.
    """
    if condition:
        with torch.autograd.profiler.profile(use_cuda=use_cuda) as prof:
            yield prof
    else:
        yield None


def main():

    args = get_args()

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    metadata_path = args.metadata_path
    data_dir = args.data_dir
    num_heads = args.num_heads
    num_layers = args.num_layers
    n_order = args.n_order
    min_read_quality = args.min_read_quality
    batch_size = args.batch_size
    emb_dim = args.emb_dim
    max_sequence_length = args.max_sequence_length
    l1_lambda = args.l1_lambda
    warm_up_epochs = args.warm_up_epochs
    epochs_at_interval = args.epochs_at_interval
    iters_in_epoch = args.iters_in_epoch
    corruption_rate = args.corruption_rate
    proportion_random = args.proportion_random
    main_lr = args.main_lr
    readformer = args.readformer
    kernel_size = args.kernel_size
    checkpoint_path = f"{args.model_dir}/{args.name}_lay{num_layers}_head{num_heads}_ord{n_order}_latest.pth"
    wand_api_path = args.wandb_api_path
    profiling = args.profiling
    mixing_alpha = args.mixing_alpha

    # Map the string logging level to the actual logging level
    numeric_level = getattr(logging, args.logging.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.logging}')

    # Create handlers
    stdout_handler = logging.StreamHandler(sys.stdout)
    stderr_handler = logging.StreamHandler(sys.stderr)

    # Set levels for handlers
    stdout_handler.setLevel(numeric_level)
    stderr_handler.setLevel(logging.ERROR)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stdout_handler.setFormatter(formatter)
    stderr_handler.setFormatter(formatter)

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    # Add handlers to the logger
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)

    if not check_cuda_availability() and not torch.backends.mps.is_available():
        sys.exit(1)
    else:
        mp.set_start_method('spawn', force=True)

    # Print values to verify
    logging.info(f"metadata_path: {metadata_path}")
    logging.info(f"data_dir: {data_dir}")
    logging.info(f"num_heads: {num_heads}")
    logging.info(f"num_layers: {num_layers}")
    logging.info(f"n_order: {n_order}")
    logging.info(f"min_read_quality: {min_read_quality}")
    logging.info(f"batch_size: {batch_size}")
    logging.info(f"emb_dim: {emb_dim}")
    logging.info(f"max_sequence_length: {max_sequence_length}")
    logging.info(f"l1_lambda: {l1_lambda}")
    logging.info(f"warm_up_epochs: {warm_up_epochs}")
    logging.info(f"epochs_at_interval: {epochs_at_interval}")
    logging.info(f"iters_in_epoch: {iters_in_epoch}")
    logging.info(f"corruption_rate: {corruption_rate}")
    logging.info(f"proportion_random: {proportion_random}")
    logging.info(f"main_lr: {main_lr}")
    logging.info(f"readformer: {readformer}")
    if readformer:
        logging.info(f"kernel_size: {kernel_size}")
    # logging.info(f"corruption_scale: {args.corruption_scale}")
    logging.info(f"name: {args.name}")
    logging.info(f"mixing_alpha: {mixing_alpha}")

    if args.wandb:
        # load api key from file
        with open(wand_api_path) as f:
            api_key = f.read().strip()
        os.environ["WANDB_API_KEY"] = api_key
        wandb.login(key=api_key)
        logging.info("Logged in to wandb.")

    device = (
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    logging.info(f"Using device: {device}")

    # Enable anomaly detection
    # torch.autograd.set_detect_anomaly(True)

    if args.wandb:
        wandb.init(project=f"mlm-pretraining-{args.name}", config={
            "layers": num_layers,
            "heads": num_heads,
            "n_order": n_order,
            "kernel_size": kernel_size,
            "batch_size": batch_size,
            "emb_dim": emb_dim,
            "max_sequence_length": max_sequence_length,
            "l1_lambda": l1_lambda,
            "epochs_at_interval": epochs_at_interval,
            "iters_in_epoch": iters_in_epoch,
            "warm_up_epochs": warm_up_epochs,
            "min_read_quality": min_read_quality,
            "corruption_rate": corruption_rate,
            "proportion_random_replacement": proportion_random,
            "learning_rate_main": main_lr,
            "mixing_alpha": mixing_alpha,
        })

    mask_rate = 1.0 - 2 * proportion_random
    nucleotide_embeddings = NucleotideEmbeddingLayer(
        emb_dim // 2, mlm_mode=True
    ).apply(init_weights).to(device)
    metric_embeddings = MetricEmbedding(
        emb_dim // 2,
        name='metric', num_metrics=16
    ).apply(init_weights).to(device)
    readformer = Model(
        emb_dim=emb_dim, heads=num_heads, num_layers=num_layers,
        n_order=n_order,
        readformer=readformer, kernel_size=kernel_size
    ).apply(init_weights).train().to(device).train()

    # for layer in readformer.layers:
    #     layer.self_attention.init_scaling_vectors()
    #     layer.self_attention.freeze_scaling_vectors()
    #     layer.feed_forward.init_scaling_vector()
    #     layer.feed_forward.freeze_scaling_vector()

    classifier = MLMClassifier(
        emb_dim=emb_dim, num_classes=nucleotide_embeddings.padding_idx
    ).apply(init_weights).to(device)

    params = (
            list(nucleotide_embeddings.parameters()) +
            list(metric_embeddings.parameters()) +
            list(readformer.parameters()) +
            list(classifier.parameters())
    )
    optimiser = torch.optim.Adam(
        params, lr=main_lr,
    )

    loss_fn = MLMLoss()

    # Get nucleotide intervals up to the nucleotide threshold
    intervals = create_intervals(max_sequence_length, 256)
    # if corruption_rate == "variable":
    #     corruption_rates = create_corruption_rates(
    #         intervals, min_rate=0.15, read_length=250,
    #         scale=args.corruption_scale
    #     )
    # else:
    #     corruption_rates = [0.2] * len(intervals)

    i = 0
    j = 0
    epoch = 0
    epoch_losses = []
    best_mean_loss = float('inf')

    if args.load_latest_checkpoint:
        epoch, best_mean_loss, i, j = load_checkpoint(
            args.model_dir, args.name, readformer, classifier,
            nucleotide_embeddings,
            metric_embeddings,
            optimiser
        )
        if epoch is None:
            logging.info("No checkpoint found. Training from scratch.")
            epoch = 0

    logging.info(f"Number of intervals: {len(intervals)}")

    for interval in intervals:
        logging.info(f"Training for interval {interval}")
        data_loader = create_data_loader(
            file_paths=data_dir,
            metadata_path=metadata_path,
            nucleotide_threshold=interval,
            max_sequence_length=interval,
            batch_size=batch_size,
            min_quality=min_read_quality,
            shuffle=True,
            num_workers=get_allocated_cpus()-1,
            prefetch_factor=2
        )
        logging.info(f"Data loader created for interval {interval}")

        # Iterate through data
        for batch in data_loader:

            logging.debug(f"Processing batch {i} of data loader {j}")

            # with device_context(device):
            nucleotide_sequences = batch['nucleotide_sequences']#.to(device)
            valid_mask = (
                    nucleotide_sequences !=
                    nucleotide_embeddings.padding_idx
            )
            base_qualities = batch['base_qualities']#.to(device)
            read_qualities = batch['read_qualities']#.to(device)
            cigar_match = batch['cigar_match']#.to(device)
            cigar_insertion = batch['cigar_insertion']#.to(device)
            bitwise_flags = batch['bitwise_flags']#.to(device)
            positions = batch['positions']#.to(device)

            metrics = torch.cat(
                [
                    base_qualities.unsqueeze(-1),
                    read_qualities.unsqueeze(-1),
                    cigar_match.unsqueeze(-1),
                    cigar_insertion.unsqueeze(-1),
                    bitwise_flags
                ],
                dim=-1
            )

            mask_token_index = nucleotide_embeddings.mask_index
            positions = positions.to(device)
            valid_mask = valid_mask.to(device)
            nucleotide_sequences = nucleotide_sequences.to(device)
            with (device_context(device)):
                (
                    masked_sequences, masked_indices, replaced_indices,
                    kept_indices  # Selected for corruption but not altered.
                ) = apply_masking_with_consistent_replacements(
                    positions, nucleotide_sequences, mask_token_index,
                    rate=corruption_rate, mask_rate=mask_rate,
                    kernel_size=kernel_size, keep_rate=proportion_random,
                    random_replace_rate=proportion_random
                )
                alt_labels = get_random_alternative_labels(
                    masked_sequences[
                        masked_indices | replaced_indices | kept_indices
                    ]
                )
                lambdas = torch.from_numpy(
                    np.random.beta(mixing_alpha, mixing_alpha, size=alt_labels.size(-1))
                ).to(device=device, dtype=torch.float32).detach()
                # Label mixing to be carried out on the indices contained in
                # replaced_indices, masked_indices and kept_indices.
                masked_nucleotide_emb = nucleotide_embeddings(masked_sequences)
                # Apply the label mixing to the masked nucleotide embeddings.
                mixed_embeddings = lambdas.unsqueeze(-1) * masked_nucleotide_emb[
                    masked_indices | replaced_indices | kept_indices] + (
                        1 - lambdas.unsqueeze(-1)) * nucleotide_embeddings(alt_labels)
                masked_nucleotide_emb[
                    masked_indices | replaced_indices | kept_indices
                ] = mixed_embeddings
                # Expand lambdas over the whole input tensor for calculating the loss.
                expanded_lambdas = torch.zeros_like(positions, dtype=torch.float32)
                expanded_lambdas[
                    masked_indices | replaced_indices | kept_indices
                ] = lambdas
                metric_emb = metric_embeddings(metrics.to(device))
                metric_emb = metric_emb * (
                        1 - expanded_lambdas.unsqueeze(-1) *
                        masked_indices.unsqueeze(-1).float()
                )
                model_input = torch.cat(
                    [
                        masked_nucleotide_emb,
                        metric_emb
                    ], dim=-1
                )

                # Get the output from the model
                # Profile every 10th batch
                profile_batch = (i % 10 == 0) and profiling
                with conditional_profiler(
                        profile_batch, use_cuda=torch.cuda.is_available()
                ) as prof:
                    output = readformer(model_input, positions)
                    output = classifier(output)
                    batch_accuracy = mlm_accuracy(output, nucleotide_sequences)
                    # Main model loss and optimisation
                    # Need to adapt to incorporate the label mixing.
                    unchanged_loss = loss_fn(
                        output[valid_mask & ~masked_indices & ~replaced_indices],
                        nucleotide_sequences[valid_mask & ~masked_indices & ~replaced_indices],
                        scale_factor=0.1
                    )
                    replaced_loss = loss_fn(
                        output[valid_mask & replaced_indices],
                        nucleotide_sequences[valid_mask & replaced_indices],
                        scale_factor=1.0
                    )
                    masked_loss = loss_fn(
                        output[valid_mask & masked_indices],
                        nucleotide_sequences[valid_mask & masked_indices],
                        scale_factor=1.0
                    )
                    loss = unchanged_loss + replaced_loss + masked_loss
                    optimiser.zero_grad()
                    loss.backward()

                    if (torch.cuda.is_available()
                            and args.logging.upper() == 'DEBUG'):
                        torch.cuda.synchronize()

                    torch.nn.utils.clip_grad_norm_(
                        params, max_norm=1
                    )
                    optimiser.step()

                    if (torch.cuda.is_available()
                            and args.logging.upper() == 'DEBUG'):
                        torch.cuda.synchronize()

            if profile_batch:
                if torch.cuda.is_available():
                    profile_data = prof.key_averages().table(
                        sort_by="cuda_time_total"
                    )
                else:
                    profile_data = prof.key_averages().table(
                        sort_by="cpu_time_total"
                    )

            logging.debug(f"Loss at iteration {i}: {loss.item()}")
            logging.debug(f"Batch accuracy: {batch_accuracy}")

            if args.wandb:
                wandb.log(
                    {
                        "loss": loss.item(),
                        "batch_accuracy": batch_accuracy,
                        # "lr": scheduler.get_last_lr()[0],
                        "interval": intervals[j]
                    }
                )
                if profile_batch:
                    wandb.log({"profile_data": wandb.Html(profile_data)})

            epoch_losses.append(loss.item())
            # scheduler.step()

            i += 1

            if i % iters_in_epoch == 0:
                mean_loss = sum(epoch_losses) / len(epoch_losses)

                if args.wandb:
                    wandb.log(
                        {
                            "mean_loss": mean_loss,
                            "epoch": epoch
                        }
                    )

                if i > 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': readformer.state_dict(),
                        'classifier_state_dict': classifier.state_dict(),
                        'nucleotide_embeddings_state_dict':
                            nucleotide_embeddings.state_dict(),
                        'metric_embeddings_state_dict':
                            metric_embeddings.state_dict(),
                        # 'float_metric_embeddings_state_dict':
                        #     float_metric_embeddings.state_dict(),
                        # 'binary_metric_embeddings_state_dict':
                        #     binary_metric_embeddings.state_dict(),
                        'optimiser_state_dict': optimiser.state_dict(),
                        'mean_loss': mean_loss,
                        'i': i,
                        'j': j
                    }, checkpoint_path)
                    if args.wandb:
                        wandb.save(checkpoint_path)

                logging.info(
                    f"Epoch {epoch}: , "
                    f"Mean Epoch Loss: {mean_loss} "
                )

                epoch += 1
                epoch_losses = []

                if j < len(intervals) - 1 and epoch % epochs_at_interval == 0 and epoch > 0:
                    j += 1
                    break

    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
