import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from components.pretrain_data_streaming import create_data_loader
from components.base_model import Model, init_weights
# from components.read_embedding import (
#     NucleotideEmbeddingLayer,
#     CigarEmbeddingLayer,
#     BaseQualityEmbeddingLayer,
#     StrandEmbeddingLayer,
#     MatePairEmbeddingLayer
# )
from components.read_embedding import InputEmbeddingLayer
from components.classification_head import MLMClassifier
from components.utils import (
    apply_masking_with_consistent_replacements,
    load_validation_tensors,
    get_layerwise_param_groups
)
from components.lamb import LAMB
from components.metrics import MLMLoss, mlm_accuracy, calculate_perplexity

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
    parser.add_argument('--emb_dim', type=int, default=256,
                        help='Embedding dimension.')
    parser.add_argument('--max_sequence_length', type=int,
                        default=1024,
                        help='Maximum sequence length.')
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
        '--num_hyena', type=int, default=1,
        help='Number of consecutive Hyena layers in each readformer block.'
    )
    parser.add_argument(
        '--max_iters', type=int, default=20000,
        help='Maximum number of iterations.'
    )
    parser.add_argument(
        '--num_attention', type=int, default=2,
        help='Number of attention layers in each readformer block.'
    )
    parser.add_argument(
        '--validation_dir', type=str, required=True,
        help='Directory containing saved validation tensors.'
    )
    parser.add_argument(
        '--adam', action='store_true',
        help='Use Adam optimizer instead of LAMB.'
    )
    parser.add_argument(
        '--project', type=str, default='readformer',
        help='Name of the wandb project.'
    )

    args = parser.parse_args()
    return args


def load_checkpoint(
        # model_dir, model_name,
        checkpoint_path,
        model, classifier, base_quality_classifier,
        cigar_classifier,
        input_embedding,
        optimiser
):
    # checkpoint_path = os.path.join(model_dir, model_name)

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        base_quality_classifier.load_state_dict(checkpoint['base_quality_classifier_state_dict'])
        cigar_classifier.load_state_dict(checkpoint['cigar_classifier_state_dict'])
        input_embedding.load_state_dict(checkpoint['input_embedding_state_dict'])
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        epoch = checkpoint['epoch']
        mean_loss = checkpoint['mean_loss']
        i = checkpoint.get('i', None)
        # j = checkpoint['j']
        run_id = checkpoint.get('wandb_run_id', None)
        logging.info(f"Loaded checkpoint '{checkpoint_path}' (epoch {epoch}, mean loss {mean_loss})")
        return epoch, mean_loss, i, run_id
    else:
        logging.error(f"No checkpoint found at '{checkpoint_path}'")
        return None, None, None, None


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
    iters_in_epoch = args.iters_in_epoch
    corruption_rate = args.corruption_rate
    proportion_random = args.proportion_random
    main_lr = args.main_lr
    readformer = args.readformer
    kernel_size = args.kernel_size
    wand_api_path = args.wandb_api_path
    num_hyena = args.num_hyena
    num_attention = args.num_attention
    checkpoint_path = (
        f"{args.model_dir}/{emb_dim}d_{num_layers}l_{num_hyena}h_"
        f"{num_attention}a_{num_heads}h.pth"
    )
    max_base_quality = 40

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
    logging.info(f"num_hyena_per_layer: {num_hyena}")
    logging.info(f"num_attention_per_layer: {num_attention}")
    logging.info(f"n_order: {n_order}")
    logging.info(f"min_read_quality: {min_read_quality}")
    logging.info(f"batch_size: {batch_size}")
    logging.info(f"emb_dim: {emb_dim}")
    logging.info(f"max_sequence_length: {max_sequence_length}")
    # logging.info(f"epochs_at_interval: {epochs_at_interval}")
    logging.info(f"iters_in_epoch: {iters_in_epoch}")
    logging.info(f"corruption_rate: {corruption_rate}")
    logging.info(f"proportion_random: {proportion_random}")
    logging.info(f"main_lr: {main_lr}")
    logging.info(f"readformer: {readformer}")
    if readformer:
        logging.info(f"kernel_size: {kernel_size}")
    # logging.info(f"corruption_scale: {args.corruption_scale}")
    logging.info(f"name: {args.name}")
    logging.info(f"model_dir: {args.model_dir}")
    logging.info(f"wandb project: {args.project}")

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

    mask_rate = 1.0 - 2 * proportion_random

    input_embedding = InputEmbeddingLayer(
        emb_dim, max_quality=max_base_quality
    ).to(device).train()

    readformer = Model(
        emb_dim=emb_dim, heads=num_heads, num_layers=num_layers,
        n_order=n_order,
        readformer=readformer, kernel_size=kernel_size,
        num_hyena=num_hyena, num_attention=num_attention
    ).apply(init_weights).to(device).train()
    # Don't train the self attention yet.
    # readformer.set_use_positionwise_self_attention(False)

    classifier = MLMClassifier(
        emb_dim=emb_dim, num_classes=input_embedding.nucleotide_embeddings.padding_idx
    ).apply(init_weights).to(device).train()

    base_quality_classifier = nn.Linear(
        emb_dim, max_base_quality + 1).to(device).train()

    cigar_classifier = nn.Linear(
        emb_dim, input_embedding.cigar_embeddings.num_embeddings
    ).to(device).train()

    min_lr = main_lr / 3
    param_groups = get_layerwise_param_groups(
        readformer, main_lr, min_lr
    )
    # Add the embedding layers to the parameter groups
    embedding_params = list(input_embedding.parameters())
    param_groups.append({
        "params": embedding_params,
        "lr": min_lr
    })

    # Add the classifier to the parameter groups
    classifier_params = (
            list(classifier.parameters()) +
            list(base_quality_classifier.parameters())
    )
    param_groups.append({
        "params": classifier_params,
        "lr": main_lr
    })
    max_lr_list = [group['lr'] for group in param_groups]

    if not args.adam:
        optimiser = LAMB(
            param_groups, eps=1e-9, weight_decay=0.05, adam=False,
            adaptive_noise=True, noise_std=0.1, use_curvature=True,
            # sharpness_aware=True, rho=0.03
        )
    else:
        optimiser = AdamW(
            param_groups, eps=1e-9, weight_decay=0.05
        )

    loss_fn = MLMLoss()
    metric_loss_fn = MLMLoss()
    cigar_loss_fn = MLMLoss()

    i = 0
    last_step = i
    epoch = 0
    epoch_losses = []
    best_mean_loss = float('inf')

    if args.load_latest_checkpoint:
        epoch, best_mean_loss, i, _ = load_checkpoint(
            checkpoint_path,
            readformer, classifier,
            base_quality_classifier,
            input_embedding,
            optimiser
        )
        if epoch is None:
            logging.info("No checkpoint found.")
            # Raise an error
            raise FileNotFoundError("No checkpoint found.")
        else:
            if i is not None:
                last_step = i
                i = i + 1
            else:
                i = (epoch + 1) * iters_in_epoch + 1
                last_step = i - 1
    else:
        logging.info("Training from scratch.")
        run_id = None

    if args.wandb:
        wandb.init(
            project=f"{args.project}",
            config={
                "layers": num_layers,
                "num_hyena_per_layer": num_hyena,
                "num_attention_per_layer": num_attention,
                "heads": num_heads,
                "n_order": n_order,
                "kernel_size": kernel_size,
                "batch_size": batch_size,
                "emb_dim": emb_dim,
                "max_sequence_length": max_sequence_length,
                "iters_in_epoch": iters_in_epoch,
                "min_read_quality": min_read_quality,
                "corruption_rate": corruption_rate,
                "proportion_random_replacement": proportion_random,
                "learning_rate_main": main_lr,
                "optimiser": "LAMB" if not args.adam else "Adam",
            },
            resume=False,
            # id=run_id
        )
        run_id = wandb.run.id
    if args.load_latest_checkpoint:
        scheduler = OneCycleLR(
            optimiser, max_lr=max_lr_list,
            total_steps=args.max_iters,
            pct_start=0.3, anneal_strategy='cos', cycle_momentum=False,
            last_epoch=last_step
        )
    else:
        scheduler = OneCycleLR(
            optimiser, max_lr=max_lr_list,
            total_steps=args.max_iters,
            pct_start=0.3, anneal_strategy='cos', cycle_momentum=False
        )

    data_loader = create_data_loader(
        file_paths=data_dir,
        metadata_path=metadata_path,
        nucleotide_threshold=max_sequence_length,
        max_sequence_length=max_sequence_length,
        batch_size=batch_size,
        min_quality=min_read_quality,
        base_quality_pad_idx=input_embedding.base_quality_embeddings.padding_idx,
        cigar_pad_idx=input_embedding.cigar_embeddings.padding_idx,
        position_pad_idx=-1,
        is_first_pad_idx=input_embedding.mate_pair_embeddings.padding_idx,
        mapped_to_reverse_pad_idx=input_embedding.strand_embeddings.padding_idx,
        shuffle=True,
        num_workers=get_allocated_cpus() - 1,
        # num_workers=0,
        prefetch_factor=2
    )
    logging.info(f"Data loader created")

    validation_batch = load_validation_tensors(args.validation_dir)

    validation_valid_mask = validation_batch['valid_positions'].to(device)
    validation_positions = validation_batch['positions'].to(device)
    validation_masked_sequences = validation_batch['masked_sequences'].to(device)
    validation_masked_indices = validation_batch['masked_indices'].to(device)
    validation_replaced_indices = validation_batch['replaced_indices'].to(device)
    validation_masked_cigar_encodings = validation_batch['masked_cigar_encodings'].to(device)
    validation_masked_base_qualities = validation_batch['masked_base_qualities'].to(device)
    validation_replaced_base_qualities = validation_batch['replaced_base_qual'].to(device)
    validation_replaced_cigar_encodings = validation_batch['replaced_cigar'].to(device)
    validation_masked_mapped_to_reverse = validation_batch['masked_mapped_to_reverse'].to(device)
    validation_masked_is_first = validation_batch['masked_is_first'].to(device)

    # ground truth
    validation_nucleotide_sequences = validation_batch['nucleotide_sequences'].to(device)
    validation_base_qualities = validation_batch['base_qualities'].clamp(0, max_base_quality).to(device)
    validation_cigar_encodings = validation_batch['cigar_encodings'].to(device)
    del validation_batch

    # Iterate through data
    for batch in data_loader:
        # Turn on the dropout layers
        input_embedding.train()
        readformer.train()
        classifier.train()
        base_quality_classifier.train()

        logging.debug(f"Processing batch {i}")

        nucleotide_sequences = batch['nucleotide_sequences']
        valid_mask = (
                nucleotide_sequences !=
                input_embedding.nucleotide_embeddings.padding_idx
        )
        base_qualities = batch['base_qualities']
        cigar_encodings = batch['cigar_encoding']
        positions = batch['positions']
        is_first = batch['is_first']
        mapped_reverse = batch['mapped_to_reverse']

        positions = positions.to(device)
        valid_mask = valid_mask.to(device)
        nucleotide_sequences = nucleotide_sequences.to(device)
        base_qualities = base_qualities.clamp(0, max_base_quality).to(device)
        cigar_encodings = cigar_encodings.to(device)
        is_first = is_first.to(device)
        mapped_reverse = mapped_reverse.to(device)

        with ((device_context(device))):
            (
                masked_sequences, masked_indices, replaced_indices
            ) = apply_masking_with_consistent_replacements(
                nucleotide_sequences, input_embedding.nucleotide_embeddings.mask_index,
                rate=corruption_rate, mask_rate=mask_rate,
                replace_rate=proportion_random,
                kernel_size=kernel_size, split=0.5
            )
            replaced_bases = apply_masking_with_consistent_replacements(
                nucleotide_sequences, input_embedding.nucleotide_embeddings.mask_index,
                rate=corruption_rate, mask_rate=mask_rate,
                replace_rate=proportion_random,
                kernel_size=kernel_size, split=0.5
            )[-1]
            replaced_cigar = apply_masking_with_consistent_replacements(
                nucleotide_sequences, input_embedding.nucleotide_embeddings.mask_index,
                rate=corruption_rate, mask_rate=mask_rate,
                replace_rate=proportion_random,
                kernel_size=kernel_size, split=0.5
            )[-1]

            # remove any overlap from replacement masks and the masked indices.
            replaced_bases[masked_indices] = False
            replaced_cigar[masked_indices] = False

            num_replaced = replaced_indices.sum().item()

            masked_cigar_encodings = cigar_encodings.clone().to(device)
            masked_cigar_encodings[masked_indices] = input_embedding.cigar_embeddings.mask_index
            masked_cigar_encodings[~valid_mask] = input_embedding.cigar_embeddings.padding_idx
            # replace the masked indices with num_replaced random indices
            num_replaced_cigar = replaced_cigar.sum().item()
            masked_cigar_encodings[replaced_cigar] = torch.randint(
                0, 4, (num_replaced_cigar,), dtype=torch.int32, device=device
            )

            masked_base_qualities = base_qualities.clone().to(device)
            masked_base_qualities[masked_indices] = input_embedding.base_quality_embeddings.mask_idx
            num_replaced_bases = replaced_bases.sum().item()
            masked_base_qualities[replaced_bases] = torch.randint(
                0, 50, (num_replaced_bases,), dtype=torch.int32, device=device
            )

            masked_is_first = is_first.clone().to(device)
            masked_is_first[masked_indices] = input_embedding.mate_pair_embeddings.mask_index
            masked_is_first[~valid_mask] = input_embedding.mate_pair_embeddings.padding_idx
            masked_mapped_reverse = mapped_reverse.clone().to(device)
            masked_mapped_reverse[masked_indices] = input_embedding.strand_embeddings.mask_index
            masked_mapped_reverse[~valid_mask] = input_embedding.strand_embeddings.padding_idx

            model_input = input_embedding(
                masked_sequences, masked_cigar_encodings,
                masked_base_qualities, masked_mapped_reverse,
                masked_is_first,
                to_be_masked=masked_indices,
                to_be_padded=~valid_mask
            )
            # Get the output from the model
            output = readformer(model_input, positions)
            base_quality_output = base_quality_classifier(
                output
            )
            cigar_output = cigar_classifier(
                output
            )
            output = classifier(output)

            masked_accuracy = mlm_accuracy(
                output[masked_indices], nucleotide_sequences[masked_indices]
            )
            replaced_accuracy = mlm_accuracy(
                output[replaced_indices], nucleotide_sequences[replaced_indices]
            )

            identity_loss = loss_fn(
                output[masked_indices | replaced_indices],
                nucleotide_sequences[masked_indices | replaced_indices],
                scale_factor=1
            )
            base_quality_loss = metric_loss_fn(
                base_quality_output[masked_indices | replaced_bases],
                base_qualities[masked_indices | replaced_bases],
                scale_factor=1
            )
            cigar_loss = cigar_loss_fn(
                cigar_output[masked_indices | replaced_cigar],
                cigar_encodings[masked_indices | replaced_cigar],
                scale_factor=1
            )

            train_perplexity = calculate_perplexity(
                output[masked_indices | replaced_indices],
                nucleotide_sequences[masked_indices | replaced_indices]
            )

            train_base_quality_perplexity = calculate_perplexity(
                base_quality_output[masked_indices | replaced_bases],
                base_qualities[masked_indices | replaced_bases]
            )

            train_cigar_perplexity = calculate_perplexity(
                cigar_output[masked_indices | replaced_cigar],
                cigar_encodings[masked_indices | replaced_cigar]
            )

            loss = identity_loss + base_quality_loss + cigar_loss
            optimiser.zero_grad()
            loss.backward()

            if (torch.cuda.is_available()
                    and args.logging.upper() == 'DEBUG'):
                torch.cuda.synchronize()

            optimiser.step()
            try:
                scheduler.step()
            except ValueError as e:
                if i >= args.max_iters:
                    pass
                else:
                    logging.error(f"Error in scheduler: {e}")

            if (torch.cuda.is_available()
                    and args.logging.upper() == 'DEBUG'):
                torch.cuda.synchronize()

            input_embedding.eval()
            readformer.eval()
            classifier.eval()
            base_quality_classifier.eval()

            # Validation forward pass.
            with torch.no_grad():
                val_model_input = input_embedding(
                    validation_masked_sequences,
                    validation_masked_cigar_encodings,
                    validation_masked_base_qualities,
                    validation_masked_mapped_to_reverse,
                    validation_masked_is_first,
                    to_be_masked=validation_masked_indices,
                    to_be_padded=~validation_valid_mask
                )
                # Forward pass
                val_output = readformer(val_model_input, validation_positions)

                val_pred_nucleotide = classifier(val_output)
                val_pred_base_quality = base_quality_classifier(
                    val_output
                )
                val_pred_cigar = cigar_classifier(
                    val_output
                )

                val_identity_loss = loss_fn(
                    val_pred_nucleotide[
                        validation_masked_indices | validation_replaced_indices],
                    validation_nucleotide_sequences[
                        validation_masked_indices | validation_replaced_indices],
                    scale_factor=1
                )
                val_base_quality_loss = metric_loss_fn(
                    val_pred_base_quality[
                        validation_masked_indices |
                        validation_replaced_base_qualities
                    ],
                    validation_base_qualities[
                        validation_masked_indices |
                        validation_replaced_base_qualities
                    ],
                    scale_factor=1
                )
                val_cigar_loss = cigar_loss_fn(
                    val_pred_cigar[
                        validation_masked_indices |
                        validation_replaced_cigar_encodings
                    ],
                    validation_cigar_encodings[
                        validation_masked_indices |
                        validation_replaced_cigar_encodings
                    ],
                    scale_factor=1
                )

                val_loss = val_identity_loss + val_base_quality_loss + val_cigar_loss

                # Compute validation statistics
                val_masked_accuracy = mlm_accuracy(
                    val_pred_nucleotide[
                        validation_masked_indices | validation_replaced_indices],
                    validation_nucleotide_sequences[
                        validation_masked_indices | validation_replaced_indices]
                )
                val_replaced_accuracy = mlm_accuracy(
                    val_pred_nucleotide[
                        validation_valid_mask & validation_replaced_indices],
                    validation_nucleotide_sequences[
                        validation_valid_mask & validation_replaced_indices]
                )

                val_perplexity = calculate_perplexity(
                    val_pred_nucleotide[
                        validation_masked_indices | validation_replaced_indices],
                    validation_nucleotide_sequences[
                        validation_masked_indices | validation_replaced_indices]
                )

                val_base_quality_perplexity = calculate_perplexity(
                    val_pred_base_quality[
                        validation_masked_indices |
                        validation_replaced_base_qualities
                    ],
                    validation_base_qualities[
                        validation_masked_indices |
                        validation_replaced_base_qualities
                    ]
                )

                val_cigar_perplexity = calculate_perplexity(
                    val_pred_cigar[
                        validation_masked_indices |
                        validation_replaced_cigar_encodings
                    ],
                    validation_cigar_encodings[
                        validation_masked_indices |
                        validation_replaced_cigar_encodings
                    ]
                )

        logging.debug(
            f"Train loss at iteration {i}: {loss.item():.5f}, "
            f"val loss: {val_loss.item():.5f}")
        # Learning rates for each group
        for num, group in enumerate(optimiser.param_groups):
            logging.debug(f"LR group {num}: {group['lr']}")
        logging.debug(
            f"Masked accuracy: {masked_accuracy:.5f}, "
            f"val masked accuracy: {val_masked_accuracy:.5f}")
        logging.debug(
            f"Replaced accuracy: {replaced_accuracy:.5f}, "
            f"val replaced accuracy: {val_replaced_accuracy:.5f}")
        logging.debug(
            f"Train perplexity: {train_perplexity:.5f}, "
            f"val perplexity: {val_perplexity:.5f}")
        logging.debug(
            f"Base quality loss: {base_quality_loss.item():.5f}, "
            f"val base quality loss: {val_base_quality_loss.item():.5f}")
        logging.debug(
            f"Train base quality perplexity: {train_base_quality_perplexity:.5f}, "
            f"val base quality perplexity: {val_base_quality_perplexity:.5f}")

        if args.wandb:
            wandb.log(
                {
                    "loss": loss.item(),
                    "masked_accuracy": masked_accuracy,
                    "replaced_accuracy": replaced_accuracy,
                    "train_perplexity": train_perplexity,
                    "base_quality_loss": base_quality_loss.item(),
                    "cigar_loss": cigar_loss.item(),
                    "val_loss": val_loss.item(),
                    "val_masked_accuracy": val_masked_accuracy,
                    "val_replaced_accuracy": val_replaced_accuracy,
                    "val_perplexity": val_perplexity,
                    "val_base_quality_loss": val_base_quality_loss.item(),
                    "val_cigar_loss": val_cigar_loss.item(),
                    "train_base_quality_perplexity": train_base_quality_perplexity,
                    "val_base_quality_perplexity": val_base_quality_perplexity,
                    "train_cigar_perplexity": train_cigar_perplexity,
                    "val_cigar_perplexity": val_cigar_perplexity
                },
                step=i
            )
            for num, group in enumerate(optimiser.param_groups):
                wandb.log({f'LR_group_{num}': group['lr']}, step=i)

        epoch_losses.append(loss.item())

        if i % iters_in_epoch == 0 or i == args.max_iters:
            mean_loss = sum(epoch_losses) / len(epoch_losses)

            if args.wandb:
                wandb.log(
                    {
                        "mean_loss": mean_loss,
                        "epoch": epoch
                    }
                )

            if i > 0:
                update = {
                    'epoch': epoch,
                    'model_state_dict': readformer.state_dict(),
                    'classifier_state_dict': classifier.state_dict(),
                    'base_quality_classifier_state_dict':
                        base_quality_classifier.state_dict(),
                    'cigar_classifier_state_dict':
                        cigar_classifier.state_dict(),
                    'input_embedding_state_dict':
                        input_embedding.state_dict(),
                    'optimiser_state_dict': optimiser.state_dict(),
                    'mean_loss': mean_loss,
                    'i': i,
                }
                if args.wandb:
                    update['wandb_run_id'] = run_id
                torch.save(update, checkpoint_path)
                if args.wandb:
                    wandb.save(checkpoint_path)

            logging.info(
                f"Epoch {epoch}: , "
                f"Mean Epoch Loss: {mean_loss} "
            )

            epoch += 1
            epoch_losses = []

        if i >= args.max_iters:
            break

        i += 1

    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
