import argparse
import torch
from torch.optim import AdamW
import os
import sys

import logging
import multiprocessing as mp

# Import the necessary modules from your components
from components.base_model import Model
from components.read_embedding import (
    InputEmbeddingLayer, NucleotideEmbeddingLayer)
from components.finetune_data_streaming import create_finetuning_dataloader
from components.classification_head import BetaDistributionClassifier
# from components.utils import get_effective_number, get_layerwise_param_groups
from components.metrics import ValidationWriter
# from components.empirical_bayes import EmpiricalBayes
from pretrain_readwise_only import device_context, check_cuda_availability


def get_args():
    parser = argparse.ArgumentParser(
        description="Load a pre-trained model for fine-tuning."
    )
    parser.add_argument(
        '--name', type=str, required=True,
        help='Name of the model'
    )

    parser.add_argument(
        '--project', type=str, default='readformer_finetuning',
        help='Name of the project for Weights & Biases.'
    )

    # Model parameters (should match the pre-trained model)
    parser.add_argument('--emb_dim', type=int, default=1024,
                        help='Embedding dimension.')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads.')
    parser.add_argument('--num_layers', type=int, default=12,
                        help='Number of layers in the model.')
    parser.add_argument('--n_order', type=int, default=4,
                        help='Order of Hyena convolutions.')
    parser.add_argument('--kernel_size', type=int, default=15,
                        help='Kernel size for the Hyena block.')
    parser.add_argument('--num_hyena', type=int, default=1,
                        help='Number of consecutive Hyena layers in each block.')
    parser.add_argument('--num_attention', type=int, default=2,
                        help='Number of attention layers in each block.')
    parser.add_argument(
        '--readformer', action='store_true',
        help='Use readformer model configuration.'
    )

    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size for fine-tuning.'
    )

    parser.add_argument(
        '--finetune_save_dir', type=str,
        help='Directory in which to save the fine-tuned model.',
        required=True
    )
    parser.add_argument(
        '--finetune_metadata_dir', type=str,
        help='Path to save the fine-tuning metadata.',
        required=True
    )
    parser.add_argument(
        '--mutation_bam_path', type=str,
        help='Path to the BAM file containing the mutation reads.',
        required=True
    )
    parser.add_argument(
        '--artefact_bam_path', type=str,
        help='Path to the BAM file containing the artefact reads.',
        required=True
    )
    parser.add_argument(
        '--fold', type=int, default=0,
        help='Fold number for cross-validation.'
    )
    parser.add_argument(
        '--validation_output_dir', type=str,
        help='Directory to save validation tensors.', required=True
    )
    parser.add_argument(
        '--max_read_length', type=int, default=151,
        help='Maximum read length to consider.'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable debug logging.'
    )

    parser.add_argument(
        '--max_base_quality', type=int, default=50,
        help='Maximum base quality value.'
    )

    parser.add_argument(
        '--no_reference', action='store_true',
        help='If set, do not use reference base embedding.'
    )

    args = parser.parse_args()
    return args


# TODO: Align this script with predict. A lot of redundancy.

def instantiate_model(args, device):
    input_embedding = InputEmbeddingLayer(
        args.emb_dim, args.max_base_quality
    ).to(device)

    readformer_model = Model(
        emb_dim=args.emb_dim, heads=args.num_heads, num_layers=args.num_layers,
        n_order=args.n_order,
        readformer=args.readformer, kernel_size=args.kernel_size,
        num_hyena=args.num_hyena, num_attention=args.num_attention
    ).to(device)

    return (
        input_embedding, readformer_model
    )


def load_pretrained_model(args, device):
    # Instantiate the models (nucleotide_embeddings, metric_embeddings, readformer)

    (
        input_embedding, readformer_model
    ) = instantiate_model(args, device)

    # Load the checkpoint
    if not os.path.isfile(args.pre_trained_path):
        logging.error(f"No checkpoint found at '{args.pre_trained_path}'")
        sys.exit(1)
    checkpoint = torch.load(args.pre_trained_path, map_location=device)

    # Load state_dicts
    input_embedding.load_state_dict(checkpoint['input_embedding_state_dict'])
    readformer_model.load_state_dict(checkpoint['model_state_dict'])

    logging.info(f"Loaded pre-trained model from '{args.pre_trained_path}'")

    return (
        input_embedding, readformer_model
    )


def load_latest_checkpoint(args, device):
    # Load the latest checkpoint from the finetune save directory
    if not os.path.isdir(args.finetune_save_dir):
        # TODO: add phase number to the save name
        logging.error(f"No directory found at '{args.finetune_save_dir}'")
        sys.exit(1)

    # models are saved in the format phase_{phase_index:03}.pth. Sort the files
    # by phase index and get the latest one.
    checkpoints = sorted(
        [
            f for f in os.listdir(args.finetune_save_dir)
            if f.endswith('.pth')
        ],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )

    latest_checkpoint = os.path.join(args.finetune_save_dir, checkpoints[-1])

    if not os.path.isfile(latest_checkpoint):
        logging.error(f"No checkpoint found at '{latest_checkpoint}'")
        sys.exit(1)

    checkpoint = torch.load(latest_checkpoint, map_location=device)

    # Load the models
    (
        input_embedding, readformer_model
    ) = instantiate_model(args, device)

    if not args.no_reference:
        ref_base_embedding = NucleotideEmbeddingLayer(
            args.emb_dim, mlm_mode=True
        ).to(device)
        ref_base_embedding.load_state_dict(checkpoint['ref_base_embedding_state_dict'])
    else:
        ref_base_embedding = None

    classifier = BetaDistributionClassifier(
        input_dim=args.emb_dim, hidden_dim=args.emb_dim // 2,
        using_reference_embedding=not args.no_reference
    ).to(device)

    # Load state_dicts
    input_embedding.load_state_dict(checkpoint['input_embedding_state_dict'])
    readformer_model.load_state_dict(checkpoint['model_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])

    if not args.no_reference:
        # Load the optimiser state
        optimiser = AdamW(
            list(input_embedding.parameters())
            + list(readformer_model.parameters())
            + list(ref_base_embedding.parameters())
            + list(classifier.parameters())
        )
    else:
        # Load the optimiser state
        optimiser = AdamW(
            list(input_embedding.parameters())
            + list(readformer_model.parameters())
            + list(classifier.parameters())
        )
    optimiser.load_state_dict(checkpoint['optimiser_state_dict'])

    # Get the epoch and iteration to be used to continue training and
    # parameterise the scheduler
    epoch = checkpoint['epoch']
    i = checkpoint['iteration']

    logging.info(f"Loaded latest checkpoint from '{latest_checkpoint}'")

    return (
        input_embedding, readformer_model,
        ref_base_embedding, classifier, optimiser, epoch, i
    )


# TODO: implement this in the code below to make ready for the HPC.
def get_allocated_cpus():
    cpus = int(os.getenv('LSB_DJOB_NUMPROC', '1'))
    logging.info(f"Allocated CPUs: {cpus}")
    return cpus


def unfreeze_layers_by_epoch(param_groups, epoch, ignore_groups=[]):
    """
    Unfreeze layers progressively based on the epoch number.

    :param param_groups:
        A list of parameter groups ordered from top-most layer to bottom-most.
    :param epoch:
        Current epoch number (1-based). At epoch=1, top-most layer is unfrozen.
        At epoch=2, top two layers are unfrozen, and so forth.
    :param ignore_groups:
        A list of indices of groups to ignore and keep unfrozen.
    """
    for i, group in enumerate(param_groups):
        # Layers with index < epoch are unfrozen, others remain frozen
        requires_grad = (i < epoch) | (i in ignore_groups)
        for p in group['params']:
            p.requires_grad = requires_grad


def load_checkpoint_by_phase(args, device, phase):
    """
    Load a specific checkpoint based on the provided phase number.

    Args:
        args (Namespace):
            Parsed command-line arguments containing configurations.
        device (torch.device):
            The device to map the model tensors.
        phase (int):
            The specific phase number to load the checkpoint for validation.

    Returns:
        tuple: A tuple containing the instantiated and loaded model components:
            (input_embedding, readformer_model, ref_base_embedding, classifier)
    """
    # Verify that the finetune_save_dir exists
    if not os.path.isdir(args.finetune_save_dir):
        logging.error(f"No directory found at '{args.finetune_save_dir}'")
        sys.exit(1)

    # Construct the expected checkpoint filename based on the phase
    filename = f"phase_{phase:03d}.pth"
    checkpoint_path = os.path.join(args.finetune_save_dir, filename)

    # Check if the specific checkpoint file exists
    if not os.path.isfile(checkpoint_path):
        logging.error(f"No checkpoint found for phase {phase} at '{checkpoint_path}'")
        sys.exit(1)

    # Load the checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        logging.error(f"Failed to load checkpoint '{checkpoint_path}': {e}")
        sys.exit(1)

    # Instantiate the model components
    try:
        input_embedding, readformer_model = instantiate_model(args, device)
        if not args.no_reference:
            ref_base_embedding = NucleotideEmbeddingLayer(
                args.emb_dim, mlm_mode=True
            ).to(device)
        else:
            ref_base_embedding = None
        classifier = BetaDistributionClassifier(
            input_dim=args.emb_dim, hidden_dim=args.emb_dim // 2,
            using_reference_embedding=not args.no_reference,
        ).to(device)
    except Exception as e:
        logging.error(f"Failed to instantiate model components: {e}")
        sys.exit(1)

    # Load state dictionaries into the models
    try:
        input_embedding.load_state_dict(checkpoint['input_embedding_state_dict'])
        readformer_model.load_state_dict(checkpoint['model_state_dict'])
        if not args.no_reference:
            ref_base_embedding.load_state_dict(checkpoint['ref_base_embedding_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
    except KeyError as e:
        logging.error(f"Missing key in checkpoint '{checkpoint_path}': {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading state_dicts: {e}")
        sys.exit(1)

    logging.info(f"Successfully loaded checkpoint for phase {phase} from '{checkpoint_path}'")

    return input_embedding, readformer_model, ref_base_embedding, classifier


def main():
    args = get_args()

    logging.basicConfig(
        level=logging.INFO if args.debug is not True else logging.DEBUG,
        format='%(levelname)s: %(message)s'
    )

    if not check_cuda_availability() and not torch.backends.mps.is_available():
        sys.exit(1)
    else:
        mp.set_start_method('spawn', force=True)

    # Set device
    device = (
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    logging.info(f"Using device: {device}")

    # get the number of files in the finetuning save directory
    num_files = len([
        f for f in os.listdir(args.finetune_save_dir)
        if f.endswith('.pth')
    ])

    for phase_index in range(num_files):

        # Load this phases saved model.
        (
            input_embedding, readformer_model, ref_base_embedding, classifier
        ) = load_checkpoint_by_phase(args, device, phase_index)

        validation_dataset = create_finetuning_dataloader(
            csv_path=f'{args.finetune_metadata_dir}/test_fold_{args.fold}.csv',
            artefact_bam_path=args.artefact_bam_path,
            mutation_bam_path=args.mutation_bam_path,
            batch_size=args.batch_size,
            base_quality_pad_idx=input_embedding.base_quality_embeddings.padding_idx,
            cigar_pad_idx=input_embedding.cigar_embeddings.padding_idx,
            is_first_pad_idx=input_embedding.mate_pair_embeddings.padding_idx,
            mapped_to_reverse_pad_idx=input_embedding.strand_embeddings.padding_idx,
            position_pad_idx=-1,
            # Only one epoch for validation but we need to loop through it after
            # training epoch multiple times.
            max_read_length=args.max_read_length,
            shuffle=False,
            # num_workers=0
            num_workers=min(get_allocated_cpus() - 1, 8),
            prefetch_factor=1
        )

        with ValidationWriter(
                args.fold, phase_index, args.validation_output_dir
        ) as writer:
            with torch.no_grad():
                # turn off dropouts for all layers during validation
                input_embedding.eval()
                readformer_model.eval()
                classifier.eval()
                if not args.no_reference:
                    ref_base_embedding.eval()

                for validation_batch in validation_dataset:
                    nucleotide_sequences = validation_batch['nucleotide_sequences'].to(device)
                    base_qualities = validation_batch['base_qualities'].to(device)
                    cigar_encoding = validation_batch['cigar_encoding'].to(device)
                    is_first = validation_batch['is_first'].to(device)
                    mapped_to_reverse = validation_batch['mapped_to_reverse'].to(device)
                    positions = validation_batch['positions'].to(device)
                    # read_support = validation_batch['read_support'].to(device)
                    # num_in_class = validation_batch['num_in_class'].to(device)
                    labels = validation_batch['labels'].to(device)
                    if not args.no_reference:
                        reference = validation_batch['reference'].to(device)
                    mutation_positions = validation_batch['mut_pos'].to(device)
                    mutation_positions = torch.unsqueeze(mutation_positions, -1)

                    chr_ = validation_batch['chr']
                    read_id = validation_batch['read_id']
                    ref = validation_batch['ref']
                    alt = validation_batch['alt']
                    is_reverse = validation_batch['is_reverse']

                    del validation_batch

                    idx = torch.nonzero(positions == mutation_positions, as_tuple=True)

                    batch_size = nucleotide_sequences.size(0)
                    if idx[0].numel() == batch_size:
                        mask = torch.ones(batch_size, dtype=torch.bool, device=device)
                    else:
                        keep_rows = set(idx[0].tolist())
                        batch_idx = torch.arange(batch_size, device=device)
                        mask = torch.tensor(
                            [i in keep_rows for i in batch_idx],
                            dtype=torch.bool,
                            device=device
                        )

                        nucleotide_sequences = nucleotide_sequences[mask]
                        base_qualities = base_qualities[mask]
                        cigar_encoding = cigar_encoding[mask]
                        is_first_flags = is_first_flags[mask]
                        mapped_to_reverse_flags = mapped_to_reverse_flags[mask]
                        positions = positions[mask]
                        if reference is not None:
                            reference = reference[mask]
                        mutation_positions = mutation_positions[mask]

                    model_input = input_embedding(
                        nucleotide_sequences, cigar_encoding,
                        base_qualities,
                        mapped_to_reverse, is_first
                    )

                    readformer_out = readformer_model(model_input, positions)

                    row_idx = torch.arange(readformer_out.size(0), device=device)
                    col_idx = (positions == mutation_positions).nonzero(as_tuple=False)[:, 1]
                    selected_h = readformer_out[row_idx, col_idx]

                    if not args.no_reference:
                        reference_embs = ref_base_embedding(reference).squeeze(-2)
                    else:
                        reference_embs = None

                    # Get indices of the mutation positions.
                    # indices = torch.nonzero(positions == mutation_positions, as_tuple=True)
                    #
                    # if indices[0].shape[0] != args.batch_size:
                    #     # Figure out which sequence is missing
                    #     missing_indices = torch.tensor(
                    #         list(set(range(args.batch_size)) - set(indices[0].tolist())))
                    #     remaining_indices = torch.tensor(
                    #         list(set(range(args.batch_size)) - set(missing_indices.tolist())))
                    #
                    #     # keep references and labels of the remaining sequences
                    #     if not args.no_reference:
                    #         reference_embs = reference_embs[remaining_indices]
                    #     labels = labels[remaining_indices]
                    #     # Get list from tensor of missing indices
                    #     chr_ = [chr_[i] for i in remaining_indices.tolist()]
                    #     mutation_positions = [mutation_positions.tolist()[i] for i in remaining_indices.tolist()]
                    #     ref = [ref[i] for i in remaining_indices.tolist()]
                    #     alt = [alt[i] for i in remaining_indices.tolist()]
                    #     is_reverse = [is_reverse[i] for i in remaining_indices.tolist()]
                    #     read_id = [read_id[i] for i in remaining_indices.tolist()]

                    alphas, betas, = classifier(
                        selected_h,
                        reference_embs
                    )

                    alphas = alphas.squeeze(-1)
                    betas = betas.squeeze(-1)

                    mask = mask.tolist()

                    chr_ = [c for c, k in zip(chr_, mask) if k]
                    ref = [r for r, k in zip(ref, mask) if k]
                    alt = [a for a, k in zip(alt, mask) if k]
                    is_reverse = [s for s, k in zip(is_reverse, mask) if k]
                    read_id = [rid for rid, k in zip(read_id, mask) if k]


                    writer.write(
                        alphas.detach(),
                        betas.detach(),
                        labels.detach().to(torch.int32),
                        chr_, mutation_positions, ref, alt, is_reverse,
                        read_id
                    )


if __name__ == '__main__':
    main()
