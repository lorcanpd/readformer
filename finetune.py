import argparse
import torch
from torch.optim import AdamW
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
import os
import sys

import logging
import wandb
from tabulate import tabulate
import multiprocessing as mp

# Import the necessary modules from your components
from components.base_model import Model
from components.read_embedding import (
    InputEmbeddingLayer, NucleotideEmbeddingLayer)
from components.finetune_data_streaming import create_finetuning_dataloader
from components.classification_head import BetaDistributionClassifier
from components.utils import get_effective_number, get_layerwise_param_groups
from components.metrics import (
    BetaBernoulliLoss, FineTuningMetrics, compute_load_balance_loss)
# from components.scheduler import CosineAnnealingScheduler
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
        '--epochs', type=int, default=10,
        help='Number of epochs for fine-tuning.'
    )
    parser.add_argument(
        '--base_lr', type=float, default=1e-4,
        help='The base learning rate for fine-tuning.'
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size for fine-tuning.'
    )

    # Checkpoint parameters
    parser.add_argument(
        '--pre_trained_path', type=str,
        help=(
            'Path to the pre-trained model checkpoint. If not supplied the '
            'model will be initialised with random weights. and trained from '
            'scratch.'
        )
    )
    parser.add_argument(
        '--finetune_save_path', type=str,
        help='Path to save the fine-tuned model. Also used for checkpointing.',
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
        '--max_read_length', type=int, default=100,
        help='Maximum read length to consider.'
    )

    parser.add_argument(
        '--num_workers', type=int, default=0,
        help='Number of workers for data loading.'
    )

    parser.add_argument(
        '--prefetch_factor', type=int, default=2,
        help='Prefetch factor for data loading.'
    )

    parser.add_argument(
        '--wandb', action='store_true',
        help='Use Weights & Biases for logging.'
    )

    parser.add_argument(
        '--wandb_api_path', type=str, default='.wandb_api',
        help='Path to the wandb api key file.'
    )
    parser.add_argument(
        '--load_latest_checkpoint', action='store_true',
        help='Load the latest checkpoint from the finetune save directory.'
    )

    parser.add_argument(
        '--debug', action='store_true',
        help='Enable debug logging.'
    )

    parser.add_argument(
        '--phases_per_epoch', type=int, default=1,
        help=(
            'Number of phases to divide each epoch into. Each phase will have '
            'its own annealing cycle and validation.'
        )
    )

    args = parser.parse_args()
    return args


def instantiate_model(args, device):
    input_embedding = InputEmbeddingLayer(
        args.emb_dim
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
    if not os.path.isdir(args.finetune_save_path):
        logging.error(f"No directory found at '{args.finetune_save_path}'")
        sys.exit(1)

    latest_checkpoint = args.finetune_save_path

    if not os.path.isfile(latest_checkpoint):
        logging.error(f"No checkpoint found at '{latest_checkpoint}'")
        sys.exit(1)

    checkpoint = torch.load(latest_checkpoint, map_location=device)

    # Load the models
    (
        input_embedding, readformer_model
    ) = instantiate_model(args, device)

    ref_base_embedding = NucleotideEmbeddingLayer(
        args.emb_dim, mlm_mode=True
    ).to(device)

    classifier = BetaDistributionClassifier(
        input_dim=args.emb_dim, hidden_dim=args.emb_dim // 2
    ).to(device)

    # Load state_dicts
    input_embedding.load_state_dict(checkpoint['input_embedding_state_dict'])
    readformer_model.load_state_dict(checkpoint['model_state_dict'])
    ref_base_embedding.load_state_dict(checkpoint['ref_base_embedding_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])

    # Load the optimiser state
    optimiser = AdamW(
        list(input_embedding.parameters())
        + list(readformer_model.parameters())
        + list(ref_base_embedding.parameters())
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


def main():
    args = get_args()

    logging.basicConfig(
        level=logging.INFO if args.debug is not True else logging.DEBUG,
        format='%(levelname)s: %(message)s'
    )

    if args.wandb:
        with open(args.wandb_api_path, 'r') as f:
            wandb_api_key = f.read().strip()
        os.environ['WANDB_API_KEY'] = wandb_api_key
        wandb.login(key=wandb_api_key)
        logging.info("Logged into Weights & Biases.")

        wandb.init(
            project=args.project,
            config={
                "layers": args.num_layers,
                "heads": args.num_heads,
                "emb_dim": args.emb_dim,
                "n_order": args.n_order,
                "kernel_size": args.kernel_size,
                "num_hyena_per_layer": args.num_hyena,
                "num_attention_per_layer": args.num_attention,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "base_lr": args.base_lr,
                "fold": args.fold
            },
            resume=False
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

    loss_fn = BetaBernoulliLoss(reduction='mean')

    iter_train_metrics = FineTuningMetrics(
        thresholds=[round(x * 0.1, 1) for x in range(2, 8)],
        device=device
    )
    epoch_train_metrics = FineTuningMetrics(
        thresholds=[round(x * 0.1, 1) for x in range(1, 10)],
        device=device
    )
    validation_metrics = FineTuningMetrics(
        thresholds=[round(x * 0.1, 1) for x in range(1, 10)],
        device=device, store_predictions=True
    )

    if args.load_latest_checkpoint:
        (
            input_embedding, readformer_model,
            ref_base_embedding, classifier, optimiser, start_epoch, i
        ) = load_latest_checkpoint(args, device)
    else:
        if args.pre_trained_path:
            # Load the pre-trained model weights only
            (
                input_embedding, readformer_model
            ) = load_pretrained_model(args, device)
            # Fine-tuning from pre-trained but no previous fine-tune steps done
            start_epoch = 0
            i = 0
        else:
            # Training from scratch (no pre-training)
            (
                input_embedding, readformer_model
            ) = instantiate_model(args, device)
            start_epoch = 0
            i = 0

        ref_base_embedding = NucleotideEmbeddingLayer(
            args.emb_dim, mlm_mode=True
        ).to(device)

        classifier = BetaDistributionClassifier(
            input_dim=args.emb_dim,
            hidden_dim=args.emb_dim // 2
        ).to(device)

        min_lr = args.base_lr / 3
        param_groups = get_layerwise_param_groups(
            readformer_model, args.base_lr, min_lr
        )

        embedding_params = list(input_embedding.parameters())
        param_groups.append({
            'params': embedding_params,
            'lr': min_lr
        })

        classifier_params = (
                list(classifier.parameters())
                + list(ref_base_embedding.parameters())
        )
        param_groups.append({
            'params': classifier_params,
            'lr': args.base_lr
        })

        optimiser = AdamW(param_groups, eps=1e-9, weight_decay=0.05)

    dataset = create_finetuning_dataloader(
        csv_path=f'{args.finetune_metadata_dir}/train_fold_{args.fold}.csv',
        artefact_bam_path=args.artefact_bam_path,
        mutation_bam_path=args.mutation_bam_path,
        batch_size=args.batch_size,
        base_quality_pad_idx=input_embedding.base_quality_embeddings.padding_idx,
        cigar_pad_idx=input_embedding.cigar_embeddings.padding_idx,
        is_first_pad_idx=input_embedding.mate_pair_embeddings.padding_idx,
        mapped_to_reverse_pad_idx=input_embedding.strand_embeddings.padding_idx,
        position_pad_idx=-1,
        max_read_length=args.max_read_length,
        shuffle=True,
        # num_workers=0,
        num_workers=get_allocated_cpus() // 2,
        prefetch_factor=1

    )

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
        num_workers=get_allocated_cpus() // 2,
        prefetch_factor=1
    )

    iters_in_epoch = len(dataset)
    steps_per_phase = iters_in_epoch // args.phases_per_epoch

    last_absolute_iter = start_epoch * iters_in_epoch + i

    if last_absolute_iter == 0:
        last_absolute_iter = -1
    else:
        last_absolute_iter = last_absolute_iter % steps_per_phase

    # scheduler = CosineAnnealingScheduler(
    #     optimiser, steps_per_phase,
    #     peak_pct=0.3, eta_min=1e-5,
    #     last_epoch=last_absolute_iter,
    #     max_lr=[group['lr'] for group in optimiser.param_groups]
    # )
    max_lr_list = [group['lr'] for group in optimiser.param_groups]
    scheduler = OneCycleLR(
        optimiser, max_lr=max_lr_list, total_steps=steps_per_phase,
        pct_start=0.3, anneal_strategy='cos', cycle_momentum=False,
        last_epoch=last_absolute_iter
    )


    # if args.pre_trained_path:
    #     # freeze all pre-trained layers
    #     for param in nucleotide_embeddings.parameters():
    #         param.requires_grad = False
    #     for param in cigar_embeddings.parameters():
    #         param.requires_grad = False
    #     for param in base_quality_embeddings.parameters():
    #         param.requires_grad = False
    #     for param in strand_embeddings.parameters():
    #         param.requires_grad = False
    #     for param in mate_pair_embeddings.parameters():
    #         param.requires_grad = False
    #     for param in gate_projection.parameters():
    #         param.requires_grad = False
    #     for param in feature_projection.parameters():
    #         param.requires_grad = False
    #     for param in readformer_model.parameters():
    #         param.requires_grad = False

    best_validation_loss = float('inf')
    for epoch in range(start_epoch, args.epochs):

        for batch in dataset:

            if i % steps_per_phase == 0:
                # scheduler = CosineAnnealingScheduler(
                #     optimiser, steps_per_phase,
                #     peak_pct=0.3, eta_min=1e-5,
                #     last_epoch=-1,
                #     max_lr=[group['lr'] for group in optimiser.param_groups]
                # )
                scheduler = OneCycleLR(
                    optimiser, max_lr=max_lr_list, total_steps=steps_per_phase,
                    pct_start=0.3, anneal_strategy='cos', cycle_momentum=False,
                    last_epoch=-1
                )

            phase_index = (i // steps_per_phase) + (epoch * args.phases_per_epoch)

            if args.pre_trained_path:
                unfreeze_layers_by_epoch(
                    optimiser.param_groups, phase_index,
                    # The classifier and ref_base_embedding are always unfrozen.
                    # They are attached at the last element of the param_groups
                    # list.
                    ignore_groups=[len(optimiser.param_groups) - 1]
                )

            input_embedding.train()
            readformer_model.train()
            classifier.train()
            ref_base_embedding.train()

            epoch_loss = []
            nucleotide_sequences = batch['nucleotide_sequences'].to(device)
            base_qualities = batch['base_qualities'].to(device)
            cigar_encoding = batch['cigar_encoding'].to(device)
            is_first = batch['is_first'].to(device)
            mapped_to_reverse = batch['mapped_to_reverse'].to(device)
            positions = batch['positions'].to(device)
            read_support = batch['read_support'].to(device)
            num_in_class = batch['num_in_class'].to(device)
            labels = batch['labels'].to(device)
            reference = batch['reference'].to(device)
            mutation_positions = batch['mut_pos'].to(device)
            mutation_positions = torch.unsqueeze(mutation_positions, -1)
            del batch

            with device_context(device):
                # Forward pass through the model to compute the loss and train the model
                model_input = input_embedding(
                    nucleotide_sequences, cigar_encoding, base_qualities,
                    mapped_to_reverse, is_first
                )
                readformer_out = readformer_model(model_input, positions)
                reference_embs = ref_base_embedding(reference).squeeze(-2)

                # Get indices of the mutation positions.
                indices = torch.nonzero(positions == mutation_positions, as_tuple=True)

                if indices[0].shape[0] != args.batch_size:
                    # Figure out which sequence is missing
                    missing_indices = torch.tensor(
                        list(set(range(args.batch_size)) - set(indices[0].tolist())))
                    remaining_indices = torch.tensor(
                        list(set(range(args.batch_size)) - set(missing_indices.tolist())))

                    # keep references and labels of the remaining sequences
                    reference_embs = reference_embs[remaining_indices]
                    labels = labels[remaining_indices]
                    read_support = read_support[remaining_indices]
                    num_in_class = num_in_class[remaining_indices]

                classifier_in = readformer_out[indices]

                alphas, betas, gate_alpha_scores, gate_beta_scores = classifier(
                    classifier_in,
                    reference_embs
                )



                eff_class_num = get_effective_number(num_in_class)
                eff_mut_num = get_effective_number(read_support)
                loss_weight = 1 / (eff_class_num * eff_mut_num + 1e-12)

                alphas = alphas.squeeze(-1)
                betas = betas.squeeze(-1)

                loss = loss_fn(alphas, betas, labels, loss_weight) * 10

                expert_balance_loss = compute_load_balance_loss(
                    gate_alpha_scores, gate_beta_scores, loss_weight
                )

                loss += expert_balance_loss

                # convert labels tensor to torch.int32
                labels = labels.to(torch.int32)

                iter_train_metrics.update(
                    alphas.detach(),
                    betas.detach(),
                    labels.detach().to(torch.int32)
                )
                epoch_train_metrics.update(
                    alphas.detach(),
                    betas.detach(),
                    labels.detach().to(torch.int32)
                )

                iter_train_metric_dict = iter_train_metrics.compute()
                iter_train_metrics.reset()

                # Backward pass
                optimiser.zero_grad()
                loss.backward()

                # Update weights
                optimiser.step()
                scheduler.step()

                # Compute the loss
                loss = loss.item()
                epoch_loss.append(loss)

                # Log the training metrics
                logging.debug(f"Training metrics after at epoch {epoch} iteration {i}:")
                logging.debug(f"Learning rate: {max(scheduler.get_lr()):.6f}")
                logging.debug(f"Loss: {loss}")
                table_data = []
                for key in iter_train_metrics.thresholds:
                    table_data.append([
                        key,
                        iter_train_metric_dict[f'Precision@{key}'],
                        iter_train_metric_dict[f'Recall@{key}'],
                        iter_train_metric_dict[f'F1-Score@{key}']
                    ])

                # Define the table headers
                headers = ["Threshold", "Precision", "Recall", "F1"]

                # Log the table
                logging.debug(
                    "\n" + tabulate(table_data, headers=headers, floatfmt=".5f"))
                # Lower is better for Brier Score and Calibration Error (ECE) and
                # higher is better for ROC AUC and PR AUC.
                logging.debug(f"ROC AUC: {iter_train_metric_dict['ROC AUC']:.5f}")
                logging.debug(f"PR AUC: {iter_train_metric_dict['PR AUC']:.5f}")
                logging.debug(
                    f"Brier Score: {iter_train_metric_dict['Brier Score']:.5f}")
                logging.debug(
                    f"Calibration Error (ECE): "
                    f"{iter_train_metric_dict['Calibration Error (ECE)']:.5f}"
                )
                logging.debug("\n")
                if args.wandb:
                    # Log the iteration metrics to Weights & Biases. I think only
                    # the loss is useful here.
                    wandb.log(
                        {
                            "iter_loss": loss,
                            "iter_largest_lr": max(scheduler.get_lr())
                        }
                    )

                i += 1

                if i > 0 and (i % steps_per_phase == 0):

                    validation_losses = []
                    # turn off dropouts for all layers during validation
                    input_embedding.eval()
                    readformer_model.eval()
                    classifier.eval()
                    ref_base_embedding.eval()

                    with torch.no_grad():
                        for validation_batch in validation_dataset:
                            nucleotide_sequences = validation_batch['nucleotide_sequences'].to(device)
                            base_qualities = validation_batch['base_qualities'].to(device)
                            cigar_encoding = validation_batch['cigar_encoding'].to(device)
                            is_first = validation_batch['is_first'].to(device)
                            mapped_to_reverse = validation_batch['mapped_to_reverse'].to(device)
                            positions = validation_batch['positions'].to(device)
                            read_support = validation_batch['read_support'].to(device)
                            num_in_class = validation_batch['num_in_class'].to(device)
                            labels = validation_batch['labels'].to(device)
                            reference = validation_batch['reference'].to(device)
                            mutation_positions = validation_batch['mut_pos'].to(device)
                            mutation_positions = torch.unsqueeze(mutation_positions, -1)

                            chr_ = validation_batch['chr']
                            read_id = validation_batch['read_id']
                            ref = validation_batch['ref']
                            alt = validation_batch['alt']
                            is_reverse = validation_batch['is_reverse']

                            del validation_batch

                            model_input = input_embedding(
                                nucleotide_sequences, cigar_encoding,
                                base_qualities,
                                mapped_to_reverse, is_first
                            )

                            readformer_out = readformer_model(model_input, positions)

                            # Get indices of the mutation positions.
                            indices = torch.nonzero(positions == mutation_positions, as_tuple=True)

                            if indices[0].shape != args.batch_size:
                                # Figure out which sequence is missing
                                missing_indices = torch.tensor(
                                    list(set(range(args.batch_size)) - set(indices[0].tolist())))
                                remaining_indices = torch.tensor(
                                    list(set(range(args.batch_size)) - set(missing_indices.tolist())))

                                # keep references and labels of the remaining sequences
                                reference_embs = reference_embs[remaining_indices]
                                labels = labels[remaining_indices]
                                read_support = read_support[remaining_indices]
                                num_in_class = num_in_class[remaining_indices]

                            classifier_in = readformer_out[indices]

                            alphas, betas, _, _ = classifier(
                                classifier_in,
                                reference_embs
                            )

                            eff_class_num = get_effective_number(num_in_class)
                            eff_mut_num = get_effective_number(read_support)
                            loss_weight = 1 / (eff_class_num * eff_mut_num + 1e-12)

                            alphas = alphas.squeeze(-1)
                            betas = betas.squeeze(-1)

                            validation_losses.append(
                                loss_fn(alphas, betas, labels, loss_weight))

                            # convert labels tensor to torch.int32
                            validation_metrics.update(
                                alphas.detach(),
                                betas.detach(),
                                labels.detach().to(torch.int32),
                                chr_, mutation_positions, ref, alt, is_reverse,
                                read_id
                            )

                        validation_metric_dict = validation_metrics.compute()
                        validation_metrics.write_predictions_to_csv(
                            phase_index, epoch, args.fold,
                            args.validation_output_dir
                        )
                        validation_metrics.reset()
                        epoch_train_metric_dict = epoch_train_metrics.compute()
                        epoch_train_metrics.reset()

                        # Log the validation metrics
                        logging.info(
                            f"Validation metrics after phase {phase_index} "
                            f"(There are {args.phases_per_epoch} phases per "
                            f"epoch. This is epoch {epoch}):"
                        )
                        table_data = []
                        for key in validation_metrics.thresholds:
                            table_data.append([
                                key,
                                validation_metric_dict[f'Precision@{key}'],
                                validation_metric_dict[f'Recall@{key}'],
                                validation_metric_dict[f'F1-Score@{key}']
                            ])

                        # Define the table headers
                        headers = ["Threshold", "Precision", "Recall", "F1"]

                        # Log the table
                        logging.info(
                            "\n" + tabulate(table_data, headers=headers, floatfmt=".5f"))
                        logging.info(f"ROC AUC: {validation_metric_dict['ROC AUC']:.5f}")
                        logging.info(f"PR AUC: {validation_metric_dict['PR AUC']:.5f}")
                        logging.info(
                            f"Brier Score: {validation_metric_dict['Brier Score']:.5f}")
                        logging.info(
                            f"Calibration Error (ECE): "
                            f"{validation_metric_dict['Calibration Error (ECE)']:.5f}"
                            f"\n\n"
                        )

                        if args.wandb:
                            log_entry = {}
                            for metric_name, metric_value in validation_metric_dict.items():
                                log_entry[f"Validation {metric_name}"] = metric_value

                            for metric_name, metric_value in epoch_train_metric_dict.items():
                                log_entry[f"Training {metric_name}"] = metric_value

                            log_entry["Validation ROC Curve"] = wandb.plot.roc_curve(
                                validation_metric_dict['labels'],
                                validation_metric_dict['Predictions']
                            )
                            log_entry["Validation PR Curve"] = wandb.plot.pr_curve(
                                validation_metric_dict['labels'],
                                validation_metric_dict['Predictions']
                            )
                            log_entry["Validation Loss"] = torch.mean(
                                torch.tensor(validation_losses)).item()
                            log_entry["Training Loss"] = torch.mean(
                                torch.tensor(epoch_loss)).item()

                            wandb.log(
                                log_entry,
                                step=phase_index
                            )

                        # Save the model checkpoint, should save everything loaded by the
                        # load_latest_checkpoint function.
                        update = {
                            'input_embedding_state_dict': input_embedding.state_dict(),
                            'model_state_dict': readformer_model.state_dict(),
                            'ref_base_embedding_state_dict': ref_base_embedding.state_dict(),
                            'classifier_state_dict': classifier.state_dict(),
                            'optimiser_state_dict': optimiser.state_dict(),
                            'epoch': epoch,
                            'iteration': i
                        }
                        torch.save(update, args.finetune_save_path)
                        if args.wandb:
                            wandb.save(args.finetune_save_path)

            if i == iters_in_epoch:
                epoch += 1
                i = 0



    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()

#  python finetune.py --emb_dim 128 --num_heads 8 --num_layers 1 --n_order 2 --kernel_size 7 --num_hyena 24 --num_attention 0 --readformer --checkpoint_path models/LOAD_FOR_TEST_emb128_lyrs1_num_hy24_num_att0_heads8.pth

