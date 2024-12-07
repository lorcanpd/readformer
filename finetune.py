import argparse
import torch
from torch.optim import AdamW
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

import logging
import wandb
from tabulate import tabulate
# import numpy as np

# Import the necessary modules from your components
from components.base_model import Model
from components.read_embedding import (
    NucleotideEmbeddingLayer,
    CigarEmbeddingLayer,
    BaseQualityEmbeddingLayer,
    StrandEmbeddingLayer,
    MatePairEmbeddingLayer
)
from components.finetune_data_streaming import create_finetuning_dataloader
from components.classification_head import BetaDistributionClassifier
from components.utils import get_effective_number
from components.metrics import BetaBernoulliLoss, FineTuningMetrics
from components.scheduler import CosineAnnealingScheduler


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

    args = parser.parse_args()
    return args


def instantiate_model(args, device):
    nucleotide_embeddings = NucleotideEmbeddingLayer(
        args.emb_dim, mlm_mode=True
    ).to(device)
    cigar_embeddings = CigarEmbeddingLayer(
        args.emb_dim // 4
    ).to(device)
    base_quality_embeddings = BaseQualityEmbeddingLayer(
        args.emb_dim // 4
    ).to(device)
    strand_embeddings = StrandEmbeddingLayer(
        args.emb_dim // 4
    ).to(device)
    mate_pair_embeddings = MatePairEmbeddingLayer(
        args.emb_dim // 4
    ).to(device)

    gate_projection = nn.Linear(args.emb_dim, args.emb_dim).to(device)
    feature_projection = nn.Linear(args.emb_dim, args.emb_dim).to(device)

    readformer_model = Model(
        emb_dim=args.emb_dim, heads=args.num_heads, num_layers=args.num_layers,
        n_order=args.n_order,
        readformer=args.readformer, kernel_size=args.kernel_size,
        num_hyena=args.num_hyena, num_attention=args.num_attention
    ).to(device)

    return (
        nucleotide_embeddings, cigar_embeddings, base_quality_embeddings,
        strand_embeddings, mate_pair_embeddings,
        gate_projection, feature_projection, readformer_model
    )


def load_pretrained_model(args, device):
    # Instantiate the models (nucleotide_embeddings, metric_embeddings, readformer)

    (
        nucleotide_embeddings, cigar_embeddings, base_quality_embeddings,
        strand_embeddings, mate_pair_embeddings,
        gate_projection, feature_projection, readformer_model
    ) = instantiate_model(args, device)

    # Load the checkpoint
    if not os.path.isfile(args.pre_trained_path):
        logging.error(f"No checkpoint found at '{args.pre_trained_path}'")
        sys.exit(1)
    checkpoint = torch.load(args.pre_trained_path, map_location=device)

    # Load state_dicts
    nucleotide_embeddings.load_state_dict(checkpoint['nucleotide_embeddings_state_dict'])
    cigar_embeddings.load_state_dict(checkpoint['cigar_embeddings_state_dict'])
    base_quality_embeddings.load_state_dict(checkpoint['base_quality_embeddings_state_dict'])
    strand_embeddings.load_state_dict(checkpoint['strand_embeddings_state_dict'])
    mate_pair_embeddings.load_state_dict(checkpoint['mate_pair_embeddings_state_dict'])
    gate_projection.load_state_dict(checkpoint['gate_projection_state_dict'])
    feature_projection.load_state_dict(checkpoint['feature_projection_state_dict'])
    readformer_model.load_state_dict(checkpoint['model_state_dict'])

    logging.info(f"Loaded pre-trained model from '{args.pre_trained_path}'")

    return (
        nucleotide_embeddings, cigar_embeddings, base_quality_embeddings,
        strand_embeddings, mate_pair_embeddings,
        gate_projection, feature_projection, readformer_model
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
        nucleotide_embeddings, cigar_embeddings, base_quality_embeddings,
        strand_embeddings, mate_pair_embeddings,
        gate_projection, feature_projection, readformer_model
    ) = instantiate_model(args, device)

    ref_base_embedding = NucleotideEmbeddingLayer(
        args.emb_dim, mlm_mode=True
    ).to(device)

    classifier = BetaDistributionClassifier(
        input_dim=args.emb_dim, hidden_dim=args.emb_dim // 2
    ).to(device)

    # Load state_dicts
    nucleotide_embeddings.load_state_dict(checkpoint['nucleotide_embeddings_state_dict'])
    cigar_embeddings.load_state_dict(checkpoint['cigar_embeddings_state_dict'])
    base_quality_embeddings.load_state_dict(checkpoint['base_quality_embeddings_state_dict'])
    strand_embeddings.load_state_dict(checkpoint['strand_embeddings_state_dict'])
    mate_pair_embeddings.load_state_dict(checkpoint['mate_pair_embeddings_state_dict'])
    gate_projection.load_state_dict(checkpoint['gate_projection_state_dict'])
    feature_projection.load_state_dict(checkpoint['feature_projection_state_dict'])
    readformer_model.load_state_dict(checkpoint['model_state_dict'])
    ref_base_embedding.load_state_dict(checkpoint['ref_base_embedding_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])

    # Load the optimiser state
    optimiser = AdamW(
        list(nucleotide_embeddings.parameters())
        + list(cigar_embeddings.parameters())
        + list(base_quality_embeddings.parameters())
        + list(strand_embeddings.parameters())
        + list(mate_pair_embeddings.parameters())
        + list(gate_projection.parameters())
        + list(feature_projection.parameters())
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
        nucleotide_embeddings, cigar_embeddings, base_quality_embeddings,
        strand_embeddings, mate_pair_embeddings,
        gate_projection, feature_projection, readformer_model,
        ref_base_embedding, classifier, optimiser, epoch, i
    )


def get_allocated_cpus():
    cpus = int(os.getenv('LSB_DJOB_NUMPROC', '1'))
    logging.info(f"Allocated CPUs: {cpus}")
    return cpus


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


    # Set device
    device = (
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    logging.info(f"Using device: {device}")

    if not args.load_latest_checkpoint:
        if args.pre_trained_path:
            # Load the pre-trained model
            (
                nucleotide_embeddings, cigar_embeddings, base_quality_embeddings,
                strand_embeddings, mate_pair_embeddings,
                gate_projection, feature_projection, readformer_model
            ) = load_pretrained_model(args, device)
        else:
            (
                nucleotide_embeddings, cigar_embeddings, base_quality_embeddings,
                strand_embeddings, mate_pair_embeddings,
                gate_projection, feature_projection, readformer_model
            ) = instantiate_model(args, device)

        ref_base_embedding = NucleotideEmbeddingLayer(
            args.emb_dim, mlm_mode=True
        ).to(device)

        classifier = BetaDistributionClassifier(
            input_dim=args.emb_dim,
            hidden_dim=args.emb_dim // 2
        ).to(device)

        epoch = 0
        i = 0

        initial_lr = args.base_lr

        params = (
                list(nucleotide_embeddings.parameters())
                + list(cigar_embeddings.parameters())
                + list(base_quality_embeddings.parameters())
                + list(strand_embeddings.parameters())
                + list(mate_pair_embeddings.parameters())
                + list(gate_projection.parameters())
                + list(feature_projection.parameters())
                + list(readformer_model.parameters())
                + list(ref_base_embedding.parameters())
                + list(classifier.parameters())
        )

        optimiser = AdamW(
            params, lr=initial_lr, eps=1e-9, weight_decay=0.05
        )

        start_epoch = 0
    else:
        (
            nucleotide_embeddings, cigar_embeddings, base_quality_embeddings,
            strand_embeddings, mate_pair_embeddings,
            gate_projection, feature_projection, readformer_model,
            ref_base_embedding, classifier, optimiser, start_epoch, i
        ) = load_latest_checkpoint(args, device)


    # TODO: make the freezing of weights based upon epoch. For example, freeze
    #  all pre trained weights for the first epoch, and then for the next epoch
    #  unfreeze the topmost layer. For the next epoch, unfreeze the top two
    #  layers, and so on. This will allow the model to learn the new data
    #  distribution while still retaining the knowledge from the pre-trained
    #  model.

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

    # TODO: Look into the prefetch parameters!
    dataset = create_finetuning_dataloader(
        csv_path=f'{args.finetune_metadata_dir}/train_fold_{args.fold}.csv',
        artefact_bam_path=args.artefact_bam_path,
        mutation_bam_path=args.mutation_bam_path,
        batch_size=args.batch_size,
        max_read_length=args.max_read_length,
        shuffle=True,
        num_workers=args.num_workers,
    )

    validation_dataset = create_finetuning_dataloader(
        csv_path=f'{args.finetune_metadata_dir}/test_fold_{args.fold}.csv',
        artefact_bam_path=args.artefact_bam_path,
        mutation_bam_path=args.mutation_bam_path,
        batch_size=args.batch_size,
        # Only one epoch for validation but we need to loop through it after
        # training epoch multiple times.
        max_read_length=args.max_read_length,
        shuffle=False,
        num_workers=args.num_workers
    )

    iters_in_epoch = len(dataset)

    # If the model has been loaded from a checkpoint the scheduler should be
    # parameterised with the iteration and epoch.
    last_absolute_iter = start_epoch * iters_in_epoch + i
    if last_absolute_iter == 0:
        last_absolute_iter = -1

    scheduler = CosineAnnealingScheduler(
        optimiser, iters_in_epoch, peak_pct=0.3, eta_min=1e-5,
        last_epoch=last_absolute_iter
    )
    for epoch in range(start_epoch, args.epochs):
        for batch in dataset:
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

            # Forward pass through the model to compute the loss and train the model
            nucleotide_embs = nucleotide_embeddings(nucleotide_sequences)
            reference_embs = ref_base_embedding(reference).squeeze(-2)

            metric_encoding = torch.cat(
                [
                    cigar_embeddings(cigar_encoding),
                    base_quality_embeddings(base_qualities),
                    strand_embeddings(mapped_to_reverse),
                    mate_pair_embeddings(is_first)
                ],
                dim=-1
            )

            gate = torch.sigmoid(gate_projection(metric_encoding))
            swish_out = F.silu(feature_projection(nucleotide_embs))
            model_input = swish_out * gate + nucleotide_embs
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

            alphas, betas = classifier(
                classifier_in,
                reference_embs
            )

            eff_class_num = get_effective_number(num_in_class)
            eff_mut_num = get_effective_number(read_support)
            loss_weight = 1 / (eff_class_num * eff_mut_num + 1e-12)

            alphas = alphas.squeeze(-1)
            betas = betas.squeeze(-1)

            loss = loss_fn(alphas, betas, labels, loss_weight) * 1000000
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
            logging.debug(f"Learning rate: {scheduler.get_lr()[0]:.6f}")
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

            i += 1
            if i == iters_in_epoch:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif device.type == 'mps':
                    pass
                epoch += 1
                i = 0
                validation_losses = []
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

                        # Forward pass through the model to compute the loss and train the model
                        nucleotide_embs = nucleotide_embeddings(nucleotide_sequences)
                        reference_embs = ref_base_embedding(reference).squeeze(-2)

                        metric_encoding = torch.cat(
                            [
                                cigar_embeddings(cigar_encoding),
                                base_quality_embeddings(base_qualities),
                                strand_embeddings(mapped_to_reverse),
                                mate_pair_embeddings(is_first)
                            ],
                            dim=-1
                        )

                        gate = torch.sigmoid(gate_projection(metric_encoding))

                        swish_out = F.silu(feature_projection(nucleotide_embs))

                        model_input = swish_out * gate + nucleotide_embs

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

                        alphas, betas = classifier(
                            classifier_in,
                            reference_embs
                        )

                        eff_class_num = get_effective_number(num_in_class)
                        eff_mut_num = get_effective_number(read_support)
                        loss_weight = 1 / (eff_class_num * eff_mut_num + 1e-12)

                        alphas = alphas.squeeze(-1)
                        betas = betas.squeeze(-1)

                        validation_losses.append(
                            loss_fn(alphas, betas, labels, loss_weight) * 100000)

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
                    epoch, args.name, args.fold, args.validation_output_dir
                )
                validation_metrics.reset()
                epoch_train_metric_dict = epoch_train_metrics.compute()
                epoch_train_metrics.reset()

                # Log the validation metrics
                logging.info(f"Validation metrics after epoch {epoch}:")
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
                    for metric_name, metric_value in validation_metric_dict.items():
                        wandb.log(
                            f"Validation {metric_name}", metric_value, step=epoch)
                    for metric_name, metric_value in epoch_train_metric_dict.items():
                        wandb.log(
                            f"Training {metric_name}", metric_value, step=epoch
                        )
                    # plot validation ROC curve with all predicted probabilities
                    wandb.log(
                        "Validation ROC Curve",
                        wandb.plot.roc_curve(
                            validation_metric_dict['labels'],
                            validation_metric_dict['Predictions']
                        ),
                        step=epoch
                    )
                    # plot validation PR curve with all predicted probabilities
                    wandb.log(
                        "Validation PR Curve",
                        wandb.plot.pr_curve(
                            validation_metric_dict['labels'],
                            validation_metric_dict['Predictions']
                        ),
                        step=epoch
                    )
                    wandb.log(
                        "Validation Loss",
                        torch.mean(torch.tensor(validation_losses)).item(),
                        step=epoch
                    )
                    wandb.log(
                        "Training Loss",
                        loss,
                        step=epoch
                    )

                # Save the model checkpoint, should save everything loaded by the
                # load_latest_checkpoint function.
                update = {
                    'nucleotide_embeddings_state_dict': nucleotide_embeddings.state_dict(),
                    'cigar_embeddings_state_dict': cigar_embeddings.state_dict(),
                    'base_quality_embeddings_state_dict': base_quality_embeddings.state_dict(),
                    'strand_embeddings_state_dict': strand_embeddings.state_dict(),
                    'mate_pair_embeddings_state_dict': mate_pair_embeddings.state_dict(),
                    'gate_projection_state_dict': gate_projection.state_dict(),
                    'feature_projection_state_dict': feature_projection.state_dict(),
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

    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()

#  python finetune.py --emb_dim 128 --num_heads 8 --num_layers 1 --n_order 2 --kernel_size 7 --num_hyena 24 --num_attention 0 --readformer --checkpoint_path models/LOAD_FOR_TEST_emb128_lyrs1_num_hy24_num_att0_heads8.pth

