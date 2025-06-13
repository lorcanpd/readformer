import argparse
import torch
import os
import sys
import csv

import logging
import multiprocessing as mp

from components.base_model import Model
from components.read_embedding import InputEmbeddingLayer, NucleotideEmbeddingLayer
from components.predict_data_streaming import create_prediction_dataloader
from components.classification_head import BetaDistributionClassifier
from pretrain_readwise_only import device_context, check_cuda_availability


def get_args():
    parser = argparse.ArgumentParser(
        description="Predicting mutations from BAM files."
    )
    parser.add_argument(
        '--csv_path', type=str, required=True,
        help=(
            'Path to the CSV file containing the candidate positions and their '
            'respective read ids.'
        )
    )
    parser.add_argument(
        '--bam_path', type=str, required=True,
        help='Path to the BAM file containing the reads.'
    )
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to the trained model.'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Directory to save the predictions.'
    )
    parser.add_argument(
        '--batch_size', type=int, default=200,
        help='Batch size for prediction.'
    )
    parser.add_argument(
        '--max_read_length', type=int, default=151,
        help='Maximum read length.'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Debug mode.'
    )

    # Model parameters
    parser.add_argument(
        '--emb_dim', type=int, default=1024,
        help='Embedding dimension.'
    )
    parser.add_argument(
        '--num_heads', type=int, default=8,
        help='Number of attention or hyena heads.'
    )
    parser.add_argument(
        '--num_layers', type=int, default=12,
        help='Number of transformer layers.'
    )
    parser.add_argument(
        '--n_order', type=int, default=4,
        help='Order of hyena convolutions.'
    )
    parser.add_argument(
        '--kernel_size', type=int, default=15,
        help='Kernel size for hyena block.'
    )
    parser.add_argument(
        '--num_hyena', type=int, default=1,
        help='Number of consecutive Hyena layers in each block.'
    )
    parser.add_argument(
        '--num_attention', type=int, default=1,
        help='Number of consecutive attention layers in each block.'
    )
    parser.add_argument(
        '--readformer', action='store_true',
        help='Use Readformer model configuration.'
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

    if args.no_reference:
        ref_base_embedding = None
    else:
        ref_base_embedding = NucleotideEmbeddingLayer(
            args.emb_dim, mlm_mode=True
        ).to(device)
    classifier = BetaDistributionClassifier(
        input_dim=args.emb_dim, hidden_dim=args.emb_dim // 2,
        using_reference_embedding=not args.no_reference,
    ).to(device)

    return (
        input_embedding, readformer_model, ref_base_embedding, classifier
    )


def load_pretrained_model(args, device):
    (
        input_embedding, readformer_model, ref_base_embedding, classifier
    ) = instantiate_model(args, device)

    # Load the checkpoint
    if not os.path.isfile(args.model_path):
        logging.error(f"No saved model found at '{args.model_path}'")
        sys.exit(1)
    checkpoint = torch.load(args.model_path, map_location=device)

    # Load state_dicts
    try:
        input_embedding.load_state_dict(checkpoint['input_embedding_state_dict'])
        readformer_model.load_state_dict(checkpoint['model_state_dict'])
        if not args.no_reference:
            ref_base_embedding.load_state_dict(checkpoint['ref_base_embedding_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
    except KeyError as e:
        logging.error(f"Missing key in the checkpoint '{args.model_path}': {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading checkpoint '{args.model_path}': {e}")
        sys.exit(1)

    logging.info(f"Loaded pre-trained model from '{args.model_path}'")

    return (
        input_embedding, readformer_model, ref_base_embedding, classifier
    )


def get_allocated_cpus():
    cpus = int(os.getenv('LSB_DJOB_NUMPROC', '1'))
    logging.info(f"Allocated CPUs: {cpus}")
    return cpus


class PredictionWriter:
    """
    Class to write the predictions to a CSV file.
    """

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.csv_file = None
        self.csv_writer = None

        self._initialise_csv()

    def _initialise_csv(self):
        self.csv_file = open(
            os.path.join(self.output_dir, 'predictions.csv'), 'w'
        )
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'chr', 'pos', 'ref', 'alt', 'mapped_to_reverse', 'read_id',
            'mutation_type', 'alpha', 'beta'
        ])

    def update(
            self, alpha, beta, chr_=None, pos=None, ref=None, alt=None,
            mapped_to_reverse=None, read_id=None, mutation_type=None
    ):
        batch_size = alpha.shape[0]
        alpha = alpha.detach().cpu().numpy()
        beta = beta.detach().cpu().numpy()

        chr_ = chr_ if chr_ is not None else ['NA'] * batch_size
        pos = pos if pos is not None else ['NA'] * batch_size
        if torch.is_tensor(pos):
            pos = pos.detach().cpu().numpy()
        # if pos is nested array, flatten it
        pos = [item for sublist in pos for item in sublist]
        ref = ref if ref is not None else ['N'] * batch_size
        alt = alt if alt is not None else ['N'] * batch_size
        mapped_to_reverse = mapped_to_reverse if mapped_to_reverse is not None \
            else ["NA"] * batch_size
        read_id = read_id if read_id is not None else ['NA'] * batch_size

        for i in range(batch_size):
            self.csv_writer.writerow([
                chr_[i], pos[i], ref[i], alt[i], mapped_to_reverse[i],
                read_id[i], mutation_type[i], alpha[i], beta[i]
            ])

    def close(self):
        """
        Closes the CSV file if it is open.
        """
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

    def __enter__(self):
        """
        Enables usage of the class with the 'with' statement.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Ensures the CSV file is closed when exiting the 'with' block.
        """
        self.close()


def main():
    args = get_args()

    logging.basicConfig(
        level=logging.INFO if args.debug is not True else logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if not check_cuda_availability() and not torch.backends.mps.is_available():
        logging.error("CUDA or MPS are not available.")
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

    # Load the model
    (
        input_embedding, readformer_model, ref_base_embedding, classifier
    ) = load_pretrained_model(args, device)

    # Create the data loader
    dataloader = create_prediction_dataloader(
        csv_path=args.csv_path,
        bam_path=args.bam_path,
        batch_size=args.batch_size,
        base_quality_pad_idx=input_embedding.base_quality_embeddings.padding_idx,
        cigar_pad_idx=input_embedding.cigar_embeddings.padding_idx,
        is_first_pad_idx=input_embedding.mate_pair_embeddings.padding_idx,
        mapped_to_reverse_pad_idx=input_embedding.strand_embeddings.padding_idx,
        position_pad_idx=-1,
        max_read_length=args.max_read_length,
        shuffle=False,
        num_workers=get_allocated_cpus() - 4,  # Reserve 4 for the main process and the writer
        prefetch_factor=1
    )

    # Set the model to evaluation mode
    input_embedding.eval()
    readformer_model.eval()
    if not args.no_reference:
        ref_base_embedding.eval()
    classifier.eval()

    with (
        PredictionWriter(args.output_dir) as writer, device_context(device),
        torch.no_grad()
    ):
        for batch in dataloader:
            nucleotide_sequences = batch['nucleotide_sequences'].to(device)
            base_qualities = batch['base_qualities'].to(device)
            cigar_encoding = batch['cigar_encoding'].to(device)
            is_first_flags = batch['is_first'].to(device)
            mapped_to_reverse_flags = batch['mapped_to_reverse'].to(device)

            positions = batch['positions'].to(device)

            if args.no_reference:
                reference = None
            else:
                reference = batch['reference'].to(device)

            mutation_positions = batch['mut_pos'].to(device).unsqueeze(-1)

            chr_ = batch['chr']
            read_id = batch['read_id']
            ref = batch['ref']
            alt = batch['alt']
            is_reverse = batch['is_reverse']
            mutation_type = batch['mutation_type']

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

            del batch

            # Forward pass
            model_input = input_embedding(
                nucleotide_sequences, cigar_encoding, base_qualities,
                mapped_to_reverse_flags, is_first_flags
            )

            readformer_out = readformer_model(
                model_input, positions
            )

            row_idx = torch.arange(readformer_out.size(0), device=device)
            col_idx = (positions == mutation_positions).nonzero(as_tuple=False)[:, 1]
            selected_h = readformer_out[row_idx, col_idx]

            if args.no_reference:
                reference_embs = None
            else:
                reference_embs = ref_base_embedding(reference).squeeze(-2)

            # indices = torch.nonzero(
            #     positions == mutation_positions, as_tuple=True
            # )
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
            #     # Get list from tensor of missing indices
            #     chr_ = [chr_[i] for i in remaining_indices.tolist()]
            #     mutation_positions = [
            #         mutation_positions.tolist()[i]
            #         for i in remaining_indices.tolist()
            #     ]
            #     ref = [ref[i] for i in remaining_indices.tolist()]
            #     alt = [alt[i] for i in remaining_indices.tolist()]
            #     is_reverse = [is_reverse[i] for i in remaining_indices.tolist()]
            #     read_id = [read_id[i] for i in remaining_indices.tolist()]
            #     mutation_type = [
            #         mutation_type[i] for i in remaining_indices.tolist()
            #     ]
            #
            # classifier_input = readformer_out[indices]

            alpha, beta = classifier(selected_h, reference_embs)

            # use the idx to remove the missing chr_, pos, ref, alt, is_reverse, read_id, mutation_type

            mask = mask.tolist()
            chr_ = [c for c, k in zip(chr_, mask) if k]
            ref = [r for r, k in zip(ref, mask) if k]
            alt = [a for a, k in zip(alt, mask) if k]
            is_reverse = [s for s, k in zip(is_reverse, mask) if k]
            read_id = [rid for rid, k in zip(read_id, mask) if k]
            mutation_type = [t for t, k in zip(mutation_type, mask) if k]

            writer.update(
                alpha.squeeze(-1), beta.squeeze(-1), chr_=chr_,
                pos=mutation_positions, ref=ref, alt=alt,
                mapped_to_reverse=is_reverse, read_id=read_id,
                mutation_type=mutation_type
            )

    logging.info("Predictions complete.")


if __name__ == '__main__':
    main()
