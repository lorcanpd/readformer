import argparse
import torch
import torch.nn as nn
import os
import sys
import logging
import numpy as np

# Import the necessary modules from your components
from components.base_model import Model
from components.read_embedding import (
    NucleotideEmbeddingLayer,
    CigarEmbeddingLayer,
    BaseQualityEmbeddingLayer,
    SequencingDirectionEmbeddingLayer,
    ReadReversalEmbeddingLayer
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Load a pre-trained model for fine-tuning."
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

    # Checkpoint parameters
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the pre-trained model checkpoint.')

    args = parser.parse_args()
    return args


def load_pretrained_model(args, device):
    # Instantiate the models (nucleotide_embeddings, metric_embeddings, readformer)
    nucleotide_embeddings = NucleotideEmbeddingLayer(
        args.emb_dim, mlm_mode=True
    ).to(device)
    cigar_embeddings = CigarEmbeddingLayer(
        args.emb_dim // 4
    ).to(device)
    base_quality_embeddings = BaseQualityEmbeddingLayer(
        args.emb_dim // 4
    ).to(device)
    sequencing_direction_embeddings = SequencingDirectionEmbeddingLayer(
        args.emb_dim // 4
    ).to(device)
    read_reversal_embeddings = ReadReversalEmbeddingLayer(
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

    # Load the checkpoint
    if not os.path.isfile(args.checkpoint_path):
        logging.error(f"No checkpoint found at '{args.checkpoint_path}'")
        sys.exit(1)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    # Load state_dicts
    nucleotide_embeddings.load_state_dict(checkpoint['nucleotide_embeddings_state_dict'])
    cigar_embeddings.load_state_dict(checkpoint['cigar_embeddings_state_dict'])
    base_quality_embeddings.load_state_dict(checkpoint['base_quality_embeddings_state_dict'])
    sequencing_direction_embeddings.load_state_dict(checkpoint['sequencing_direction_embeddings_state_dict'])
    read_reversal_embeddings.load_state_dict(checkpoint['read_reversal_embeddings_state_dict'])
    gate_projection.load_state_dict(checkpoint['gate_projection_state_dict'])
    feature_projection.load_state_dict(checkpoint['feature_projection_state_dict'])
    readformer_model.load_state_dict(checkpoint['model_state_dict'])

    logging.info(f"Loaded pre-trained model from '{args.checkpoint_path}'")

    return (
        nucleotide_embeddings, cigar_embeddings, base_quality_embeddings,
        sequencing_direction_embeddings, read_reversal_embeddings,
        gate_projection, feature_projection, readformer_model
    )


def main():
    args = get_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load the pre-trained model
    (
        nucleotide_embeddings, cigar_embeddings, base_quality_embeddings,
        sequencing_direction_embeddings, read_reversal_embeddings,
        gate_projection, feature_projection, readformer_model
    ) = load_pretrained_model(args, device)

    breakpoint()

    # # Freeze or unfreeze layers as needed
    # if args.freeze_embeddings:
    #     logging.info("Freezing embedding layers.")
    #     for param in nucleotide_embeddings.parameters():
    #         param.requires_grad = False
    #     for param in metric_embeddings.parameters():
    #         param.requires_grad = False


if __name__ == '__main__':
    main()

#  python finetune.py --emb_dim 128 --num_heads 8 --num_layers 1 --n_order 2 --kernel_size 7 --num_hyena 24 --num_attention 0 --readformer --checkpoint_path models/LOAD_FOR_TEST_emb128_lyrs1_num_hy24_num_att0_heads8.pth