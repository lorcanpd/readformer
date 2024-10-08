import torch
from torch.utils.data import DataLoader
import argparse
import os
from components.data_streaming import create_data_loader
from components.pretrain_utils import apply_masking_with_consistent_replacements


def get_args():
    parser = argparse.ArgumentParser(
        description="Extract a masked validation batch and save tensors."
    )

    # Adding arguments
    parser.add_argument(
        '--metadata_path', type=str, required=True,
        help='Path to the metadata file for data loading.'
    )
    parser.add_argument(
        '--data_dir', type=str, required=True,
        help='Directory containing the data files.'
    )
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for validation data extraction.')
    parser.add_argument('--max_sequence_length', type=int, default=1024,
                        help='Maximum sequence length for data loading.')
    parser.add_argument('--min_read_quality', type=int, default=30,
                        help='Minimum read quality.')
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Directory where the masked validation tensors will be saved.'
    )
    parser.add_argument(
        '--corruption_rate', type=float, default=0.15,
        help='Rate at which bases are selected for masking/replacement.'
    )
    parser.add_argument(
        '--proportion_random', type=float, default=0.1,
        help='Proportion of corrupted labels to be assigned random nucleotides.'
    )
    parser.add_argument('--kernel_size', type=int, default=15,
                        help='Kernel size for the Hyena block.')
    parser.add_argument('--mask_token_index', type=int, default=0,
                        help='Index for the mask token.')

    return parser.parse_args()


def save_tensors(tensor_dict, output_dir):
    """Save the provided tensors to disk."""
    os.makedirs(output_dir, exist_ok=True)
    for name, tensor in tensor_dict.items():
        tensor_path = os.path.join(output_dir, f"{name}.pt")
        torch.save(tensor, tensor_path)
        print(f"Saved {name} to {tensor_path}")


def main():
    args = get_args()

    # Create data loader for validation
    data_loader = create_data_loader(
        file_paths=args.data_dir,
        metadata_path=args.metadata_path,
        nucleotide_threshold=args.max_sequence_length,
        max_sequence_length=args.max_sequence_length,
        batch_size=args.batch_size,
        min_quality=args.min_read_quality,
        shuffle=True,
        num_workers=0
    )

    print("Data loader created.")

    # Get a validation batch
    validation_batch = next(iter(data_loader))

    # Extract necessary tensors from the batch
    validation_positions = validation_batch['positions']
    validation_nucleotide_sequences = validation_batch['nucleotide_sequences']
    validation_base_qualities = validation_batch['base_qualities']
    validation_read_qualities = validation_batch['read_qualities']
    validation_cigar_match = validation_batch['cigar_match']
    validation_cigar_insertion = validation_batch['cigar_insertion']
    validation_bitwise_flags = validation_batch['bitwise_flags']

    # Combine metrics into one tensor
    validation_metrics = torch.cat(
        [
            validation_base_qualities.unsqueeze(-1),
            validation_read_qualities.unsqueeze(-1),
            validation_cigar_match.unsqueeze(-1),
            validation_cigar_insertion.unsqueeze(-1),
            validation_bitwise_flags
        ],
        dim=-1
    )

    # Apply masking to the validation batch
    (
        val_masked_sequences, val_masked_indices, val_replaced_indices,
        val_kept_indices
    ) = apply_masking_with_consistent_replacements(
        validation_positions, validation_nucleotide_sequences,
        args.mask_token_index, rate=args.corruption_rate,
        mask_rate=1.0 - 2 * args.proportion_random,
        keep_rate=args.proportion_random, replace_rate=args.proportion_random,
        kernel_size=args.kernel_size, split=0.5
    )

    # Prepare tensors to save
    tensors_to_save = {
        'positions': validation_positions,
        'masked_sequences': val_masked_sequences,
        'masked_indices': val_masked_indices,
        'replaced_indices': val_replaced_indices,
        'kept_indices': val_kept_indices,
        'metrics': validation_metrics,
        'nucleotide_sequences': validation_nucleotide_sequences  # Ground truth
    }

    # Save tensors to the specified output directory
    save_tensors(tensors_to_save, args.output_dir)


if __name__ == '__main__':
    main()
