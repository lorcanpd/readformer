import argparse
import os
from components.empirical_bayes import EmpiricalBayes


def get_args():
    parser = argparse.ArgumentParser(
        description="Empirical Bayes Validation Analysis"
    )
    parser.add_argument(
        '--fold', type=int, required=True,
        help='Fold number for cross-validation.'
    )
    parser.add_argument(
        '--validation_output_dir', type=str,
        required=True, help='Directory to save validation outputs.'
    )
    parser.add_argument(
        '--desired_fdr', type=float, default=0.01,
        help='Desired false discovery rate.'
    )
    parser.add_argument(
        '--sample_size', type=int, default=None,
        help='Number of samples to use for fitting GMM.'
    )
    parser.add_argument(
        '--random_seed', type=int, default=42,
        help='Random seed for reproducibility.'
    )

    return parser.parse_args()


def main():
    args = get_args()

    # get the number of prediction files in the directory that start with
    # fold_{fold} and end with predictions.csv
    num_files = len(
        [
            f for f in os.listdir(args.validation_output_dir)
            if f.startswith(f"fold_{args.fold}")
                and f.endswith("predictions.csv")
        ]
    )

    for phase_index in range(num_files):
        # Instantiate the EmpiricalBayes class
        eb = EmpiricalBayes(
            fold=args.fold,
            phase_index=phase_index,
            validation_output_dir=args.validation_output_dir,
            desired_fdr=args.desired_fdr,
            sample_size=args.sample_size,
            random_state=args.random_seed
        )

        # Run the Empirical Bayes analysis
        eb.run()


if __name__ == '__main__':
    main()
