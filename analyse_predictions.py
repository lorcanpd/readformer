#!/usr/bin/env python3
"""
Combined Pipeline:
  - Runs Empirical Bayes analysis on model predictions (computing z-scores, local FDR, and threshold).
  - Produces a histogram of z-scores.
  - Then, depending on the flag --with_ground_truth:
      * If True: uses the EB-derived cutoff to label artefacts and mutations and produces:
            - COSMIC signature plots (stacked if labels available)
            - Read candidate position heatmaps and faceted PDFs based on candidate alt-call positions.
            - Uses the provided reference FASTA to look up the trinucleotide context (and thus mutation type) for each mutation.
      * If False: uses fixed prior mutation probabilities to compute posterior probabilities,
            labels mutations (using e.g. post_prob > 0.99),
            produces a COSMIC signature plot, a combined trinucleotide matrix TSV,
            and the candidate read position plots.
  - The candidate read analysis uses only the predictions file and the sample BAM file.

Directory structure example:
  PD31012b/
  ├── 256d_2l_5h_1a_16h
  │   └── fold_0
  │         └── predictions.csv
  ├── PD31012b.bam
  └── PD31012b.bam.bai

Usage examples:
  With ground truth:
    python combined_pipeline.py --validation_output_dir /path/to/out --with_ground_truth --fold 0 --bam /path/to/PD31012b.bam --reference_fasta /path/to/hs37d5.fa.gz

  Without ground truth:
    python combined_pipeline.py --validation_output_dir /path/to/out --model 256d_2l_5h_1a_16h --fold 0 --bam /path/to/PD31012b.bam
"""

import os
import argparse
import logging
import gc
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pysam
# from scipy.stats import beta

# import normal cumulative distribution function
# from scipy.stats import norm
from scipy.special import digamma, polygamma
# Import the EmpiricalBayes class from your components module.
# (Ensure PYTHONPATH is set so that components/empirical_bayes.py is accessible.)
from components.empirical_bayes import EmpiricalBayes

# ---------------------------
# Helper functions for candidate read extraction
# (Adapted from explore_reads.py)
# ---------------------------
def extract_read_by_id_obj(bam_obj, chromosome: str, position: int, read_id: str) -> dict:
    """
    Extract a specific read from an open BAM object given its ID and genomic coordinates.
    (Positions are 0–based.)
    """
    adjusted_chr = chromosome
    if chromosome not in bam_obj.references:
        adjusted_chr = chromosome[3:] if chromosome.startswith("chr") else "chr" + chromosome
        if adjusted_chr not in bam_obj.references:
            raise ValueError(f"Chromosome {adjusted_chr} not found in BAM file.")
    try:
        reads = bam_obj.fetch(adjusted_chr, position, position + 1)
    except ValueError as e:
        raise ValueError(f"Error fetching reads from {adjusted_chr}:{position} - {e}")
    for read in reads:
        if read.query_name == read_id:
            return {
                "query_name": read.query_name,
                "bitwise_flags": read.flag,
                "reference_start": read.reference_start,
                "mapping_quality": read.mapping_quality,
                "cigar": read.cigarstring,
                "template_length": read.template_length,
                "query_sequence": read.query_sequence,
                "query_qualities": read.query_qualities,
                "tags": read.tags,
                "positions": read.get_reference_positions(full_length=True),
                "is_first_in_pair": read.is_read1
            }
    return {}

def reorient_read(read: dict) -> dict:
    """Reorient a read so its 5'-to-3' direction is left-to-right.
       (If the read is on the reverse strand, reverse-complement its sequence, qualities, and positions.)"""
    if read["bitwise_flags"] & 0x10:  # bit flag for reverse strand
        comp = str.maketrans("ACGTacgt", "TGCAtgca")
        read["query_sequence"] = read["query_sequence"].translate(comp)[::-1]
        read["query_qualities"] = read["query_qualities"][::-1]
        read["positions"] = read["positions"][::-1]
    return read

def get_alt_call_index(read: dict, mutation_genomic_position: int) -> int:
    """
    Return the index in the read corresponding to the mutation position (0-based).
    If not found, return -1.
    """
    try:
        return read["positions"].index(mutation_genomic_position)
    except ValueError:
        return -1

# ---------------------------
# Functions for trinucleotide context and mutation category lookup (for ground truth)
# ---------------------------
def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a nucleotide sequence."""
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(comp)[::-1]

def get_trinuc_context(fasta, chrom: str, pos: int) -> str:
    """
    Get the tri-nucleotide context for a mutation.
    pos is 1-based and the function fetches the base at (pos-1) and one base upstream and downstream.
    Returns a 3-letter string (e.g., 'ACA'); if unavailable, returns "NNN".
    """
    try:
        seq = fasta.fetch(chrom, pos - 1, pos + 2)
        if len(seq) == 3:
            return seq.upper()
    except Exception:
        pass
    return "NNN"

def get_mutation_category(trinuc: str, ref: str, alt: str) -> str:
    """
    Given a trinucleotide context and the ref/alt alleles, return the mutation category in COSMIC format.
    If the ref allele is C or T, use the trinucleotide as is.
    Otherwise, compute the reverse complement.
    """
    if ref in ['C', 'T']:
        # return f"{trinuc[0]}[{ref}>{alt}]{trinuc[2]}"
        return f"{trinuc}{alt}"
    else:
        rc_trinuc = reverse_complement(trinuc)
        rc_ref = reverse_complement(ref)
        rc_alt = reverse_complement(alt)
        # return f"{rc_trinuc[0]}[{rc_ref}>{rc_alt}]{rc_trinuc[2]}"
        return f"{rc_trinuc}{rc_alt}"


# ---------------------------
# Functions from scratch_52.py (for COSMIC signature plotting)
# ---------------------------
def aggregate_unique_mutations(df, mutation_flag):
    """
    Filters the DataFrame based on the mutation_flag and aggregates unique mutations
    by chr, pos, and mutation_type.
    """
    filtered_df = df[df["called_mutation"] == mutation_flag]
    unique_mutations = filtered_df.drop_duplicates(subset=["chr", "pos", "mutation_type"])
    mutation_counts = unique_mutations.groupby("mutation_type").size()
    return mutation_counts


def convert_to_cosmic_format(atag_format):
    """Converts an ATAG mutation format (e.g. 'ACAG') to COSMIC format (e.g. 'A[C>G]A')."""
    left = atag_format[0]
    ref = atag_format[1]
    right = atag_format[2]
    alt = atag_format[3]
    return f"{left}[{ref}>{alt}]{right}"


def plot_cosmic_spectra(mutation_counts, artefact_counts, cosmic_96, substitution_types, color_map, save_name=None):
    """Plots COSMIC-style trinucleotide spectra for mutations and artefacts as a lattice plot."""
    plt.clf()

    # Create figure & axes
    fig, axes = plt.subplots(2, 1, figsize=(20, 12), sharex=True, sharey=False)

    # Define substitution colors
    colors = {
        'C>A': color_map[0],
        'C>G': color_map[1],
        'C>T': color_map[2],
        'T>A': color_map[3],
        'T>C': color_map[4],
        'T>G': color_map[5]
    }

    def get_bar_colors(order):
        bar_colors = []
        for mut in order:
            # Extract substitution from something like "A[C>G]T"
            substitution = mut.split('[')[1].split(']')[0]
            bar_colors.append(colors[substitution])
        return bar_colors

    # Separate maxima for top & bottom
    top_max = max(mutation_counts.values)
    bottom_max = max(artefact_counts.values)

    # Bar colors for each row
    mutation_bar_colors = get_bar_colors(cosmic_96)
    artefact_bar_colors = get_bar_colors(cosmic_96)

    # Plot top axis (Mutations)
    axes[0].bar(range(96), mutation_counts.values,
                color=mutation_bar_colors, width=0.8, edgecolor='black')
    axes[0].set_title("Predicted Mutations: COSMIC 96 Trinucleotide Spectrum",
                      fontsize=16, weight='bold')
    axes[0].set_ylabel("Count", fontsize=14)
    axes[0].yaxis.grid(True, linestyle='--', alpha=0.7)
    axes[0].set_ylim(0, top_max * 1.2)

    # Plot bottom axis (Artefacts)
    axes[1].bar(range(96), artefact_counts.values,
                color=artefact_bar_colors, width=0.8, edgecolor='black')
    axes[1].set_title("Predicted Artefacts: COSMIC 96 Trinucleotide Spectrum",
                      fontsize=16, weight='bold')
    axes[1].set_ylabel("Count", fontsize=14)
    axes[1].yaxis.grid(True, linestyle='--', alpha=0.7)
    axes[1].set_ylim(0, bottom_max * 1.2)

    # X labels only on bottom axis
    axes[1].set_xticks(range(96))
    axes[1].set_xticklabels(cosmic_96, rotation=90, fontsize=8)
    axes[1].set_xlabel("Trinucleotide Context", fontsize=14)

    # Place substitution type labels on the top axis using top_max
    for i, substitution in enumerate(substitution_types):
        position = i * 16 + 8
        axes[0].text(position, top_max * 1.05, substitution,
                     fontsize=14, weight='bold', ha='center',
                     color=colors[substitution])

    # Add a text box displaying the total count of mutations and artefacts
    mutation_total = mutation_counts.sum()
    artefact_total = artefact_counts.sum()
    total_text = f"Total Mutations: {mutation_total}\nTotal Artefacts: {artefact_total}"
    fig.text(0.95, 0.5, total_text, fontsize=14, weight='bold', va='center', ha='right')
    # Adjust layout: reserve a bit of space at the top to fit text
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_name:
        plt.savefig(save_name, dpi=300)

    # Close figure completely
    plt.close('all')
    logging.info(f"COSMIC spectra plot saved to {save_name}")

# ---------------------------
# Functions from explore_reads.py (for candidate read position plots)
# ---------------------------
def plot_alt_positions_heatmap(alt_positions, labels, out_file):
    """
    Plot a heatmap of alt call positions grouped by label.
    """
    unique_labels = np.unique(labels)
    data = {}
    max_pos = 0
    for lbl in unique_labels:
        positions = [pos for pos, l in zip(alt_positions, labels) if pos >= 0 and l is not None]
        data[lbl] = positions
        if positions:
            max_pos = max(max_pos, max(positions))
    bins = np.arange(0, max_pos + 2)
    heatmap_data = []
    for lbl in unique_labels:
        hist, _ = np.histogram(data[lbl], bins=bins, density=True)
        heatmap_data.append(hist)
    heatmap_data = np.array(heatmap_data)
    fig, ax = plt.subplots(figsize=(10, 6))
    hm = sns.heatmap(heatmap_data, cmap="viridis",
                     cbar_kws={'orientation': 'horizontal', 'pad': 0.2},
                     yticklabels=[f"{l}" for l in unique_labels],
                     xticklabels=bins[:-1],
                     ax=ax)
    ax.set_xlabel("Alt Call Position")
    ax.set_ylabel("Group")
    ax.set_title("Heatmap of Alt Call Positions")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    logging.info(f"Alt call positions heatmap saved to {out_file}")


def plot_facet_alt_positions_heatmap_pdf(merged_df, out_file):
    """
    Produce a multi-page PDF of facet heatmaps of alt call positions by trinucleotide context.
    Assumes merged_df contains columns: 'alt_call_position', 'mutation_category', and 'trinuc'.
    """
    from matplotlib.backends.backend_pdf import PdfPages
    mutation_types = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
    bases = ['A', 'C', 'G', 'T']
    cosmic_order = [f"{l}[{mut}]{r}" for mut in mutation_types for l in bases for r in bases]
    if "full_context" not in merged_df.columns:
        merged_df["full_context"] = merged_df["mutation_category"]
    df = merged_df[merged_df["alt_call_position"] >= 0].copy()
    max_pos = int(df["alt_call_position"].max()) if not df["alt_call_position"].empty else 0
    bins = np.arange(0, max_pos + 2)
    norm_hist_dict = {}
    for context in cosmic_order:
        positions = df.loc[df["full_context"] == context, "alt_call_position"]
        if len(positions) == 0:
            raw_hist = np.zeros(len(bins) - 1)
        else:
            raw_hist, _ = np.histogram(positions, bins=bins)
        if raw_hist.max() > 0:
            norm_hist_dict[context] = raw_hist / raw_hist.max()
        else:
            norm_hist_dict[context] = raw_hist
    contexts_per_page = 16
    n_pages = int(np.ceil(len(cosmic_order) / contexts_per_page))
    pp = PdfPages(out_file)
    for page in range(n_pages):
        page_contexts = cosmic_order[page * contexts_per_page:(page + 1) * contexts_per_page]
        fig, axs = plt.subplots(nrows=16, ncols=1, figsize=(8, 16 * 0.4), sharex=True)
        for i, context in enumerate(page_contexts):
            ax = axs[i]
            heat_data = norm_hist_dict[context].reshape(1, -1)
            im = ax.imshow(heat_data, aspect="auto", cmap="viridis", vmin=0, vmax=1)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_ylabel(context, rotation=0, ha='right', va='center', fontsize=8, labelpad=8)
        for j in range(i + 1, 16):
            axs[j].axis("off")
        cbar = fig.colorbar(im, ax=axs.tolist(), orientation="horizontal", fraction=0.05, pad=0.15)
        cbar.set_label("Normalised Count", fontsize=10)
        plt.tight_layout()
        pp.savefig(fig)
        plt.close(fig)
    pp.close()
    logging.info(f"Facet alt call positions heatmaps saved to multi-page PDF: {out_file}")

# ---------------------------
# Main pipeline function
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Combined Empirical Bayes and Candidate Read Analysis Pipeline"
    )
    parser.add_argument(
        '--validation_output_dir', type=str, required=True,
        help="Directory to save outputs (plots, CSVs, matrices)."
    )
    parser.add_argument(
        '--with_ground_truth', action='store_true',
        help="Use ground truth labels with EB FDR cutoff. Otherwise use fixed priors."
    )
    parser.add_argument(
        '--fold', type=int, default=None,
        help="Fold number (used when with_ground_truth=True for cross-validation)."
    )
    parser.add_argument(
        '--sample_size', type=int, default=None,
        help="Number of samples to use for fitting GMM."
    )
    parser.add_argument(
        '--desired_fdr', type=float, default=0.01,
        help="Desired false discovery rate for EB analysis."
    )
    parser.add_argument(
        '--random_seed', type=int, default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        '--alpha_prior', type=float, default=0.0,
        help="Alpha prior for beta distribution."
    )
    parser.add_argument(
        '--beta_prior', type=float, default=0.0,
        help="Beta prior for beta distribution."
    )
    # For prior-based analysis (when with_ground_truth is False)
    # parser.add_argument(
    #     '--model', type=str, default=None,
    #     help="Model name (e.g. '256d_2l_5h_1a_16h') used to locate predictions."
    # )
    # Candidate read analysis uses the sample BAM file.
    parser.add_argument(
        '--bam', type=str, default=None,
        help="Path to sample BAM file for candidate read analysis."
    )
    # For ground truth analysis, use the reference FASTA to look up the mutation context.
    parser.add_argument(
        '--reference_fasta', type=str, default=None,
        help="Path to reference fasta file (e.g. hs37d5.fa.gz) used when ground truth is enabled."
    )

    parser.add_argument(
        '--sample_id', type=str, default=None,
        help="Sample ID for the analysis"
    )

    args = parser.parse_args()

    os.makedirs(args.validation_output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(os.path.join(args.validation_output_dir, "combined_pipeline.log"))]
    )

    # ---------------------------
    # Step 1: Run Empirical Bayes analysis
    # ---------------------------
    logging.info("Starting Empirical Bayes analysis.")
    if args.with_ground_truth:
        if args.fold is None:
            logging.error("Fold number must be provided when using ground truth.")
            return
        prediction_files = [
            f for f in os.listdir(args.validation_output_dir)
            if f.startswith(f"fold_{args.fold}") and f.endswith("predictions.csv")
        ]
        if not prediction_files:
            logging.error("No prediction files found for the specified fold in the output directory.")
            return
        for phase_index in range(len(prediction_files)):
            eb = EmpiricalBayes(
                fold=args.fold,
                phase_index=phase_index,
                validation_output_dir=args.validation_output_dir,
                desired_fdr=args.desired_fdr,
                sample_size=args.sample_size,
                random_state=args.random_seed,
                alpha_prior=args.alpha_prior,
                beta_prior=args.beta_prior,
                with_ground_truth=True
            )
            eb.run()
    else:
        eb = EmpiricalBayes(
            validation_output_dir=args.validation_output_dir,
            desired_fdr=args.desired_fdr,
            sample_size=args.sample_size,
            random_state=args.random_seed,
            alpha_prior=args.alpha_prior,
            beta_prior=args.beta_prior,
            with_ground_truth=False
        )
        eb.run()

    # ---------------------------
    # Step 2: Downstream analysis using the predictions file output by EB
    # ---------------------------
    if args.with_ground_truth:
        pred_filename = f"fold_{args.fold}_phase_000_predictions_with_zlfdr.csv"
    else:
        pred_filename = "predictions_with_zlfdr.csv"
    pred_path = os.path.join(args.validation_output_dir, pred_filename)
    try:
        predictions = pd.read_csv(pred_path)
    except Exception as e:
        logging.error(f"Error loading predictions file {pred_path}: {e}")
        return

    # If ground truth is used and a reference FASTA is provided, update mutation types using the reference.
    if args.with_ground_truth and args.reference_fasta:
        logging.info("Updating mutation types using reference FASTA for ground truth analysis.")
        fasta = pysam.FastaFile(args.reference_fasta)
        def lookup_mutation_type(row):
            trinuc = get_trinuc_context(fasta, str(row["chr"]), int(row["pos"]))
            return get_mutation_category(trinuc, row["ref"], row["alt"])
        predictions["mutation_type"] = predictions.apply(lookup_mutation_type, axis=1)
        fasta.close()

    substitution_types = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
    bases = ['A', 'C', 'G', 'T']
    cosmic_96 = []
    for substitution in substitution_types:
        ref_base, alt_base = substitution.split('>')
        for left in bases:
            for right in bases:
                mutation_type = f"{left}[{ref_base}>{alt_base}]{right}"
                cosmic_96.append(mutation_type)
    assert len(cosmic_96) == 96, f"Expected 96 COSMIC mutation types, got {len(cosmic_96)}"
    color_map = ['#5cb4e6', '#000000', '#e62725', '#bababa', '#70bf41', '#de7b85']



    # use alpha and beta to get the bernoulli probability
    predictions['bernoulli_prob'] = predictions['alpha'] / (predictions['alpha'] + predictions['beta'])

    # confidence = 0.95
    # lower_quantile = (1 - confidence) / 2
    # predictions['lower_bound'] = beta.ppf(lower_quantile, predictions['alpha'], predictions['beta'])

    # predictions['tail_prob'] = 1 - beta.cdf(0.95, predictions['alpha'], predictions['beta'])
    #
    # filter_threshold = 0.9
    #
    # predictions['tail_prob'] >= filter_threshold

    # predictions['tail_prob'] = 1 - beta.cdf(0.975, predictions['alpha'], predictions['beta'])



    # predictions['new_z_score'] = np.log(predictions['alpha'] / predictions['beta']) / np.sqrt((1 / predictions['alpha']) + (1 / predictions['beta']))

    # breakpoint()

    # Compute the naive log odds: log(alpha/beta)
    predictions['log_odds_exact'] = digamma(predictions['alpha']) - digamma(predictions['beta'])

    # Compute the analytical standard error from the trigamma functions:
    # SE = sqrt(psi1(alpha) + psi1(beta))
    predictions['analytical_SE'] = np.sqrt(polygamma(1, predictions['alpha']) + polygamma(1, predictions['beta']))

    # Compute the adjusted z-score, scaling the naive log odds by the analytical standard error.
    predictions['z_exact'] = predictions['log_odds_exact'] / predictions['analytical_SE']

    # max(predictions[predictions['lower_bound'] >= 0.97].z_score)
    #
    # max(predictions.z_score)

    # breakpoint()

    # (A) Fixed prior analysis when ground truth is False.
    # if not args.with_ground_truth:
    # drop duplicate rows
    predictions.drop_duplicates(subset=["chr", "pos", "ref", "alt", "read_id", "mutation_type"], inplace=True)
    # remove non-canonical positions
    def is_canonical(chrom):
        canonical_list = [f'{i}' for i in range(1, 23)] + ['X', 'Y']
        return str(chrom) in canonical_list

    predictions = predictions[predictions["chr"].apply(is_canonical)]

    # for prob in [0.95, 0.96, 0.975, 0.98, 0.99]:
    #     copy_of_predictions = predictions.copy()
    #
    #     # filter so that only bernoulli_prob > 0.99 is kept
    #     copy_of_predictions['called_mutation'] = copy_of_predictions["tail_prob"] >= prob
    #     prob_str = str(prob).replace(".", "p")
    #     mutation_type_counts = aggregate_unique_mutations(copy_of_predictions, mutation_flag=True)
    #     artefact_type_counts = aggregate_unique_mutations(copy_of_predictions, mutation_flag=False)
    #     mutation_type_counts.index = mutation_type_counts.index.map(convert_to_cosmic_format)
    #     artefact_type_counts.index = artefact_type_counts.index.map(convert_to_cosmic_format)
    #     mutation_type_counts = mutation_type_counts.reindex(cosmic_96, fill_value=0)
    #     artefact_type_counts = artefact_type_counts.reindex(cosmic_96, fill_value=0)
    #     cosmic_plot_file = os.path.join(args.validation_output_dir,
    #                                     f"cosmic96_spectra_lattice_tail_prob_geq_{prob_str}.png")
    #     plot_cosmic_spectra(mutation_type_counts, artefact_type_counts, cosmic_96, substitution_types, color_map,
    #                         save_name=cosmic_plot_file)
    #     mutation_type_counts = mutation_type_counts.reindex(cosmic_96, fill_value=0)
    #     mutation_type_counts.index = [
    #         f"{parts[0]}{parts[1]}{parts[3]}{parts[2]}"
    #         for parts in mutation_type_counts.index.str.split(r'[\[\]>]')
    #     ]
    #     # add sample name as the column header
    #     if args.sample_id:
    #         # mutation_type_counts = mutation_type_counts.rename(args.sample_id)
    #         mutation_type_counts = mutation_type_counts.to_frame(name=args.sample_id)
    #
    #     out_matrix = os.path.join(args.validation_output_dir, f"Trinucleotide_Matrix_tail_prob_geq_{prob_str}.tsv")
    #     mutation_type_counts.to_csv(out_matrix, sep="\t", index=True, header=True)
    #
    #     logging.info(f"Saved combined trinucleotide matrix for tail prob mass >= 0.975 with greater than {prob} "
    #                  f"probability to {out_matrix}")

    # mut_priors = [
    #     # 105/3_000_000_000,  # 105 mutations in 3 billion bases
    #     # 110/3_000_000_000,  # 110 mutations in 3 billion bases
    #     # 115/3_000_000_000,  # 115 mutations in 3 billion bases
    #     # 120/3_000_000_000,  # 120 mutations in 3 billion bases
    #     # 125/3_000_000_000,  # 125 mutations in 3 billion bases
    #     # 3/100_000_000,  # 3 mutations in 100 million bases
    #     # 2.8/100_000_000,  # 2.8 mutations in 100 million bases
    #     1.5/100_000_000,  # 1.5 mutations in 100 million bases
    #     2/100_000_000,  # 2 mutations in 100 million bases
    #     2.1/100_000_000,  # 2.1 mutations in 100 million bases
    #     2.2/100_000_000,  # 2.2 mutations in 100 million bases
    #     2.3/100_000_000,  # 2.3 mutations in 100 million bases
    #     2.4/100_000_000,  # 2.4 mutations in 100 million bases
    #     2.5/100_000_000,  # 2.5 mutations in 100 million bases
    # ]
    # if 'z_score' not in predictions.columns:
    #     logging.error("Predictions file does not contain 'z_score' column.")
    #     return
    # for prior_mut_prob in mut_priors:
    #     copy_of_predictions = predictions.copy()
    #     prior_str = str(prior_mut_prob).replace("/", "_").replace(".", "p")
    #     logging.info(f"Processing fixed prior: {prior_mut_prob} (prior_str={prior_str})")
    #     log_prior_odds = np.log(prior_mut_prob / (1 - prior_mut_prob))
    #     copy_of_predictions['post_prob'] = 1 / (1 + np.exp(-copy_of_predictions['z_exact'] - log_prior_odds))
    #     copy_of_predictions['called_mutation'] = copy_of_predictions['post_prob'] > 0.99
    #     # out_pred_file = os.path.join(args.validation_output_dir, f"predictions_with_zlfdr_prior_{prior_str}.csv")
    #     # predictions.to_csv(out_pred_file, index=False)
    #     # logging.info(f"Saved updated predictions for prior {prior_mut_prob} to {out_pred_file}")
    #     mutation_type_counts = aggregate_unique_mutations(copy_of_predictions, mutation_flag=True)
    #     artefact_type_counts = aggregate_unique_mutations(copy_of_predictions, mutation_flag=False)
    #     mutation_type_counts.index = mutation_type_counts.index.map(convert_to_cosmic_format)
    #     artefact_type_counts.index = artefact_type_counts.index.map(convert_to_cosmic_format)
    #     mutation_type_counts = mutation_type_counts.reindex(cosmic_96, fill_value=0)
    #     artefact_type_counts = artefact_type_counts.reindex(cosmic_96, fill_value=0)
    #     cosmic_plot_file = os.path.join(args.validation_output_dir, f"cosmic96_spectra_lattice_prior_mut_{prior_str}.png")
    #     plot_cosmic_spectra(mutation_type_counts, artefact_type_counts, cosmic_96, substitution_types, color_map, save_name=cosmic_plot_file)
    #     mutation_type_counts = mutation_type_counts.reindex(cosmic_96, fill_value=0)
    #     mutation_type_counts.index = [
    #         f"{parts[0]}{parts[1]}{parts[3]}{parts[2]}"
    #         for parts in mutation_type_counts.index.str.split(r'[\[\]>]')
    #     ]
    #     # add sample name as the column header
    #     if args.sample_id:
    #         # mutation_type_counts = mutation_type_counts.rename(args.sample_id)
    #         mutation_type_counts = mutation_type_counts.to_frame(name=args.sample_id)
    #
    #     out_matrix = os.path.join(args.validation_output_dir, f"Trinucleotide_Matrix_prior_{prior_str}.tsv")
    #     mutation_type_counts.to_csv(out_matrix, sep="\t", index=True, header=True)
    #
    #     logging.info(f"Saved combined trinucleotide matrix for prior {prior_mut_prob} to {out_matrix}")

    if 'z_score' not in predictions.columns:
        logging.error("Predictions file does not contain 'z_score' column.")
        return

    for z_score in ["z_score", "z_exact", "bernoulli_prob"]:
        max_z_score = predictions[z_score].max()
        increment = 5
        limit = math.floor(max_z_score) * increment
        z_cutoffs = [x / increment for x in range(limit, limit + increment, 1)]

        for z_cutoff in z_cutoffs:
            # # TODO: skip for now.
            # continue
            copy_of_predictions = predictions.copy()
            # prior_str = str(prior_mut_prob).replace("/", "_").replace(".", "p")
            # logging.info(f"Processing fixed prior: {prior_mut_prob} (prior_str={prior_str})")
            # log_prior_odds = np.log(prior_mut_prob / (1 - prior_mut_prob))
            # copy_of_predictions['post_prob'] = 1 / (1 + np.exp(-copy_of_predictions['z_score'] - log_prior_odds))
            copy_of_predictions['called_mutation'] = copy_of_predictions[z_score] >= z_cutoff
            # out_pred_file = os.path.join(args.validation_output_dir, f"predictions_with_zlfdr_prior_{prior_str}.csv")
            # predictions.to_csv(out_pred_file, index=False)
            # logging.info(f"Saved updated predictions for prior {prior_mut_prob} to {out_pred_file}")
            mutation_type_counts = aggregate_unique_mutations(copy_of_predictions, mutation_flag=True)
            artefact_type_counts = aggregate_unique_mutations(copy_of_predictions, mutation_flag=False)

            mutation_type_counts.index = mutation_type_counts.index.map(convert_to_cosmic_format)
            artefact_type_counts.index = artefact_type_counts.index.map(convert_to_cosmic_format)
            mutation_type_counts = mutation_type_counts.reindex(cosmic_96, fill_value=0)
            artefact_type_counts = artefact_type_counts.reindex(cosmic_96, fill_value=0)
            cosmic_plot_file = os.path.join(args.validation_output_dir, f"cosmic96_spectra_lattice_{z_score}_{z_cutoff}.png")
            plot_cosmic_spectra(mutation_type_counts, artefact_type_counts, cosmic_96, substitution_types, color_map, save_name=cosmic_plot_file)
            # mutation_type_counts = mutation_type_counts.reindex(cosmic_96, fill_value=0)
            mutation_type_counts.index = [
                f"{parts[0]}{parts[1]}{parts[3]}{parts[2]}"
                for parts in mutation_type_counts.index.str.split(r'[\[\]>]')
            ]
            # add sample name as the column header
            if args.sample_id:
                mutation_type_counts = mutation_type_counts.to_frame(name=args.sample_id)

            out_matrix = os.path.join(args.validation_output_dir, f"Trinucleotide_Matrix_{z_score}_{z_cutoff}.tsv")
            mutation_type_counts.to_csv(out_matrix, sep="\t", index=True, header=True)
            logging.info(f"Saved combined trinucleotide matrix for z-score cutoff of {z_cutoff} to {out_matrix}")

            # prior_str = str(prior_mut_prob).replace("/", "_").replace(".", "p")
            # logging.info(f"Processing fixed prior: {prior_mut_prob} (prior_str={prior_str})")
            # log_prior_odds = np.log(prior_mut_prob / (1 - prior_mut_prob))
            # copy_of_predictions['post_prob'] = 1 / (1 + np.exp(-copy_of_predictions['z_score'] - log_prior_odds))

        for top_n in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
            copy_of_predictions = predictions.copy()
            # set called mutation to True for the top n mutations by z_score
            copy_of_predictions['called_mutation'] = False
            copy_of_predictions.loc[copy_of_predictions[z_score].nlargest(top_n).index, 'called_mutation'] = True
            # create a subset of the top n mutations by the z_score
            # copy_of_predictions['called_mutation'] = copy_of_predictions[z_score] >= z_cutoff
            # out_pred_file = os.path.join(args.validation_output_dir, f"predictions_with_zlfdr_prior_{prior_str}.csv")
            # predictions.to_csv(out_pred_file, index=False)
            # logging.info(f"Saved updated predictions for prior {prior_mut_prob} to {out_pred_file}")
            mutation_type_counts = aggregate_unique_mutations(copy_of_predictions, mutation_flag=True)
            artefact_type_counts = aggregate_unique_mutations(copy_of_predictions, mutation_flag=False)

            mutation_type_counts.index = mutation_type_counts.index.map(convert_to_cosmic_format)
            artefact_type_counts.index = artefact_type_counts.index.map(convert_to_cosmic_format)
            mutation_type_counts = mutation_type_counts.reindex(cosmic_96, fill_value=0)
            artefact_type_counts = artefact_type_counts.reindex(cosmic_96, fill_value=0)
            cosmic_plot_file = os.path.join(args.validation_output_dir, f"cosmic96_spectra_lattice_{z_score}_top_{top_n}.png")
            plot_cosmic_spectra(mutation_type_counts, artefact_type_counts, cosmic_96, substitution_types, color_map, save_name=cosmic_plot_file)
            # mutation_type_counts = mutation_type_counts.reindex(cosmic_96, fill_value=0)
            mutation_type_counts.index = [
                f"{parts[0]}{parts[1]}{parts[3]}{parts[2]}"
                for parts in mutation_type_counts.index.str.split(r'[\[\]>]')
            ]
            # add sample name as the column header
            if args.sample_id:
                mutation_type_counts = mutation_type_counts.to_frame(name=args.sample_id)

            out_matrix = os.path.join(args.validation_output_dir, f"Trinucleotide_Matrix_{z_score}_top_{top_n}.tsv")
            mutation_type_counts.to_csv(out_matrix, sep="\t", index=True, header=True)
            logging.info(f"Saved combined trinucleotide matrix for the {top_n} largest {z_score} values to {out_matrix}")
    if args.with_ground_truth and args.reference_fasta:
        # When ground truth is used, use the predictions (with updated mutation types) to plot spectra.
        mutation_type_counts = aggregate_unique_mutations(predictions, mutation_flag=True)
        artefact_type_counts = aggregate_unique_mutations(predictions, mutation_flag=False)
        mutation_type_counts.index = mutation_type_counts.index.map(convert_to_cosmic_format)
        artefact_type_counts.index = artefact_type_counts.index.map(convert_to_cosmic_format)
        mutation_type_counts = mutation_type_counts.reindex(cosmic_96, fill_value=0)
        artefact_type_counts = artefact_type_counts.reindex(cosmic_96, fill_value=0)
        # Check tht there are 96 rows in the mutation type counts
        if len(mutation_type_counts) != 96:
            logging.error(f"Expected 96 mutation types, got {len(mutation_type_counts)}")
            return
        # Ensure the mutation type counts are in the same order as cosmic_96
        cosmic_plot_file = os.path.join(args.validation_output_dir, "cosmic96_spectra_lattice.png")
        plot_cosmic_spectra(mutation_type_counts, artefact_type_counts, cosmic_96, substitution_types, color_map, save_name=cosmic_plot_file)

    # breakpoint()
    # Build the COSMIC 96 ordered list of mutation types (same as before)
    substitution_types = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
    bases = ['A', 'C', 'G', 'T']
    _cosmic_96 = []
    for substitution in substitution_types:
        ref_base, alt_base = substitution.split('>')
        for left in bases:
            for right in bases:
                mutation_type = f"{left}{ref_base}{right}{alt_base}"
                _cosmic_96.append(mutation_type)
    assert len(_cosmic_96) == 96, f"Expected 96 COSMIC mutation types, got {len(_cosmic_96)}"
    color_map = ['#5cb4e6', '#000000', '#e62725', '#bababa', '#70bf41', '#de7b85']

    # Define grid dimensions: 6 rows (one per substitution type) x 16 columns (one per context within each type)
    nrows, ncols = 6, 16
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.2, nrows * 1.2), sharex=True, sharey=True)

    # Loop over each mutation type in cosmic_96 (which is already in the desired order)
    for i, mutation_type in enumerate(_cosmic_96):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]

        # Subset the predictions DataFrame for the current mutation type
        subset = predictions[predictions['mutation_type'] == mutation_type]

        # Determine the substitution category.
        # For a mutation type like "GCGT", the ref is the second character and the alt is the fourth.
        substitution = mutation_type[1] + '>' + mutation_type[3]
        try:
            color_idx = substitution_types.index(substitution)
            color = color_map[color_idx]
        except ValueError:
            color = 'grey'

        # Plot the histogram of z-scores if there are any entries
        if not subset.empty:
            ax.hist(subset['z_score'], bins=100, color=color)

        # Set a small title so you can see the mutation type (optional)
        ax.set_title(mutation_type, fontsize=6)
        ax.tick_params(axis='both', which='both', labelsize=4)

    # Add common axis labels
    fig.text(0.5, 0.04, 'z-score', ha='center', fontsize=8)
    fig.text(0.04, 0.5, 'Frequency', va='center', rotation='vertical', fontsize=8)

    plt.tight_layout(rect=[0.05, 0.05, 1, 1])
    hist_lattice_path = os.path.join(args.validation_output_dir, 'z_score_hist_lattice.png')
    plt.savefig(hist_lattice_path, dpi=300)
    plt.close()

    # (B) Candidate read position analysis using the predictions file and sample BAM.
    if args.bam:
        logging.info("Running candidate read position analysis using predictions and sample BAM.")
        bam_obj = pysam.AlignmentFile(args.bam, "rb")
        # For each prediction, extract candidate alt call position using the read_id.
        for idx, row in predictions.iterrows():
            chrom = str(row["chr"])
            pos = int(row["pos"])
            read_id = row["read_id"]
            try:
                read_info = extract_read_by_id_obj(bam_obj, chrom, pos, read_id)
            except Exception as e:
                logging.warning(f"Error fetching read {read_id} at {chrom}:{pos} - {e}")
                read_info = {}
            if read_info:
                read_info = reorient_read(read_info)
                alt_idx = get_alt_call_index(read_info, pos)
                predictions.loc[idx, "alt_call_position"] = alt_idx
                predictions.loc[idx, "is_first_in_pair"] = read_info["is_first_in_pair"]

        bam_obj.close()

        logging.info("Extracted candidate alt call positions from BAM and updated predictions.")
        # Group by called mutation status if available.
        # if "called_mutation" in predictions.columns:
        #     labels = predictions["called_mutation"].apply(lambda x: "Mutation" if x else "Artefact").tolist()
        # else:
        #     labels = predictions["read_id"].apply(lambda x: "Group 1" if "1" in x else "Group 2").tolist()

        # loop through the three different prior mutation probabilities
        mut_priors = [1/1000, 1/10000, 1/100000]
        if 'z_score' not in predictions.columns:
            logging.error("Predictions file does not contain 'z_score' column.")
            return
        for prior_mut_prob in mut_priors:
            copy_of_predictions = predictions.copy()
            prior_str = str(prior_mut_prob).replace("/", "_").replace(".", "p")
            logging.info(f"Processing fixed prior: {prior_mut_prob} (prior_str={prior_str})")
            log_prior_odds = np.log(prior_mut_prob / (1 - prior_mut_prob))
            copy_of_predictions['post_prob'] = 1 / (1 + np.exp(-copy_of_predictions['z_score'] - log_prior_odds))
            copy_of_predictions['called_mutation'] = copy_of_predictions['post_prob'] > 0.99

            copy_of_predictions = copy_of_predictions[
                (copy_of_predictions["called_mutation"] == True)
                & (copy_of_predictions["alt_call_position"] is not None)
            ]

            labels = copy_of_predictions["is_first_in_pair"]

            heatmap_out = os.path.join(args.validation_output_dir, f"alt_positions_heatmap_{prior_str}.png")
            plot_alt_positions_heatmap(copy_of_predictions["alt_call_position"].tolist(), labels, heatmap_out)
            copy_of_predictions["full_context"] = copy_of_predictions["mutation_category"].apply(convert_to_cosmic_format)
            facet_pdf_out = os.path.join(args.validation_output_dir, f"facet_alt_positions_{prior_str}.pdf")
            plot_facet_alt_positions_heatmap_pdf(copy_of_predictions, facet_pdf_out)

    logging.info("Combined pipeline completed successfully.")


if __name__ == '__main__':
    main()


