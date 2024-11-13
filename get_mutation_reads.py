#!/usr/bin/env python3

import heapq
import argparse
import pypdf.errors
import os
import multiprocessing
import pysam
import pandas as pd
from collections import defaultdict
import shutil
import tempfile
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for HPC or headless environments
import matplotlib.pyplot as plt
from SigProfilerAssignment import Analyzer as Analyze


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Extract mutation supporting reads and compile metadata into a VCF file.'
    )
    parser.add_argument(
        '--mutations_csv', required=True,
        help='Path to the CSV file containing mutations (nfs_muts.csv).'
    )
    parser.add_argument(
        '--sample_links_csv', required=True,
        help='Path to the CSV file mapping sample IDs to BAM file paths (sample_nfs_links.csv).'
    )
    parser.add_argument(
        '--reference_fasta', required=True,
        help='Path to the reference FASTA file.'
    )
    parser.add_argument(
        '--output_bam', required=True,
        help='Output path for the composite BAM file.'
    )
    parser.add_argument(
        '--output_vcf', required=True,
        help='Output path for the composite VCF file.'
    )
    parser.add_argument(
        '--min_mapping_quality', type=int,
        default=30
    )
    parser.add_argument(
        '--num_threads', type=int, default=4,
        help='Number of threads to use (default: 4).'
    )
    parser.add_argument(
        '--temp_dir',
        required=True,
        help='Path to directory where temporary files should be created.'
    )
    args = parser.parse_args()
    return args


def read_mutations(mutations_csv):
    """
    Read mutations from CSV and return a DataFrame.
    """
    mutations = pd.read_csv(mutations_csv)
    return mutations


def read_sample_links(sample_links_csv):
    """
    Read sample IDs and construct BAM file paths based on sample IDs.
    """
    sample_links = pd.read_csv(sample_links_csv)
    sample_to_bam = {}
    for sample_id in sample_links['sample_id'].unique():
        bam_path = os.path.join('/nfs', f'{sample_id}.bam')
        # bam_path = os.path.join('GIAB_BAM/NIST_HiSeq_2500_2x148bp', f'{sample_id}.bam')
        sample_to_bam[sample_id] = bam_path

    # breakpoint()
    return sample_to_bam


def detect_chrom_prefix(bam_file):
    """
    Detect whether chromosome names in a BAM file are prefixed with 'chr'.
    """
    with pysam.AlignmentFile(bam_file, "rb") as bam:
        for ref in bam.references[:10]:  # Check the first 10 references
            if ref.startswith("chr"):
                return "chr"
    return ""


def detect_chrom_prefix_fasta(fasta_file):
    """
    Detect whether chromosome names in a FASTA file are prefixed with 'chr'.
    """
    with pysam.FastaFile(fasta_file) as fasta:
        for ref in fasta.references[:10]:  # Check the first 10 references
            if ref.startswith("chr"):
                return "chr"
    return ""


def classify_mutation(left_context, right_context, ref_base, alt_base):
    """
    Classify a mutation based on the reference and alternate bases, and the
    context of the mutation. Determines the trinucleotide context for
    mutation signature analysis.
    """
    # Convert bases to uppercase to ensure consistency
    ref_base = ref_base.upper()
    alt_base = alt_base.upper()
    left_context = left_context.upper()
    right_context = right_context.upper()

    # Define the six standard mutation types
    mutation_types = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']

    # Complementary bases mapping
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}

    # Get the trinucleotide context
    trinuc = left_context + ref_base + right_context

    # Check for valid nucleotides
    if any(base not in 'ACGT' for base in trinuc + alt_base):
        return None  # Invalid base encountered

    # Normalize to pyrimidine context
    if ref_base in ['C', 'T']:
        mutation = f"{ref_base}>{alt_base}"
        trinuc_context = f"{left_context}[{mutation}]{right_context}"
    else:
        # Complement the trinucleotide and mutation
        ref_base_comp = complement[ref_base]
        alt_base_comp = complement[alt_base]
        left_context_comp = complement[right_context]
        right_context_comp = complement[left_context]
        mutation = f"{ref_base_comp}>{alt_base_comp}"
        trinuc_context = f"{left_context_comp}[{mutation}]{right_context_comp}"

    # Verify that the mutation is one of the standard types
    if mutation not in mutation_types:
        return None  # Invalid mutation type

    return trinuc_context


def process_sample(args_tuple):
    (
        sample_id, sample_mutations, bam_path, reference_fasta, temp_dir,
        chrom_prefix, fasta_prefix, min_mapping_quality
    ) = args_tuple

    sample_mutations = sample_mutations.sort_values(['Chr', 'Pos'])

    # Output file paths
    temp_bam_path = os.path.join(temp_dir, f"{sample_id}_supporting_reads.bam")
    temp_vcf_path = os.path.join(temp_dir, f"{sample_id}.vcf")

    # Check if BAM file exists
    if not os.path.exists(bam_path):
        print(
            f"BAM file {bam_path} for sample {sample_id} not found. Skipping.")
        return {}, temp_bam_path, temp_vcf_path  # Return empty counts

    # Open BAM file
    bam_file = pysam.AlignmentFile(bam_path, 'rb')

    contigs = {
        contig: length
        for contig, length in zip(bam_file.references, bam_file.lengths)
    }

    # Create output BAM file
    output_bam = pysam.AlignmentFile(
        temp_bam_path, 'wb', template=bam_file)

    # Open reference FASTA
    fasta = pysam.FastaFile(reference_fasta)

    # Create VCF header
    vcf_header = create_vcf_header(contigs)

    # Create output VCF file
    vcf_out = pysam.VariantFile(temp_vcf_path, 'w', header=vcf_header)

    mutation_counts = defaultdict(int)

    for idx, mutation in sample_mutations.iterrows():
        chrom = str(mutation['Chr'])
        pos = int(mutation['Pos']) - 1  # Convert to 0-based
        ref = mutation['Ref']
        alt = mutation['Alt']

        chrom_with_prefix = chrom_prefix + chrom

        # Fetch reads overlapping the position
        try:
            reads = bam_file.fetch(chrom_with_prefix, pos, pos + 1)
        # Get reads with a mapping quality greater than
        except ValueError:
            print(
                f"Chromosome {chrom_with_prefix} not found in BAM file for "
                f"sample {sample_id}. Skipping mutation at position {pos + 1}.")
            continue

        supporting_reads = []
        supporting_read_ids = []
        for read in reads:
            if read.is_unmapped or read.is_duplicate:
                continue
            if read.mapping_quality < min_mapping_quality:
                continue
            # Get read base at position
            read_pos = read.get_reference_positions(full_length=False)
            if pos in read_pos:
                idx_in_read = read_pos.index(pos)
                base = read.query_sequence[idx_in_read]
                if base.upper() == alt.upper():
                    supporting_reads.append(read)
                    supporting_read_ids.append(read.query_name)

        if len(supporting_reads) < 2:
            continue

        # Extract context for the mutation

        context_seq = get_trinucleotide_context_sequence(
            fasta, chrom, pos, fasta_prefix)
        if context_seq == 'NNN':
            print(
                f"Could not obtain context for mutation at {chrom}:{pos + 1} in "
                f"sample {sample_id}.")
            continue
        left_base, ref_base_context, right_base = context_seq[0], context_seq[1], context_seq[2]

        # Classify the mutation
        trinuc_context = classify_mutation(
            left_base, right_base, ref_base_context, alt)
        if trinuc_context is None:
            continue  # Skip if classification fails

        # Write supporting reads to per-sample BAM
        for read in supporting_reads:
            output_bam.write(read)

        # Update mutation counts
        mutation_counts[trinuc_context] += 1

        # Create VCF record
        vcf_record = vcf_out.new_record()
        vcf_record.chrom = chrom_with_prefix
        vcf_record.pos = pos + 1  # VCF is 1-based
        vcf_record.id = '.'
        vcf_record.ref = ref
        vcf_record.alts = [alt]
        vcf_record.qual = None
        vcf_record.filter.add('PASS')
        vcf_record.info['SAMPLE'] = sample_id
        vcf_record.info['CONTEXT'] = trinuc_context
        # Add read IDs to INFO field
        vcf_record.info['READ_IDS'] = ','.join(list(set(supporting_read_ids)))
        # Optionally, include total reads at position
        vcf_record.info['TOTAL_READS_AT_POS'] = bam_file.count(
            chrom_with_prefix, pos, pos + 1)

        vcf_out.write(vcf_record)

    bam_file.close()
    output_bam.close()
    vcf_out.close()
    fasta.close()

    return mutation_counts, temp_bam_path, temp_vcf_path


def create_vcf_header(contigs):
    """
    Create a VCF header with necessary INFO fields and contig definitions.
    """
    header = pysam.VariantHeader()
    header.add_line('##fileformat=VCFv4.2')
    header.add_line(
        '##INFO=<ID=SAMPLE,Number=1,Type=String,Description="Sample ID">'
    )
    header.add_line(
        '##INFO=<ID=CONTEXT,Number=1,Type=String,'
        'Description="Trinucleotide context of the mutation">')
    header.add_line(
        '##INFO=<ID=READ_IDS,Number=.,Type=String,'
        'Description="Comma-separated list of read IDs supporting the '
        'mutation">')
    header.add_line(
        '##INFO=<ID=TOTAL_READS_AT_POS,Number=1,Type=Integer,'
        'Description="Total number of reads at the position">')

    # Add contig definitions
    for contig_name, contig_length in contigs.items():
        header.contigs.add(contig_name, length=contig_length)

    return header


def get_trinucleotide_context_sequence(fasta, chrom, pos, chrom_prefix):
    """
    Get the trinucleotide sequence around the mutation.
    """
    chrom_with_prefix = chrom_prefix + chrom
    try:
        seq = fasta.fetch(chrom_with_prefix, pos - 1, pos + 2).upper()
    except ValueError:
        seq = 'NNN'
    if len(seq) == 3:
        return seq
    else:
        return 'NNN'  # If context cannot be obtained


def merge_bam_files(temp_bam_files, output_bam_path):
    """
    Merge per-sample BAM files into a composite BAM file.
    """
    temp_bam_files = [f for f in temp_bam_files if os.path.exists(f)]
    if len(temp_bam_files) == 0:
        print("No BAM files to merge.")
        return
    merged_unsorted_bam = output_bam_path + '.unsorted.bam'
    pysam.merge('-f', merged_unsorted_bam, *temp_bam_files)
    pysam.sort('-o', output_bam_path, merged_unsorted_bam)
    pysam.index(output_bam_path)
    os.remove(merged_unsorted_bam)


# def merge_vcf_files(temp_vcf_files, output_vcf_path):
#     """
#     Merge per-sample VCF files into a composite VCF file.
#     """
#     temp_vcf_files = [f for f in temp_vcf_files if os.path.exists(f)]
#     if len(temp_vcf_files) == 0:
#         print("No VCF files to merge.")
#         return
#
#     merged_unsorted_vcf = output_vcf_path + '.unsorted.vcf'
#
#     # Concatenate VCF files
#     with open(merged_unsorted_vcf, 'w') as outfile:
#         for idx, vcf_file in enumerate(temp_vcf_files):
#             with open(vcf_file, 'r') as infile:
#                 for line in infile:
#                     if line.startswith('#'):
#                         if idx == 0:
#                             outfile.write(line)
#                         continue
#                     outfile.write(line)
#
#     # Sort the merged VCF
#     sorted_vcf = output_vcf_path + '.sorted.vcf'
#     pysam.sort('-o', sorted_vcf, merged_unsorted_vcf)
#     # Compress and index the VCF file
#     pysam.tabix_compress(output_vcf_path, output_vcf_path + '.gz', force=True)
#     pysam.tabix_index(output_vcf_path + '.gz', preset='vcf', force=True)
#
#     os.remove(merged_unsorted_vcf)
#     os.remove(sorted_vcf)


def merge_vcf_files(temp_vcf_files, output_vcf_path):
    """
    Merge per-sample VCF files into a sorted composite VCF file using a k-way
    merge.
    """
    temp_vcf_files = [f for f in temp_vcf_files if os.path.exists(f)]
    if len(temp_vcf_files) == 0:
        print("No VCF files to merge.")
        return

    # Open all temp VCF files
    vcf_iters = []
    for vcf_file in temp_vcf_files:
        vcf = pysam.VariantFile(vcf_file, 'r')
        vcf_iters.append(iter(vcf))

    # Initialize the heap
    heap = []
    for idx, vcf_iter in enumerate(vcf_iters):
        try:
            record = next(vcf_iter)
            heap.append((record.chrom, record.pos, idx, record))
        except StopIteration:
            continue  # Empty VCF file

    # Define a chromosome order function (optional but recommended)
    def chrom_order(chrom):
        try:
            return int(chrom.replace('chr', '').replace('Chr', ''))
        except ValueError:
            # Handle non-numeric chromosomes (e.g., chrX, chrY, chrM)
            if chrom in ['chrX', 'ChrX']:
                return 23
            elif chrom in ['chrY', 'ChrY']:
                return 24
            elif chrom in ['chrM', 'ChrM', 'chrMT', 'ChrMT']:
                return 25
            else:
                return 100  # Arbitrary high number for unknown chromosomes

    # Heapify the initial records
    heapq.heapify(heap)

    # Open the output VCF file
    with pysam.VariantFile(output_vcf_path, 'w', header=vcf_iters[0].header) as out_vcf:
        while heap:
            # Pop the smallest item from the heap
            chrom, pos, idx, record = heapq.heappop(heap)
            out_vcf.write(record)

            # Fetch the next record from the same VCF iterator
            try:
                next_record = next(vcf_iters[idx])
                heapq.heappush(
                    heap,
                    (next_record.chrom, next_record.pos, idx, next_record)
                )
            except StopIteration:
                continue  # No more records in this iterator

    # Compress and index the merged VCF using pysam
    pysam.tabix_compress(output_vcf_path, output_vcf_path + '.gz', force=True)
    pysam.tabix_index(output_vcf_path + '.gz', preset='vcf', force=True)

    # Clean up the uncompressed merged VCF
    os.remove(output_vcf_path)

    print(f"Composite VCF file created and indexed at {output_vcf_path}.gz")


def plot_mutational_spectra(mutation_counts, output_dir, title=""):
    """
    Plot the mutational spectra conforming to the Sanger COSMIC mutational
    spectra format.
    """
    # Define the 96 trinucleotide contexts in COSMIC order
    mutation_types = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
    bases = ['A', 'C', 'G', 'T']
    trinucleotide_contexts = []

    for mut_type in mutation_types:
        for left_base in bases:
            for right_base in bases:
                context = f"{left_base}[{mut_type}]{right_base}"
                trinucleotide_contexts.append(context)

    # Initialize counts for all 96 contexts to zero
    counts = {context: 0 for context in trinucleotide_contexts}

    # Update counts with mutation_counts
    for context, count in mutation_counts.items():
        if context in counts:
            counts[context] += count
        else:
            print(
                f"Warning: Unexpected trinucleotide context '{context}' found "
                f"in mutation_counts.")

    # Create a DataFrame for plotting
    df = pd.DataFrame({
        'Trinucleotide': trinucleotide_contexts,
        'Count': [counts[context] for context in trinucleotide_contexts]
    })

    # Define mutation types and their corresponding colors (COSMIC standard)
    mutation_type_colors = {
        'C>A': '#2EBAED',  # Light blue
        'C>G': '#000000',  # Black
        'C>T': '#DE1F27',  # Red
        'T>A': '#A59B8F',  # Gray
        'T>C': '#98C21D',  # Green
        'T>G': '#ED6BA1'  # Pink
    }

    # Extract mutation type from trinucleotide context
    def get_mutation_type(trinuc):
        # trinuc format: X[Y>Z]W
        return trinuc.split('[')[1].split(']')[0]

    df['Mutation_Type'] = df['Trinucleotide'].apply(get_mutation_type)

    # Assign colors based on mutation type
    df['Color'] = df['Mutation_Type'].map(mutation_type_colors)

    # Create the plot
    plt.figure(figsize=(20, 6))
    bars = plt.bar(
        range(len(df['Trinucleotide'])), df['Count'], color=df['Color'],
        edgecolor='black')

    # Customize the plot
    plt.xlabel('Trinucleotide Context', fontsize=14)
    plt.ylabel('Mutation Count', fontsize=14)
    plt.title(f'{title} Mutational Spectra', fontsize=16)
    plt.xticks(
        range(len(df['Trinucleotide'])), df['Trinucleotide'],
        rotation=90, fontsize=6)

    # Add vertical lines to separate mutation types
    mutation_type_positions = [i * 16 for i in range(1, 6)]  # Positions between mutation types
    for pos in mutation_type_positions:
        plt.axvline(x=pos - 0.5, color='grey', linestyle='--', linewidth=0.5)

    # Add mutation type labels
    for i, mut_type in enumerate(mutation_types):
        plt.text(
            (i * 16) + 8,  # Position in the middle of the group
            max(df['Count']) * 1.05,  # Slightly above the top of the y-axis
            mut_type, ha='center', fontsize=12, fontweight='bold'
        )

    plt.tight_layout()
    # Save the plot to a file
    output_plot_path = os.path.join(output_dir, f"{title}_mutational_spectra.png")
    plt.savefig(output_plot_path, dpi=300)
    print(f"Mutational spectra plot saved to {output_plot_path}")


def assign_known_signatures(
        mutation_counts, output_dir, genome_build='GRCh37', name='Sample1'
):
    """
    Assign known mutational signatures to the observed mutational spectra using SigProfilerAssignment.
    """
    # Prepare input data
    contexts = list(mutation_counts.keys())
    counts = list(mutation_counts.values())
    sample_name = name

    # Create DataFrame
    df = pd.DataFrame({sample_name: counts}, index=contexts)
    df.index.name = 'MutationType'

    # Ensure correct trinucleotide order
    mutation_types = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
    bases = ['A', 'C', 'G', 'T']
    trinucleotide_contexts = [
        f"{left_base}[{mut_type}]{right_base}"
        for mut_type in mutation_types
        for left_base in bases
        for right_base in bases
    ]
    df = df.reindex(trinucleotide_contexts)
    df.fillna(0, inplace=True)

    # Save to file
    input_file = os.path.join(output_dir, f"{name}_mutational_spectra.txt")
    df.to_csv(input_file, sep='\t')

    # Run SigProfilerAssignment
    output_dir_sample = os.path.join(output_dir, name)
    if not os.path.exists(output_dir_sample):
        os.makedirs(output_dir_sample)

    try:
        Analyze.cosmic_fit(
            samples=input_file,
            output=output_dir_sample,
            input_type="matrix",
            genome_build=genome_build,
            context_type="96",
            cosmic_version=3.3,
            make_plots=True,
            sample_reconstruction_plots='pdf',
            verbose=True
        )
    except pypdf.errors.DeprecationError:
        pass


def main():
    args = parse_arguments()

    temp_dir = tempfile.mkdtemp(dir=args.temp_dir)

    # Read mutations and sample links
    mutations = read_mutations(args.mutations_csv)
    sample_to_bam = read_sample_links(args.sample_links_csv)

    # Group mutations by sample
    mutations_by_sample = mutations.groupby('sample_id')

    fasta_prefix = detect_chrom_prefix_fasta(args.reference_fasta)
    # Prepare tasks for multiprocessing
    tasks = []
    for sample_id, sample_mutations in mutations_by_sample:
        if sample_id not in sample_to_bam:
            print(
                f"BAM file for sample {sample_id} not found in sample_to_bam "
                f"mapping. Skipping.")
            continue
        bam_path = sample_to_bam[sample_id]
        chrom_prefix = detect_chrom_prefix(bam_path)
        tasks.append((
            sample_id,
            sample_mutations,
            bam_path,
            args.reference_fasta,
            temp_dir,
            chrom_prefix,
            fasta_prefix,
            args.min_mapping_quality
        ))

    # Process samples in parallel
    pool = multiprocessing.Pool(processes=args.num_threads)
    results = pool.map(process_sample, tasks)
    pool.close()
    pool.join()

    # Process samples sequentially for debugging.
    # results = []
    # for task in tasks:
    #     results.append(process_sample(task))

    # Collect per-sample BAM and VCF files
    temp_bam_files = []
    temp_vcf_files = []
    total_mutation_counts = defaultdict(int)

    for mutation_counts, temp_bam_path, temp_vcf_path in results:
        # Aggregate mutation counts
        for context, count in mutation_counts.items():
            total_mutation_counts[context] += count
        # Collect temp files
        if os.path.exists(temp_bam_path):
            temp_bam_files.append(temp_bam_path)
        if os.path.exists(temp_vcf_path):
            temp_vcf_files.append(temp_vcf_path)

    # Merge per-sample BAM files
    merge_bam_files(temp_bam_files, args.output_bam)

    # Merge per-sample VCF files
    merge_vcf_files(temp_vcf_files, args.output_vcf)

    # Remove temporary directory
    shutil.rmtree(temp_dir)

    print("Processing complete.")
    print(f"Composite BAM file saved to {args.output_bam}")
    print(f"Composite VCF file saved to {args.output_vcf}.gz")

    # Plot mutational spectra
    if total_mutation_counts:
        output_dir = os.path.dirname(args.output_vcf)
        plot_mutational_spectra(total_mutation_counts, output_dir, title="Aggregated")
        # Run SigProfilerAssignment

        assign_known_signatures(
            total_mutation_counts, output_dir, genome_build='GRCh37',
            name='Aggregated_Sample'
        )

    else:
        print("No mutations found to plot mutational spectra.")


if __name__ == '__main__':
    main()

# python get_mutation_reads.py \
#   --mutations_csv TEST_DATA/dummy_nfs_muts.csv \
#   --sample_links_csv TEST_DATA/dummy_sample_nfs.csv \
#   --reference_fasta Reference/hs37d5.fa.gz \
#   --output_bam TEST_DATA/composite.bam \
#   --output_vcf TEST_DATA/composite.vcf \
#   --num_threads 2 \
#   --temp_dir TEST_DATA
