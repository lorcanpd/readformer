#!/usr/bin/env python

import argparse

# import pypdf.errors
import pysam
import concurrent.futures
from collections import defaultdict
import os
import tempfile
import shutil
import matplotlib
# This is needed to make plots on the HPC.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from SigProfilerAssignment import Analyzer as Analyze

"""
Identify likely sequencing artifacts in Illumina BAM file by comparing to 
PacBio BAM file. This script identifies positions where an alternate allele is
supported by exactly one read in the Illumina BAM file, but not supported by
any reads in the PacBio BAM file. The script outputs a BAM file containing the
artifact reads and a VCF file containing the positions of the artifacts.
"""


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Identify likely sequencing artifacts in Illumina BAM file '
                    'by comparing to PacBio BAM file.'
    )
    parser.add_argument(
        '--illumina_bam', required=True,
        help='Path to the Illumina BAM file.'
    )
    parser.add_argument(
        '--pacbio_bam', required=True,
        help='Path to the PacBio BAM file.'
    )
    parser.add_argument(
        '--bed_file', required=True,
        help='Path to the high-confidence BED file.'
    )
    parser.add_argument(
        '--reference_fasta', required=True,
        help='Path to the reference FASTA file.'
    )
    parser.add_argument(
        '--output_bam', required=True,
        help='Output path for the artifact BAM file.'
    )
    parser.add_argument(
        '--output_vcf', required=True,
        help='Output path for the VCF file.'
    )
    parser.add_argument(
        '--mapq_threshold', type=int, default=30,
        help='Mapping quality threshold (default: 30).'
    )
    parser.add_argument(
        '--baseq_threshold', type=int, default=20,
        help='Base quality threshold (default: 20).'
    )
    parser.add_argument(
        '--num_threads', type=int, default=4,
        help='Number of threads to use (default: 4).'
    )
    parser.add_argument(
        '--low_complexity_bed',
        help='Path to BED file of low-complexity regions to exclude.'
    )
    parser.add_argument(
        '--temp_dir',
        required=True,
        help='Path to directory where temporary files should be created.'
    )
    args = parser.parse_args()
    return args


def chrom_sort_key(chrom):
    """
    Sort chromosome names in the order: 1, 2, ..., 22, X, Y, MT
    :param chrom:
        Chromosome name.
    :return:
        Integer key for sorting.
    """
    # Convert chromosome names to integers when possible
    if chrom.isdigit():
        return int(chrom)
    elif chrom == 'X':
        return 23
    elif chrom == 'Y':
        return 24
    elif chrom == 'MT' or chrom == 'M':
        return 25
    else:
        # For other cases, return a high value to place them at the end
        return 100 + hash(chrom)


def detect_chrom_prefix(bam_file):
    """
    Detect whether chromosome names in a BAM file are prefixed with 'chr'.

    :param bam_file: str
        Path to the BAM file.
    :return: str
        'chr' if chromosomes are prefixed with 'chr', otherwise an empty string.
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


def read_bed_file(bed_file):
    """
    Read a BED file and return a list of regions, indicated by the start and end
    positions in each line.
    :param bed_file:
        Path to the BED file.
    :return:
        List of regions as (chrom, start, end) tuples.
    """
    regions = []
    with open(bed_file, 'r') as f:
        for line in f:
            if line.strip() == '':
                continue
            if line.startswith('#'):
                continue
            fields = line.strip().split()
            if len(fields) < 3:
                continue
            chrom, start, end = fields[0], int(fields[1]), int(fields[2])
            regions.append((chrom, start, end))
    # Sort regions by chromosome and start position
    regions.sort(key=lambda x: (chrom_sort_key(x[0]), x[1]))
    return regions


def is_in_low_complexity_region(chrom, pos, low_complexity_regions):
    """
    Check if a position is in a low-complexity region.

    :param chrom:
        chromosome name.
    :param pos:
        position.
    :param low_complexity_regions:
        dictionary of low-complexity regions.
    :return:
        True if the position is in a low-complexity region, False otherwise.
    """
    if low_complexity_regions is None:
        return False
    regions = low_complexity_regions.get(chrom, [])
    for start, end in regions:
        if start <= pos < end:
            return True
    return False


def partition_regions(regions, num_chunks):
    """
    A function to partition regions into chunks of approximately equal size, so
    that each core has a similar amount of workload.

    :param regions:
        The list of regions to partition.
    :param num_chunks:
        The number of chunks to partition the regions into.
    :return:
        A list of chunks, where each chunk is a list of regions.
    """
    # Calculate total size
    total_size = sum(end - start for chrom, start, end in regions)
    target_chunk_size = total_size / num_chunks

    # Partition regions into chunks
    chunks = []
    current_chunk = []
    current_chunk_size = 0
    for region in regions:
        chrom, start, end = region
        region_size = end - start
        if current_chunk_size + region_size > target_chunk_size and len(chunks) < num_chunks - 1:
            # Start a new chunk
            chunks.append(current_chunk)
            current_chunk = [region]
            current_chunk_size = region_size
        else:
            current_chunk.append(region)
            current_chunk_size += region_size
    # Add the last chunk
    chunks.append(current_chunk)
    return chunks


def check_pacbio_support(
        pacbio_bam, chrom, pos, alt_base, mapq_threshold, baseq_threshold
):
    """
    Check if the alternate allele is supported by PacBio data at a given
    position.

    :param pacbio_bam:
        Path to the PacBio BAM file.
    :param chrom:
        Chromosome name.
    :param pos:
        Position.
    :param alt_base:
        Alternate base.
    :param mapq_threshold:
        Mapping quality threshold.
    :param baseq_threshold:
        Base quality threshold.
    :return:
        Tuple of (reference count, alternate count, total coverage) if the
        alternate allele is supported, None otherwise.
    """
    # Initialise counts
    pacbio_base_counts = defaultdict(int)
    total_coverage = 0

    # Get pileup at this position in PacBio BAM
    try:
        pileupcolumns = pacbio_bam.pileup(
            chrom,
            pos,
            pos + 1,
            min_mapping_quality=mapq_threshold,
            truncate=True,
            stepper='samtools',
        )
    except ValueError as e:
        # Handle case where region is invalid
        return None

    for pileupcolumn in pileupcolumns:
        if pileupcolumn.pos != pos:
            continue  # Not the position we're interested in
        for pileupread in pileupcolumn.pileups:
            if pileupread.is_del or pileupread.is_refskip:
                continue
            if pileupread.query_position is None:
                continue  # Skip if query position is None
            read = pileupread.alignment
            if read.mapping_quality < mapq_threshold:
                continue
            base = read.query_sequence[pileupread.query_position]
            baseq = read.query_qualities[pileupread.query_position]
            if baseq < baseq_threshold:
                continue
            pacbio_base_counts[base] += 1
            total_coverage += 1
    if total_coverage < 30:
        return None  # Coverage too low
    pacbio_alt_count = pacbio_base_counts.get(alt_base, 0)
    pacbio_ref_count = total_coverage - pacbio_alt_count
    return pacbio_ref_count, pacbio_alt_count, total_coverage


def create_vcf_record(chrom, pos, base_counts, ref_base, alt_base, illumina_read_name, position_on_read, pacbio_support):
    """
    Create a VCF record for an artifact position.

    :param chrom:
        Chromosome name.
    :param pos:
        Position.
    :param base_counts:
        Dictionary of base counts at the position.
    :param ref_base:
        Reference base.
    :param alt_base:
        Alternate base.
    :param illumina_read_name:
        Read name supporting the alternate allele.
    :param position_on_read:
        Position of the alternate allele on the read.
    :param pacbio_support:
        Tuple of (reference count, alternate count, total coverage) from PacBio.
    :return:
        Dictionary representing the VCF record.
    """
    if ref_base is None:
        ref_base = 'N'

    illumina_ref_count = sum(
        count for base, count in base_counts.items() if base != alt_base)
    # get key of the highest value in the dictionary
    illumina_alt_count = base_counts[alt_base]
    illumina_total_coverage = sum(base_counts.values())

    pacbio_ref_count, pacbio_alt_count, pacbio_total_coverage = pacbio_support
    # get the position of the base on the read (e.g. the first element on a
    # read is 0)

    # Build VCF record
    vcf_record = {
        'chrom': chrom,
        'pos': pos + 1,  # VCF is 1-based
        'id': '.',
        'ref': ref_base,
        'alt': alt_base,
        'qual': '.',
        'filter': '.',
        'info': '.',
        'illumina_ref_count': illumina_ref_count,
        'illumina_alt_count': illumina_alt_count,
        'illumina_total_coverage': illumina_total_coverage,
        'illumina_read_name': illumina_read_name,
        'position_on_read': position_on_read,
        'pacbio_ref_count': pacbio_ref_count,
        'pacbio_alt_count': pacbio_alt_count,
        'pacbio_total_coverage': pacbio_total_coverage
    }
    return vcf_record


def write_vcf_record(vcf_file, record):
    """
    Write a VCF record to a file.

    :param vcf_file:
        The VCF file object.
    :param record:
        Dictionary representing the VCF record.
    :return:
        None
    """
    info_fields = [
        f"ILLUMINA_REF_COUNT={record['illumina_ref_count']}",
        f"ILLUMINA_ALT_COUNT={record['illumina_alt_count']}",
        f"ILLUMINA_TOTAL_COVERAGE={record['illumina_total_coverage']}",
        f"ILLUMINA_READ_NAME={record['illumina_read_name']}",
        f"ELEMENT_ON_READ={record['position_on_read']}",
        f"PACBIO_REF_COUNT={record['pacbio_ref_count']}",
        f"PACBIO_ALT_COUNT={record['pacbio_alt_count']}",
        f"PACBIO_TOTAL_COVERAGE={record['pacbio_total_coverage']}"
    ]
    info = ';'.join(info_fields)
    vcf_line = '\t'.join([
        record['chrom'],
        str(record['pos']),
        record['id'],
        record['ref'],
        record['alt'],
        record['qual'],
        record['filter'],
        info
    ])
    vcf_file.write(vcf_line + '\n')


def classify_mutation(left_context, right_context, ref_base, alt_base):
    """
    Classify a mutation based on the reference and alternate bases, and the
    context of the mutation. Determines the trinucleotide context for
    mutation signature analysis.

    :param left_context: str
        Allele on the left side of the mutation (preceding base).
    :param right_context: str
        Allele on the right side of the mutation (following base).
    :param ref_base: str
        Reference base at the mutation site.
    :param alt_base: str
        Alternate base at the mutation site.
    :return: str or None
        Trinucleotide context for the mutation (e.g., 'A[C>A]A') or
        None if the mutation type is invalid.
    """

    # Convert bases to uppercase to ensure consistency
    ref_base = ref_base.upper()
    alt_base = alt_base.upper()

    # Define the six standard mutation types
    mutation_types = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']

    # Complementary bases mapping
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}

    trinuc_context = None  # Initialize trinuc_context

    # Case 1: Reference base is 'C' or 'T'
    if ref_base in ['C', 'T']:
        mutation = f"{ref_base}>{alt_base}"
        if mutation not in mutation_types:
            # If mutation is not standard, take complement
            try:
                ref_base_comp = complement[ref_base]
                alt_base_comp = complement[alt_base]
            except KeyError:
                # Invalid base encountered
                return None
            mutation = f"{ref_base_comp}>{alt_base_comp}"
            trinuc_context = f"{left_context}[{mutation}]{right_context}"
        else:
            # Standard mutation type; assign trinuc_context directly
            trinuc_context = f"{left_context}[{mutation}]{right_context}"
    else:
        # Case 2: Reference base is not 'C' or 'T'; complement to represent pyrimidine context
        try:
            ref_base_comp = complement[ref_base]
            alt_base_comp = complement[alt_base]
        except KeyError:
            # Invalid base encountered
            return None
        mutation = f"{ref_base_comp}>{alt_base_comp}"
        trinuc_context = f"{complement.get(right_context, 'N')}[{mutation}]{complement.get(left_context, 'N')}"

    # Verify that the mutation is one of the standard types
    if mutation not in mutation_types:
        return None  # Invalid mutation type

    # Construct the trinucleotide key (e.g., 'A[C>A]A')
    if len(trinuc_context) < 7:
        # Trinucleotide context is too short; invalid format
        return None

    trinuc_key = f"{trinuc_context[0]}[{mutation}]{trinuc_context[6]}"

    # Validate the format of trinuc_key
    if len(trinuc_key) != 7 or trinuc_key[1] != '[' or trinuc_key[5] != ']':
        return None

    return trinuc_key



def get_majority_base_from_reads(
        illumina_bam, chrom, pos, mapq_threshold, baseq_threshold,
        exclude_read_name
):
    """
    Get the majority base at a given position from the reads in the BAM file.

    :param illumina_bam:
        Path to the Illumina BAM file.
    :param chrom:
        Chromosome name.
    :param pos:
        Position.
    :param mapq_threshold:
        Mapping quality threshold.
    :param baseq_threshold:
        Base quality threshold.
    :param exclude_read_name:
        Read name to exclude from the base count.
    :return:
        Majority base at the position, or None if no base is supported.
    """
    base_counts = defaultdict(int)
    try:
        for pileupcolumn in illumina_bam.pileup(
                chrom, pos, pos + 1, min_base_quality=baseq_threshold,
                min_mapping_quality=mapq_threshold, truncate=True
        ):
            for pileupread in pileupcolumn.pileups:
                if pileupread.is_del or pileupread.is_refskip:
                    continue
                read = pileupread.alignment
                if read.query_name == exclude_read_name:
                    continue  # Exclude the artifact read
                if read.mapping_quality < mapq_threshold:
                    continue
                base = read.query_sequence[pileupread.query_position]
                baseq = read.query_qualities[pileupread.query_position]
                if baseq < baseq_threshold:
                    continue
                base_counts[base.upper()] += 1
            break  # We only need the first pileupcolumn
    except ValueError:
        return None  # Position invalid or no coverage
    if base_counts:
        majority_base = max(base_counts.items(), key=lambda x: x[1])[0]
        return majority_base
    else:
        return None


def process_chunk(
        args_tuple
):
    (
        chunk, args, temp_dir, chunk_id, first_bam_chrom_prefix,
        second_bam_chrom_prefix, reference_fasta, fasta_prefix
    ) = args_tuple
    illumina_bam_path = args.illumina_bam
    pacbio_bam_path = args.pacbio_bam
    mapq_threshold = args.mapq_threshold
    baseq_threshold = args.baseq_threshold
    low_complexity_regions = args.low_complexity_regions  # dict of chrom -> list of (start, end)

    # Open BAM files
    illumina_bam = pysam.AlignmentFile(illumina_bam_path, 'rb')
    pacbio_bam = pysam.AlignmentFile(pacbio_bam_path, 'rb')

    # Open reference FASTA file
    reference_fasta = pysam.FastaFile(reference_fasta)


    # Temporary BAM and VCF files for this chunk
    temp_bam_path = os.path.join(temp_dir, f"temp_{chunk_id:09d}.bam")
    temp_bam = pysam.AlignmentFile(
        temp_bam_path, 'wb', template=illumina_bam)
    temp_vcf_path = os.path.join(temp_dir, f"temp_{chunk_id:09d}.vcf")
    temp_vcf_file = open(temp_vcf_path, 'w')

    # Write VCF header
    temp_vcf_file.write('##fileformat=VCFv4.2\n')
    temp_vcf_file.write('#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n')

    artefact_count = 0
    mutation_counts = defaultdict(int)

    for region in chunk:
        chrom, start, end = region
        illumina_chrom = first_bam_chrom_prefix + chrom
        pacbio_chrom = second_bam_chrom_prefix + chrom

        # Fetch Illumina pileup columns in this region
        for pileupcolumn in illumina_bam.pileup(
                illumina_chrom, start, end, min_base_quality=baseq_threshold,
                min_mapping_quality=mapq_threshold, truncate=True
        ):
            # TODO ADD GLOBAL NAMING CONVENTION CHECK.
            pos = pileupcolumn.pos
            # Check if position is in low-complexity region
            if is_in_low_complexity_region(chrom, pos, low_complexity_regions):
                continue
            # Get base counts
            base_counts = defaultdict(int)
            read_bases = {}
            reads_at_pos = {}
            read_positions = {}
            for pileupread in pileupcolumn.pileups:
                if pileupread.is_del or pileupread.is_refskip:
                    continue
                read = pileupread.alignment
                if read.mapping_quality < mapq_threshold:
                    continue
                base = read.query_sequence[pileupread.query_position]
                baseq = read.query_qualities[pileupread.query_position]
                if baseq < baseq_threshold:
                    continue
                base_counts[base] += 1
                read_bases[read.query_name] = base
                reads_at_pos[read.query_name] = read  # Store the read
                read_positions[read.query_name] = pileupread.query_position

            # Identify positions where alternate alleles are supported by exactly one read
            if len(base_counts) > 1:
                # Check that illumina coverage is at least 30
                if sum(base_counts.values()) >= 40:
                    for base, count in base_counts.items():
                        if count == 1:
                            # This base is supported by exactly one read
                            # Get the read name supporting this base
                            for read_name, read_base in read_bases.items():
                                if read_base == base:
                                    # Now check PacBio data at this position
                                    pacbio_support = check_pacbio_support(
                                        pacbio_bam, pacbio_chrom, pos, base,
                                        mapq_threshold, baseq_threshold
                                    )
                                    if pacbio_support is None:
                                        # PacBio coverage too low, skip
                                        continue
                                    else:
                                        (
                                            pacbio_ref_count, pacbio_alt_count,
                                            pacbio_total_coverage
                                        ) = pacbio_support
                                        if pacbio_alt_count == 0 and pacbio_total_coverage >= 50:
                                            # No support in PacBio, sufficient coverage
                                            # Collect read and VCF record
                                            artifact_read = reads_at_pos[read_name]
                                            # Get the index of the base on the read
                                            # position_on_read = artifact_read.get_reference_positions().index(pos)
                                            position_on_read = read_positions[read_name]

                                            # Determine the 5' end of the read
                                            if artifact_read.is_reverse:
                                                # Check position on the read is
                                                # within 100bp of the 5' end, which
                                                # if the read is reversed is the end
                                                # of the read
                                                read_length = len(artifact_read.query_sequence)
                                                if read_length - 100 > position_on_read:
                                                    continue
                                            else:
                                                # Check position on the read is
                                                # within 100bp of the 5' end
                                                if position_on_read >= 100:
                                                    continue

                                            majority_base = max(base_counts, key=base_counts.get)
                                            # Get tri-nucleotide context from the reference
                                            trinuc = reference_fasta.fetch(chrom, pos - 1, pos + 2)
                                            ref_base = trinuc[1]

                                            # If ref base doesn't match the majority base, skip
                                            if ref_base != majority_base:
                                                continue

                                            left_base = trinuc[0]
                                            right_base = trinuc[2]

                                            # Check if we have valid bases
                                            if None in [left_base, right_base, ref_base]:
                                                continue

                                            temp_bam.write(artifact_read)

                                            # Classify the mutation
                                            mutation_type = classify_mutation(
                                                left_base, right_base, ref_base,
                                                base
                                            )
                                            if mutation_type is not None:
                                                mutation_counts[mutation_type] += 1
                                            vcf_record = create_vcf_record(
                                                chrom, pos, base_counts,
                                                ref_base, base,
                                                read_name, position_on_read,
                                                pacbio_support
                                            )
                                            write_vcf_record(temp_vcf_file, vcf_record)
                                            artefact_count += 1
                                    break  # Only need to process the one read supporting this base
    illumina_bam.close()
    pacbio_bam.close()
    temp_bam.close()
    temp_vcf_file.close()
    return artefact_count, mutation_counts


def merge_bam_files(temp_bam_files, output_bam_path):
    # Merge the temporary BAM files into an unsorted BAM file
    unsorted_bam_path = output_bam_path + '.unsorted.bam'
    pysam.merge('-f', unsorted_bam_path, *temp_bam_files)

    # Sort the merged BAM file
    sorted_bam_path = output_bam_path  # Use the desired output path
    pysam.sort('-o', sorted_bam_path, unsorted_bam_path)

    # Index the sorted BAM file
    pysam.index(sorted_bam_path)

    # Remove the unsorted merged BAM file
    os.remove(unsorted_bam_path)


def merge_vcf_files(temp_vcf_files, output_vcf_path):

    temp_vcf_files.sort()

    with open(output_vcf_path, 'w') as outfile:
        # Write the VCF header once
        header = (
            '##fileformat=VCFv4.2\n'
            '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n'
        )
        outfile.write(header)

        # Copy contents of each temp VCF file to the output file
        for vcf_file in temp_vcf_files:
            with open(vcf_file, 'r') as infile:
                shutil.copyfileobj(infile, outfile)


def plot_mutational_spectra(mutation_counts, out_dir, title=""):
    """
    Plot the mutational spectra conforming to the Sanger COSMIC mutational
    spectra format.

    :param mutation_counts: dict
        Dictionary mapping trinucleotide contexts (e.g., 'A[C>A]A') to counts.
    :return: None
    """

    # Define the 96 trinucleotide contexts in COSMIC order
    mutation_types = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
    bases = ['A', 'C', 'G', 'T']
    trinucleotide_contexts = []

    for mut_type in mutation_types:
        ref_base = mut_type[0]
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
                f"Warning: Unexpected trinucleotide context '{context}' "
                f"found in mutation_counts.")

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
        'T>G': '#ED6BA1'   # Pink
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
        range(len(df['Trinucleotide'])), df['Count'], color=df['Color'], edgecolor='black')

    # Customize the plot
    plt.xlabel('Trinucleotide Context', fontsize=14)
    plt.ylabel('Mutation Count', fontsize=14)
    plt.title('Mutational Spectra', fontsize=16)
    plt.xticks(range(len(df['Trinucleotide'])), df['Trinucleotide'], rotation=90, fontsize=6)

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

    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color, edgecolor='black', label=mut_type)
        for mut_type, color in mutation_type_colors.items()
    ]
    # plt.legend(
    #     handles=legend_elements, title='Mutation Type',
    #     bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add title to the plot
    plt.title(f"{title} mutational spectra", fontsize=16)

    plt.tight_layout()
    # plt.show()

    # Uncomment the following lines to save the plot to a file
    output_plot_path = f"{out_dir}/artefact_mutational_spectra.png"
    plt.savefig(output_plot_path, dpi=300)
    print(f"Mutational spectra plot saved to {output_plot_path}")


def assign_known_signatures(
        mutation_counts, output_dir, genome_build='GRCh37', name='Sample1',
        exclude_non_artefact=False
):
    """
    Assign known mutational signatures, including artefact signatures, to the
    observed mutational spectra.

    Parameters:
    - mutation_counts: dict
        Dictionary mapping trinucleotide contexts (e.g., 'A[C>A]A') to counts.
    - output_dir: str
        Path to the directory where results will be saved.
    - genome_build: str
        Genome build ('GRCh37' or 'GRCh38'). Default is 'GRCh37'.

    Returns:
    - None
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
    input_file = f"{output_dir}/{name}_known_mutational_spectra.txt"
    df.to_csv(input_file, sep='\t')

    os.mkdir(f"{output_dir}/{name}")

    # Run SigProfilerAssignment
    if exclude_non_artefact:
        exclude_subgroups =[
                'MMR_deficiency_signatures', 'POL_deficiency_signatures',
                'HR_deficiency_signatures', 'BER_deficiency_signatures',
                'Chemotherapy_signatures', 'Immunosuppressants_signatures',
                'Treatment_signatures',  # 'APOBEC_signatures',
                'Tobacco_signatures', 'UV_signatures', 'AA_signatures',
                'Colibactin_signatures',  # 'Artifact_signatures'
                'Lymphoid_signatures'
            ]
    else:
        exclude_subgroups = []

    try:
        Analyze.cosmic_fit(
            samples=input_file,
            output=f"{output_dir}/{name}",
            input_type="matrix",
            genome_build=genome_build,
            context_type="96",
            cosmic_version=3.3,
            make_plots=True,
            sample_reconstruction_plots='pdf',
            verbose=True,
            exclude_signature_subgroups=exclude_subgroups
        )
    except Exception as e:
        pass


def main():
    args = parse_arguments()

    if args.temp_dir:
        temp_dir = tempfile.mkdtemp(dir=args.temp_dir)
    else:
        raise ValueError("Temporary directory not specified.")

    # detect if chromosome names are prefixed with 'chr' in the BAM files
    illumina_chrom_prefix = detect_chrom_prefix(args.illumina_bam)
    pacbio_chrom_prefix = detect_chrom_prefix(args.pacbio_bam)
    # Read BED files
    high_conf_regions = read_bed_file(args.bed_file)  # TODO: If testing locally take a subset of this.

    reference_fasta = args.reference_fasta

    reference_prefix = detect_chrom_prefix_fasta(reference_fasta)

    if args.low_complexity_bed:
        low_complexity_regions = read_bed_file(args.low_complexity_bed)
    else:
        low_complexity_regions = None

    args.low_complexity_regions = low_complexity_regions  # Pass it in args

    chunks = partition_regions(high_conf_regions, args.num_threads)

    # Prepare arguments for process_chromosome
    tasks = []
    for i, chunk in enumerate(chunks):
        tasks.append(
            (
                chunk, args, temp_dir, i, illumina_chrom_prefix,
                pacbio_chrom_prefix, reference_fasta, reference_prefix
            )
        )

    total_artefact_counts = []
    total_mutation_counts = defaultdict(int)
    # Process chromosomes in parallel
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.num_threads) as executor:
        results = executor.map(
            process_chunk, tasks)
    for artefact_count, mutation_count in results:
        total_artefact_counts.append(artefact_count)
        for mutation_type, count in mutation_count.items():
            total_mutation_counts[mutation_type] += count

    # # Non-parallel version for debugging
    # for task in tasks[3:]:
    #     artefact_count, mutation_count = process_chunk(
    #         task
    #     )
    #     total_artefact_counts.append(artefact_count)
    #     for mutation_type, count in mutation_count.items():
    #         total_mutation_counts[mutation_type] += count

    # Collect per-chunk BAM and VCF files
    temp_bam_files = [
        os.path.join(temp_dir, f"temp_{i:09d}.bam") for i in range(len(chunks))]
    temp_vcf_files = [
        os.path.join(temp_dir, f"temp_{i:09d}.vcf") for i in range(len(chunks))]

    if temp_bam_files:
        merge_bam_files(temp_bam_files, args.output_bam)
    else:
        print("No artifact reads found. Output BAM file will not be created.")

    if temp_vcf_files:
        merge_vcf_files(temp_vcf_files, args.output_vcf)
    else:
        print("No artifact reads found. Output VCF file will not be created.")

    # Remove temporary directory
    shutil.rmtree(temp_dir)

    # BGZIP compress the VCF file
    bgzipped_vcf = args.output_vcf + '.gz'
    pysam.tabix_compress(args.output_vcf, bgzipped_vcf, force=True)

    # TABIX index the compressed VCF file
    pysam.tabix_index(bgzipped_vcf, preset='vcf', force=True)

    # Remove the uncompressed VCF file
    os.remove(args.output_vcf)

    # Print total number of candidates found
    total_candidates = sum(total_artefact_counts)
    print("Total number of candidates found:", total_candidates)

    print("Mutation counts:")
    for mutation_type in sorted(total_mutation_counts.keys()):
        print(f"{mutation_type}: {total_mutation_counts[mutation_type]}")

    if total_candidates > 0:
        # Plot the mutational spectra
        plot_mutational_spectra(
            total_mutation_counts, args.temp_dir,
            title="candidate illumina artefacts"
        )

        # Assign known signatures
        assign_known_signatures(
            total_mutation_counts, args.temp_dir, genome_build='GRCh37',
            name="only_artefact_signatures", exclude_non_artefact=True
        )
        assign_known_signatures(
            total_mutation_counts, args.temp_dir, genome_build='GRCh37',
            name="all_signatures", exclude_non_artefact=False
        )


if __name__ == '__main__':
    import time
    start = time.time()
    main()
    end = time.time()
    # Print time take in hours, minutes and seconds
    print(
        f"Time taken (H:M:S): "
        f"{time.strftime('%H:%M:%S', time.gmtime(end - start))}, "
        f" or {end - start:.2f} seconds."
    )
