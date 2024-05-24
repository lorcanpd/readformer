
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pysam
import re
import random


def estimate_initial_window(
        bam_file, chromosome, position, nucleotide_threshold, initial_window=100
) -> int:
    """
    Estimate an initial search window size based on the desired nucleotide
    threshold, to ensure that the number of nucleotides fetched from the BAM
    file is just below the threshold.
    """
    # Estimate read density around the position
    start, end = position - initial_window, position + initial_window
    total_nucleotides_in_window = sum(
        read.query_alignment_length for read
        in bam_file.fetch(chromosome, start, end)
    )
    window_size = end - start

    if window_size > 0:
        density_per_base = total_nucleotides_in_window / window_size
    else:
        density_per_base = 0
    if density_per_base > 0:
        estimated_needed_window = 0.85 * nucleotide_threshold / density_per_base
    else:
        estimated_needed_window = estimate_initial_window(
            bam_file, chromosome, position, nucleotide_threshold,
            initial_window * 2
        )

    return int(estimated_needed_window)


def extract_reads_from_position_onward(
        bam_file_path, chromosome, position, nucleotide_threshold
) -> dict:
    """
    Iteratively extract reads from a BAM file starting at a given position,
    until a nucleotide threshold is reached. A sample coverage can be specified
    to randomly select reads to match the desired coverage.

    :param bam_file_path: Path to the BAM file.
    :param chromosome: Chromosome name.
    :param position: Starting genomic position.
    :param nucleotide_threshold: Nucleotide coverage threshold.

    :return: A dictionary of reads fetched from the BAM file.
    """

    bam_file = pysam.AlignmentFile(bam_file_path, "rb")
    fetched_reads = {}
    total_nucleotides = 0

    for read in bam_file.fetch(
            chromosome, position, position + nucleotide_threshold
    ):
        if total_nucleotides + read.query_length > nucleotide_threshold:
            break  # Stop if the next read exceeds the threshold

        if read.query_name not in fetched_reads:  # Check for read ID instead of read object
            fetched_reads[read.query_name] = {
                "bitwise_flags": read.flag,
                "reference_start": read.reference_start,
                "mapping_quality": read.mapping_quality,
                "cigar": read.cigarstring,
                "template_length": read.template_length,
                "query_sequence": read.query_sequence,
                "query_qualities": read.query_qualities,
                "tags": read.tags
            }
            total_nucleotides += read.query_length

    bam_file.close()
    return fetched_reads


def extract_reads_around_position(
        bam_file_path, chromosome, position, nucleotide_threshold
) -> dict:
    """
    Extract reads and their metadata from a BAM file around a given genomic
    position on a given chromosome, until the number of nucleotides represented
    reaches a given threshold. This includes expanding the search window and
    including reads adjacent to the given position if necessary.

    :param bam_file_path:
        Path to the BAM file.
    :param chromosome:
        Chromosome name.
    :param position:
        Genomic position.
    :param nucleotide_threshold:
        The minimum number of nucleotides covered by the fetched reads.

    :return:
        A list of reads fetched from the BAM file.
    """
    bam_file = pysam.AlignmentFile(bam_file_path, "rb")

    window = estimate_initial_window(
        bam_file, chromosome, position, nucleotide_threshold
    )
    start, end = position - window, position + window

    fetched_reads = set()
    for read in bam_file.fetch(chromosome, start, end):
        fetched_reads.add(read)

    bam_file.close()

    # Construct the dictionary from the fetched reads
    read_dict = {}
    for read in fetched_reads:
        read_dict[read.query_name] = {
            "bitwise_flags": read.flag,
            # "reference_name": bam_file.get_reference_name(read.reference_id),
            "reference_start": read.reference_start,
            "mapping_quality": read.mapping_quality,
            "cigar": read.cigarstring,
            # "next_reference_name": bam_file.get_reference_name(
            #     read.next_reference_id) if read.next_reference_id >= 0 else None,
            # "next_reference_start": read.next_reference_start,
            "template_length": read.template_length,
            "query_sequence": read.query_sequence,
            "query_qualities": read.query_qualities,
            "tags": read.tags
        }

    return read_dict


def decode_binary_flags_to_vector(flags, num_flags=12):
    """
    Convert bitwise flags to a binary feature vector.

    :param flags: Integer representing the bitwise flags from a SAM/BAM read
        alignment.
    :param num_flags: The number of flags to consider (default is 12, covering
        standard SAM flags).

    :return binary_vector: A list of integers (1 or 0) representing the binary
        feature vectors. The elemnents represent, in order, the following
        features:
            - 0: read paired
            - 1: read mapped in proper pair
            - 2: read unmapped
            - 3: mate unmapped
            - 4: read reverse strand
            - 5: mate reverse strand
            - 6: first in pair
            - 7: second in pair
            - 8: not primary alignment
            - 9: read fails platform/vendor quality checks
            - 10: read is PCR or optical duplicate
            - 11: supplementary alignment
    """
    binary_vector = [(flags >> i) & 1 for i in range(num_flags)]
    return binary_vector


# def decode_bq_to_numerical(bq_string):
#     """
#     Decode a 'BQ' quality score string (ASCII characters) into a numerical
#     vector of Phred scores.
#
#     :param bq_string: A string representing the 'BQ' quality scores in ASCII
#         encoding.
#
#     :return numerical_scores: A list of integers representing the Phred quality
#         scores.
#     """
#     # Convert each character in the BQ string to its Phred quality score
#     numerical_scores = [ord(char) - 33 for char in bq_string]
#     # convert to probabilities
#     numerical_scores = [10 ** (-score / 10) for score in numerical_scores]
#     return numerical_scores


def cigar_to_binary_vector(cigar, read_length):
    """
    Decode a CIGAR string into a binary vector.

    :param cigar: The CIGAR string.
    :param read_length: The length of the read.

    :return binary_vector: A binary list where 1 indicates an aligned base and
        0 indicates a non-aligned base.
    """
    match_vector = [0] * read_length
    insertion_vector = [0] * read_length
    soft_clipping_vector = [0] * read_length

    current_position = 0  # Track current position in the read
    for length, op in re.findall('(\d+)([MID])', cigar):
        length = int(length)
        if op == 'M':
            for i in range(length):
                if current_position + i < read_length:
                    match_vector[current_position + i] = 1
            current_position += length
        elif op == 'I':
            # Insertion affects the position before it occurs
            if current_position < read_length:
                insertion_vector[current_position] = 1
            # No increment to current_position since insertions don't consume
            # reference bases
        elif op == 'S':
            for i in range(length):
                if current_position + i < read_length:
                    soft_clipping_vector[current_position + i] = 1
            current_position += length

    return match_vector, insertion_vector


def sample_positions(n, sex):
    """
    Sample genomic positions from the human genome.

    :param n:
        The number of positions to sample.
    :param sex:
        The sex of the individual from which to sample the positions.
    :return:
        A list of sampled genomic positions.
    """
    chromosome_ranges = {
        '1': (1, 248956422),
        '2': (2, 242193529),
        '3': (3, 198295559),
        '4': (4, 190214555),
        '5': (5, 181538259),
        '6': (6, 170805979),
        '7': (7, 159345973),
        '8': (8, 145138636),
        '9': (9, 138394717),
        '10': (10, 133797422),
        '11': (11, 135086622),
        '12': (12, 133275309),
        '13': (13, 114364328),
        '14': (14, 107043718),
        '15': (15, 101991189),
        '16': (16, 90338345),
        '17': (17, 83257441),
        '18': (18, 80373285),
        '19': (19, 58617616),
        '20': (20, 64444167),
        '21': (21, 46709983),
        '22': (22, 50818468),
        'X': (23, 156040895)
    }
    if sex == "male":
        chromosome_ranges['Y'] = (24, 57227415)

    # Randomly sample positions from the genome giving each chromosome a probability
    # proportional to its length
    chromosome_probs = np.array([val[1] for val in chromosome_ranges.values()])
    chromosome_probs = chromosome_probs / chromosome_probs.sum()
    chromosome_names = list(chromosome_ranges.keys())
    random_chromosomes = np.random.choice(
        chromosome_names, size=n, p=chromosome_probs
    )
    # Randomly sample positions (given by the chromosome_ranges) from each chromosome
    random_positions = [
        np.random.randint(chromosome_ranges[chromosome][1])
        for chromosome in random_chromosomes
    ]

    return list(zip(random_chromosomes, random_positions))


def get_read_info(read_dict):
    """
    Extracts information from the read dictionary and returns a new dictionary
    containing only the necessary information for the model.

    :param read_dict:
        A dictionary containing read information.
    :return final_dict:
        A dictionary containing only the necessary information for the model.
    """
    final_dict = {}
    for read, values in read_dict.items():

        cigar_match_vector, cigar_insertion_vector = cigar_to_binary_vector(
            values['cigar'], len(values['query_sequence'])
        )
        final_dict[read] = {
            # Convert mapping quality to a probability.
            'mapping_quality': 10 ** (-values['mapping_quality'] / 10),
            'query_sequence': values['query_sequence'],
            'binary_flag_vector':  decode_binary_flags_to_vector(
                values['bitwise_flags']
            ),
            # Convert base qualities to probabilities.
            'base_qualities': [
                10 ** (-bq / 10) for bq in values['query_qualities']
            ],
            'cigar_match_vector': cigar_match_vector,
            'cigar_insertion_vector': cigar_insertion_vector,
            'positions': list(
                range(
                    values['reference_start'],
                    values['reference_start'] + len(values['query_sequence'])
                )
            )

        }

    return final_dict



# #
# bam_file_path = 'TEST_DATA/HG002.GRCh38.2x250.bam'
#
# positions = sample_positions(2, sex='male')
#
# read_dict = extract_reads_from_position_onward(
#     bam_file_path, 'chr' + positions[1][0], positions[1][1], 1024*12
# )
#
# read_info = get_read_info(read_dict)
#
# from components.plots import plot_nucleotide_coverage, plot_mapping_and_base_quality_histogram
#
# plot_nucleotide_coverage(read_dict, positions[1][1])
