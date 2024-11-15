from typing import Dict, Any

import numpy as np
import pysam
import re


def extract_random_read_from_position(
        bam_file_path: str,
        chromosome: str,
        position: int,
        min_quality: int = 0
) -> dict[Any, dict[str, Any]] | None:
    """
    Extract a random read from a BAM file starting at a given position.

    :param bam_file_path:
        Path to the BAM file.
    :param chromosome:
        Chromosome name.
    :param position:
        Starting genomic position (0-based).
    :param min_quality:
        Minimum mapping quality threshold.

    :return:
        A dictionary containing information about the randomly selected read.
        Returns None if no read meets the criteria.
    """

    bam_file = pysam.AlignmentFile(bam_file_path, "rb")

    # Adjust chromosome name if necessary
    adjusted_chromosome = chromosome
    if chromosome not in bam_file.references:
        if chromosome.startswith("chr"):
            adjusted_chromosome = chromosome[3:]  # Try without 'chr'
        else:
            adjusted_chromosome = "chr" + chromosome  # Try with 'chr'
        if adjusted_chromosome not in bam_file.references:
            bam_file.close()
            raise ValueError(
                f"Chromosome {adjusted_chromosome} not found in the BAM file."
            )

    try:
        iter_reads = bam_file.fetch(adjusted_chromosome, position)
    except ValueError as e:
        bam_file.close()
        raise ValueError(
            f"Error fetching reads from {adjusted_chromosome}:{position} - {e}"
        )

    selected_read = None
    count = 0
    max_reads = 40  # Maximum number of reads to consider for randomness
    n = np.random.randint(0, max_reads + 1)  # Random integer between 0 and 40 inclusive

    for read in iter_reads:
        if read.mapping_quality < min_quality:
            continue

        if count == n:
            selected_read = read
            break

        selected_read = read
        count += 1

        if count >= max_reads:
            break

    bam_file.close()

    if selected_read is None:
        return None

    return {
        selected_read.query_name: {
            "bitwise_flags": selected_read.flag,
            "reference_start": selected_read.reference_start,
            "mapping_quality": selected_read.mapping_quality,
            "cigar": selected_read.cigarstring,
            "template_length": selected_read.template_length,
            "query_sequence": selected_read.query_sequence,
            "query_qualities": selected_read.query_qualities,
            "tags": selected_read.tags
        }
    }


def decode_orientation_flags(flags):
    """
    Extract essential flags related to read orientation.
    """
    orientation_info = {
        # 'is_paired': (flags >> 0) & 1,             # 0x1
        'proper_pair': (flags >> 1) & 1,           # 0x2
        'is_reverse': (flags >> 4) & 1,            # 0x10
        'is_first_in_pair': (flags >> 6) & 1,      # 0x40
        'is_second_in_pair': (flags >> 7) & 1      # 0x80
    }
    return orientation_info


def cigar_to_integer_encoding(cigar, read_length):
    """
    Convert a CIGAR string into an integer encoded list based on each position's
    CIGAR operation.

    :param cigar: The CIGAR string.
    :param read_length: The length of the read.

    :return encoding:
        A list of integers representing the CIGAR operation at each read
        position.
        Each integer corresponds to an operation as per the mapping:
        'M': 0, 'I': 1, 'S': 2, '=': 3, 'X': 4
    """
    # Define CIGAR operation to integer mapping
    op_mapping = {
        'M': 0,
        'I': 1,
        'S': 2,
        '=': 3,
        'X': 4
    }
    none_code = -1

    encoding = []

    # Parse CIGAR string
    for length, op in re.findall(r'(\d+)([MIDNSHP=X])', cigar):
        length = int(length)
        if op in op_mapping:
            # These operations consume read bases
            encoding.extend([op_mapping[op]] * length)
        else:
            # Operations that do not consume read bases are skipped
            # Do not extend the encoding
            continue

    # If encoding length doesn't match read_length, pad with 'none'
    if len(encoding) < read_length:
        encoding.extend([none_code] * (read_length - len(encoding)))
    elif len(encoding) > read_length:
        encoding = encoding[:read_length]

    return encoding


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

    # Randomly sample positions from the genome giving each chromosome a
    # probability proportional to its length
    chromosome_probs = np.array([val[1] for val in chromosome_ranges.values()])
    chromosome_probs = chromosome_probs / chromosome_probs.sum()
    chromosome_names = list(chromosome_ranges.keys())
    random_chromosomes = np.random.choice(
        chromosome_names, size=n, p=chromosome_probs
    )
    # Randomly sample positions (given by the chromosome_ranges) from each
    # chromosome
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

        final_dict[read] = {
            'query_sequence': values['query_sequence'],
            'base_qualities': values['query_qualities'],
            'orientation_flags': decode_orientation_flags(
                values['bitwise_flags']
            ),
            'cigar_encoding': cigar_to_integer_encoding(
                values['cigar'], len(values['query_sequence'])
            ),
            'positions': list(
                range(
                    values['reference_start'],
                    values['reference_start'] + len(values['query_sequence'])
                )
            )
        }

    return final_dict
