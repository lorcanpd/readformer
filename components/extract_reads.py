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
    max_reads = 40  # Maximum number of reads to consider

    for read in iter_reads:
        if read.mapping_quality < min_quality:
            continue

        if count >= max_reads:
            break

        selected_read = read
        count += 1

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
            "tags": selected_read.tags,
            "positions": selected_read.get_reference_positions(full_length=True)
        }
    }


def decode_orientation_flags(flags):
    """
    Extract essential flags related to read orientation.
    """
    orientation_info = {
        'is_paired': (flags & 0x1) != 0,
        'proper_pair': (flags & 0x2) != 0,
        'is_reverse': (flags & 0x10) != 0,
        'is_first_in_pair': (flags & 0x40) != 0,
        'is_second_in_pair': (flags & 0x80) != 0
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
        'M': 0,  # Alignment match or mismatch
        'I': 1,  # Insertion to the reference
        'S': 2,  # Soft clip
        '=': 3,  # Sequence match
        'X': 4   # Sequence mismatch
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
    """
    final_dict = {}
    for read_name, values in read_dict.items():
        # Decode orientation flags
        orientation_flags = decode_orientation_flags(values['bitwise_flags'])

        # Extract reference positions
        positions = values['positions']
        # If None value in positions, replace with -1 (these are insertions)
        positions = [-1 if pos is None else pos for pos in positions]

        final_dict[read_name] = {
            'query_sequence': values['query_sequence'],
            'base_qualities': values['query_qualities'],
            'orientation_flags': orientation_flags,
            'cigar_encoding': cigar_to_integer_encoding(
                values['cigar'], len(values['query_sequence'])
            ),
            'positions': positions
        }

    return final_dict


# TODO: Double check that the position information for the reads is correct.
#  This is important be casue we will be uses the positions to index the read
#  to index the mdoel outptus for classification.
def extract_read_by_id(
        bam_file_path: str,
        chromosome: str,
        position: int,
        read_id: str
) -> Dict[str, Any] | None:
    """
    Extract a specific read from a BAM file given its ID and genomic
    coordinates. Take advantage of the fact that BAM files are indexed by
    genomic coordinates, rather than read IDs.

    :param bam_file_path:
        Path to the BAM file.
    :param chromosome:
        Chromosome name.
    :param position:
        Genomic position (0-based).
    :param read_id:
        ID of the read to retrieve.

    :return:
        A dictionary containing information about the read.
        Returns None if no read matches the criteria.
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
        iter_reads = bam_file.fetch(adjusted_chromosome, position, position+1)
    except ValueError as e:
        bam_file.close()
        raise ValueError(
            f"Error fetching reads from {adjusted_chromosome}:{position} - {e}"
        )

    out_dict = {}
    for read in iter_reads:
        if read.query_name == read_id:
            bam_file.close()
            out_dict[read_id] = {
                "query_name": read.query_name,
                "bitwise_flags": read.flag,
                "reference_start": read.reference_start,
                "mapping_quality": read.mapping_quality,
                "cigar": read.cigarstring,
                "template_length": read.template_length,
                "query_sequence": read.query_sequence,
                "query_qualities": read.query_qualities,
                "tags": read.tags,
                "positions": read.get_reference_positions(full_length=True)
            }
            break

    return out_dict

