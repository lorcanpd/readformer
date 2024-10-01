#!/usr/bin/env python

import argparse
import pysam
import concurrent.futures
from collections import defaultdict
import os
import tempfile
import shutil


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Identify likely sequencing artifacts in Illumina BAM file by comparing to PacBio BAM file.')
    parser.add_argument('--illumina_bam', required=True, help='Path to the Illumina BAM file.')
    parser.add_argument('--pacbio_bam', required=True, help='Path to the PacBio BAM file.')
    parser.add_argument('--bed_file', required=True, help='Path to the high-confidence BED file.')
    parser.add_argument('--output_bam', required=True, help='Output path for the artifact BAM file.')
    parser.add_argument('--output_vcf', required=True, help='Output path for the VCF file.')
    parser.add_argument('--mapq_threshold', type=int, default=30, help='Mapping quality threshold (default: 30).')
    parser.add_argument('--baseq_threshold', type=int, default=20, help='Base quality threshold (default: 20).')
    parser.add_argument('--num_threads', type=int, default=4, help='Number of threads to use (default: 4).')
    parser.add_argument('--low_complexity_bed', help='Path to BED file of low-complexity regions to exclude.')
    args = parser.parse_args()
    return args


def read_bed_file(bed_file):
    regions = defaultdict(list)
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
            regions[chrom].append((start, end))
    return regions


def is_in_low_complexity_region(chrom, pos, low_complexity_regions):
    if low_complexity_regions is None:
        return False
    regions = low_complexity_regions.get(chrom, [])
    for start, end in regions:
        if start <= pos < end:
            return True
    return False


def check_pacbio_support(
        pacbio_bam, chrom, pos, alt_base, mapq_threshold, baseq_threshold
):
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
    if total_coverage < 10:
        return None  # Coverage too low
    pacbio_alt_count = pacbio_base_counts.get(alt_base, 0)
    pacbio_ref_count = total_coverage - pacbio_alt_count
    return pacbio_ref_count, pacbio_alt_count, total_coverage


def create_vcf_record(chrom, pos, base_counts, alt_base, illumina_read_name, position_on_read, pacbio_support):
    # # Get reference base (set as 'N' if not available)
    ref_base = 'N'

    illumina_ref_count = sum(count for base, count in base_counts.items() if base != alt_base)
    # get key of the highest value in the dictionary
    ref_base = max(base_counts, key=base_counts.get)
    illumina_alt_count = base_counts[alt_base]
    illumina_total_coverage = sum(base_counts.values())
    pacbio_ref_count, pacbio_alt_count, pacbio_total_coverage = pacbio_support
    # get the position of the base on the read (e.g. the first element on a read is 0)

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


def process_chromosome(args_tuple):
    chrom, regions, args, temp_dir = args_tuple
    illumina_bam_path = args.illumina_bam
    pacbio_bam_path = args.pacbio_bam
    mapq_threshold = args.mapq_threshold
    baseq_threshold = args.baseq_threshold
    low_complexity_regions = args.low_complexity_regions  # dict of chrom -> list of (start, end)

    # Open BAM files
    illumina_bam = pysam.AlignmentFile(illumina_bam_path, 'rb')
    pacbio_bam = pysam.AlignmentFile(pacbio_bam_path, 'rb')

    # Temporary BAM file for this chromosome
    temp_bam_path = os.path.join(temp_dir, f"temp_{chrom}.bam")
    temp_bam = pysam.AlignmentFile(temp_bam_path, 'wb', template=illumina_bam)

    # Collect VCF records
    vcf_records = []

    # For all regions in this chromosome
    for start, end in regions[:4]: # Limit to first 4 regions for testing
        # Fetch Illumina pileup columns in this region
        for pileupcolumn in illumina_bam.pileup(
                chrom, start, end, min_base_quality=baseq_threshold,
                min_mapping_quality=mapq_threshold, truncate=True
        ):
            pos = pileupcolumn.pos
            # Check if position is in low-complexity region
            if is_in_low_complexity_region(chrom, pos, low_complexity_regions):
                continue
            # Get base counts
            base_counts = defaultdict(int)
            read_bases = {}
            reads_at_pos = {}
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

            # Identify positions where alternate alleles are supported by exactly one read
            if len(base_counts) > 1:
                for base, count in base_counts.items():
                    if count == 1:
                        # This base is supported by exactly one read
                        # Get the read name supporting this base
                        for read_name, read_base in read_bases.items():
                            if read_base == base:
                                # Now check PacBio data at this position
                                pacbio_support = check_pacbio_support(
                                    pacbio_bam, chrom, pos, base,
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
                                    if pacbio_alt_count == 0 and pacbio_total_coverage >= 30:
                                        # No support in PacBio, sufficient coverage
                                        # Collect read and VCF record
                                        artifact_read = reads_at_pos[read_name]
                                        # geet the index of the base on the read
                                        position_on_read = artifact_read.get_reference_positions().index(pos)
                                        temp_bam.write(artifact_read)
                                        vcf_record = create_vcf_record(
                                            chrom, pos, base_counts, base,
                                            read_name, position_on_read,
                                            pacbio_support
                                        )
                                        vcf_records.append(vcf_record)
                                break  # Only need to process the one read supporting this base
    illumina_bam.close()
    pacbio_bam.close()
    temp_bam.close()
    return vcf_records


# def write_vcf_file(vcf_path, vcf_records_list):
#     with open(vcf_path, 'w') as vcf_file:
#         # Write VCF header
#         vcf_file.write('##fileformat=VCFv4.2\n')
#         vcf_file.write('#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n')
#         for vcf_records in vcf_records_list:
#             for record in vcf_records:
#                 info_fields = [
#                     f"ILLUMINA_REF_COUNT={record['illumina_ref_count']}",
#                     f"ILLUMINA_ALT_COUNT={record['illumina_alt_count']}",
#                     f"ILLUMINA_TOTAL_COVERAGE={record['illumina_total_coverage']}",
#                     f"ILLUMINA_READ_NAME={record['illumina_read_name']}",
#                     f"PACBIO_REF_COUNT={record['pacbio_ref_count']}",
#                     f"PACBIO_ALT_COUNT={record['pacbio_alt_count']}",
#                     f"PACBIO_TOTAL_COVERAGE={record['pacbio_total_coverage']}"
#                 ]
#                 info = ';'.join(info_fields)
#                 vcf_line = '\t'.join([
#                     record['chrom'],
#                     str(record['pos']),
#                     record['id'],
#                     record['ref'],
#                     record['alt'],
#                     record['qual'],
#                     record['filter'],
#                     info
#                 ])
#                 vcf_file.write(vcf_line + '\n')


def write_vcf_file(vcf_path, vcf_records_list):
    # Flatten the list of VCF records into a single list
    all_vcf_records = [record for vcf_records in vcf_records_list for record in vcf_records]

    # Define a custom sort key function for chromosomes
    def chrom_sort_key(chrom):
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

    # Sort the records by chromosome and position
    all_vcf_records.sort(key=lambda r: (chrom_sort_key(r['chrom']), r['pos']))

    with open(vcf_path, 'w') as vcf_file:
        # Write VCF header
        vcf_file.write('##fileformat=VCFv4.2\n')
        vcf_file.write('#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n')
        for record in all_vcf_records:
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


# def merge_bam_files(temp_bam_files, output_bam_path):
#     # Use pysam.merge to merge the BAM files
#     pysam.merge('-f', output_bam_path, *temp_bam_files)
#     # Index the merged BAM file
#     pysam.index(output_bam_path)

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


def main():
    args = parse_arguments()

    # Read BED files
    high_conf_regions = read_bed_file(args.bed_file)
    if args.low_complexity_bed:
        low_complexity_regions = read_bed_file(args.low_complexity_bed)
    else:
        low_complexity_regions = None
    args.low_complexity_regions = low_complexity_regions  # Pass it in args

    # Create a temporary directory to store temporary BAM files
    temp_dir = tempfile.mkdtemp()

    # Prepare arguments for process_chromosome
    tasks = []
    for chrom in high_conf_regions:
        regions = high_conf_regions[chrom]
        tasks.append((chrom, regions, args, temp_dir))

    # Process chromosomes in parallel
    vcf_records_list = []
    # with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_threads) as executor:
    #     results = executor.map(process_chromosome, tasks)
    #     for vcf_records in results:
    #         vcf_records_list.append(vcf_records)

    # Non-parallel version
    for task in tasks:
        vcf_records = process_chromosome(task)
        vcf_records_list.append(vcf_records)

    # Merge temporary BAM files
    temp_bam_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.bam')]
    if temp_bam_files:
        merge_bam_files(temp_bam_files, args.output_bam)
    else:
        print("No artifact reads found. Output BAM file will not be created.")

    # Remove temporary directory
    shutil.rmtree(temp_dir)

    # Write VCF file
    write_vcf_file(args.output_vcf, vcf_records_list)

    # BGZIP compress the VCF file
    bgzipped_vcf = args.output_vcf + '.gz'
    pysam.tabix_compress(args.output_vcf, bgzipped_vcf, force=True)

    # TABIX index the compressed VCF file
    pysam.tabix_index(bgzipped_vcf, preset='vcf', force=True)

    # Remove the uncompressed VCF file
    os.remove(args.output_vcf)

    # Print total number of candidates found
    total_candidates = sum(len(vcf_records) for vcf_records in vcf_records_list)
    print("Total number of candidates found:", total_candidates)


if __name__ == '__main__':
    main()
