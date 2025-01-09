#!/usr/bin/env python

import argparse
import pysam
from collections import defaultdict
import pandas as pd
import os


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract variant reads from a single BAM, compare to a reference, "
                    "and produce a CSV with read-level details."
    )
    parser.add_argument(
        "--bam", required=True,
        help="Path to the BAM file."
    )
    parser.add_argument(
        "--bed_file", required=True,
        help="Path to a BED file specifying regions to search."
    )
    parser.add_argument(
        "--reference_fasta", required=True,
        help="Path to the reference FASTA file."
    )
    parser.add_argument(
        "--output_csv", required=True,
        help="Output CSV file path."
    )
    parser.add_argument(
        "--mapq_threshold", type=int, default=30,
        help="Minimum read mapping quality. Default=30."
    )
    parser.add_argument(
        "--baseq_threshold", type=int, default=20,
        help="Minimum base quality. Default=20."
    )
    parser.add_argument(
        "--min_coverage", type=int, default=2,
        help="Minimum coverage at a site for consideration. Default=2."
    )
    parser.add_argument(
        "--low_complexity_bed",
        help="Path to a BED of low-complexity regions to exclude."
    )
    args = parser.parse_args()
    return args


def read_bed_file(bed_file):
    """
    Read a BED file and return a list of (chrom, start, end) tuples.
    """
    regions = []
    with open(bed_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 3:
                continue
            chrom, start, end = fields[0], int(fields[1]), int(fields[2])
            regions.append((chrom, start, end))
    return regions


def build_low_complexity_dict(bed_file):
    """
    Read low-complexity regions into a dict: chrom -> list of (start, end).
    """
    if bed_file is None:
        return None
    regions = read_bed_file(bed_file)
    lc_dict = {}
    for chrom, start, end in regions:
        lc_dict.setdefault(chrom, []).append((start, end))
    return lc_dict


# TODO change this to an avoid region function.
def is_in_low_complexity_region(chrom, pos, lc_dict):
    """
    Check if 'pos' (0-based) in 'chrom' is within low-complexity regions.
    """
    if lc_dict is None or chrom not in lc_dict:
        return False
    for (start, end) in lc_dict[chrom]:
        if start <= pos < end:
            return True
    return False


def get_trinucleotide_context(fasta, chrom, pos0):
    """
    Return the trinucleotide context for the 0-based position 'pos0' on 'chrom'.
    If out of bounds, use 'N'.

    Example return: 'ACA', 'TCT', etc.
    """
    # We'll fetch [pos-1 : pos+2], but each coordinate might be out of range
    left_base = "N"
    ref_base = "N"
    right_base = "N"

    try:
        if pos0 - 1 >= 0:
            left_base = fasta.fetch(chrom, pos0 - 1, pos0).upper()
    except:
        pass

    try:
        ref_base = fasta.fetch(chrom, pos0, pos0 + 1).upper()
    except:
        pass

    try:
        right_base = fasta.fetch(chrom, pos0 + 1, pos0 + 2).upper()
    except:
        pass

    # If any fetch hits out-of-bound, they remain 'N'
    return f"{left_base}{ref_base}{right_base}"


def main():
    args = parse_arguments()

    # 1. Read the target regions
    regions = read_bed_file(args.bed_file)
    if not regions:
        raise ValueError(f"No valid regions found in {args.bed_file}.")

    # 2. Build low complexity dictionary (optional)
    lc_dict = build_low_complexity_dict(args.low_complexity_bed)

    # 3. Open BAM and reference
    bam = pysam.AlignmentFile(args.bam, "rb")
    ref_fasta = pysam.FastaFile(args.reference_fasta)

    # 4. Prepare to collect results
    records = []

    # 5. Iterate over regions
    for chrom, start, end in regions:
        # pysam pileup uses 0-based start, end
        for pileupcolumn in bam.pileup(
            chrom, start, end,
            min_base_quality=args.baseq_threshold,
            min_mapping_quality=args.mapq_threshold,
            truncate=True
        ):
            pos0 = pileupcolumn.pos  # 0-based
            if is_in_low_complexity_region(chrom, pos0, lc_dict):
                continue

            # Gather base counts
            base_counts = defaultdict(int)
            read_bases = {}
            read_positions = {}

            for pileupread in pileupcolumn.pileups:
                # if the base is deletion or refskip, skip
                if pileupread.is_del or pileupread.is_refskip:
                    continue
                read = pileupread.alignment
                if read.mapping_quality < args.mapq_threshold:
                    continue
                base = read.query_sequence[pileupread.query_position]
                bq = read.query_qualities[pileupread.query_position]
                if bq < args.baseq_threshold:
                    continue

                base = base.upper()
                base_counts[base] += 1
                read_bases[read.query_name] = base
                read_positions[read.query_name] = pileupread.query_position

            coverage = sum(base_counts.values())
            if coverage < args.min_coverage:
                continue

            # Fetch the reference base from FASTA
            ref_base = ref_fasta.fetch(chrom, pos0, pos0 + 1).upper() or "N"

            # For every alt base present at this position
            for alt_base, alt_count in base_counts.items():
                if alt_base == ref_base:
                    continue  # skip if it's the same as reference
                # We'll find the reference count
                ref_count = base_counts.get(ref_base, 0)

                # For each read that carries this alt_base
                # we create a row with position_on_read
                for read_name, base in read_bases.items():
                    if base == alt_base:
                        # position_on_read is the 0-based offset in that read
                        pos_on_read = read_positions[read_name]

                        # Build the trinucleotide context (0-based)
                        trinuc = get_trinucleotide_context(ref_fasta, chrom, pos0)

                        row = {
                            "chrom": chrom,
                            "pos": pos0 + 1,  # convert to 1-based
                            "ref": ref_base,
                            "alt": alt_base,
                            "read_id": read_name,
                            "trinucleotide_context": trinuc,
                            "ref_count": ref_count,
                            "alt_count": alt_count,
                            "position_on_read": pos_on_read
                        }
                        records.append(row)

    bam.close()
    ref_fasta.close()

    # 6. Convert to DataFrame and save
    df = pd.DataFrame(records)
    if df.empty:
        print("No variants found under the given thresholds.")
    else:
        print(f"Found {len(df)} read-level variant entries.")
    df.to_csv(args.output_csv, index=False)
    print(f"CSV saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
