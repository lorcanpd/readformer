#!/usr/bin/env python

import argparse
import pysam
from collections import defaultdict
import pandas as pd
import concurrent.futures


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract single-read variant positions from a BAM file, "
                    "compare to reference, output CSV. Parallelized with ProcessPool."
    )
    parser.add_argument(
        "--bam", required=True,
        help="Path to the BAM file."
    )
    parser.add_argument(
        "--bed_file",
        help="Path to a BED file specifying regions to search (0-based). "
             "If not provided, the script will process all contigs in the BAM."
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
        "--min_coverage", type=int, default=20,
        help="Minimum coverage at a site for consideration. Default=20."
    )
    parser.add_argument(
        "--avoid_bed",
        help="Path to a BED of regions (e.g., low complexity) to exclude (0-based)."
    )
    parser.add_argument(
        "--num_threads", type=int, default=4,
        help="Number of worker processes to use. Default=4."
    )
    return parser.parse_args()


def read_bed_file(bed_file):
    """
    Read a BED file and return a list of (chrom, start, end) tuples (0-based).
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


def build_avoid_region_dict(bed_file):
    """
    Read 'avoid' regions into a dict: chrom -> list of (start, end).
    (All 0-based coordinates)
    """
    if not bed_file:
        return None
    regions = read_bed_file(bed_file)
    avoid_dict = {}
    for chrom, start, end in regions:
        avoid_dict.setdefault(chrom, []).append((start, end))
    return avoid_dict


def is_in_avoid_region(chrom, pos0, avoid_dict):
    """
    Check if 'pos0' (0-based) in 'chrom' is within any region in avoid_dict.
    """
    if avoid_dict is None or chrom not in avoid_dict:
        return False
    for (start, end) in avoid_dict[chrom]:
        if start <= pos0 < end:
            return True
    return False


def chunk_bed_regions(regions, num_chunks=1):
    """
    Given a list of (chrom, start, end), produce 'num_chunks' sub-lists.
    A simple approach: sort by size, then allocate greedily.
    """
    # Sort the regions by their length descending
    regions = sorted(regions, key=lambda r: r[2]-r[1], reverse=True)
    buckets = [[] for _ in range(num_chunks)]
    bucket_sizes = [0]*num_chunks
    for region in regions:
        length = region[2]-region[1]
        # Place region in the bucket with the smallest total size so far
        idx = bucket_sizes.index(min(bucket_sizes))
        buckets[idx].append(region)
        bucket_sizes[idx] += length
    return buckets


def chunk_all_contigs(bam_file, num_chunks=1):
    """
    If no BED file is provided, create intervals covering all contigs
    from the BAM header. Then chunk them similarly.
    """
    af = pysam.AlignmentFile(bam_file, "rb")
    contigs = []
    for i, contig_name in enumerate(af.references):
        length = af.lengths[i]
        contigs.append((contig_name, 0, length))
    af.close()
    # Now chunk similarly
    return chunk_bed_regions(contigs, num_chunks)


def detect_chrom_prefix_bam(bam_file):
    """
    Detect whether chromosome names in a BAM file are prefixed with 'chr'.
    Returns 'chr' or ''.
    """
    with pysam.AlignmentFile(bam_file, "rb") as bam:
        for ref in bam.references[:10]:
            if ref.startswith("chr"):
                return "chr"
    return ""


def detect_chrom_prefix_fasta(fasta_file):
    """
    Detect whether chromosome names in a FASTA file are prefixed with 'chr'.
    Returns 'chr' or ''.
    """
    with pysam.FastaFile(fasta_file) as f:
        for ref in f.references[:10]:
            if ref.startswith("chr"):
                return "chr"
    return ""


def detect_chrom_prefix_bed(bed_file):
    """
    Detect whether chromosome names in a BED file are prefixed with 'chr'.
    Returns 'chr' or ''.
    """
    if not bed_file:
        return ""
    with open(bed_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            chrom = line.split()[0]
            if chrom.startswith("chr"):
                return "chr"
    return ""


def unify_chromosome_name(chrom, from_prefix, to_prefix):
    """
    Convert a chromosome name from one prefix style to another.
    E.g., unify_chromosome_name("chr1", "chr", "") -> "1"
          unify_chromosome_name("1", "", "chr") -> "chr1"
    """
    if from_prefix == to_prefix:
        return chrom

    if from_prefix == "chr" and to_prefix == "":
        # Remove 'chr' if present
        if chrom.startswith("chr"):
            return chrom[3:]
        else:
            return chrom
    elif from_prefix == "" and to_prefix == "chr":
        # Add 'chr' if not present
        if not chrom.startswith("chr"):
            return "chr" + chrom
        else:
            return chrom
    else:
        return chrom  # fallback if any unexpected scenario


def get_ICAMS_trinucleotide_context(fasta, chrom, pos0, alt_base):
    """
    Return the trinucleotide context around pos0 (0-based) on 'chrom'.
    If out of bounds, use 'N'. E.g., 'ACA', 'TCT', etc.
    PySAM's FastaFile.fetch(contig, start, end) is 0-based, end-exclusive.
    So 'fetch(chrom, pos0, pos0+1)' returns the base at pos0.
    The complement of the bases, including the alt base, is obtained if the
    middle base is not C or T, and the trinucleotide triplet is reversed.
    To conform to the ICAMS trinucleotide context, the alt base is appended
    to the end of the trinucleotide (e.g. 'ACAG')
    """
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    left_base = "N"

    if pos0 - 1 >= 0:
        left_base = fasta.fetch(chrom, pos0 - 1, pos0).upper()
    mid_base = fasta.fetch(chrom, pos0, pos0 + 1).upper() or "N"
    right_base = fasta.fetch(chrom, pos0 + 1, pos0 + 2).upper() or "N"

    # If the middle base is not C or T, flip the entire trinucleotide
    # to its reverse complement so that the middle becomes C or T.
    if mid_base not in ['C', 'T']:
        triplet = left_base + mid_base + right_base
        rc = ''.join(complement[b] for b in reversed(triplet))
        left_base, mid_base, right_base = rc[0], rc[1], rc[2]
        alt_base = complement[alt_base]

    return f"{left_base}{mid_base}{right_base}{alt_base}"


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

def process_chunk(
    chunk_regions, bam_file, ref_fasta_file,
    avoid_dict, bam_prefix, ref_prefix, from_prefix, avoid_bed_prefix,
    mapq_threshold, baseq_threshold, min_coverage
):
    """
    A function to process a chunk of regions (list of (chrom, start, end)) in one worker.
    Returns a list of dictionaries, each corresponding to a single-read variant.
    """
    results = []
    bam = pysam.AlignmentFile(bam_file, "rb")
    ref_fasta = pysam.FastaFile(ref_fasta_file)

    for (bed_chrom, bed_start, bed_end) in chunk_regions:
        # Convert BED chromosome to match the BAM prefix for pileup
        bam_chrom = unify_chromosome_name(bed_chrom, from_prefix, bam_prefix)
        # Convert also to the reference prefix for reference fetch
        ref_chrom = unify_chromosome_name(bed_chrom, from_prefix, ref_prefix)

        # Similarly unify for the avoid bed
        avoid_chrom = unify_chromosome_name(
            bed_chrom, from_prefix, avoid_bed_prefix)

        for pileupcolumn in bam.pileup(
            bam_chrom, bed_start, bed_end,
            min_base_quality=baseq_threshold,
            min_mapping_quality=mapq_threshold,
            truncate=True
        ):
            pos0 = pileupcolumn.pos
            if is_in_avoid_region(avoid_chrom, pos0, avoid_dict):
                continue

            # Gather base counts
            base_counts = defaultdict(int)
            read_bases = {}
            read_positions = {}

            for pileupread in pileupcolumn.pileups:
                if pileupread.is_del or pileupread.is_refskip:
                    continue
                read = pileupread.alignment
                if read.mapping_quality < mapq_threshold:
                    continue
                idx_on_read = pileupread.query_position
                if idx_on_read is None:
                    continue
                base = read.query_sequence[idx_on_read]
                bq = read.query_qualities[idx_on_read]
                if bq < baseq_threshold:
                    continue

                base = base.upper()
                base_counts[base] += 1
                read_bases[read.query_name] = base
                read_positions[read.query_name] = idx_on_read

            coverage = sum(base_counts.values())
            if coverage < min_coverage:
                continue

            # Fetch the reference base
            ref_base = ref_fasta.fetch(ref_chrom, pos0, pos0 + 1).upper() or "N"

            # We only track single-read variants (alt_count == 1),
            # but you can remove "alt_count == 1" if you want everything.
            for alt_base, alt_count in base_counts.items():
                if alt_base == ref_base or alt_count != 1:
                    continue
                ref_count = base_counts.get(ref_base, 0)

                # Build up records
                for read_name, base in read_bases.items():
                    if base == alt_base:
                        pos_on_read = read_positions[read_name]
                        try:
                            trinuc = get_ICAMS_trinucleotide_context(
                                ref_fasta, ref_chrom, pos0, alt_base
                            )
                        except KeyError:
                            # If any of the context bases are not the canonical
                            # ACGT then skip this variant
                            continue
                        # If "chr" prefix is present on the bed_chrom, remove it
                        if bed_chrom.startswith("chr"):
                            bed_chrom = bed_chrom[3:]
                        row = {
                            "chrom": bed_chrom,
                            "pos": pos0,
                            "ref": ref_base,
                            "alt": alt_base,
                            "read_id": read_name,
                            "mutation_type": trinuc,
                            "ref_count": ref_count,
                            "alt_count": alt_count,
                            "position_on_read": pos_on_read
                        }
                        results.append(row)

    bam.close()
    ref_fasta.close()
    return results


def main():
    args = parse_arguments()

    # 1) Possibly read or create regions
    if args.bed_file:
        bed_regions = read_bed_file(args.bed_file)
        if not bed_regions:
            raise ValueError(f"No valid regions in {args.bed_file}")
        # Partition the bed regions
        region_chunks = chunk_bed_regions(bed_regions, args.num_threads)
    else:
        # If no bed_file is provided, chunk the entire contigs from the BAM
        region_chunks = chunk_all_contigs(args.bam, args.num_threads)

    # 2) Build avoid dict
    avoid_dict = build_avoid_region_dict(args.avoid_bed)
    avoid_bed_prefix = detect_chrom_prefix_bed(args.avoid_bed) if args.avoid_bed else ""

    # 3) Detect prefixes for the BAM, FASTA, BED
    bam_prefix = detect_chrom_prefix_bam(args.bam)
    ref_prefix = detect_chrom_prefix_fasta(args.reference_fasta)
    # If we have a bed_file, see if it has 'chr'
    # (we unify from "" in worker to the right prefix).
    if args.bed_file:
        bed_prefix = detect_chrom_prefix_bed(args.bed_file)
        from_prefix = bed_prefix
    else:
        from_prefix = bam_prefix

    # 4) Parallel processing
    all_records = []
    # We'll define the arguments that remain constant
    worker_args = {
        "bam_file": args.bam,
        "ref_fasta_file": args.reference_fasta,
        "avoid_dict": avoid_dict,
        "bam_prefix": bam_prefix,
        "ref_prefix": ref_prefix,
        "from_prefix": from_prefix,
        "avoid_bed_prefix": avoid_bed_prefix,
        "mapq_threshold": args.mapq_threshold,
        "baseq_threshold": args.baseq_threshold,
        "min_coverage": args.min_coverage
    }

    with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.num_threads) as executor:
        futures = []
        for chunk in region_chunks:
            futures.append(
                executor.submit(process_chunk, chunk, **worker_args)
            )
        for fut in concurrent.futures.as_completed(futures):
            chunk_result = fut.result()
            all_records.extend(chunk_result)

    # 5) Convert to DataFrame, save
    df = pd.DataFrame(all_records)
    if df.empty:
        print("No variants found under the given thresholds.")
    else:
        print(f"Found {len(df)} read-level variant entries.")
        # sort df by chrom, pos using chrom_sort_key
        df['chrom_sort_key'] = df['chrom'].apply(chrom_sort_key)
        df = df.sort_values(['chrom_sort_key', 'pos']).reset_index(drop=True)
        df = df.drop(columns=['chrom_sort_key'])
        df.to_csv(args.output_csv, index=False)
        print(f"CSV saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
