#!/usr/bin/env python

"""
nanoseq_pileup_chunks.py
------------------------
Creates a BED file of covered intervals from a NanoSeq BAM file using pysam's pileup.
Each contig is subdivided into fixed-size chunks for parallel processing.
The resulting chunk-BED files are merged in lex-sorted filename order.

Requirements:
  - Python 3
  - pysam (pip install pysam)
  - A properly indexed BAM file (.bai alongside .bam)
"""

import os
import sys
import argparse
import tempfile
import shutil
from concurrent.futures import ProcessPoolExecutor
import pysam


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a BED file of genomic intervals with coverage >= threshold "
            "from a NanoSeq BAM file, using pysam pileup in parallel. "
            "Each contig is split into fixed-size chunks for parallel coverage "
            "calculation."
        )
    )
    parser.add_argument("-i", "--input_bam", required=True,
                        help="Input NanoSeq BAM file (must be indexed).")
    parser.add_argument("-o", "--output_bed", required=True,
                        help="Output BED file of covered intervals.")
    parser.add_argument("-t", "--threshold", type=int, default=1,
                        help="Minimum coverage threshold (default=1).")
    parser.add_argument("-q", "--mapq_filter", type=int, default=30,
                        help="Minimum mapping quality for pileup (default=30).")
    parser.add_argument("-n", "--num_cores", type=int, default=1,
                        help="Number of parallel processes (default=1).")
    parser.add_argument("-c", "--chunk_size", type=int,
                        default=10000000,
                        help="Chunk size in bp (default=10,000,000).")
    return parser.parse_args()


def coverage_to_intervals_pileup(pileup_columns, cov_threshold, outfile):
    """
    Given a pysam pileup iterator (pileup_columns), stream through each column:
      - reference_name -> chrom
      - reference_pos  -> 0-based position
      - nsegments      -> coverage
    For coverage >= cov_threshold, merge into intervals, writing them to
    `outfile` in BED format.

    BED intervals are 0-based, half-open [start, end).
    """
    current_chrom = None
    in_region = False
    region_start = 0
    last_pos = 0

    for column in pileup_columns:
        chrom = column.reference_name
        pos_0 = column.reference_pos  # 0-based
        depth = column.nsegments  # total reads covering this site

        # If we switched to a new contig, finalize intervals for the old one
        if chrom != current_chrom:
            if in_region and current_chrom is not None:
                # close off the previous interval
                outfile.write(f"{current_chrom}\t{region_start}\t{last_pos}\n")
            current_chrom = chrom
            in_region = False

        if depth >= cov_threshold:
            if not in_region:
                in_region = True
                region_start = pos_0
        else:
            # coverage < threshold
            if in_region:
                # finalize the interval up to the previous position
                outfile.write(f"{current_chrom}\t{region_start}\t{pos_0}\n")
                in_region = False

        last_pos = pos_0 + 1

    # If the last contig ended while still in a region
    if in_region and current_chrom is not None:
        outfile.write(f"{current_chrom}\t{region_start}\t{last_pos}\n")


def run_pileup_for_chunk(
        bam_path, contig, start, end, mapq, cov_threshold, chunk_bed_path
):
    """
    Process a single chunk of (contig, start, end) using pysam pileup.
    Convert coverage >= cov_threshold into intervals, then write to chunk_bed_path.
    """
    # Open the BAM in this process
    with pysam.AlignmentFile(bam_path, "rb") as bam_file:
        # Build the pileup
        pileup_iter = bam_file.pileup(
            contig=contig,
            start=start,
            end=end,
            stepper="all",
            truncate=True,  # don't go beyond [start, end)
            min_mapping_quality=mapq
        )

        # Write intervals for coverage >= cov_threshold to chunk_bed_path
        with open(chunk_bed_path, "w") as outbed:
            coverage_to_intervals_pileup(pileup_iter, cov_threshold, outbed)

    return chunk_bed_path  # for reference if needed


def main():
    args = parse_args()
    bam_path = args.input_bam
    out_bed = args.output_bed
    cov_threshold = args.threshold
    mapq_filter = args.mapq_filter
    num_cores = args.num_cores
    chunk_size = args.chunk_size

    # Ensure output directory exists
    outdir = os.path.dirname(os.path.abspath(out_bed)) or "."
    os.makedirs(outdir, exist_ok=True)

    # Create a temporary directory inside the outdir
    tmpdir = tempfile.mkdtemp(prefix="nanoseq_pileup_", dir=outdir)

    # 1) Retrieve contigs & lengths from the BAM file
    with pysam.AlignmentFile(bam_path, "rb") as bamfile:
        contigs = list(bamfile.references)
        lengths = list(bamfile.lengths)

    # 2) Build a list of tasks: (contig, start, end, chunk_bed_filename)
    tasks = []
    for i, (ctg, ctg_len) in enumerate(zip(contigs, lengths)):
        chunk_start = 0
        while chunk_start < ctg_len:
            chunk_end = min(chunk_start + chunk_size, ctg_len)
            # Zero-pad contig index for sorting
            ctg_idx_str = f"{i:07d}"
            start_str = f"{chunk_start:012d}"
            end_str = f"{chunk_end:012d}"
            chunk_filename = os.path.join(
                tmpdir,
                f"{ctg_idx_str}_{ctg}_{start_str}_{end_str}.bed"
            )
            tasks.append((ctg, chunk_start, chunk_end, chunk_filename))
            chunk_start = chunk_end

    # 3) Process each chunk in parallel
    futures = []
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        for (ctg, cstart, cend, chunkbed) in tasks:
            futures.append(executor.submit(
                run_pileup_for_chunk,
                bam_path, ctg, cstart, cend,
                mapq_filter, cov_threshold,
                chunkbed
            ))

        # Wait for tasks to complete, raise exceptions if any
        for f in futures:
            _ = f.result()

    # 4) Reassemble the final BED by lex-sorting the chunk files
    chunk_files = sorted(os.listdir(tmpdir))
    with open(out_bed, "w") as out_f:
        for cf in chunk_files:
            if not cf.endswith(".bed"):
                continue
            bed_path = os.path.join(tmpdir, cf)
            with open(bed_path, "r") as bed_in:
                shutil.copyfileobj(bed_in, out_f)

    # 5) Clean up
    shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"Done. Coverage BED written to: {out_bed}")


if __name__ == "__main__":
    main()
