#!/usr/bin/env python3
"""
This script processes each donor sequentially. For a given donor, it reads the
donor’s BED file line‐by‐line and enqueues each region (as a task) into a bounded
multiprocessing queue. A fixed number of worker processes (one per core) are spawned;
each repeatedly pulls a task from the queue and processes it (writing its results
to core‐specific temporary BAM/VCF files). Once the donor’s BED file has been fully
processed (and a sentinel is sent for each worker), the workers are joined. Finally,
all the core-specific temporary files (from all donors) are merged into final output
files.
"""

import argparse
import pysam
import os
import sys
import shutil
import tempfile
from collections import defaultdict
import multiprocessing
import matplotlib
import subprocess

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


# =============================================================================
# Global variables for worker processes (to be set once per worker).
global_vcf_header = None
global_ref_fasta = None   # reference FASTA file path
global_min_mapq = None
global_min_baseq = None
global_nano_threshold = None
global_temp_dir = None
global_outputs = {}       # keys: "mutation_bam", "artefact_bam", "mutation_vcf", "artefact_vcf"
global_bam_template = None  # pysam.AlignmentFile used as template


def parse_args():
    p = argparse.ArgumentParser(
        description="Process (donor, region) tasks via a producer-consumer model with a bounded queue."
    )
    p.add_argument("--illumina_bam", nargs='+', required=True,
                   help="Paths to Illumina BAM files (one per donor).")
    p.add_argument("--nanoseq_bam", nargs='+', required=True,
                   help="Paths to NanoSeq BAM files (one per donor).")
    p.add_argument("--nanoseq_bed", nargs='+', required=True,
                   help="Paths to NanoSeq BED files (one per donor).")
    p.add_argument("--reference_fasta", required=True,
                   help="Reference FASTA (e.g. hs37d5.fa.gz).")
    p.add_argument("--output_prefix", default="combined",
                   help="Prefix for final merged output files (default: 'combined').")
    p.add_argument("--min_mapq", type=int, default=30,
                   help="Minimum mapping quality for a read (default=30).")
    p.add_argument("--min_baseq", type=int, default=25,
                   help="Minimum base quality for an alt call (default=25).")
    p.add_argument("--nano_threshold", type=int, default=2,
                   help="Minimum complete duplex bundles supporting alt in NanoSeq (default=2).")
    p.add_argument("--illumina_single_alt", type=int, default=1,
                   help="If Illumina alt count equals this then consider as single-read alt (default=1).")
    p.add_argument("--temp_dir", required=True,
                   help="Directory in which to create a temporary subdirectory for core-specific outputs.")
    p.add_argument("--num_cores", type=int, default=1,
                   help="Number of CPU cores to use for each donor.")
    p.add_argument("--queue_size", type=int, default=1000,
                   help="Maximum number of tasks to hold in the queue (default=1000).")
    return p.parse_args()


def create_vcf_header(ref_bam):
    header = pysam.VariantHeader()
    header.add_line("##fileformat=VCFv4.2")
    for i, contig in enumerate(ref_bam.references):
        length = ref_bam.lengths[i]
        header.contigs.add(contig, length=length)
    header.add_line("##INFO=<ID=DONOR,Number=1,Type=String,Description=\"Donor label\">")
    # header.add_line("##INFO=<ID=TYPE,Number=1,Type=String,Description=\"mutation or artefact\">")
    header.add_line("##INFO=<ID=READ_IDS,Number=.,Type=String,Description=\"Read IDs supporting the variant\">")
    header.add_line("##INFO=<ID=ILLUMINA_ALT_COUNT,Number=1,Type=Integer,Description=\"Illumina alt count\">")
    header.add_line("##INFO=<ID=ILLUMINA_FORWARD,Number=1,Type=Integer,Description=\"Illumina alt count on forward strand\">")
    header.add_line("##INFO=<ID=ILLUMINA_REVERSE,Number=1,Type=Integer,Description=\"Illumina alt count on reverse strand\">")
    header.add_line("##INFO=<ID=NANO_ALT_COUNT,Number=1,Type=Integer,Description=\"Alt supporting duplex bundle count in NanoSeq\">")
    # header.add_line("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO")
    return header


def get_ref_base(fasta, chrom, pos):
    try:
        ref = fasta.fetch(chrom, pos, pos + 1)
        if len(ref) == 1:
            return ref.upper()
    except Exception:
        pass
    return "N"


def get_trinuc_context(fasta, chrom, pos):
    try:
        seq = fasta.fetch(chrom, pos - 1, pos + 2)
        if len(seq) == 3:
            return seq.upper()
    except Exception:
        pass
    return "NNN"


def worker_init(
        vcf_header, ref_fasta, min_mapq, min_baseq, nano_thr, temp_dir,
        bam_template_path
):
    """
    This initialiser sets the global variables for each worker.
    """
    global global_vcf_header, global_ref_fasta, global_min_mapq, global_min_baseq
    global global_nano_threshold, global_temp_dir, global_outputs, global_bam_template

    # global_vcf_header = pysam.VariantHeader()
    # for line in vcf_header_str.strip().splitlines():
    #     if line.startswith("##"):
    #         global_vcf_header.add_line(line)
    global_vcf_header = vcf_header
    global_ref_fasta = ref_fasta
    global_min_mapq = min_mapq
    global_min_baseq = min_baseq
    global_nano_threshold = nano_thr
    global_temp_dir = temp_dir

    # Open the Illumina bam as template for writing output BAMs.
    global_bam_template = pysam.AlignmentFile(bam_template_path, "rb")

    pid = os.getpid()
    mut_bam_path = os.path.join(temp_dir, f"core_{pid}.mutation.bam")
    art_bam_path = os.path.join(temp_dir, f"core_{pid}.artefact.bam")
    mut_vcf_path = os.path.join(temp_dir, f"core_{pid}.mutation.vcf")
    art_vcf_path = os.path.join(temp_dir, f"core_{pid}.artefact.vcf")

    global_outputs["mutation_bam"] = pysam.AlignmentFile(
        mut_bam_path, "wb", template=global_bam_template)
    global_outputs["artefact_bam"] = pysam.AlignmentFile(
        art_bam_path, "wb", template=global_bam_template)
    global_outputs["mutation_vcf"] = pysam.VariantFile(
        mut_vcf_path, "w", header=global_vcf_header)
    global_outputs["artefact_vcf"] = pysam.VariantFile(
        art_vcf_path, "w", header=global_vcf_header)


def classify_mutation_type(left, ref_base, right, alt):
    if len(left) != 1 or len(ref_base) != 1 or len(right) != 1 or len(alt) != 1:
        return None
    ref = ref_base.upper()
    alt = alt.upper()
    left = left.upper()
    right = right.upper()
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    if ref in ['C', 'T']:
        mut_label = f"{ref}>{alt}"
    else:
        ref = complement.get(ref, 'N')
        alt = complement.get(alt, 'N')
        left = complement.get(right, 'N')
        right = complement.get(left, 'N')
        mut_label = f"{ref}>{alt}"
    return f"{left}[{mut_label}]{right}"


def pileup_duplex_bundle_count(nanoseq_bam, chrom, pos, alt):
    bundle_support = defaultdict(lambda: {"f1r2": 0, "f2r1": 0})
    for read in nanoseq_bam.fetch(chrom, pos, pos + 1):
        if read.mapping_quality < 30:
            continue
        ref_positions = read.get_reference_positions()
        if pos not in ref_positions:
            continue
        qpos = ref_positions.index(pos)
        if read.query_qualities[qpos] < 10:
            continue
        if read.query_sequence[qpos].upper() != alt.upper():
            continue
        try:
            rb = read.get_tag("RB")
        except KeyError:
            continue
        if read.is_reverse:
            if read.is_read1:
                bundle_support[rb]["f2r1"] += 1
            else:
                bundle_support[rb]["f1r2"] += 1
        else:
            if read.is_read1:
                bundle_support[rb]["f1r2"] += 1
            else:
                bundle_support[rb]["f2r1"] += 1
    duplex_count = 0
    for counts in bundle_support.values():
        if counts["f1r2"] >= 2 and counts["f2r1"] >= 2:
            duplex_count += 1
    return duplex_count


def extract_illumina_alt_reads(illumina_bam, chrom, pos, alt):
    alt_reads = []
    for col in illumina_bam.pileup(
            chrom, pos, pos + 1, truncate=True,
            min_mapping_quality=global_min_mapq, stepper='all'
    ):
        if col.pos != pos:
            continue
        for pread in col.pileups:
            if pread.is_del or pread.is_refskip:
                continue
            read = pread.alignment
            if read.mapping_quality < global_min_mapq:
                continue
            qpos = pread.query_position
            if qpos is None or read.query_qualities[qpos] < global_min_baseq:
                continue
            if read.query_sequence[qpos].upper() == alt.upper():
                alt_reads.append(read)
        break
    # print(f"Extracted {len(alt_reads)} alt reads from Illumina.")
    return alt_reads


def process_region(task):
    """
    Task tuple is: (donor_label, illumina_bam_path, nanoseq_bam_path, region)
    """
    donor_label, illumina_bam_path, nanoseq_bam_path, region = task
    chrom, start, end = region

    illumina_bam = pysam.AlignmentFile(illumina_bam_path, "rb")
    nanoseq_bam = pysam.AlignmentFile(nanoseq_bam_path, "rb")
    fasta = pysam.FastaFile(global_ref_fasta)

    for col in illumina_bam.pileup(
            chrom, start, end, truncate=True,
            min_mapping_quality=global_min_mapq, stepper='all'
    ):
        pos = col.pos
        if not (start <= pos < end):
            print(f"Unable to process as pos {pos} outside region: {start}-{end}")
            continue
        # print(f"Processing {chrom}:{pos}")
        base_counts = defaultdict(lambda: {"forward": 0, "reverse": 0})
        for pread in col.pileups:
            if pread.is_del or pread.is_refskip:
                continue
            read = pread.alignment
            # require proper paired reads
            if not read.is_paired or (read.is_paired and not read.is_proper_pair):
                continue
            if read.mapping_quality < global_min_mapq:
                continue
            qpos = pread.query_position
            if qpos is None or read.query_qualities[qpos] < global_min_baseq:
                continue
            base = read.query_sequence[qpos].upper()
            if read.is_reverse:
                base_counts[base]["reverse"] += 1
            else:
                base_counts[base]["forward"] += 1

        # Only process positions with enough overall depth.
        total_depth = sum(sum(d.values()) for d in base_counts.values())
        if total_depth < 40:
            continue

        ref_base = get_ref_base(fasta, chrom, pos)
        for alt_base, counts in base_counts.items():
            if alt_base == ref_base:
                continue
            alt_count = counts["forward"] + counts["reverse"]
            # Determine whether both strands contribute.
            is_balanced = (counts["forward"] > 0 and counts["reverse"] > 0)
            # Get duplex support from NanoSeq.
            duplex_count = pileup_duplex_bundle_count(nanoseq_bam, chrom, pos, alt_base)

            if duplex_count >= global_nano_threshold:
                alt_reads = extract_illumina_alt_reads(illumina_bam, chrom, pos, alt_base)
                for r in alt_reads:
                    global_outputs["mutation_bam"].write(r)
                rec = global_outputs["mutation_vcf"].new_record()
                rec.info.pop("END", None)
                rec.chrom = chrom
                rec.pos = pos + 1  # Convert to 1-based.
                rec.ref = ref_base
                rec.alts = (alt_base,)
                rec.id = "."
                rec.qual = None
                rec.filter.add("PASS")
                rec.info["DONOR"] = donor_label
                rec.info["READ_IDS"] = ",".join(r.query_name for r in alt_reads)
                rec.info["ILLUMINA_ALT_COUNT"] = alt_count
                rec.info["NANO_ALT_COUNT"] = duplex_count
                rec.info["ILLUMINA_FORWARD"] = counts["forward"]
                rec.info["ILLUMINA_REVERSE"] = counts["reverse"]
                global_outputs["mutation_vcf"].write(rec)
            elif is_balanced or (0 < duplex_count < global_nano_threshold):
                continue
            else:
                alt_reads = extract_illumina_alt_reads(
                    illumina_bam, chrom, pos, alt_base
                )
                for r in alt_reads:
                    global_outputs["artefact_bam"].write(r)
                rec = global_outputs["artefact_vcf"].new_record()
                rec.info.pop("END", None)
                rec.chrom = chrom
                rec.pos = pos + 1
                rec.ref = ref_base
                rec.alts = (alt_base,)
                rec.id = "."
                rec.qual = None
                rec.filter.add("PASS")
                rec.info["DONOR"] = donor_label
                rec.info["READ_IDS"] = ",".join(r.query_name for r in alt_reads)
                rec.info["ILLUMINA_ALT_COUNT"] = alt_count
                rec.info["NANO_ALT_COUNT"] = duplex_count
                rec.info["ILLUMINA_FORWARD"] = counts["forward"]
                rec.info["ILLUMINA_REVERSE"] = counts["reverse"]
                global_outputs["artefact_vcf"].write(rec)
    illumina_bam.close()
    nanoseq_bam.close()
    fasta.close()
    return donor_label, region


def worker_loop(
        task_queue, vcf_header, ref_fasta, min_mapq, min_baseq, nano_thr,
        temp_dir, bam_template_path,
        # illumina_single_alt
):
    """
    worker loop for producer-consumer model.
    """
    # Each worker calls the initialiser once.
    from collections import defaultdict  # in case not already imported
    worker_init(
        vcf_header, ref_fasta, min_mapq, min_baseq, nano_thr, temp_dir,
        bam_template_path
    )
    # Loop over tasks from the queue.
    while True:
        task = task_queue.get()
        if task is None:
            break  # Sentinel received.
        # task is (donor_label, illumina_bam_path, nanoseq_bam_path, region)
        process_region(task)
    # Worker loop ends; close worker-specific output files.
    for f in global_outputs.values():
        f.close()


def main():
    global args
    args = parse_args()

    donors = len(args.illumina_bam)
    if not (donors == len(args.nanoseq_bam) == len(args.nanoseq_bed)):
        print("ERROR: Mismatch in number of donors!", file=sys.stderr)
        sys.exit(1)

    # Create a temporary directory (inside user-specified temp_dir)
    temp_dir = tempfile.mkdtemp(dir=args.temp_dir)
    print(f"Using temporary directory: {temp_dir}")

    # Use the first Illumina BAM as template for headers.
    first_illumina = pysam.AlignmentFile(args.illumina_bam[0], "rb")
    bam_template_path = args.illumina_bam[0]
    vcf_header = create_vcf_header(first_illumina)
    # header_str = str(vcf_header)
    first_illumina.close()

    # For each donor, process the donor's BED file using a producer-consumer model.
    # The tasks for a donor will be read one line at a time and put into a bounded queue.
    for i in range(donors):
        donor_label = os.path.basename(args.illumina_bam[i]).split('.')[0]
        print(f"Processing donor: {donor_label}")
        # Create a new bounded queue for this donor.
        task_queue = multiprocessing.Queue(maxsize=args.queue_size)
        # Start worker processes for this donor.
        workers = []
        for _ in range(args.num_cores):
            p = multiprocessing.Process(
                target=worker_loop,
                args=(
                    task_queue, vcf_header, args.reference_fasta, args.min_mapq,
                    args.min_baseq, args.nano_threshold, temp_dir,
                    bam_template_path,
                    # args.illumina_single_alt
                )
            )
            p.start()
            workers.append(p)
        # Open the donor's BED file (using pysam.TabixFile to stream lines)
        bed_path = args.nanoseq_bed[i]
        with pysam.TabixFile(bed_path) as tbx:
            # for j, line in enumerate(tbx.fetch()):
            for line in tbx.fetch():
                parts = line.split()
                if len(parts) < 3:
                    continue
                chrom, start, end = parts[0], int(parts[1]), int(parts[2])
                task = (
                    donor_label, args.illumina_bam[i], args.nanoseq_bam[i],
                    (chrom, start, end)
                )
                task_queue.put(task)  # This will block if the queue is full.
        # After queuing all tasks for this donor, put one sentinel (None) per
        # worker. Signalling the end of tasks for this donor.
        for _ in range(args.num_cores):
            task_queue.put(None)
        # Wait for all workers for this donor to finish.
        for p in workers:
            p.join()
        print(f"Donor {donor_label} finished.")

    # At this point, all donors have been processed.
    # Now merge the core-specific temporary files in temp_dir.
    mut_bam_files = []
    art_bam_files = []
    mut_vcf_files = []
    art_vcf_files = []
    for fname in os.listdir(temp_dir):
        if fname.endswith(".mutation.bam"):
            mut_bam_files.append(os.path.join(temp_dir, fname))
        elif fname.endswith(".artefact.bam"):
            art_bam_files.append(os.path.join(temp_dir, fname))
        elif fname.endswith(".mutation.vcf"):
            mut_vcf_files.append(os.path.join(temp_dir, fname))
        elif fname.endswith(".artefact.vcf"):
            art_vcf_files.append(os.path.join(temp_dir, fname))

    def merge_files(output_path, file_list, file_type):
        if file_type == "bam":
            pysam.merge("-f", "tmp_" + output_path, *file_list)
            pysam.sort("-o", output_path, "tmp_" + output_path)
            pysam.index(output_path)
            os.remove("tmp_" + output_path)
        elif file_type == "vcf":
            # First, create a merged unsorted VCF file by streaming records
            # from each file. Assume that all VCFs share the same header; here
            # we get the header from the first file.
            with pysam.VariantFile(file_list[0], "r") as vcf_in:
                header = vcf_in.header
            merged_unsorted = output_path + ".merged.unsorted.vcf"
            with pysam.VariantFile(
                merged_unsorted, "w", header=header
            ) as vcf_out:
                for f in file_list:
                    with pysam.VariantFile(f, "r") as vcf_in:
                        for rec in vcf_in:
                            vcf_out.write(rec)
            # bcftools sort is disk-based and avoids loading all records into
            # memory.
            tmp_sorted = output_path + ".tmp.vcf"
            subprocess.check_call(
                ["bcftools", "sort", "-o", tmp_sorted, merged_unsorted]
            )
            # Compress and index the sorted VCF using pysam.
            pysam.tabix_compress(tmp_sorted, output_path, force=True)
            pysam.tabix_index(output_path, preset="vcf", force=True)
            os.remove(merged_unsorted)
            os.remove(tmp_sorted)
        else:
            raise ValueError("Unknown file type: " + file_type)

    final_mut_bam = os.path.join(
        args.temp_dir, f"{args.output_prefix}.mutation.merged.bam"
    )
    final_art_bam = os.path.join(
        args.temp_dir, f"{args.output_prefix}.artefact.merged.bam"
    )
    final_mut_vcf = os.path.join(
        args.temp_dir, f"{args.output_prefix}.mutation.merged.vcf.gz"
    )
    final_art_vcf = os.path.join(
        args.temp_dir, f"{args.output_prefix}.artefact.merged.vcf.gz"
    )

    print("Merging core-specific BAMs...")
    if mut_bam_files:
        merge_files(final_mut_bam, mut_bam_files, "bam")
    if art_bam_files:
        merge_files(final_art_bam, art_bam_files, "bam")
    print("Merging core-specific VCFs...")
    if mut_vcf_files:
        merge_files(final_mut_vcf, mut_vcf_files, "vcf")
    if art_vcf_files:
        merge_files(final_art_vcf, art_vcf_files, "vcf")

    # Clean up the temporary directory.
    shutil.rmtree(temp_dir)
    print("All done.")
    print("Final merged mutation BAM:", final_mut_bam)
    print("Final merged artefact BAM:", final_art_bam)
    print("Final merged mutation VCF:", final_mut_vcf)
    print("Final merged artefact VCF:", final_art_vcf)


if __name__ == "__main__":
    main()
