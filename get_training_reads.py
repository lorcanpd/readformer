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
import subprocess
import heapq  # For k-way merging VCFs using a heap

# =============================================================================
# Global variables for worker processes.
global_vcf_header = None
global_min_mapq = None
global_min_baseq = None
global_nano_threshold = None
global_temp_dir = None
global_outputs = {}       # keys: "mutation_bam", "artefact_bam", "mutation_vcf", "artefact_vcf"
global_bam_template = None  # pysam.AlignmentFile used as template

# New globals to hold open file handles for the donor.
global_illumina_bam = None
global_nanoseq_bam = None
global_fasta_handle = None  # Reference FASTA handle


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
    header.add_line("##INFO=<ID=READ_IDS,Number=.,Type=String,Description=\"Read IDs supporting the variant\">")
    header.add_line("##INFO=<ID=ILLUMINA_ALT_COUNT,Number=1,Type=Integer,Description=\"Illumina alt count\">")
    header.add_line("##INFO=<ID=ILLUMINA_FORWARD,Number=1,Type=Integer,Description=\"Illumina alt count on forward strand\">")
    header.add_line("##INFO=<ID=ILLUMINA_REVERSE,Number=1,Type=Integer,Description=\"Illumina alt count on reverse strand\">")
    header.add_line("##INFO=<ID=NANO_ALT_COUNT,Number=1,Type=Integer,Description=\"Alt supporting duplex bundle count in NanoSeq\">")
    return header


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


def worker_init(vcf_header, ref_fasta_path, min_mapq, min_baseq, nano_thr, temp_dir,
                bam_template_path, illumina_bam_path, nanoseq_bam_path):
    """
    Initialiser that sets global variables and opens the donor-specific file handles.
    """
    global global_vcf_header, global_min_mapq, global_min_baseq, global_nano_threshold, global_temp_dir
    global global_outputs, global_bam_template, global_illumina_bam, global_nanoseq_bam, global_fasta_handle

    global_vcf_header = vcf_header
    global_min_mapq = min_mapq
    global_min_baseq = min_baseq
    global_nano_threshold = nano_thr
    global_temp_dir = temp_dir

    global_bam_template = pysam.AlignmentFile(bam_template_path, "rb")
    # Open the donor-specific BAM files and reference FASTA once.
    global_illumina_bam = pysam.AlignmentFile(illumina_bam_path, "rb")
    global_nanoseq_bam = pysam.AlignmentFile(nanoseq_bam_path, "rb")
    global_fasta_handle = pysam.FastaFile(ref_fasta_path)

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


def process_region(task):
    """
    Processes a region using a cached read collection and open file handles.
    Task tuple: (donor_label, illumina_bam_path, nanoseq_bam_path, region)
    """
    donor_label, _, _, region = task
    chrom, start, end = region

    # Fetch the entire reference sequence for this region once.
    region_ref = global_fasta_handle.fetch(chrom, start, end)

    for col in global_illumina_bam.pileup(
            chrom, start, end, truncate=True,
            min_mapping_quality=global_min_mapq, stepper='all'):
        pos = col.pos
        if not (start <= pos < end):
            continue

        base_counts = defaultdict(lambda: {"forward": 0, "reverse": 0})
        read_cache = defaultdict(list)

        # Process each read in the pileup column.
        for pread in col.pileups:
            if pread.is_del or pread.is_refskip:
                continue
            read = pread.alignment
            qpos = pread.query_position
            # Streamlined filtering conditions.
            if qpos is None:
                continue
            if read.mapping_quality < global_min_mapq or read.query_qualities[qpos] < global_min_baseq:
                continue
            if not (read.is_paired and read.is_proper_pair):
                continue

            base = read.query_sequence[qpos].upper()
            if read.is_reverse:
                base_counts[base]["reverse"] += 1
            else:
                base_counts[base]["forward"] += 1
            read_cache[base].append(read)

        total_depth = sum(sum(d.values()) for d in base_counts.values())
        if total_depth < 40:
            continue

        try:
            # Get the reference base from the cached region sequence.
            ref_base = region_ref[pos - start].upper()
        except IndexError:
            ref_base = "N"

        for alt_base, counts in base_counts.items():
            if alt_base == ref_base:
                continue
            alt_count = counts["forward"] + counts["reverse"]
            is_balanced = (counts["forward"] > 0 and counts["reverse"] > 0)
            duplex_count = pileup_duplex_bundle_count(global_nanoseq_bam, chrom, pos, alt_base)

            if duplex_count >= global_nano_threshold:
                alt_reads = read_cache[alt_base]
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
                alt_reads = read_cache[alt_base]
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
    return donor_label, region


def worker_loop(task_queue, vcf_header, ref_fasta_path, min_mapq, min_baseq, nano_thr,
                temp_dir, bam_template_path):
    """
    Worker loop that initialises file handles from the first task and then
    processes subsequent tasks.
    """
    # Retrieve the first task to initialise global file handles.
    first_task = task_queue.get()
    if first_task is None:
        return
    donor_label, illumina_bam_path, nanoseq_bam_path, _ = first_task
    worker_init(vcf_header, ref_fasta_path, min_mapq, min_baseq, nano_thr, temp_dir,
                bam_template_path, illumina_bam_path, nanoseq_bam_path)
    # Process the first task.
    process_region(first_task)
    # Process remaining tasks.
    while True:
        task = task_queue.get()
        if task is None:
            break
        process_region(task)

    # Close open file handles and output files.
    global_illumina_bam.close()
    global_nanoseq_bam.close()
    global_fasta_handle.close()
    for f in global_outputs.values():
        f.close()


def merge_files(output_path, file_list, file_type):
    """
    Merge files of a given type into output_path.
    For BAM, we merge and then sort using pysam.
    For VCF, we perform a k-way merge using a heap (qheap) so that only one record
    per file is in memory.
    """
    output_dir = os.path.dirname(output_path)
    output_filename = os.path.basename(output_path)

    if file_type == "bam":
        tmp_file = os.path.join(output_dir, "tmp_" + output_filename)
        pysam.merge("-f", tmp_file, *file_list)
        pysam.sort("-o", output_path, tmp_file)
        pysam.index(output_path)
        os.remove(tmp_file)
    elif file_type == "vcf":
        # Use a k-way merge with heapq.
        # Get the header from the first VCF.
        with pysam.VariantFile(file_list[0], "r") as vcf_in:
            header = vcf_in.header
        out_vcf = pysam.VariantFile(output_path, "w", header=header)
        # Build an ordering dictionary for contigs from the header.
        header_order = {contig: idx for idx, contig in enumerate(header.contigs)}
        heap = []
        file_iters = []
        # Open each VCF file and push its first record.
        for f in file_list:
            vcf_in = pysam.VariantFile(f, "r")
            file_iters.append(vcf_in)
            try:
                rec = next(vcf_in)
                # Use header_order to get a numerical ordering for the contig.
                key = (header_order.get(rec.contig, 9999), rec.pos)
                heapq.heappush(heap, (key, rec, vcf_in))
            except StopIteration:
                vcf_in.close()
        # Merge records in sorted order.
        while heap:
            key, rec, vcf_in = heapq.heappop(heap)
            out_vcf.write(rec)
            try:
                next_rec = next(vcf_in)
                next_key = (header_order.get(next_rec.contig, 9999), next_rec.pos)
                heapq.heappush(heap, (next_key, next_rec, vcf_in))
            except StopIteration:
                vcf_in.close()
        out_vcf.close()
    else:
        raise ValueError("Unknown file type: " + file_type)


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
    first_illumina.close()

    # Process each donor using a producer-consumer model.
    for i in range(donors):
        donor_label = os.path.basename(args.illumina_bam[i]).split('.')[0]
        print(f"Processing donor: {donor_label}")
        task_queue = multiprocessing.Queue(maxsize=args.queue_size)
        workers = []
        for _ in range(args.num_cores):
            p = multiprocessing.Process(
                target=worker_loop,
                args=(
                    task_queue, vcf_header, args.reference_fasta, args.min_mapq,
                    args.min_baseq, args.nano_threshold, temp_dir,
                    bam_template_path,
                )
            )
            p.start()
            workers.append(p)
        bed_path = args.nanoseq_bed[i]
        with pysam.TabixFile(bed_path) as tbx:
            # for j, line in enumerate(tbx.fetch()):
            for line in tbx.fetch():
                parts = line.split()
                if len(parts) < 3:
                    continue
                chrom, start, end = parts[0], int(parts[1]), int(parts[2])
                task = (donor_label, args.illumina_bam[i], args.nanoseq_bam[i], (chrom, start, end))
                task_queue.put(task)
                # # For debugging or testing: limit to 1000 regions.
                # if j == 1000:
                #     break
        for _ in range(args.num_cores):
            task_queue.put(None)
        for p in workers:
            p.join()
        print(f"Donor {donor_label} finished.")

    # Gather core-specific temporary files.
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

    # Merge the files using the revised merge_files() function.
    final_mut_bam = os.path.join(args.temp_dir, f"{args.output_prefix}.mutation.merged.bam")
    final_art_bam = os.path.join(args.temp_dir, f"{args.output_prefix}.artefact.merged.bam")
    final_mut_vcf = os.path.join(args.temp_dir, f"{args.output_prefix}.mutation.merged.vcf.gz")
    final_art_vcf = os.path.join(args.temp_dir, f"{args.output_prefix}.artefact.merged.vcf.gz")

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

    shutil.rmtree(temp_dir)
    print("All done.")
    print("Final merged mutation BAM:", final_mut_bam)
    print("Final merged artefact BAM:", final_art_bam)
    print("Final merged mutation VCF:", final_mut_vcf)
    print("Final merged artefact VCF:", final_art_vcf)


if __name__ == "__main__":
    main()
