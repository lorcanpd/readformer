import os
import pandas as pd
import pysam
import random
from concurrent.futures import ProcessPoolExecutor
import tempfile
import shutil


def split_vcf_into_chunks(vcf_file, temp_dir, num_chunks=10):
    """
    Split the input VCF file into smaller chunks for parallel processing using
    its index file.

    :param vcf_file:
        Path to the input VCF file.
    :param num_chunks:
        Number of chunks to split into.
    :return:
        List of paths to the chunk files.
    """
    vcf = pysam.VariantFile(vcf_file, 'r')

    # Get all contigs and their lengths from the header
    contigs = [
        (contig, header.length) for contig, header in vcf.header.contigs.items()
    ]

    # Calculate total number of records in the VCF file
    total_records = sum(len(list(vcf.fetch(contig))) for contig, _ in contigs)

    # Determine chunk size
    chunk_size = max(1, total_records // num_chunks)

    chunk_count = 0
    chunk_files = []
    records = []

    # strip extension from vcf file basename
    vcf_name = os.path.basename(vcf_file).split('.')[0]

    for contig, length in contigs:
        for start in range(0, length, chunk_size):
            end = min(start + chunk_size, length)
            for record in vcf.fetch(contig, start, end):
                records.append(record)
                if len(records) >= chunk_size:
                    chunk_file = write_chunk(
                        vcf_name, records, vcf.header, chunk_count, temp_dir
                    )
                    chunk_files.append(chunk_file)
                    records = []
                    chunk_count += 1

    # Write remaining records if any
    if records:
        chunk_file = write_chunk(
            vcf_name, records, vcf.header, chunk_count, temp_dir
        )
        chunk_files.append(chunk_file)

    return chunk_files


def write_chunk(vcf_name, records, header, chunk_count, temp_dir):
    """
    Write a chunk of records to a temporary VCF file.

    :param vcf_name:
        Name of the VCF file.
    :param records:
        List of VariantRecords.
    :param header:
        VCF header.
    :param chunk_count:
        Chunk index.
    :return:
        Path to the chunk file.
    """
    chunk_file = os.path.join(temp_dir, f'{vcf_name}_chunk_{chunk_count}.vcf')
    with pysam.VariantFile(chunk_file, 'w', header=header) as vcf_out:
        for record in records:
            vcf_out.write(record)
    return chunk_file


def read_metadata(metadata_file):
    return pd.read_csv(metadata_file)


def extract_variants(vcf_file):
    variants = []
    with pysam.VariantFile(vcf_file) as vcf_in:
        for record in vcf_in:
            if record.alts is not None:
                for alt in record.alts:
                    variants.append({
                        'Chr': record.chrom,
                        'Pos': record.pos,
                        'Ref': record.ref,
                        'Alt': alt
                    })
    return pd.DataFrame(variants)


def fetch_reads(bam_file, chrom, pos, ref, alt, min_quality):
    bam = pysam.AlignmentFile(bam_file, "rb")
    supporting_reads = []
    other_reads = []
    for read in bam.fetch(chrom, pos - 1, pos):
        if read.is_unmapped or read.mapping_quality < min_quality:
            continue
        read_pos = read.get_reference_positions()
        if pos - 1 in read_pos:
            read_base = read.query_sequence[read_pos.index(pos - 1)]
            read_id = read.query_name
            if read_base == alt:
                supporting_reads.append(read_id)
            elif read_base == ref:
                other_reads.append(read_id)
    bam.close()
    return supporting_reads, other_reads


def generate_samples(
        bam_file, variants, read_supports, sequencing_depths,
        samples_per_combination, min_quality
):
    sample_data = []
    for index, row in variants.iterrows():
        chrom = row['Chr']
        pos = row['Pos']
        ref = row['Ref']
        alt = row['Alt']

        supporting_reads, other_reads = fetch_reads(
            bam_file, chrom, pos, ref, alt, min_quality
        )

        for total_depth in sequencing_depths:
            for num_support_reads in read_supports:
                supporting_reads_for_num_support = supporting_reads.copy()

                for _ in range(samples_per_combination):
                    if len(
                            supporting_reads_for_num_support
                    ) < num_support_reads:
                        break  # Not enough supporting reads
                    sampled_supporting_reads = random.sample(
                        supporting_reads_for_num_support, num_support_reads
                    )
                    remainder = total_depth - num_support_reads
                    if remainder > len(other_reads):
                        break
                    sampled_other_reads = random.sample(
                        other_reads, remainder
                    )

                    sample_data.append({
                        'bam': bam_file,
                        'Chr': chrom,
                        'Pos': pos,
                        'Ref': ref,
                        'Alt': alt,
                        'num_support_reads': num_support_reads,
                        'total_depth': total_depth,
                        'supporting_read_ids': sampled_supporting_reads,
                        'other_read_ids': sampled_other_reads
                    })
                    # Ensure diversity by removing sampled reads from the pools
                    supporting_reads = [
                        read for read in supporting_reads_for_num_support
                        if read not in set(sampled_supporting_reads)
                    ]
    return sample_data


def process_bam_vcf_pair(
        bam_file, vcf_file, read_supports, sequencing_depths,
        samples_per_combination, min_quality
):
    variants = extract_variants(vcf_file)
    return generate_samples(
        bam_file, variants, read_supports, sequencing_depths,
        samples_per_combination, min_quality
    )


def main():
    metadata_file = 'GIAB_VCFs/finetuning_negatives_creation_metadata.csv'
    bam_dir = 'GIAB_BAM/illumina_2x250bps'
    vcf_dir = 'GIAB_VCFs/artefacts'
    output_csv = 'GIAB_VCFs/artefacts/test_finetuning_false_samples_dataset.csv'
    intermediate_dir = tempfile.mkdtemp()

    read_supports = [1, 2, 3, 4, 5]
    sequencing_depths = [10, 15, 20, 25, 30]
    samples_per_combination = 3
    min_quality = 30  # Minimum read quality threshold

    metadata = read_metadata(metadata_file)
    all_samples = []

    if not os.path.exists(intermediate_dir):
        os.makedirs(intermediate_dir)

    # Check that there are 8 cores available for parallel processing. If not,
    # reduce the number of cores to those available - which ever is less.
    total_cores = os.cpu_count()
    process_pool_cores = min(8, total_cores)

    with ProcessPoolExecutor(max_workers=process_pool_cores) as executor:
        futures = []
        for index, row in metadata.iterrows():
            bam_file = os.path.join(bam_dir, row['bam'])
            vcf_file = os.path.join(vcf_dir, row['vcf'])

            # Check VCF has an index file, if not use pysam to create one
            if not os.path.exists(f"{vcf_file}.tbi"):
                print(f"Indexing VCF file: {vcf_file}")
                pysam.tabix_index(vcf_file, preset='vcf')

            # Split the VCF file into chunks
            print(f"Splitting VCF into chunks: {vcf_file}")
            chunk_files = split_vcf_into_chunks(
                vcf_file, intermediate_dir, num_chunks=100
            )

            for chunk_file in chunk_files:
                print(f"Processing BAM: {bam_file} VCF Chunk: {chunk_file}")

                futures.append(executor.submit(
                    process_bam_vcf_pair, bam_file, chunk_file, read_supports,
                    sequencing_depths, samples_per_combination, min_quality
                ))

        for future in futures:
            all_samples.extend(future.result())

    # Combine intermediate results
    output_df = pd.DataFrame(all_samples)
    output_df.to_csv(output_csv, index=False)
    print(f"Output CSV created: {output_csv}")

    # Calculate and print the total number of samples created at each
    # depth/read support combination
    summary = output_df.groupby(
        ['num_support_reads', 'total_depth']
    ).size().reset_index(name='count')
    print(
        "\nSummary of samples created at each depth/read support combination:"
    )
    print(summary)

    # Remove intermediate directory and its contents.
    shutil.rmtree(intermediate_dir)


if __name__ == "__main__":
    main()
