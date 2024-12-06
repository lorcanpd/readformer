import pysam
import pandas as pd
import os
from tqdm import tqdm
import hashlib
import csv

from components.extract_reads import extract_read_by_id, decode_orientation_flags

# Configuration
K_FOLDS = 5
MUTATIONS_VCF_PATH = 'TEST_DATA/fine_tuning/mutation_reads.vcf.gz'
MUTATIONS_BAM_PATH = 'TEST_DATA/fine_tuning/mutation_reads.bam'
ARTEFACTS_VCF_PATH = 'TEST_DATA/fine_tuning/HG002_artefacts_cleaned.vcf.gz'
ARTEFACTS_BAM_PATH = 'TEST_DATA/fine_tuning/HG002_artefacts.bam'
OUTPUT_DIR = 'TEST_DATA/fine_tuning/cross_validation_splits'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_fold_id(identifier, k=K_FOLDS):
    """
    Assign a fold based on the hash of the identifier.
    """
    hash_digest = hashlib.md5(identifier.encode()).hexdigest()
    return int(hash_digest, 16) % k


# def count_entries_per_fold(vcf_path, k=K_FOLDS):
#     """
#     Count the number of entries per fold.
#     """
#     counts = {fold: 0 for fold in range(k)}
#     vcf_in = pysam.VariantFile(vcf_path)
#     for record in vcf_in:
#         if 'READ_IDS' in record.info:
#             chrom = record.chrom
#             pos = record.pos
#             ref = record.ref
#             alt = ','.join([str(a) for a in record.alts])
#             mutation_id = f"{chrom}:{pos}:{ref}>{alt}"
#             fold_id = get_fold_id(mutation_id)
#             counts[fold_id] += 1
#         else:
#             read_id = record.info.get(
#                 'ILLUMINA_READ_NAME', [])[0]
#             if read_id:
#                 fold_id = get_fold_id(read_id)
#                 counts[fold_id] += 1
#     return counts


def process_mutations_vcf(vcf_path, output_dir, k=K_FOLDS):
    """
    Process mutations VCF and assign each mutation to a fold.
    """
    print("Processing Mutations VCF...")
    writers = {}
    files = {}
    headers = [
        'CHROM', 'POS', 'ID', 'REF', 'ALT', 'READ_ID', 'mapped_to_reverse',
        'read_support',
    ]
    for fold in range(k):
        out_path = os.path.join(output_dir, f'mutations_fold_{fold}.csv')
        files[fold] = open(out_path, 'w', newline='')
        writers[fold] = csv.writer(files[fold])

        writers[fold].writerow(headers)

    vcf_in = pysam.VariantFile(vcf_path)
    bam_file = pysam.AlignmentFile(MUTATIONS_BAM_PATH, 'rb')

    for record in tqdm(vcf_in, desc="Mutations"):
        chrom = record.chrom
        pos = record.pos - 1
        ref = record.ref
        alt = ','.join([str(a) for a in record.alts])
        mutation_id = f"{chrom}:{pos}:{ref}>{alt}"
        fold_id = get_fold_id(mutation_id)
        read_ids = record.info.get('READ_IDS', [])
        if isinstance(read_ids, str):
            read_ids = read_ids.split(',')

        # Fetch all reads overlapping the position once
        reads_at_pos = bam_file.fetch(chrom, pos, pos + 1)
        read_id_to_read = {}
        for read in reads_at_pos:
            read_id_to_read[read.query_name] = read

        rows_to_write = []
        num_reverse = 0
        num_forward = 0
        for read_id in read_ids:
            read = read_id_to_read.get(read_id)
            if read is not None:
                mapped_to_reverse = read.is_reverse

                # positions = read.get_reference_positions(full_length=True)
                # query_sequence = read.query_sequence
                #
                # try:
                #     index_in_read = positions.index(pos)
                #     base_in_read = query_sequence[index_in_read]
                # except ValueError:
                #     # Position not found in read
                #     continue

                # get aligned pairs
                aligned_pairs = read.get_aligned_pairs(matches_only=False)
                base_in_read = None

                for query_pos, ref_pos in aligned_pairs:
                    if ref_pos == pos:
                        if query_pos is not None:
                            base_in_read = read.query_sequence[query_pos]
                        break

                if base_in_read.upper() != alt.upper():
                    # Due to bug in the extraction code.
                    continue

                if mapped_to_reverse:
                    num_reverse += 1
                else:
                    num_forward += 1

                rows_to_write.append(
                    [chrom, pos, '.', ref, alt, read_id, mapped_to_reverse]
                )

            else:
                continue

        for row in rows_to_write:
            if row[-1]:
                row.append(num_reverse)
            else:
                row.append(num_forward)
            writers[fold_id].writerow(row)

    for fold in range(k):
        files[fold].close()

    # print("Mutations VCF processing completed.")
    # # counts = count_entries_per_fold(vcf_path, k)
    # return counts


# def process_artefacts_vcf(vcf_path, output_dir, k=K_FOLDS):
#     """
#     Process artefacts VCF and assign each artefact to a fold.
#     """
#     print("Processing Artefacts VCF...")
#     writers = {}
#     files = {}
#     for fold in range(k):
#         out_path = os.path.join(
#             output_dir, f'artefacts_fold_{fold}.csv')
#         files[fold] = open(out_path, 'w', newline='')
#         writers[fold] = csv.writer(files[fold])
#         headers = [
#             'CHROM', 'POS', 'ID', 'REF', 'ALT', 'READ_ID', 'mapped_to_reverse',
#             'read_support',
#         ]
#         writers[fold].writerow(headers)
#
#     vcf_in = pysam.VariantFile(vcf_path)
#     for record in tqdm(vcf_in, desc="Artefacts"):
#         chrom = record.chrom
#         pos = record.pos - 1  # VCF is 1-based, so we subtract 1 for 0-based used by BAM
#         ref = record.ref
#         alt = ','.join([str(a) for a in record.alts])
#         read_id = record.info.get('ILLUMINA_READ_NAME', [])[0]
#
#         if not read_id:
#             continue
#
#         fold_id = get_fold_id(read_id)
#
#         # get mapped_to_reverse from BAM file
#         read_dict = extract_read_by_id(ARTEFACTS_BAM_PATH, chrom, pos, read_id)
#         if read_dict is not None:
#             mapped_to_reverse = decode_orientation_flags(
#                 read_dict['bitwise_flags']
#             )['is_reverse']
#         else:
#             breakpoint()
#
#         row = [
#             chrom, pos, '.', ref, alt, read_id, mapped_to_reverse, 1
#         ]
#         writers[fold_id].writerow(row)
#
#     for fold in range(k):
#         files[fold].close()
#
#     # print("Artefacts VCF processing completed.")
#     # # counts = count_entries_per_fold(vcf_path, k)
#     # return counts

def process_artefacts_vcf(vcf_path, output_dir, k=K_FOLDS):
    """
    Process artefacts VCF and assign each artefact to a fold.
    """
    print("Processing Artefacts VCF...")
    writers = {}
    files = {}
    headers = [
        'CHROM', 'POS', 'ID', 'REF', 'ALT', 'READ_ID', 'mapped_to_reverse',
        'read_support',
    ]
    for fold in range(k):
        out_path = os.path.join(
            output_dir, f'artefacts_fold_{fold}.csv')
        files[fold] = open(out_path, 'w', newline='')
        writers[fold] = csv.writer(files[fold])
        writers[fold].writerow(headers)

    vcf_in = pysam.VariantFile(vcf_path)
    bam_file = pysam.AlignmentFile(ARTEFACTS_BAM_PATH, 'rb')

    # Create a cache to store reads at positions
    position_cache = {}

    for record in tqdm(vcf_in, desc="Artefacts"):
        chrom = record.chrom
        pos = record.pos - 1  # VCF is 1-based, so we subtract 1 for 0-based used by BAM
        ref = record.ref
        alt = ','.join([str(a) for a in record.alts])
        read_id = record.info.get('ILLUMINA_READ_NAME')

        if not read_id:
            continue

        fold_id = get_fold_id(read_id)

        # Normalise contig names if necessary
        if chrom not in bam_file.references:
            adjusted_chrom = 'chr' + chrom
            if adjusted_chrom not in bam_file.references:
                print(f"Contig {chrom} not found in BAM file.")
                continue
        else:
            adjusted_chrom = chrom

        # Use a cache to avoid fetching the same position multiple times
        cache_key = (adjusted_chrom, pos)
        if cache_key in position_cache:
            read_id_to_read = position_cache[cache_key]
        else:
            # Fetch all reads overlapping the position once
            try:
                reads_at_pos = bam_file.fetch(adjusted_chrom, pos, pos + 1)
            except ValueError as e:
                print(f"Error fetching reads for {adjusted_chrom}:{pos} - {e}")
                continue
            read_id_to_read = {read.query_name: read for read in reads_at_pos}
            position_cache[cache_key] = read_id_to_read

        read = read_id_to_read.get(read_id)
        if read is not None:
            mapped_to_reverse = read.is_reverse

            row = [
                chrom, pos, '.', ref, alt, read_id, mapped_to_reverse, 1
            ]
            writers[fold_id].writerow(row)
        else:
            # Read not found; could be due to a mismatch or the read not overlapping the position
            continue

    bam_file.close()
    for fold in range(k):
        files[fold].close()


def count_class_samples(output_dir, k=K_FOLDS):

    mutation_class_counts = {}
    artefact_class_counts = {}

    for fold in range(k):
        mutations_path = os.path.join(
            output_dir, f'mutations_fold_{fold}.csv')
        artefacts_path = os.path.join(
            output_dir, f'artefacts_fold_{fold}.csv')

        # Count the number of unique mutation IDs for mutations
        mutations_df = pd.read_csv(mutations_path)
        # This should be the intersection of (chr, pos) and mapped_to_reverse.
        num_mutation_classes = mutations_df.groupby(
            ['CHROM', 'POS', 'mapped_to_reverse']).ngroups
        mutation_class_counts[fold] = num_mutation_classes

        # Count the number of artefacts for artefacts
        artefacts_df = pd.read_csv(artefacts_path)
        num_artefact_classes = len(artefacts_df)
        artefact_class_counts[fold] = num_artefact_classes


    return mutation_class_counts, artefact_class_counts


def combine_fold_csvs(
        output_dir, k=K_FOLDS, mutation_class_counts=None,
        artefact_class_counts=None
):
    """
    For each fold, combine train and test CSVs with a classification label.
    Mutation reads have label 1.0 and artefact reads have label 0.0.
    """
    for fold in range(k):
        print(f"Combining CSVs for Fold {fold}...")

        # Paths to individual test CSVs
        mutations_path = os.path.join(output_dir, f'mutations_fold_{fold}.csv')
        artefacts_path = os.path.join(output_dir, f'artefacts_fold_{fold}.csv')
        test_combined_path = os.path.join(output_dir, f'test_fold_{fold}.csv')

        # Paths to individual train CSVs (excluding current fold)
        train_combined_path = os.path.join(output_dir, f'train_fold_{fold}.csv')

        with open(
                test_combined_path, 'w', newline=''
        ) as test_out, open(
            train_combined_path, 'w', newline=''
        ) as train_out:
            test_writer = csv.writer(test_out)
            train_writer = csv.writer(train_out)

            # Write header with label
            header = [
                'CHROM', 'POS', 'ID', 'REF', 'ALT', 'READ_ID',
                'mapped_to_reverse', 'read_support', 'num_in_class', 'label'
            ]
            test_writer.writerow(header)
            train_writer.writerow(header)

            # Write mutations test with label 1.0
            with open(mutations_path, 'r') as mut_test:
                reader = csv.reader(mut_test)
                next(reader)  # Skip header
                for row in reader:
                    row.append(mutation_class_counts[fold])
                    row.append(1.0)
                    test_writer.writerow(row)

            # Write artefacts test with label 0.0
            with open(artefacts_path, 'r') as art_test:
                reader = csv.reader(art_test)
                next(reader)  # Skip header
                for row in reader:
                    row.append(artefact_class_counts[fold])
                    row.append(0.0)
                    test_writer.writerow(row)

            train_mut_count = sum(
                mutation_class_counts.values()) - mutation_class_counts[fold]
            train_art_count = sum(
                artefact_class_counts.values()) - artefact_class_counts[fold]

            # Iterate over all folds except the current one for training data
            for other_fold in range(k):
                if other_fold == fold:
                    continue

                # Write mutations train with label 1.0
                mutations_train_path = os.path.join(
                    output_dir, f'mutations_fold_{other_fold}.csv')
                with open(mutations_train_path, 'r') as mut_train:
                    reader = csv.reader(mut_train)
                    next(reader)  # Skip header
                    for row in reader:
                        row.append(train_mut_count)
                        row.append(1.0)
                        train_writer.writerow(row)

                # Write artefacts train with label 0.0
                artefacts_train_path = os.path.join(
                    output_dir, f'artefacts_fold_{other_fold}.csv')
                with open(artefacts_train_path, 'r') as art_train:
                    reader = csv.reader(art_train)
                    next(reader)  # Skip header
                    for row in reader:
                        row.append(train_art_count)
                        row.append(0.0)
                        train_writer.writerow(row)

        print(
            f"Fold {fold} CSVs created: train_fold_{fold}.csv and "
            f"test_fold_{fold}.csv with classification labels.")


def main():
    process_mutations_vcf(
        MUTATIONS_VCF_PATH, OUTPUT_DIR, K_FOLDS)
    process_artefacts_vcf(
        ARTEFACTS_VCF_PATH, OUTPUT_DIR, K_FOLDS)

    # get class counts for each fold
    mutation_class_counts, artefect_class_counts = count_class_samples(
        OUTPUT_DIR, K_FOLDS)

    combine_fold_csvs(
        OUTPUT_DIR, K_FOLDS, mutation_class_counts, artefect_class_counts)
    print("All cross-validation splits have been created.")


if __name__ == "__main__":
    main()
