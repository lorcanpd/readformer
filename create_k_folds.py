import pysam
import pandas as pd
import os
from tqdm import tqdm
import hashlib
import csv

# Configuration
K_FOLDS = 5
MUTATIONS_VCF_PATH = 'TEST_DATA/fine_tuning/mutation_reads.vcf.gz'
ARTEFACTS_VCF_PATH = 'TEST_DATA/fine_tuning/HG002_artefacts_cleaned.vcf.gz'
OUTPUT_DIR = 'TEST_DATA/fine_tuning/cross_validation_splits'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_fold_id(identifier, k=K_FOLDS):
    """
    Assign a fold based on the hash of the identifier.
    """
    hash_digest = hashlib.md5(identifier.encode()).hexdigest()
    return int(hash_digest, 16) % k


def count_entries_per_fold(vcf_path, k=K_FOLDS):
    """
    Count the number of entries per fold.
    """
    counts = {fold: 0 for fold in range(k)}
    vcf_in = pysam.VariantFile(vcf_path)
    for record in vcf_in:
        if 'READ_IDS' in record.info:
            chrom = record.chrom
            pos = record.pos
            ref = record.ref
            alt = ','.join([str(a) for a in record.alts])
            mutation_id = f"{chrom}:{pos}:{ref}>{alt}"
            fold_id = get_fold_id(mutation_id)
            counts[fold_id] += 1
        else:
            read_id = record.info.get('ILLUMINA_READ_NAME', [])[0]
            if read_id:
                fold_id = get_fold_id(read_id)
                counts[fold_id] += 1
    return counts


def process_mutations_vcf(vcf_path, output_dir, k=K_FOLDS):
    """
    Process mutations VCF and assign each mutation to a fold.
    """
    print("Processing Mutations VCF...")
    writers = {}
    files = {}
    headers = [
        'CHROM', 'POS', 'ID', 'REF', 'ALT', 'READ_ID',
        'read_support'
    ]
    for fold in range(k):
        out_path = os.path.join(output_dir, f'mutations_fold_{fold}.csv')
        files[fold] = open(out_path, 'w', newline='')
        writers[fold] = csv.writer(files[fold])

        writers[fold].writerow(headers)

    vcf_in = pysam.VariantFile(vcf_path)
    for record in tqdm(vcf_in, desc="Mutations"):
        chrom = record.chrom
        pos = record.pos
        ref = record.ref
        alt = ','.join([str(a) for a in record.alts])
        mutation_id = f"{chrom}:{pos}:{ref}>{alt}"
        fold_id = get_fold_id(mutation_id)
        read_ids = record.info.get('READ_IDS', [])
        read_support = len(read_ids)
        if isinstance(read_ids, str):
            read_ids = read_ids.split(',')
        for read_id in read_ids:
            row = [
                chrom, pos, '.', ref, alt, read_id, read_support
            ]
            writers[fold_id].writerow(row)

    for fold in range(k):
        files[fold].close()

    print("Mutations VCF processing completed.")
    counts = count_entries_per_fold(vcf_path, k)
    return counts


def process_artefacts_vcf(vcf_path, output_dir, k=K_FOLDS):
    """
    Process artefacts VCF and assign each artefact to a fold.
    """
    print("Processing Artefacts VCF...")
    writers = {}
    files = {}
    for fold in range(k):
        out_path = os.path.join(
            output_dir, f'artefacts_fold_{fold}.csv')
        files[fold] = open(out_path, 'w', newline='')
        writers[fold] = csv.writer(files[fold])
        headers = [
            'CHROM', 'POS', 'ID', 'REF', 'ALT', 'READ_ID',
            'read_support'
        ]
        writers[fold].writerow(headers)


    vcf_in = pysam.VariantFile(vcf_path)
    for record in tqdm(vcf_in, desc="Artefacts"):
        chrom = record.chrom
        pos = record.pos
        ref = record.ref
        alt = ','.join([str(a) for a in record.alts])
        read_id = record.info.get('ILLUMINA_READ_NAME', [])[0]

        if not read_id:
            continue

        fold_id = get_fold_id(read_id)

        row = [
            chrom, pos, '.', ref, alt, read_id, 1
        ]
        writers[fold_id].writerow(row)

    for fold in range(k):
        files[fold].close()

    print("Artefacts VCF processing completed.")
    counts = count_entries_per_fold(vcf_path, k)
    return counts


def combine_fold_csvs(output_dir, k=K_FOLDS, mutation_class_counts=None, artefact_class_counts=None):
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
                'CHROM', 'POS', 'ID', 'REF', 'ALT', 'READ_ID', 'read_support',
                'num_in_class', 'label'
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
                artefacts_train_path = os.path.join(output_dir, f'artefacts_fold_{other_fold}.csv')
                with open(artefacts_train_path, 'r') as art_train:
                    reader = csv.reader(art_train)
                    next(reader)  # Skip header
                    for row in reader:
                        row.append(train_art_count)
                        row.append(0.0)
                        train_writer.writerow(row)

        print(f"Fold {fold} CSVs created: train_fold_{fold}.csv and test_fold_{fold}.csv with classification labels.")

def main():
    mutation_class_counts = process_mutations_vcf(
        MUTATIONS_VCF_PATH, OUTPUT_DIR, K_FOLDS)
    artefect_class_counts = process_artefacts_vcf(
        ARTEFACTS_VCF_PATH, OUTPUT_DIR, K_FOLDS)
    combine_fold_csvs(
        OUTPUT_DIR, K_FOLDS, mutation_class_counts, artefect_class_counts)
    print("All cross-validation splits have been created.")


if __name__ == "__main__":
    main()
