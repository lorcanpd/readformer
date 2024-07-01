import os
import pysam
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def extract_snp_variants(vcf_file):
    """
    Extract variants from a VCF file into a DataFrame.

    :param vcf_file:
        Path to the VCF file.
    :return:
        DataFrame containing chromosome, position, reference, and alternate
        alleles.
    """
    print(f"Extracting SNP variants from {vcf_file}")
    variants = []
    with pysam.VariantFile(vcf_file) as vcf_in:
        for record in vcf_in:
            if record.alts is not None and all(
                    len(allele) == 1 for allele
                    in [record.ref] + list(record.alts)
            ):  # Only consider SNPs
                for alt in record.alts:
                    variants.append({
                        'Chr': record.chrom,
                        'Pos': record.pos,
                        'Ref': record.ref,
                        'Alt': alt
                    })
    return pd.DataFrame(variants)


def find_discrepancies(illumina_vcf, pacbio_vcf, threadpool_cores):
    """
    Find discrepancies between Illumina and PacBio VCF files and return a
    DataFrame of unique Illumina variants.

    :param illumina_vcf:
        Path to the Illumina VCF file.
    :param pacbio_vcf:
        Path to the PacBio VCF file.
    :return:
        DataFrame of variants unique to the Illumina VCF.
    """
    with ThreadPoolExecutor(max_workers=threadpool_cores) as executor:
        print(f'Extracting variants from {illumina_vcf} and {pacbio_vcf}...')
        illumina_future = executor.submit(extract_snp_variants, illumina_vcf)
        pacbio_future = executor.submit(extract_snp_variants, pacbio_vcf)

        illumina_variants = illumina_future.result()
        pacbio_variants = pacbio_future.result()

    # check both use chr1, chr2, ... instead of 1, 2, ... if not change
    if 'chr' not in illumina_variants['Chr'][0]:
        illumina_variants['Chr'] = 'chr' + illumina_variants['Chr']
    if 'chr' not in pacbio_variants['Chr'][0]:
        pacbio_variants['Chr'] = 'chr' + pacbio_variants['Chr']

    # ensure no mitochondrial variants and viral variants
    illumina_variants = illumina_variants[~illumina_variants['Chr'].str.contains('chrM')]
    pacbio_variants = pacbio_variants[~pacbio_variants['Chr'].str.contains('chrM')]
    illumina_variants = illumina_variants[~illumina_variants['Chr'].str.contains('chrEBV')]
    pacbio_variants = pacbio_variants[~pacbio_variants['Chr'].str.contains('chrEBV')]

    print(f"Number of Illumina variants: {len(illumina_variants)}")
    print(f"Number of PacBio variants: {len(pacbio_variants)}")
    # Merge on chromosome, position, reference, and alternate alleles to find
    # common variants
    merged_variants = pd.merge(
        illumina_variants, pacbio_variants,
        on=['Chr', 'Pos', 'Ref', 'Alt'],
        how='outer', indicator=True
    )

    # Filter for variants unique to the Illumina VCF
    illumina_unique = merged_variants[merged_variants['_merge'] == 'left_only']

    print(f"Found {len(illumina_unique)} unique Illumina variants")
    return illumina_unique[['Chr', 'Pos', 'Ref', 'Alt']]


def create_artefact_vcf(individual, threadpool_cores):
    """
    Create a VCF file of likely Illumina artefacts for an individual.

    :param individual:
        The individual (e.g., 'HG003').
    """
    illumina_vcf = os.path.join(
        illumina_dir, f'{individual}-250bp-All-good_S1.genome.vcf.gz'
    )
    illumina_vcf = illumina_vcf.replace(
        f'{individual}-250bp-All-good_S1.genome.vcf.gz',
        f'{individual}run02_S1.genome.vcf.gz') if not (
        os.path.exists(illumina_vcf)
    ) else illumina_vcf

    pacbio_vcf = os.path.join(
        pacbio_dir, f'{individual}.pbmm2.hs37d5.DeepVariant.v090.vcf.gz'
    )
    artefact_vcf = os.path.join(
        artefacts_dir, f'{individual}_illumina_artefacts.vcf.gz'
    )

    # Find discrepancies
    print(f'Finding discrepancies for {individual}...')
    illumina_unique_variants = find_discrepancies(illumina_vcf, pacbio_vcf, threadpool_cores)

    # Read the Illumina VCF to get the header
    with pysam.VariantFile(illumina_vcf, 'r') as vcf_in:
        header = vcf_in.header

    # Write unique Illumina variants to a new VCF
    with pysam.VariantFile(artefact_vcf, 'w', header=header) as vcf_out:
        for _, row in illumina_unique_variants.iterrows():
            # record = header.new_record(chrom=row['Chr'], pos=row['Pos'], id='.', ref=row['Ref'], alts=[row['Alt']], qual=None, filter='PASS', info={})
            record = header.new_record(
                contig=row['Chr'],
                start=row['Pos'] - 1,
                stop=row['Pos'],
                alleles=(row['Ref'], row['Alt']),
                id='.',
                qual=None,
                filter='PASS',
                info={}
            )
            vcf_out.write(record)


#
# def main():
#     # Define the paths to the VCF files
#     global illumina_dir
#     global pacbio_dir
#     global artefacts_dir
#     illumina_dir = 'GIAB_VCFs/illumina_2x250bps'
#     pacbio_dir = 'GIAB_VCFs/pacbio_deepvariant'
#     artefacts_dir = 'GIAB_VCFs/artefacts'
#
#     # Ensure the artefacts directory exists
#     os.makedirs(artefacts_dir, exist_ok=True)
#
#     # List of individuals to process
#     individuals = ['HG003', 'HG004']
#
#     # Get the total number of available cores
#     total_cores = os.cpu_count()
#
#     # Define the number of cores to use for ProcessPoolExecutor and ThreadPoolExecutor
#     process_pool_cores = min(8, total_cores - 3) // 2
#     thread_pool_cores = min(8, total_cores - 3) // 2
#
#     print(f"Total cores: {total_cores}")
#     print(f"Process pool cores: {process_pool_cores}")
#     print(f"Thread pool cores: {thread_pool_cores}")
#
#     # Process each individual using parallel processing
#     with ProcessPoolExecutor(max_workers=process_pool_cores) as executor:
#         executor.map(
#             lambda individual: create_artefact_vcf(
#                 individual, thread_pool_cores
#             ),
#             individuals
#         )
#
#     print("Artefact VCFs created successfully.")

def main():
    global illumina_dir
    global pacbio_dir
    global artefacts_dir
    illumina_dir = 'GIAB_VCFs/illumina_2x250bps'
    pacbio_dir = 'GIAB_VCFs/pacbio_deepvariant'
    artefacts_dir = 'GIAB_VCFs/artefacts'

    os.makedirs(artefacts_dir, exist_ok=True)

    individuals = ['HG003', 'HG004']
    total_cores = os.cpu_count()
    # process_pool_cores = min(8, total_cores - 3) // 2
    # thread_pool_cores = min(8, total_cores - 3) // 2
    thread_pool_cores = min(8, total_cores - 3)

    print(f"Total cores: {total_cores}")
    # print(f"Process pool cores: {process_pool_cores}")
    print(f"Thread pool cores: {thread_pool_cores}")

    for individual in individuals:
        create_artefact_vcf(individual, total_cores)

    print("Artefact VCFs created successfully.")


if __name__ == '__main__':
    main()
