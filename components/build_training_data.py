
import os
import pysam



class DataProcessor(object):

    def __init__(
            self, directory_path, reference_fasta_path,
            mapping_quality_threshold, base_quality_threshold, output_dir
    ):
        self.directory_path = directory_path
        self.reference = pysam.FastaFile(reference_fasta_path)
        self.mapping_quality_threshold = mapping_quality_threshold
        self.base_quality_threshold = base_quality_threshold
        self.output_dir = output_dir

    def process_cigar_operations(self, read, filename):
        chrom = read.reference_name
        ref_start_pos = read.reference_start
        read_start_pos = 0

        candidate_snvs = {}

        for operation in read.cigartuples:
            operation_type, operation_length = operation
            if operation_type == 0:  # MATCH - start and end of read are aligned.
                for i in range(operation_length):
                    try:
                        ref_base = self.reference.fetch(
                            chrom, ref_start_pos + i, ref_start_pos + i + 1
                        )
                    except KeyError:
                        if chrom.startswith("chr"):
                            chrom = chrom.replace("chr", "")
                        else:
                            chrom = "chr" + chrom
                        ref_base = self.reference.fetch(
                            chrom, ref_start_pos + i, ref_start_pos + i + 1
                        )
                    try:
                        read_base = read.query_sequence[i]
                    except IndexError:
                        breakpoint()
                    # get base quality

                    if (
                        ref_base != read_base and
                        ref_base == ref_base.upper() and
                        read.query_qualities[i] >= self.base_quality_threshold
                    ):
                        key = f"{chrom}:{ref_start_pos + i}_{ref_base}_{read_base}"
                        if key in candidate_snvs:
                            candidate_snvs[key]["read_ids"].append(read.query_name)
                        else:
                            candidate_snvs[key] = {
                                "file": filename,
                                "chromosome": chrom,
                                "position": ref_start_pos + i,
                                "ref_base": ref_base,
                                "alt_base": read_base,
                                "soft_masked": ref_base.islower(),
                                "read_ids": [read.query_name]
                            }
            # SNVs consume single bases on both reference and query
            ref_start_pos += operation_length
            read_start_pos += operation_length

        return candidate_snvs

    def process_bam_file(self, bam_file_path):
        filename = os.path.basename(bam_file_path)
        output_file_path = os.path.join(self.output_dir, f"{filename}.tsv")
        with (
            open(output_file_path, "w") as f,
            pysam.AlignmentFile(bam_file_path, "rb") as reader
        ):
            f.write("file\tchromosome\tposition\tref_base\talt_base\tread_id\n")
            for read in reader.fetch():
                if (read.mapping_quality >= mapping_quality_threshold and
                        not read.is_unmapped):
                    candidate_snvs = self.process_cigar_operations(
                        read, filename
                    )
                    # if not empty list write to temporary tsv file.
                    if candidate_snvs:
                        for snv in candidate_snvs.values():
                            f.write(
                                f"{filename}\t"
                                f"{snv['chromosome']}"
                                f"\t{snv['position']}"
                                f"\t{snv['ref_base']}"
                                f"\t{snv['alt_base']}"
                                f"\t{snv['read_ids']}\n"
                            )
                            print(
                                f"SNV found at "
                                f"{snv['chromosome']}:{snv['position']} "
                                f"with reference base {snv['ref_base']} "
                                f"and alternate base {snv['alt_base']} "
                                f"from read {snv['read_ids']}"
                            )


    def process_bam_files(self):
        for file in os.listdir(self.directory_path):
            if file.endswith(".bam"):
                self.process_bam_file(os.path.join(self.directory_path, file))


# TODO: Write something for mixing reads from different technologies for the
#  same individual at the same position.


# TODO: Start exploring how to identify regions where reads differ between
#  technologies. This is to capture the artefacts. Then store the read
#  information.


# TODO: True mutations should be present in all technologies but not be germline.
#  Think of an efficient way to capture these positions.
#  1000 genomess project has read data paired with VCF files.



# # Example usage
# directory_path = 'TEST_DATA'
# reference_fasta_path = 'TEST_DATA/Homo_sapiens.GRCh38.dna_sm.toplevel.fa'
# output_dir = directory_path
# mapping_quality_threshold = 40
# base_quality_threshold = 30
#
# data_processor = DataProcessor(
#     directory_path, reference_fasta_path, mapping_quality_threshold,
#     base_quality_threshold, output_dir
# )
#
# data_processor.process_bam_files()

