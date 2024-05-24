


# OLD TENSORFLOW IMPLEMENTATION.
#
# import tensorflow as tf
#
# from components.data_streaming import create_tf_dataset
# from components.read_embedding import NucleotideEmbeddingLayer, MetricEmbedding
#
# bam_dir = 'TEST_DATA'
# metadata_path = 'TEST_DATA/test_metadata.csv'
# # mutations_vcf_path = 'TEST_DATA/master_vcf.csv'
#
# # Create a dataset from the BAM file
# dataset = create_tf_dataset(
#     bam_dir, metadata_path, 1024, 2
# )
#
# # Create the model components
# embedding_dim = 32
# nucleotide_embeddings = NucleotideEmbeddingLayer(embedding_dim)
# float_metric_embeddings = MetricEmbedding(embedding_dim//2, name='float')
# binary_metric_embeddings = MetricEmbedding(embedding_dim//2, name='binary')
#
#
# for batch in dataset:
#
#     # REMINDER OF THE ORDER OF THE BATCH:
#     # batch_nucleotide_sequences, batch_base_qualities,
#     # batch_read_qualities, batch_cigar_match, batch_cigar_insertion,
#     # batch_bitwise_flags, batch_positions
#     binary = tf.concat(
#         [
#             batch[5],
#             tf.expand_dims(batch[3], axis=-1),
#             tf.expand_dims(batch[4], axis=-1)
#         ],
#         axis=-1
#     )
#     floating = tf.concat(
#         [
#             tf.expand_dims(batch[1], axis=-1),
#             tf.expand_dims(batch[2], axis=-1)
#         ],
#         axis=-1
#     )
#
#     nucleotide_embeddings_output = nucleotide_embeddings(batch[0])
#
#     float_metric_embeddings_output = float_metric_embeddings(floating)
#     binary_metric_embeddings_output = binary_metric_embeddings(binary)
#
#     model_inputs = nucleotide_embeddings_output + tf.concat(
#         [float_metric_embeddings_output, binary_metric_embeddings_output],
#         axis=-1
#     )
#
#     break
#
#
