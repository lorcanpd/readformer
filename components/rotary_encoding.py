
"""
Rotary Positional Encoding (RoPE) functions for use in Transformer models.
"""


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def compute_theta_vector(d_model) -> tf.Tensor:
    """
    Compute vector of theta values for rotary encoding as in RoPE paper.

    :param d_model:
        The dimensionality of the embedding vectors.
    :return:
        A tensor of shape [d_model//2] containing angles by which to rotate each
        dimension by a factor of its genomic position.
    """
    # i ranges from 1 to d_model/2
    i = tf.range(1, d_model//2 + 1, dtype=tf.float32)
    # Compute theta_i for each dimension
    theta_i = tf.pow(10000.0, -2 * (i - 1) / d_model)
    return theta_i


def compute_rotation_angles(loci_positions, d_model, theta_vector) -> tf.Tensor:
    """
    Compute rotation angles for each dimension of the embedding vectors.

    :param loci_positions:
        Genomic positions of each embedding in the batch. A tensor of shape
        [batch_size, seq_len] containing integer positions.
    :param d_model:
        The dimensionality of the embedding vectors.
    :param theta_vector:
        Vector of theta values for rotary encoding. A tensor of shape
        [d_model//2]
    :return:
        A tensor of shape [batch_size, seq_len, d_model//2, 2, 2] containing
        rotation matrices for each dimension of the embedding vectors.
    """
    # Ensure loci_positions is a batched tensor [batch_size, seq_len]
    batch_size, seq_len = tf.shape(loci_positions)

    # Expand theta_vector to match batch and sequence dimensions
    # theta_vector = compute_theta_vector(d_model)
    theta_vector = tf.reshape(
        theta_vector, [1, 1, -1]
    )  # Shape: [1, 1, d_model//2]

    # Expand loci_positions for element-wise multiplication
    loci_positions = tf.cast(
        loci_positions, tf.float32
    )[:, :, tf.newaxis]  # Shape: [batch_size, seq_len, 1]

    # Compute angles for the whole batch
    # Broadcasting to shape: [batch_size, seq_len, d_model//2]
    angles = loci_positions * theta_vector

    # Compute sine and cosine for angles
    sin_angles = tf.sin(angles)
    cos_angles = tf.cos(angles)

    # Construct rotation matrices
    rotation_matrices = tf.stack(
        [cos_angles, -sin_angles, sin_angles, cos_angles], axis=-1
    )
    # Reshape to a shape that allows for element-wise multiplication of the
    # rotation matrices with the embeddings.
    rotation_matrices = tf.reshape(
        rotation_matrices, [batch_size, seq_len, d_model // 2, 2, 2]
    )

    return rotation_matrices


def apply_dimensionwise_rotation(embeddings, rotation_matrices) -> tf.Tensor:
    """
    Apply dimension-wise rotation to the embeddings. This is done by rotating
    each dimension of the embeddings by a factor of its genomic position. The
    rotation through efficient element-wise multiplication of the embeddings
    with the rotation matrices.

    :param embeddings:
        Embedding vectors to be rotated. A tensor of shape
        [batch_size, seq_len, d_model].
    :param rotation_matrices:
        Rotation matrices for each dimension. A tensor of shape
        [batch_size, seq_len, d_model//2, 2, 2].
    :return:
        A tensor of shape [batch_size, seq_len, d_model] containing the
        rotated embeddings.
    """
    batch_size, seq_len, d_model = tf.shape(embeddings)
    half_d_model = d_model // 2

    # Reshape embeddings for rotation
    embeddings_reshaped = tf.reshape(
        embeddings,
        [batch_size, seq_len, half_d_model, 2]
    )

    # Apply rotation for the whole batch
    rotated_embeddings = tf.einsum(
        'bsij,bsijd->bsid',
        embeddings_reshaped, rotation_matrices
    )

    # Reshape back to original shape
    rotated_embeddings = tf.reshape(
        rotated_embeddings,
        [batch_size, seq_len, d_model]
    )

    return rotated_embeddings


def plot_rotations(embeddings, rotated_embeddings, loci_positions) -> None:
    batch_size, seq_len, d_model = embeddings.shape
    fig, axs = plt.subplots(1, seq_len, figsize=(seq_len * 4, 4))

    if seq_len == 1:
        axs = [axs]

    for i in range(seq_len):
        # Original embedding vector
        vec = embeddings[0, i].numpy()
        rotated_vec = rotated_embeddings[0, i].numpy()

        # Plot original vector
        axs[i].plot([0, vec[0]], [0, vec[1]], 'b-', label='Original')

        # Plot rotated vector
        axs[i].plot(
            [0, rotated_vec[0]], [0, rotated_vec[1]], 'r--', label='Rotated'
        )

        # Display angle
        angle = np.arctan2(
            rotated_vec[1] - vec[1], rotated_vec[0] - vec[0]
        ) * 180 / np.pi
        axs[i].set_title(
            f'Loci: {loci_positions[0, i].numpy()}, Rotation: {angle:.2f}°'
        )

        axs[i].set_xlim(-1.5, 1.5)
        axs[i].set_ylim(-1.5, 1.5)
        axs[i].grid(True)
        axs[i].legend()

    plt.tight_layout()
    plt.show()


# # Example usage
# batch_size = 1
# seq_len = 6
# d_model = 2
# loci_positions = tf.random.uniform(
#     (batch_size, seq_len), minval=0, maxval=seq_len, dtype=tf.int32
# )  # Random loci positions for batch
# embeddings = tf.random.uniform(
#     [batch_size, seq_len, d_model], dtype=tf.float32
# )  # Batched embeddings
#
# rotation_matrices = compute_rotation_angles(loci_positions, d_model)
#
# rotated_embeddings = apply_dimensionwise_rotation(
#     embeddings, rotation_matrices
# )
#
# plot_rotations(embeddings, rotated_embeddings, loci_positions)
