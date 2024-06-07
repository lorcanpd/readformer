import torch
import torch.nn as nn
import torch.nn.functional as F

from components.rotary_encoding import (
    compute_theta_vector, compute_rotation_angles, apply_dimensionwise_rotation
)


class CustomMaskedConv1D(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, groups, stride=1):
        super(CustomMaskedConv1D, self).__init__()
        self.groups = groups
        self.stride = stride
        # self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels_per_group = in_channels // groups
        self.out_channels_per_group = out_channels // groups
        self.kernel = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.randn(
                    1, 1, groups,
                    self.out_channels_per_group,
                    self.in_channels_per_group,
                    kernel_size
                )
            )
        )

    def forward(self, inputs, positions):
        batch_size, seq_length, _ = inputs.size()
        kernel_center = self.kernel_size // 2
        # Pad inputs and positions to handle the borders
        pad_size = kernel_center
        padded_inputs = F.pad(inputs, (0, 0, pad_size, pad_size))
        padded_positions = F.pad(positions, (pad_size, pad_size), value=-1)
        # Extract and reshape patches
        input_patches = padded_inputs.unfold(
            1, self.kernel_size, self.stride
        ).contiguous()
        position_patches = padded_positions.unfold(
            1, self.kernel_size, self.stride
        )
        # Create a mask to prevent short convolution aggregating signals from
        # bases on different reads.
        expected_positions = (
                position_patches[:, :, kernel_center].unsqueeze(2) +
                torch.arange(-kernel_center, kernel_center + 1)
        )
        mask = (position_patches == expected_positions).float().unsqueeze(-2)
        # Apply the mask
        masked_input_patches = input_patches * mask
        # Reshape the masked input patches to group the channels
        grouped_patches = masked_input_patches.reshape(
            batch_size, seq_length, self.groups, self.in_channels_per_group,
            self.kernel_size
        )
        # Perform an elementwise multiplication with each of the
        # num_out_channels kernels
        # output shape = (batch_size, seq_length, groups,
        # out_channels_per_group, in_channels_per_group, kernel_size)
        conv_result = (grouped_patches.unsqueeze(-3) * self.kernel)
        # Sum over the kernel size dimension and the in_channels_per_group
        # dimension
        # output shape = (batch_size, seq_length, groups, out_channels_per_group)
        output = conv_result.sum(dim=(-1, -2))

        if torch.isnan(output).any():
            breakpoint()

        return output


class HyenaProjection(nn.Module):
    def __init__(self, emb_dim, n_order, kernel_size):
        super(HyenaProjection, self).__init__()
        self.groups = n_order + 1
        self.emb_dim = emb_dim
        self.W = nn.Parameter(
            nn.init.kaiming_uniform_(torch.randn(emb_dim, self.groups * emb_dim))
        )
        self.custom_conv = CustomMaskedConv1D(
            kernel_size, self.groups * emb_dim, self.groups * emb_dim,
            self.groups
        )

    def forward(self, inputs, positions):
        x = inputs
        x = torch.matmul(x, self.W)
        z = self.custom_conv(x, positions)
        # Reshape z to batch_size x D x N x L
        z = z.reshape(z.size(0), self.emb_dim, self.groups, -1)

        if torch.isnan(z).any():
            breakpoint()

        # Unstack the groups for separate processing
        z = z.unbind(dim=-2)

        return z


class FeedForward(nn.Module):
    def __init__(self, emb_dim, hidden_dim=None, n_order=1, activation="gelu"):
        super(FeedForward, self).__init__()
        self.emb_dim = emb_dim
        self.n_order = n_order
        self.hidden_dim = hidden_dim if hidden_dim is not None else emb_dim
        self.W1 = nn.Parameter(
            nn.init.kaiming_uniform_(torch.randn(emb_dim, self.hidden_dim))
        )
        self.b1 = nn.Parameter(torch.zeros(self.hidden_dim))
        self.W2 = nn.Parameter(
            nn.init.kaiming_uniform_(torch.randn(self.hidden_dim, n_order * emb_dim))
        )
        self.b2 = nn.Parameter(torch.zeros(n_order * emb_dim))

        self.layer_norm = nn.LayerNorm(self.hidden_dim)

        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'swish':
            self.activation = lambda x: x * torch.sigmoid(x)
        elif activation == 'mish':
            self.activation = lambda x: x * torch.tanh(F.softplus(x))
        elif activation == 'elu':
            self.activation = F.elu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, inputs):
        x = torch.matmul(inputs, self.W1) + self.b1
        x = self.layer_norm(x)
        x = self.activation(x)
        x = torch.matmul(x, self.W2) + self.b2
        return x


class HyenaFilter(nn.Module):
    def __init__(self, emb_dim, n_order):
        super(HyenaFilter, self).__init__()
        self.emb_dim = emb_dim
        self.n_order = n_order
        self.ffn = FeedForward(emb_dim, n_order=n_order)
        self.theta_vector = compute_theta_vector(emb_dim)
        self.epsilon = 1e-8  # Small value to avoid division by zero

    def forward(self, embeddings, positions):
        rotation_matrices = compute_rotation_angles(
            positions, self.emb_dim, self.theta_vector
        )
        t = apply_dimensionwise_rotation(embeddings, rotation_matrices)
        h_hat = self.ffn(t)
        # reshape h_hat from batch_size x L x ND to batch_size x L x N x D
        h_hat = h_hat.view(
            h_hat.size(0), h_hat.size(1), self.n_order, self.emb_dim
        )

        # reshape to batch_size x N x D x L
        h_hat = h_hat.permute(0, 2, 3, 1)
        # # reshape to batch_size x N x L x D
        # h_hat = h_hat.permute(0, 2, 1, 3)
        # Normalise h_hat along the channel dimension D.
        h_hat = h_hat / (h_hat.norm(p=1, dim=-2, keepdim=True) + self.epsilon)
        # split h_hat into h1, h2, ..., hN
        h = h_hat.unbind(dim=-3)

        return h


class FFTLongConv(nn.Module):
    def __init__(self):
        super(FFTLongConv, self).__init__()

    def forward(self, inputs, filters):
        L = inputs.size(-1)
        padded_length = 2 * L  # Double the length for FFT
        # Pad the inputs and filters
        inputs_padded = F.pad(inputs, (0, padded_length - L))
        filters_padded = F.pad(filters, (0, padded_length - L))
        # Perform FFT
        inputs_fft = torch.fft.rfft(
            inputs_padded, n=padded_length, dim=-1, norm='forward'
        )
        filters_fft = torch.fft.rfft(
            filters_padded, n=padded_length, dim=-1, norm='forward'
        )
        # Element-wise multiplication in the frequency domain
        product_fft = inputs_fft * filters_fft
        # Inverse FFT to get the convolution result
        result = torch.fft.irfft(
            product_fft, n=padded_length, dim=-1, norm='forward'
        )
        # Remove padding
        result_real = result[..., :L]

        return result_real


class HyenaBlock(nn.Module):
    def __init__(self, emb_dim, n_order, kernel_size):
        super(HyenaBlock, self).__init__()
        self.n_order = n_order
        self.embedding_dim = emb_dim
        self.kernel_size = kernel_size
        self.projection = HyenaProjection(emb_dim, n_order, kernel_size)
        self.filter = HyenaFilter(emb_dim, n_order)
        self.fft_long_conv = FFTLongConv()
        self.output_projection = nn.Parameter(
            nn.init.kaiming_uniform_(torch.randn(emb_dim, emb_dim))
        )
        # TODO ADD BIAS TERM AND RESIDUALS.
        self.B = nn.Parameter(
            nn.init.kaiming_uniform_(torch.randn((n_order, 1, emb_dim, 1)))
        )

    def forward(self, embeddings, positions):
        *x, v = self.projection(embeddings, positions)
        filters = self.filter(embeddings, positions)

        for i, x_i in enumerate(x):
            h_i = filters[i]
            v = x_i * (v * self.B[i] + self.fft_long_conv(v, h_i))

        # Transpose v to shape (batch_size, seq_len, emb_dim)
        v = v.transpose(2, 1)
        # # Residual connection added as gradients might have been vanishing.
        # output = v + v.matmul(self.output_projection)
        output = v.matmul(self.output_projection)

        return output


# Test forward pass of the HyenaBlock.

# # Define the parameters
# batch_size = 1
# seq_length = 8
# embedding_dim = 4
# n_order = 3
# kernel_size = 3
#
# # Generate random data for inputs and positions
# inputs = torch.randn(batch_size, seq_length, embedding_dim)
# positions = torch.tensor([[1, 2, 3, 4, 5, 1, 2, 3]])
# # positions = torch.arange(seq_length).repeat(batch_size, 1)
#
# # Instantiate the HyenaBlock
# hyena_block = HyenaBlock(embedding_dim, n_order, kernel_size)
#
# # Test the HyenaBlock with the generated data
# output = hyena_block(inputs, positions)
#
# print("Inputs:")
# print(inputs)
# print("\nPositions:")
# print(positions)
# print("\nOutput:")
# print(output)


# OLD TENSOFRLOW IMPLEMENTATION
# import tensorflow as tf
#
# from components.rotary_encoding import (
#     compute_theta_vector, compute_rotation_angles, apply_dimensionwise_rotation
# )
#
# class FeedForward(tf.keras.layers.Layer):
#     def __init__(self, n_order, emb_dim):
#         super(FeedForward, self).__init__()
#         self.emb_dim = emb_dim
#         self.n_order = n_order
#         init = tf.keras.initializers.he_normal()
#         self.W1 = tf.Variable(
#             initial_value=init(
#                 shape=(emb_dim, emb_dim),
#                 dtype='float32'
#             ),
#             name='W1',
#             trainable=True
#         )
#         self.b1 = tf.Variable(
#             initial_value=tf.zeros(shape=(emb_dim,), dtype='float32'),
#             name='b1',
#             trainable=True
#         )
#         self.W2 = tf.Variable(
#             initial_value=init(
#                 shape=(emb_dim, n_order * emb_dim),
#                 dtype='float32'
#             ),
#             name='W2',
#             trainable=True
#         )
#         self.b2 = tf.Variable(
#             initial_value=tf.zeros(shape=(n_order * emb_dim, ), dtype='float32'),
#             name='b2',
#             trainable=True
#         )
#
#     def call(self, inputs, training=False):
#         x = tf.tensordot(inputs, self.W1, axes=1) + self.b1
#
#         if training:
#             x = tf.nn.dropout(x, rate=0.1)
#
#         x = tf.nn.relu(x)
#         x = tf.tensordot(x, self.W2, axes=1) + self.b2
#
#         return x
#
# class HyenaFilter(tf.keras.layers.Layer):
#     def __init__(self, emb_dim, n_order, **kwargs):
#         super(HyenaFilter, self).__init__(**kwargs)
#         self.theta_vector = compute_theta_vector(emb_dim)
#         self.emb_dim = emb_dim
#         self.n_order = n_order
#         self.ffn = FeedForward(n_order, emb_dim)
#
#     def call(self, inputs, training=False):
#         seq_len = tf.shape(inputs[0])[1]
#         batch_size = tf.shape(inputs[0])[0]
#         rotation_matrices = compute_rotation_angles(
#             inputs[1], self.emb_dim, self.theta_vector
#         )
#         t = apply_dimensionwise_rotation(inputs[0], rotation_matrices)
#         h_hat = self.ffn(t, training=training)
#         # reshape h_hat from batch_size x L x ND to batch_size x L x N x D
#         h_hat = tf.reshape(
#             h_hat,
#             (batch_size, seq_len, self.n_order, self.emb_dim)
#         )
#         # reshape to batch_size x N x D x L
#         h_hat = tf.transpose(h_hat, perm=[0, 2, 3, 1])
#
#         # Normalize h_hat along the channel dimension D.
#         h_hat = h_hat / tf.norm(h_hat, ord=1, axis=-2, keepdims=True)
#
#         # split h_hat into h1, h2, ..., hN
#         h = tf.unstack(h_hat, axis=-3)
#
#         return h
#
#
# def custom_masked_conv1d(
#         inputs, kernel, groups, positions, stride=1, padding='VALID'
# ):
#     """
#     Perform a custom 1D convolution that ignores certain interactions based on
#     genomic positions, ensuring that convolution is applied with padding to
#     cover all elements. The grouping is done along the embedding dimension of
#     the input tensor. This allows convolution to be performed in parallel on
#     different groups of the embedding dimension.
#
#     :param inputs:
#         Input tensor of shape
#         [batch_size, sequence_length, embedding_dimensions].
#     :param kernel:
#         Convolution kernel of shape
#         [kernel_size, embedding_dimensions/groups, out_channels].
#     :param groups:
#         Number of groups to split the embedding dimension for grouped
#         convolution.
#     :param positions:
#         A tensor of shape [batch_size, sequence_length] containing the genomic
#         positions.
#
#     :return:
#         Tensor of shape [batch_size, sequence_length, out_channels].
#     """
#
#     batch_size, sequence_length, in_channels = inputs.shape
#     kernel_size, _, out_channels = kernel.shape
#
#     in_channels_per_group = in_channels // groups
#     out_channels_per_group = out_channels // groups
#
#     kernel_center = kernel_size // 2
#
#     # Ensure in_channels and out_channels are divisible by groups
#     assert in_channels % groups == 0, "in_channels must be divisible by groups"
#     assert out_channels % groups == 0, "out_channels must be divisible by groups"
#
#     # Add padding to inputs and positions to handle edges
#     pad_size = kernel_size // 2
#     padded_inputs = tf.pad(
#         inputs, [[0, 0], [pad_size, pad_size], [0, 0]], "CONSTANT"
#     )
#     padded_positions = tf.pad(
#         positions, [[0, 0], [pad_size, pad_size]], "CONSTANT",
#         constant_values=-1
#     )
#
#     # Extract patches from padded input
#     input_patches = tf.image.extract_patches(
#         images=tf.expand_dims(padded_inputs, axis=1),
#         sizes=[1, 1, kernel_size, 1],
#         strides=[1, 1, stride, 1],
#         rates=[1, 1, 1, 1],
#         padding='VALID'
#     )
#     input_patches = tf.reshape(
#         input_patches,
#         [batch_size, sequence_length, kernel_size, in_channels]
#     )
#
#     # Prepare the mask using position information
#     position_patches = tf.image.extract_patches(
#         images=tf.expand_dims(
#             tf.expand_dims(padded_positions, axis=1),
#             axis=-1
#         ),
#         sizes=[1, 1, kernel_size, 1],
#         strides=[1, 1, stride, 1],
#         rates=[1, 1, 1, 1],
#         padding='VALID'
#     )
#     position_patches = tf.reshape(
#         position_patches,
#         [batch_size, sequence_length, kernel_size]
#     )
#
#     # Create the mask to prevent short convolution aggregating signals from
#     # bases on different reads.
#     central_positions = position_patches[:, :, kernel_center:kernel_center + 1]
#     expected_positions = central_positions + tf.range(
#         -kernel_center, kernel_center + 1, dtype=tf.int32
#     )
#     mask = tf.equal(expected_positions, position_patches)
#     mask = tf.cast(mask, tf.float32)
#
#     # Expand mask dimensions for element-wise multiplication
#     mask = tf.expand_dims(mask, axis=3)  # Add an axis for in_channels
#     mask = tf.tile(mask, [1, 1, 1, in_channels])
#
#     masked_input_patches = input_patches * mask
#     new_width = tf.shape(masked_input_patches)[-2]
#
#     reshaped_input = tf.reshape(
#         masked_input_patches,
#         [batch_size, sequence_length, new_width, groups, in_channels_per_group]
#     )
#     reshaped_kernel = tf.reshape(
#         kernel, [kernel_size, groups, in_channels_per_group, out_channels]
#     )
#     # Apply grouped convolution by using 2d conv and reshaped inputs and kernel.
#     conv = tf.nn.conv2d(
#         input=reshaped_input,
#         filters=reshaped_kernel,
#         strides=[1, 1, 1, 1],
#         padding=padding,
#         data_format='NHWC'
#     )
#     output = tf.reshape(conv, [batch_size, sequence_length, out_channels])
#
#     return output
#
#
# class HyenaProjection(tf.keras.layers.Layer):
#     def __init__(self, emb_dim, n_order, kernel_size, **kwargs):
#         super(HyenaProjection, self).__init__(**kwargs)
#         self.emb_dim = emb_dim
#         self.n_order = n_order
#         init = tf.keras.initializers.he_normal()
#         self.groups = n_order + 1
#         self.W = tf.Variable(
#             initial_value=init(
#                 shape=(emb_dim, self.groups * emb_dim),
#                 dtype='float32'
#             ),
#             name='W_linear',
#             trainable=True
#         )
#         self.conv_kernel = tf.Variable(
#             initial_value=init(
#                 shape=(
#                     kernel_size,
#                     self.groups * emb_dim,
#                     self.groups * emb_dim
#                 ),
#                 dtype='float32'
#             ),
#             name='conv_kernel',
#             trainable=True
#         )
#
#     def call(self, inputs):
#         x = inputs[0]
#         positions = inputs[1]
#         x = tf.tensordot(x, self.W, axes=1)
#         z = custom_masked_conv1d(x, self.conv_kernel, self.groups, positions)
#         # reshape batch_size, seq_len, groups*emb_dim into
#         # batch_size, groups, emb_dim, seq_len
#         z = tf.reshape(z, (-1, self.groups, self.emb_dim, tf.shape(z)[1],))
#         # split z into z1, z2, ..., zN
#         z = tf.unstack(z, axis=-3)
#
#         return z
#
# class FFTLongConv(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(FFTLongConv, self).__init__(**kwargs)
#
#     def call(self, inputs, training=False):
#         inputs, filters, bias = inputs
#         L = tf.shape(inputs)[-1]
#         padded_length = 2 * L  # Double the length for FFT
#         inputs_padded = tf.pad(
#             inputs, [[0, 0], [0, 0], [0, padded_length - L]]
#         )
#         filters_padded = tf.pad(
#             filters, [[0, 0], [0, 0], [0, padded_length - L]]
#         )
#         # FFT
#         inputs_fft = tf.signal.rfft(inputs_padded, fft_length=[padded_length])
#         filters_fft = tf.signal.rfft(filters_padded, fft_length=[padded_length])
#         # element-wise multiplication
#         product_fft = inputs_fft * filters_fft
#         # inverse FFT
#         result = tf.signal.irfft(product_fft, fft_length=[padded_length])
#         breakpoint()
#         # remove padding
#         result_real = tf.math.real(result)[..., :L]
#         # # residual connection to improve gradient flow.
#         # output = result_real + bias * inputs
#
#         return result_real
#
# class HyenaBlock(tf.keras.Model):
#     def __init__(self, emb_dim, n_order, kernel_size):
#         super(HyenaBlock, self).__init__()
#         self.n_order = n_order
#         self.embedding_dim = emb_dim
#         self.kernel_size = kernel_size
#         self.projection = HyenaProjection(emb_dim, n_order, kernel_size)
#         self.filter = HyenaFilter(emb_dim, n_order)
#         self.fft_long_conv = FFTLongConv()
#         init = tf.keras.initializers.he_normal()
#         self.B = tf.Variable(
#             initial_value=init(
#                 shape=(n_order, 1, emb_dim, 1),
#                 dtype='float32'
#             ),
#             name='bias_matrix',
#             trainable=True
#         )
#         self.B = tf.unstack(self.B, axis=-4)
#         self.output_projection = tf.Variable(
#             initial_value=init(
#                 shape=(emb_dim, emb_dim),
#                 dtype='float32'
#             ),
#             name='output_projection',
#             trainable=True
#         )
#
#     def call(self, inputs, training=False):
#         *x, v = self.projection(inputs)
#         # TODO residuals not in original model - testing will be needed to see
#         #  if they are necessary.
#         # # Residual connection
#         # v = v + tf.transpose(inputs[0], perm=[0, 2, 1])
#         filters = self.filter(inputs)
#
#         for i, x_i in enumerate(x):
#             h_i = filters[i]
#             # v = v + tf.norm(
#             v = tf.norm(
#                 x_i, ord=1, axis=-2, keepdims=True
#             ) * self.fft_long_conv(
#                 inputs=[v, h_i, self.B[i]], training=training
#             )
#
#         # Transpose v to shape (batch_size, seq_len, emb_dim)
#         v = tf.transpose(v, perm=[0, 2, 1])
#         # # Residual connection.
#         # output = v + tf.tensordot(v, self.output_projection, axes=1)
#         output = tf.tensordot(v, self.output_projection, axes=1)
#
#         return output
#
#
# from components.data_streaming import create_tf_dataset
# from components.read_embedding import NucleotideEmbeddingLayer, MetricEmbedding
#
# emb_dim = 2
#
# nucleotide_embeddings = NucleotideEmbeddingLayer(embedding_dim=emb_dim)
# float_metric_embeddings = MetricEmbedding(emb_dim//2, name='float')
# binary_metric_embeddings = MetricEmbedding(emb_dim//2, name='binary')
#
# hyena_block = HyenaBlock(emb_dim, n_order = 2, kernel_size = 3)
#
# dataset = create_tf_dataset(
#     file_paths='TEST_DATA', metadata_path='TEST_DATA/test_metadata.csv',
#     nucleotide_threshold=1024, batch_size=1
# )
#
# for batch in dataset:
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
#     positions = batch[6]
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
#     output = hyena_block([model_inputs, positions])
#
#     break


# TODO: Make sure this works.
# def fft_long_convolution(inputs, filters):
#     L = tf.shape(inputs)[-1]
#     padded_length = 2 * L  # Double the length for FFT
#
#     inputs_padded = tf.pad(inputs, [[0, 0], [0, padded_length - L], [0, 0]])
#     filters_padded = tf.pad(filters, [[0, 0], [0, padded_length - L], [0, 0]])
#
#     # FFT
#     inputs_fft = tf.signal.rfft(inputs_padded, fft_length=[padded_length])
#     filters_fft = tf.signal.rfft(filters_padded, fft_length=[padded_length])
#
#     # Element-wise multiplication
#     product_fft = inputs_fft * filters_fft
#
#     # Inverse FFT
#     result = tf.signal.irfft(product_fft, fft_length=[padded_length])
#     # Remove padding
#     result_real = tf.math.real(result)[:, :L, :]

# class FFTLongConv(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(FFTLongConv, self).__init__(**kwargs)
#
#     def call(self, inputs, filters, training=False):
#         L = tf.shape(inputs)[-1]
#         padded_length = 2 * L  # Double the length for FFT
#
#         inputs_padded = tf.pad(inputs, [[0, 0], [0, padded_length - L], [0, 0]])
#         filters_padded = tf.pad(filters, [[0, 0], [0, padded_length - L], [0, 0]])
#
#         # FFT
#         inputs_fft = tf.signal.rfft(inputs_padded, fft_length=[padded_length])
#         filters_fft = tf.signal.rfft(filters_padded, fft_length=[padded_length])
#
#         # Element-wise multiplication
#         product_fft = inputs_fft * filters_fft
#
#         # Inverse FFT
#         result = tf.signal.irfft(product_fft, fft_length=[padded_length])
#         # Remove padding
#         result_real = tf.math.real(result)[:, :L, :]
#
#         return result_real
#
#
# # Example usage


# projection = HyenaProjection(emb_dim=embedding_dim, num_layers=2, kernel_size=5)
#
# inputs = tf.random.normal([batch_size, sequence_length, embedding_dim], dtype=tf.float32)
# positions = tf.constant(
#     [[1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
#         [1, 2, 3, 4, 5, 6, 2, 3, 4, 5],
#         [1, 2, 3, 4, 5, 6, 2, 3, 4, 5],
#         [1, 2, 3, 4, 5, 6, 2, 3, 4, 5]]
# )  # Example positions with reset indicating new reads
#
# output = projection([inputs, positions])
#
#
# # Test custom masking convolution.
# batch_size, sequence_length, embedding_dim, out_channels = 4, 10, 6, 6
# kernel_size = 5
# inputs = tf.random.normal([batch_size, sequence_length, embedding_dim])
# kernel = tf.random.normal([kernel_size, embedding_dim, out_channels])
# positions = tf.constant(
#     [[1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
#      [1, 2, 3, 4, 5, 6, 2, 3, 4, 5],
#      [1, 2, 3, 4, 5, 6, 2, 3, 4, 5],
#      [1, 2, 3, 4, 5, 6, 2, 3, 4, 5]]
# )  # Example positions with reset indicating new reads
# # positions = tf.constant([[1, 2, 3, 4, 5, 1, 2, 3, 4, 5]])
#
# output = custom_masked_conv1d(inputs, kernel, 2, positions)
# print("Output Shape:", output.shape)
#


#
# class WithinRead1DConv(tf.keras.layers.Layer):
#     def __init__(self, filters, kernel_size, **kwargs):
#         super(WithinRead1DConv, self).__init__(**kwargs)
#         self.filters = filters
#         self.kernel_size = kernel_size
#
#     def build(self, input_shape):
#         self.kernel = self.add_weight(
#             shape=(self.kernel_size, input_shape[-1], self.filters),
#             initializer='he_normal',
#             trainable=True,
#             name='kernel'
#         )
#         self.bias = self.add_weight(
#             shape=(self.filters, ),
#             initializer='zeros',
#             trainable=True,
#             name='bias'
#         )
#         super(WithinRead1DConv, self).build(input_shape)
#
#     def call(self, inputs):
#         # TODO Implement the local convolution operation that smooths the input
#         #  sequence tokens using it's context. It needs to pad the edges of each
#         #  read so that non-contiguous bases are not treated as contiguous.
#
#         return tf.nn.conv1d(inputs, self.kernel, 1, 'VALID') + self.bias
#
#     def compute_output_shape(self, input_shape):
#         return input_shape[:-1] + (input_shape[-1], self.filters)
#
#     def get_config(self):
#         config = super(WithinRead1DConv, self).get_config()
#         config.update({
#             'filters': self.filters,
#             'kernel_size': self.kernel_size
#         })
#         return config
