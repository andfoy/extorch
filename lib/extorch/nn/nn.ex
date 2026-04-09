defmodule ExTorch.NN do
  @moduledoc """
  Neural network layer creation and operations.

  This module provides functions to create PyTorch `nn.Module` layers and
  run forward passes on them. Layers are created eagerly (not JIT-compiled)
  and support autograd for training.

  ## Example

      linear = ExTorch.NN.linear(784, 128)
      relu = ExTorch.NN.relu()
      input = ExTorch.randn({1, 784})
      output = input |> ExTorch.NN.forward(linear) |> ExTorch.NN.forward(relu)

  """

  alias ExTorch.NN.Layer

  # ============================================================================
  # Linear layers
  # ============================================================================

  @doc """
  Applies a linear transformation to the incoming data: $y = xA^T + b$.

  This layer implements a fully connected layer with `in_features` inputs
  and `out_features` outputs.

  ## Args
    * `in_features` (`integer`) - size of each input sample.
    * `out_features` (`integer`) - size of each output sample.
    * `opts` (`keyword`) - optional arguments:
      * `:bias` (`boolean`) - if `false`, the layer will not learn an additive bias. Default: `true`.

  ## Shape
    * Input: `{*, H_in}` where `*` means any number of dimensions including none
      and `H_in = in_features`.
    * Output: `{*, H_out}` where `H_out = out_features`.

  ## Variables
    * `weight` - the learnable weights of shape `{out_features, in_features}`.
    * `bias` - the learnable bias of shape `{out_features}`.

  ## Examples

      iex> m = ExTorch.NN.linear(20, 30)
      iex> input = ExTorch.randn({128, 20})
      iex> output = ExTorch.NN.forward(input, m)
      iex> output.size
      {128, 30}
  """
  @spec linear(integer(), integer(), keyword()) :: Layer.t()
  def linear(in_features, out_features, opts \\ []) do
    bias = Keyword.get(opts, :bias, true)
    ExTorch.Native.nn_linear(in_features, out_features, bias)
  end

  # ============================================================================
  # Convolution layers
  # ============================================================================

  @doc """
  Applies a 1D convolution over an input signal composed of several input planes.

  In the simplest case, the output value of the layer with input size `{N, C_in, L}`
  and output `{N, C_out, L_out}` can be described as:

  $out(N_i, C_{out_j}) = bias(C_{out_j}) + \\sum_{k=0}^{C_{in}-1} weight(C_{out_j}, k) \\star input(N_i, k)$

  ## Args
    * `in_channels` (`integer`) - number of channels in the input signal.
    * `out_channels` (`integer`) - number of channels produced by the convolution.
    * `kernel_size` (`integer`) - size of the convolving kernel.
    * `opts` (`keyword`) - optional arguments:
      * `:stride` (`integer`) - stride of the convolution. Default: `1`.
      * `:padding` (`integer`) - zero-padding added to both sides of the input. Default: `0`.
      * `:dilation` (`integer`) - spacing between kernel elements. Default: `1`.
      * `:groups` (`integer`) - number of blocked connections from input to output channels. Default: `1`.
      * `:bias` (`boolean`) - if `false`, the layer will not learn an additive bias. Default: `true`.

  ## Shape
    * Input: `{N, C_in, L_in}` or `{C_in, L_in}`.
    * Output: `{N, C_out, L_out}` or `{C_out, L_out}`.

  ## Examples

      iex> m = ExTorch.NN.conv1d(16, 33, 3, stride: 2)
      iex> input = ExTorch.randn({20, 16, 50})
      iex> output = ExTorch.NN.forward(input, m)
      iex> output.size
      {20, 33, 24}
  """
  @spec conv1d(integer(), integer(), integer(), keyword()) :: Layer.t()
  def conv1d(in_channels, out_channels, kernel_size, opts \\ []) do
    ExTorch.Native.nn_conv1d(
      in_channels, out_channels, kernel_size,
      Keyword.get(opts, :stride, 1),
      Keyword.get(opts, :padding, 0),
      Keyword.get(opts, :dilation, 1),
      Keyword.get(opts, :groups, 1),
      Keyword.get(opts, :bias, true)
    )
  end

  @doc """
  Applies a 2D convolution over an input signal composed of several input planes.

  In the simplest case, the output value of the layer with input size `{N, C_in, H, W}`
  and output `{N, C_out, H_out, W_out}` can be described as:

  $out(N_i, C_{out_j}) = bias(C_{out_j}) + \\sum_{k=0}^{C_{in}-1} weight(C_{out_j}, k) \\star input(N_i, k)$

  ## Args
    * `in_channels` (`integer`) - number of channels in the input image.
    * `out_channels` (`integer`) - number of channels produced by the convolution.
    * `kernel_size` (`integer`) - size of the convolving kernel.
    * `opts` (`keyword`) - optional arguments:
      * `:stride` (`integer`) - stride of the convolution. Default: `1`.
      * `:padding` (`integer`) - zero-padding added to both sides of the input. Default: `0`.
      * `:dilation` (`integer`) - spacing between kernel elements. Default: `1`.
      * `:groups` (`integer`) - number of blocked connections from input to output channels. Default: `1`.
      * `:bias` (`boolean`) - if `false`, the layer will not learn an additive bias. Default: `true`.

  ## Shape
    * Input: `{N, C_in, H_in, W_in}` or `{C_in, H_in, W_in}`.
    * Output: `{N, C_out, H_out, W_out}` or `{C_out, H_out, W_out}`.

  where $H_{out} = \\lfloor\\frac{H_{in} + 2 \\times padding - dilation \\times (kernel\\_size - 1) - 1}{stride} + 1\\rfloor$

  ## Examples

      iex> m = ExTorch.NN.conv2d(3, 16, 3, padding: 1)
      iex> input = ExTorch.randn({1, 3, 32, 32})
      iex> output = ExTorch.NN.forward(input, m)
      iex> output.size
      {1, 16, 32, 32}
  """
  @spec conv2d(integer(), integer(), integer(), keyword()) :: Layer.t()
  def conv2d(in_channels, out_channels, kernel_size, opts \\ []) do
    ExTorch.Native.nn_conv2d(
      in_channels, out_channels, kernel_size,
      Keyword.get(opts, :stride, 1),
      Keyword.get(opts, :padding, 0),
      Keyword.get(opts, :dilation, 1),
      Keyword.get(opts, :groups, 1),
      Keyword.get(opts, :bias, true)
    )
  end

  @doc """
  Applies a 3D convolution over an input signal composed of several input planes.

  ## Args
    * `in_channels` (`integer`) - number of channels in the input volume.
    * `out_channels` (`integer`) - number of channels produced by the convolution.
    * `kernel_size` (`integer`) - size of the convolving kernel.
    * `opts` (`keyword`) - optional: `:stride` (default `1`), `:padding` (default `0`),
      `:dilation` (default `1`), `:groups` (default `1`), `:bias` (default `true`).

  ## Shape
    * Input: `{N, C_in, D_in, H_in, W_in}`.
    * Output: `{N, C_out, D_out, H_out, W_out}`.
  """
  @spec conv3d(integer(), integer(), integer(), keyword()) :: Layer.t()
  def conv3d(in_channels, out_channels, kernel_size, opts \\ []) do
    ExTorch.Native.nn_conv3d(in_channels, out_channels, kernel_size,
      Keyword.get(opts, :stride, 1), Keyword.get(opts, :padding, 0),
      Keyword.get(opts, :dilation, 1), Keyword.get(opts, :groups, 1),
      Keyword.get(opts, :bias, true))
  end

  @doc """
  Applies a 1D transposed convolution operator (sometimes called "deconvolution").

  ## Args
    * `in_channels` (`integer`) - number of channels in the input signal.
    * `out_channels` (`integer`) - number of channels produced by the convolution.
    * `kernel_size` (`integer`) - size of the convolving kernel.
    * `opts` (`keyword`) - optional: `:stride` (default `1`), `:padding` (default `0`),
      `:output_padding` (default `0`), `:groups` (default `1`), `:bias` (default `true`),
      `:dilation` (default `1`).

  ## Shape
    * Input: `{N, C_in, L_in}`.
    * Output: `{N, C_out, L_out}` where
      $L_{out} = (L_{in} - 1) \\times stride - 2 \\times padding + dilation \\times (kernel\\_size - 1) + output\\_padding + 1$.
  """
  @spec conv_transpose1d(integer(), integer(), integer(), keyword()) :: Layer.t()
  def conv_transpose1d(in_channels, out_channels, kernel_size, opts \\ []) do
    ExTorch.Native.nn_conv_transpose1d(in_channels, out_channels, kernel_size,
      Keyword.get(opts, :stride, 1), Keyword.get(opts, :padding, 0),
      Keyword.get(opts, :output_padding, 0), Keyword.get(opts, :groups, 1),
      Keyword.get(opts, :bias, true), Keyword.get(opts, :dilation, 1))
  end

  @doc """
  Applies a 2D transposed convolution operator (sometimes called "deconvolution").

  ## Args
    * `in_channels` (`integer`) - number of channels in the input image.
    * `out_channels` (`integer`) - number of channels produced by the convolution.
    * `kernel_size` (`integer`) - size of the convolving kernel.
    * `opts` (`keyword`) - optional: `:stride` (default `1`), `:padding` (default `0`),
      `:output_padding` (default `0`), `:groups` (default `1`), `:bias` (default `true`),
      `:dilation` (default `1`).

  ## Shape
    * Input: `{N, C_in, H_in, W_in}`.
    * Output: `{N, C_out, H_out, W_out}`.

  ## Examples

      iex> m = ExTorch.NN.conv_transpose2d(16, 3, 3, stride: 2, padding: 1, output_padding: 1)
      iex> input = ExTorch.randn({1, 16, 4, 4})
      iex> output = ExTorch.NN.forward(input, m)
      iex> output.size
      {1, 3, 8, 8}
  """
  @spec conv_transpose2d(integer(), integer(), integer(), keyword()) :: Layer.t()
  def conv_transpose2d(in_channels, out_channels, kernel_size, opts \\ []) do
    ExTorch.Native.nn_conv_transpose2d(in_channels, out_channels, kernel_size,
      Keyword.get(opts, :stride, 1), Keyword.get(opts, :padding, 0),
      Keyword.get(opts, :output_padding, 0), Keyword.get(opts, :groups, 1),
      Keyword.get(opts, :bias, true), Keyword.get(opts, :dilation, 1))
  end

  # ============================================================================
  # Pooling layers
  # ============================================================================

  @doc """
  Applies a 1D max pooling over an input signal composed of several input planes.

  ## Args
    * `kernel_size` (`integer`) - the size of the sliding window.
    * `opts` (`keyword`) - optional arguments:
      * `:stride` (`integer`) - stride of the sliding window. Default: `kernel_size`.
      * `:padding` (`integer`) - implicit zero padding on both sides. Default: `0`.
      * `:dilation` (`integer`) - stride between elements within the sliding window. Default: `1`.
      * `:ceil_mode` (`boolean`) - use ceil instead of floor to compute output shape. Default: `false`.

  ## Shape
    * Input: `{N, C, L_in}` or `{C, L_in}`.
    * Output: `{N, C, L_out}` or `{C, L_out}` where
      $L_{out} = \\lfloor\\frac{L_{in} + 2 \\times padding - dilation \\times (kernel\\_size - 1) - 1}{stride} + 1\\rfloor$.
  """
  @spec max_pool1d(integer(), keyword()) :: Layer.t()
  def max_pool1d(kernel_size, opts \\ []) do
    ExTorch.Native.nn_max_pool1d(kernel_size,
      Keyword.get(opts, :stride, kernel_size), Keyword.get(opts, :padding, 0),
      Keyword.get(opts, :dilation, 1), Keyword.get(opts, :ceil_mode, false))
  end

  @doc """
  Applies a 2D max pooling over an input signal composed of several input planes.

  ## Args
    * `kernel_size` (`integer`) - the size of the sliding window.
    * `opts` (`keyword`) - optional arguments:
      * `:stride` (`integer`) - stride of the sliding window. Default: `kernel_size`.
      * `:padding` (`integer`) - implicit zero padding on both sides. Default: `0`.
      * `:dilation` (`integer`) - stride between elements within the sliding window. Default: `1`.
      * `:ceil_mode` (`boolean`) - use ceil instead of floor to compute output shape. Default: `false`.

  ## Shape
    * Input: `{N, C, H_in, W_in}` or `{C, H_in, W_in}`.
    * Output: `{N, C, H_out, W_out}` or `{C, H_out, W_out}`.

  ## Examples

      iex> m = ExTorch.NN.max_pool2d(2)
      iex> input = ExTorch.randn({1, 1, 8, 8})
      iex> output = ExTorch.NN.forward(input, m)
      iex> output.size
      {1, 1, 4, 4}
  """
  @spec max_pool2d(integer(), keyword()) :: Layer.t()
  def max_pool2d(kernel_size, opts \\ []) do
    ExTorch.Native.nn_max_pool2d(kernel_size,
      Keyword.get(opts, :stride, kernel_size), Keyword.get(opts, :padding, 0),
      Keyword.get(opts, :dilation, 1), Keyword.get(opts, :ceil_mode, false))
  end

  @doc """
  Applies a 1D average pooling over an input signal.

  ## Args
    * `kernel_size` (`integer`) - the size of the sliding window.
    * `opts` (`keyword`) - optional: `:stride` (default `kernel_size`), `:padding` (default `0`),
      `:ceil_mode` (default `false`), `:count_include_pad` (default `true`).

  ## Shape
    * Input: `{N, C, L_in}`. Output: `{N, C, L_out}`.
  """
  @spec avg_pool1d(integer(), keyword()) :: Layer.t()
  def avg_pool1d(kernel_size, opts \\ []) do
    ExTorch.Native.nn_avg_pool1d(kernel_size,
      Keyword.get(opts, :stride, kernel_size), Keyword.get(opts, :padding, 0),
      Keyword.get(opts, :ceil_mode, false), Keyword.get(opts, :count_include_pad, true))
  end

  @doc """
  Applies a 2D average pooling over an input signal.

  ## Args
    * `kernel_size` (`integer`) - the size of the sliding window.
    * `opts` (`keyword`) - optional: `:stride` (default `kernel_size`), `:padding` (default `0`),
      `:ceil_mode` (default `false`), `:count_include_pad` (default `true`).

  ## Shape
    * Input: `{N, C, H_in, W_in}`. Output: `{N, C, H_out, W_out}`.
  """
  @spec avg_pool2d(integer(), keyword()) :: Layer.t()
  def avg_pool2d(kernel_size, opts \\ []) do
    ExTorch.Native.nn_avg_pool2d(kernel_size,
      Keyword.get(opts, :stride, kernel_size), Keyword.get(opts, :padding, 0),
      Keyword.get(opts, :ceil_mode, false), Keyword.get(opts, :count_include_pad, true))
  end

  @doc """
  Applies a 1D adaptive average pooling over an input signal.

  The output size is `output_size` regardless of input size.

  ## Args
    * `output_size` (`integer`) - the target output size.

  ## Shape
    * Input: `{N, C, L_in}`. Output: `{N, C, output_size}`.
  """
  @spec adaptive_avg_pool1d(integer()) :: Layer.t()
  def adaptive_avg_pool1d(output_size), do: ExTorch.Native.nn_adaptive_avg_pool1d(output_size)

  @doc """
  Applies a 2D adaptive average pooling over an input signal.

  The output spatial size is `{output_h, output_w}` regardless of input size.

  ## Args
    * `output_h` (`integer`) - the target output height.
    * `output_w` (`integer`) - the target output width.

  ## Shape
    * Input: `{N, C, H_in, W_in}`. Output: `{N, C, output_h, output_w}`.

  ## Examples

      iex> m = ExTorch.NN.adaptive_avg_pool2d(1, 1)
      iex> input = ExTorch.randn({1, 64, 7, 7})
      iex> output = ExTorch.NN.forward(input, m)
      iex> output.size
      {1, 64, 1, 1}
  """
  @spec adaptive_avg_pool2d(integer(), integer()) :: Layer.t()
  def adaptive_avg_pool2d(output_h, output_w), do: ExTorch.Native.nn_adaptive_avg_pool2d(output_h, output_w)

  # ============================================================================
  # Normalization layers
  # ============================================================================

  @doc """
  Applies Batch Normalization over a 2D or 3D input (a mini-batch of 1D inputs
  with optional additional channel dimension).

  $y = \\frac{x - E[x]}{\\sqrt{Var[x] + \\epsilon}} * \\gamma + \\beta$

  The mean and standard-deviation are calculated per-dimension over the
  mini-batches and $\\gamma$ and $\\beta$ are learnable parameter vectors of
  size C (where C is the number of features).

  ## Args
    * `num_features` (`integer`) - C from an expected input of size `{N, C}` or `{N, C, L}`.
    * `opts` (`keyword`) - optional arguments:
      * `:eps` (`float`) - value added to the denominator for numerical stability. Default: `1.0e-5`.
      * `:momentum` (`float`) - the value used for the running_mean and running_var computation. Default: `0.1`.
      * `:affine` (`boolean`) - if `true`, has learnable affine parameters. Default: `true`.
      * `:track_running_stats` (`boolean`) - if `true`, tracks running mean and variance. Default: `true`.

  ## Shape
    * Input: `{N, C}` or `{N, C, L}`.
    * Output: same shape as input.
  """
  @spec batch_norm1d(integer(), keyword()) :: Layer.t()
  def batch_norm1d(num_features, opts \\ []) do
    ExTorch.Native.nn_batch_norm1d(
      num_features,
      Keyword.get(opts, :eps, 1.0e-5),
      Keyword.get(opts, :momentum, 0.1),
      Keyword.get(opts, :affine, true),
      Keyword.get(opts, :track_running_stats, true)
    )
  end

  @doc """
  Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
  with additional channel dimension).

  $y = \\frac{x - E[x]}{\\sqrt{Var[x] + \\epsilon}} * \\gamma + \\beta$

  ## Args
    * `num_features` (`integer`) - C from an expected input of size `{N, C, H, W}`.
    * `opts` (`keyword`) - same as `batch_norm1d/2`.

  ## Shape
    * Input: `{N, C, H, W}`.
    * Output: same shape as input.
  """
  @spec batch_norm2d(integer(), keyword()) :: Layer.t()
  def batch_norm2d(num_features, opts \\ []) do
    ExTorch.Native.nn_batch_norm2d(
      num_features,
      Keyword.get(opts, :eps, 1.0e-5),
      Keyword.get(opts, :momentum, 0.1),
      Keyword.get(opts, :affine, true),
      Keyword.get(opts, :track_running_stats, true)
    )
  end

  @doc """
  Applies Layer Normalization over a mini-batch of inputs.

  $y = \\frac{x - E[x]}{\\sqrt{Var[x] + \\epsilon}} * \\gamma + \\beta$

  The mean and standard-deviation are calculated over the last D dimensions,
  where D is the dimension of `normalized_shape`.

  ## Args
    * `normalized_shape` (`[integer]` or `tuple`) - input shape from an expected input.
    * `opts` (`keyword`) - optional arguments:
      * `:eps` (`float`) - value added to the denominator for numerical stability. Default: `1.0e-5`.
      * `:elementwise_affine` (`boolean`) - if `true`, has learnable per-element affine parameters. Default: `true`.

  ## Shape
    * Input: `{N, *}` where `*` matches `normalized_shape`.
    * Output: same shape as input.

  ## Examples

      iex> m = ExTorch.NN.layer_norm([10])
      iex> input = ExTorch.randn({3, 10})
      iex> output = ExTorch.NN.forward(input, m)
      iex> output.size
      {3, 10}
  """
  @spec layer_norm([integer()] | tuple(), keyword()) :: Layer.t()
  def layer_norm(normalized_shape, opts \\ []) do
    shape = if is_tuple(normalized_shape), do: Tuple.to_list(normalized_shape), else: normalized_shape
    ExTorch.Native.nn_layer_norm(
      shape,
      Keyword.get(opts, :eps, 1.0e-5),
      Keyword.get(opts, :elementwise_affine, true)
    )
  end

  @doc """
  Applies Group Normalization over a mini-batch of inputs.

  $y = \\frac{x - E[x]}{\\sqrt{Var[x] + \\epsilon}} * \\gamma + \\beta$

  The input channels are separated into `num_groups` groups, each containing
  `num_channels / num_groups` channels. Mean and standard-deviation are
  calculated separately over each group.

  ## Args
    * `num_groups` (`integer`) - number of groups to separate the channels into.
    * `num_channels` (`integer`) - number of channels expected in input.
    * `opts` (`keyword`) - optional: `:eps` (default `1.0e-5`), `:affine` (default `true`).

  ## Shape
    * Input: `{N, C, *}` where C = `num_channels`.
    * Output: same shape as input.
  """
  @spec group_norm(integer(), integer(), keyword()) :: Layer.t()
  def group_norm(num_groups, num_channels, opts \\ []) do
    ExTorch.Native.nn_group_norm(num_groups, num_channels,
      Keyword.get(opts, :eps, 1.0e-5), Keyword.get(opts, :affine, true))
  end

  @doc """
  Applies Instance Normalization over a 3D input (a mini-batch of 1D inputs).

  $y = \\frac{x - E[x]}{\\sqrt{Var[x] + \\epsilon}} * \\gamma + \\beta$

  The mean and standard-deviation are calculated per-instance per-channel.

  ## Args
    * `num_features` (`integer`) - C from an expected input of size `{N, C, L}`.
    * `opts` (`keyword`) - optional: `:eps` (default `1.0e-5`), `:momentum` (default `0.1`),
      `:affine` (default `false`), `:track_running_stats` (default `false`).

  ## Shape
    * Input: `{N, C, L}`. Output: same shape.
  """
  @spec instance_norm1d(integer(), keyword()) :: Layer.t()
  def instance_norm1d(num_features, opts \\ []) do
    ExTorch.Native.nn_instance_norm1d(num_features,
      Keyword.get(opts, :eps, 1.0e-5), Keyword.get(opts, :momentum, 0.1),
      Keyword.get(opts, :affine, false), Keyword.get(opts, :track_running_stats, false))
  end

  @doc """
  Applies Instance Normalization over a 4D input (a mini-batch of 2D inputs).

  ## Args
    * `num_features` (`integer`) - C from an expected input of size `{N, C, H, W}`.
    * `opts` (`keyword`) - same as `instance_norm1d/2`.

  ## Shape
    * Input: `{N, C, H, W}`. Output: same shape.
  """
  @spec instance_norm2d(integer(), keyword()) :: Layer.t()
  def instance_norm2d(num_features, opts \\ []) do
    ExTorch.Native.nn_instance_norm2d(num_features,
      Keyword.get(opts, :eps, 1.0e-5), Keyword.get(opts, :momentum, 0.1),
      Keyword.get(opts, :affine, false), Keyword.get(opts, :track_running_stats, false))
  end

  # ============================================================================
  # Dropout layers
  # ============================================================================

  @doc """
  During training, randomly zeroes some of the elements of the input tensor
  with probability `p` using samples from a Bernoulli distribution. Each
  channel will be zeroed out independently on every forward call.

  Furthermore, the outputs are scaled by a factor of $\\frac{1}{1-p}$ during
  training. This means that during evaluation the module simply computes an
  identity function.

  ## Args
    * `opts` (`keyword`) - optional arguments:
      * `:p` (`float`) - probability of an element to be zeroed. Default: `0.5`.
      * `:inplace` (`boolean`) - if `true`, will do this operation in-place. Default: `false`.

  ## Shape
    * Input: `{*}` (any shape).
    * Output: same shape as input.

  ## Examples

      iex> m = ExTorch.NN.dropout(p: 0.2)
      #NN.Layer<Dropout>
  """
  @spec dropout(keyword()) :: Layer.t()
  def dropout(opts \\ []) do
    ExTorch.Native.nn_dropout(
      Keyword.get(opts, :p, 0.5),
      Keyword.get(opts, :inplace, false)
    )
  end

  # ============================================================================
  # Embedding layers
  # ============================================================================

  @doc """
  A simple lookup table that stores embeddings of a fixed dictionary and size.

  This module is often used to store word embeddings and retrieve them using
  indices. The input to the module is a list of indices, and the output is the
  corresponding word embeddings.

  ## Args
    * `num_embeddings` (`integer`) - size of the embedding dictionary.
    * `embedding_dim` (`integer`) - the size of each embedding vector.
    * `opts` (`keyword`) - optional arguments:
      * `:padding_idx` (`integer` or `nil`) - if specified, the entries at
        `padding_idx` do not contribute to the gradient. Default: `nil`.

  ## Shape
    * Input: `{*}` (integer tensor of arbitrary shape).
    * Output: `{*, embedding_dim}`.

  ## Variables
    * `weight` - the learnable weights of shape `{num_embeddings, embedding_dim}`.

  ## Examples

      iex> m = ExTorch.NN.embedding(10, 3)
      iex> input = ExTorch.tensor([1, 2, 4, 5], dtype: :int64)
      iex> output = ExTorch.NN.forward(input, m)
      iex> output.size
      {4, 3}
  """
  @spec embedding(integer(), integer(), keyword()) :: Layer.t()
  def embedding(num_embeddings, embedding_dim, opts \\ []) do
    {padding_idx, has_padding} =
      case Keyword.get(opts, :padding_idx) do
        nil -> {0, false}
        idx -> {idx, true}
      end

    ExTorch.Native.nn_embedding(num_embeddings, embedding_dim, padding_idx, has_padding)
  end

  # ============================================================================
  # Recurrent layers
  # ============================================================================

  @doc """
  Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

  For each element in the input sequence, each layer computes the following function:

  $i_t = \\sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi})$

  $f_t = \\sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf})$

  $g_t = \\tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg})$

  $o_t = \\sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho})$

  $c_t = f_t \\odot c_{t-1} + i_t \\odot g_t$

  $h_t = o_t \\odot \\tanh(c_t)$

  ## Args
    * `input_size` (`integer`) - the number of expected features in the input.
    * `hidden_size` (`integer`) - the number of features in the hidden state.
    * `opts` (`keyword`) - optional arguments:
      * `:num_layers` (`integer`) - number of recurrent layers. Default: `1`.
      * `:bias` (`boolean`) - if `false`, the layer does not use bias weights. Default: `true`.
      * `:batch_first` (`boolean`) - if `true`, input/output tensors are `{batch, seq, feature}`. Default: `false`.
      * `:dropout` (`float`) - dropout probability on outputs of each layer except the last. Default: `0.0`.
      * `:bidirectional` (`boolean`) - if `true`, becomes a bidirectional LSTM. Default: `false`.

  ## Shape
    * Input: `{L, N, H_in}` or `{N, L, H_in}` when `batch_first: true`.
    * Output: `{L, N, D * H_out}` where D = 2 if bidirectional, else 1.
  """
  @spec lstm(integer(), integer(), keyword()) :: Layer.t()
  def lstm(input_size, hidden_size, opts \\ []) do
    ExTorch.Native.nn_lstm(input_size, hidden_size,
      Keyword.get(opts, :num_layers, 1), Keyword.get(opts, :bias, true),
      Keyword.get(opts, :batch_first, false), Keyword.get(opts, :dropout, 0.0),
      Keyword.get(opts, :bidirectional, false))
  end

  @doc """
  Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.

  For each element in the input sequence, each layer computes:

  $r_t = \\sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{t-1} + b_{hr})$

  $z_t = \\sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{t-1} + b_{hz})$

  $n_t = \\tanh(W_{in} x_t + b_{in} + r_t \\odot (W_{hn} h_{t-1} + b_{hn}))$

  $h_t = (1 - z_t) \\odot n_t + z_t \\odot h_{t-1}$

  ## Args
    * `input_size` (`integer`) - the number of expected features in the input.
    * `hidden_size` (`integer`) - the number of features in the hidden state.
    * `opts` (`keyword`) - same options as `lstm/3`.

  ## Shape
    * Input: `{L, N, H_in}` or `{N, L, H_in}` when `batch_first: true`.
    * Output: `{L, N, D * H_out}` where D = 2 if bidirectional, else 1.
  """
  @spec gru(integer(), integer(), keyword()) :: Layer.t()
  def gru(input_size, hidden_size, opts \\ []) do
    ExTorch.Native.nn_gru(input_size, hidden_size,
      Keyword.get(opts, :num_layers, 1), Keyword.get(opts, :bias, true),
      Keyword.get(opts, :batch_first, false), Keyword.get(opts, :dropout, 0.0),
      Keyword.get(opts, :bidirectional, false))
  end

  # ============================================================================
  # Transformer layers
  # ============================================================================

  @doc """
  Allows the model to jointly attend to information from different
  representation subspaces.

  $MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O$

  where $head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)$.

  ## Args
    * `embed_dim` (`integer`) - total dimension of the model.
    * `num_heads` (`integer`) - number of parallel attention heads. `embed_dim` will
      be split across `num_heads` (i.e. each head will have dimension `embed_dim / num_heads`).
    * `opts` (`keyword`) - optional arguments:
      * `:dropout` (`float`) - dropout probability on attention weights. Default: `0.0`.
      * `:bias` (`boolean`) - if `false`, input/output projection layers will not learn an additive bias. Default: `true`.
  """
  @spec multihead_attention(integer(), integer(), keyword()) :: Layer.t()
  def multihead_attention(embed_dim, num_heads, opts \\ []) do
    ExTorch.Native.nn_multihead_attention(embed_dim, num_heads,
      Keyword.get(opts, :dropout, 0.0), Keyword.get(opts, :bias, true))
  end

  # ============================================================================
  # Activation functions
  # ============================================================================

  @doc """
  Applies the rectified linear unit function element-wise: $ReLU(x) = \\max(0, x)$.

  ## Args
    * `opts` (`keyword`) - optional:
      * `:inplace` (`boolean`) - can optionally do the operation in-place. Default: `false`.

  ## Shape
    * Input: `{*}` (any shape).
    * Output: same shape as input.
  """
  @spec relu(keyword()) :: Layer.t()
  def relu(opts \\ []) do
    ExTorch.Native.nn_relu(Keyword.get(opts, :inplace, false))
  end

  @doc """
  Applies the Gaussian Error Linear Units function: $GELU(x) = x \\Phi(x)$

  where $\\Phi(x)$ is the cumulative distribution function for the Gaussian distribution.

  ## Shape
    * Input: `{*}`. Output: same shape as input.
  """
  @spec gelu() :: Layer.t()
  def gelu, do: ExTorch.Native.nn_gelu()

  @doc """
  Applies the element-wise Sigmoid function: $Sigmoid(x) = \\frac{1}{1 + e^{-x}}$

  ## Shape
    * Input: `{*}`. Output: same shape as input.
  """
  @spec sigmoid() :: Layer.t()
  def sigmoid, do: ExTorch.Native.nn_sigmoid()

  @doc """
  Applies the element-wise Tanh function: $Tanh(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}$

  ## Shape
    * Input: `{*}`. Output: same shape as input.
  """
  @spec tanh() :: Layer.t()
  def tanh, do: ExTorch.Native.nn_tanh()

  @doc """
  Applies the Softmax function to an n-dimensional input tensor, rescaling them
  so that the elements of the n-dimensional output tensor lie in the range [0,1] and sum to 1.

  $Softmax(x_i) = \\frac{e^{x_i}}{\\sum_j e^{x_j}}$

  ## Args
    * `dim` (`integer`) - dimension along which Softmax will be computed.

  ## Shape
    * Input: `{*}`. Output: same shape as input.
  """
  @spec softmax(integer()) :: Layer.t()
  def softmax(dim), do: ExTorch.Native.nn_softmax(dim)

  @doc """
  Applies element-wise: $LeakyReLU(x) = \\max(0, x) + negative\\_slope * \\min(0, x)$

  ## Args
    * `opts` (`keyword`) - optional:
      * `:negative_slope` (`float`) - controls the angle of the negative slope. Default: `0.01`.
      * `:inplace` (`boolean`) - do the operation in-place. Default: `false`.

  ## Shape
    * Input: `{*}`. Output: same shape as input.
  """
  @spec leaky_relu(keyword()) :: Layer.t()
  def leaky_relu(opts \\ []) do
    ExTorch.Native.nn_leaky_relu(
      Keyword.get(opts, :negative_slope, 0.01),
      Keyword.get(opts, :inplace, false)
    )
  end

  @doc """
  Applies element-wise: $ELU(x) = \\max(0, x) + \\min(0, \\alpha * (e^x - 1))$

  ## Args
    * `opts` (`keyword`) - optional:
      * `:alpha` (`float`) - the $\\alpha$ value for the ELU formulation. Default: `1.0`.
      * `:inplace` (`boolean`) - do the operation in-place. Default: `false`.

  ## Shape
    * Input: `{*}`. Output: same shape as input.
  """
  @spec elu(keyword()) :: Layer.t()
  def elu(opts \\ []) do
    ExTorch.Native.nn_elu(
      Keyword.get(opts, :alpha, 1.0),
      Keyword.get(opts, :inplace, false)
    )
  end

  @doc """
  Applies the Sigmoid Linear Unit (SiLU) function, element-wise.
  Also known as the swish function.

  $SiLU(x) = x * \\sigma(x)$ where $\\sigma(x)$ is the logistic sigmoid.

  ## Shape
    * Input: `{*}`. Output: same shape as input.
  """
  @spec silu() :: Layer.t()
  def silu, do: ExTorch.Native.nn_silu()

  @doc """
  Applies the Mish function, element-wise.

  $Mish(x) = x * \\tanh(Softplus(x))$

  ## Shape
    * Input: `{*}`. Output: same shape as input.
  """
  @spec mish() :: Layer.t()
  def mish, do: ExTorch.Native.nn_mish()

  @doc """
  Applies element-wise: $PReLU(x) = \\max(0, x) + a * \\min(0, x)$

  where $a$ is a learnable parameter.

  ## Args
    * `opts` (`keyword`) - optional:
      * `:num_parameters` (`integer`) - number of $a$ to learn. Default: `1`.

  ## Variables
    * `weight` - the learnable parameter $a$ of shape `{num_parameters}`.

  ## Shape
    * Input: `{*}`. Output: same shape as input.
  """
  @spec prelu(keyword()) :: Layer.t()
  def prelu(opts \\ []), do: ExTorch.Native.nn_prelu(Keyword.get(opts, :num_parameters, 1))

  @doc """
  Applies the LogSoftmax function: $LogSoftmax(x_i) = \\log\\left(\\frac{e^{x_i}}{\\sum_j e^{x_j}}\\right)$

  ## Args
    * `dim` (`integer`) - dimension along which LogSoftmax will be computed.

  ## Shape
    * Input: `{*}`. Output: same shape as input.
  """
  @spec log_softmax(integer()) :: Layer.t()
  def log_softmax(dim), do: ExTorch.Native.nn_log_softmax(dim)

  # ============================================================================
  # Utility layers
  # ============================================================================

  @doc """
  Flattens a contiguous range of dims into a tensor.

  ## Args
    * `opts` (`keyword`) - optional:
      * `:start_dim` (`integer`) - first dim to flatten. Default: `1`.
      * `:end_dim` (`integer`) - last dim to flatten. Default: `-1`.

  ## Shape
    * Input: `{*, S_start, ..., S_end, *}`.
    * Output: `{*, product(S_start...S_end), *}`.

  ## Examples

      iex> m = ExTorch.NN.flatten()
      iex> input = ExTorch.randn({2, 3, 4, 5})
      iex> output = ExTorch.NN.forward(input, m)
      iex> output.size
      {2, 60}
  """
  @spec flatten(keyword()) :: Layer.t()
  def flatten(opts \\ []) do
    ExTorch.Native.nn_flatten(
      Keyword.get(opts, :start_dim, 1),
      Keyword.get(opts, :end_dim, -1))
  end

  @doc """
  Unflattens a tensor dim, expanding it to a desired shape.

  ## Args
    * `dim` (`integer`) - dimension to unflatten.
    * `sizes` (`[integer]` or `tuple`) - new shape of the unflattened dimension.

  ## Shape
    * Input: `{*, S_dim, *}` where `S_dim = product(sizes)`.
    * Output: `{*, sizes[0], sizes[1], ..., *}`.
  """
  @spec unflatten(integer(), [integer()] | tuple()) :: Layer.t()
  def unflatten(dim, sizes) do
    sizes = if is_tuple(sizes), do: Tuple.to_list(sizes), else: sizes
    ExTorch.Native.nn_unflatten(dim, sizes)
  end

  # ============================================================================
  # Module operations
  # ============================================================================

  @doc """
  Run the forward pass of a layer on an input tensor.

  Applies the layer's computation to the input and returns the output tensor.
  This is the primary way to use layers created by the factory functions in
  this module.

  ## Args
    * `input` (`ExTorch.Tensor`) - input tensor.
    * `layer` (`ExTorch.NN.Layer`) - the layer to apply.

  ## Returns
  The output tensor.

  ## Examples

      iex> m = ExTorch.NN.linear(10, 5)
      iex> input = ExTorch.randn({1, 10})
      iex> output = ExTorch.NN.forward(input, m)
      iex> output.size
      {1, 5}
  """
  @spec forward(ExTorch.Tensor.t(), Layer.t()) :: ExTorch.Tensor.t()
  def forward(%ExTorch.Tensor{} = input, %Layer{} = layer) do
    ExTorch.Native.nn_forward(layer, input)
  end

  @doc """
  Get named parameters of a layer.

  Returns a list of `{name, tensor}` tuples containing the learnable
  parameters of the layer.

  ## Examples

      iex> m = ExTorch.NN.linear(10, 5)
      iex> params = ExTorch.NN.parameters(m)
      iex> length(params)
      2
  """
  @spec parameters(Layer.t()) :: [{String.t(), ExTorch.Tensor.t()}]
  def parameters(%Layer{} = layer) do
    ExTorch.Native.nn_parameters(layer)
  end

  @doc """
  Set a layer to evaluation mode.

  This has effect on certain modules like `Dropout` and `BatchNorm`
  which behave differently during training vs evaluation.
  """
  @spec eval(Layer.t()) :: :ok
  def eval(%Layer{} = layer) do
    ExTorch.Native.nn_set_eval(layer)
    :ok
  end

  @doc """
  Set a layer to training mode.

  See `eval/1` for the inverse operation.
  """
  @spec train(Layer.t()) :: :ok
  def train(%Layer{} = layer) do
    ExTorch.Native.nn_set_train(layer)
    :ok
  end

  @doc """
  Copy parameter values from a list of `{name, tensor}` tuples into a layer.

  This enables loading pre-trained weights from a JIT model or another layer.
  Parameter names must match. The copy is performed in-place under a no-grad guard.

  ## Args
    * `layer` (`ExTorch.NN.Layer`) - destination layer.
    * `params` (`[{String.t(), ExTorch.Tensor.t()}]`) - source parameters.
  """
  @spec copy_parameters(Layer.t(), [{String.t(), ExTorch.Tensor.t()}]) :: :ok
  def copy_parameters(%Layer{} = layer, params) do
    ExTorch.Native.nn_copy_parameters(layer, params)
    :ok
  end

  @doc """
  Move a layer to a different device.

  ## Args
    * `layer` (`ExTorch.NN.Layer`) - the layer to move.
    * `device` (`ExTorch.Device`) - target device (e.g., `:cpu`, `{:cuda, 0}`).

  ## Returns
  A new layer on the target device.
  """
  @spec to(Layer.t(), ExTorch.Device.device()) :: Layer.t()
  def to(%Layer{} = layer, device) do
    ExTorch.Native.nn_to_device(layer, device)
  end
end
