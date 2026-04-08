defmodule ExTorch.NN do
  @moduledoc """
  Neural network layer creation and operations.

  This module provides functions to create PyTorch nn.Module layers and
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
  # Layer factories
  # ============================================================================

  @doc """
  Create a fully connected (linear) layer.

  ## Arguments
    - `in_features` - Size of each input sample.
    - `out_features` - Size of each output sample.
    - `opts` - Keyword options:
      - `:bias` - Whether to include a learnable bias (default: `true`).
  """
  @spec linear(integer(), integer(), keyword()) :: Layer.t()
  def linear(in_features, out_features, opts \\ []) do
    bias = Keyword.get(opts, :bias, true)
    ExTorch.Native.nn_linear(in_features, out_features, bias)
  end

  @doc """
  Create a 1D convolution layer.

  ## Arguments
    - `in_channels` - Number of input channels.
    - `out_channels` - Number of output channels.
    - `kernel_size` - Size of the convolving kernel.
    - `opts` - Keyword options: `:stride`, `:padding`, `:dilation`, `:groups`, `:bias`.
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
  Create a 2D convolution layer.

  ## Arguments
    - `in_channels` - Number of input channels.
    - `out_channels` - Number of output channels.
    - `kernel_size` - Size of the convolving kernel.
    - `opts` - Keyword options: `:stride`, `:padding`, `:dilation`, `:groups`, `:bias`.
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
  Create a 1D batch normalization layer.

  ## Arguments
    - `num_features` - Number of features (C from an input of size (N, C, ...)).
    - `opts` - Keyword options: `:eps`, `:momentum`, `:affine`, `:track_running_stats`.
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
  Create a 2D batch normalization layer.

  ## Arguments
    - `num_features` - Number of features (C from an input of size (N, C, H, W)).
    - `opts` - Keyword options: `:eps`, `:momentum`, `:affine`, `:track_running_stats`.
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
  Create a layer normalization layer.

  ## Arguments
    - `normalized_shape` - Shape of the normalized dimensions (list or tuple of integers).
    - `opts` - Keyword options: `:eps`, `:elementwise_affine`.
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
  Create a dropout layer.

  ## Arguments
    - `opts` - Keyword options:
      - `:p` - Probability of an element being zeroed (default: `0.5`).
      - `:inplace` - Operate in-place (default: `false`).
  """
  @spec dropout(keyword()) :: Layer.t()
  def dropout(opts \\ []) do
    ExTorch.Native.nn_dropout(
      Keyword.get(opts, :p, 0.5),
      Keyword.get(opts, :inplace, false)
    )
  end

  @doc """
  Create an embedding layer.

  ## Arguments
    - `num_embeddings` - Size of the embedding dictionary.
    - `embedding_dim` - Size of each embedding vector.
    - `opts` - Keyword options:
      - `:padding_idx` - Index for zero-padded embeddings (default: none).
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
  # Activation functions
  # ============================================================================

  @doc "Create a ReLU activation layer."
  @spec relu(keyword()) :: Layer.t()
  def relu(opts \\ []) do
    ExTorch.Native.nn_relu(Keyword.get(opts, :inplace, false))
  end

  @doc "Create a GELU activation layer."
  @spec gelu() :: Layer.t()
  def gelu, do: ExTorch.Native.nn_gelu()

  @doc "Create a Sigmoid activation layer."
  @spec sigmoid() :: Layer.t()
  def sigmoid, do: ExTorch.Native.nn_sigmoid()

  @doc "Create a Tanh activation layer."
  @spec tanh() :: Layer.t()
  def tanh, do: ExTorch.Native.nn_tanh()

  @doc "Create a Softmax activation layer."
  @spec softmax(integer()) :: Layer.t()
  def softmax(dim), do: ExTorch.Native.nn_softmax(dim)

  @doc "Create a LeakyReLU activation layer."
  @spec leaky_relu(keyword()) :: Layer.t()
  def leaky_relu(opts \\ []) do
    ExTorch.Native.nn_leaky_relu(
      Keyword.get(opts, :negative_slope, 0.01),
      Keyword.get(opts, :inplace, false)
    )
  end

  @doc "Create an ELU activation layer."
  @spec elu(keyword()) :: Layer.t()
  def elu(opts \\ []) do
    ExTorch.Native.nn_elu(
      Keyword.get(opts, :alpha, 1.0),
      Keyword.get(opts, :inplace, false)
    )
  end

  @doc "Create a SiLU (Swish) activation layer."
  @spec silu() :: Layer.t()
  def silu, do: ExTorch.Native.nn_silu()

  @doc "Create a Mish activation layer."
  @spec mish() :: Layer.t()
  def mish, do: ExTorch.Native.nn_mish()

  @doc "Create a PReLU activation layer."
  @spec prelu(keyword()) :: Layer.t()
  def prelu(opts \\ []), do: ExTorch.Native.nn_prelu(Keyword.get(opts, :num_parameters, 1))

  @doc "Create a LogSoftmax layer."
  @spec log_softmax(integer()) :: Layer.t()
  def log_softmax(dim), do: ExTorch.Native.nn_log_softmax(dim)

  # ============================================================================
  # Additional convolutions
  # ============================================================================

  @doc "Create a 3D convolution layer."
  @spec conv3d(integer(), integer(), integer(), keyword()) :: Layer.t()
  def conv3d(in_channels, out_channels, kernel_size, opts \\ []) do
    ExTorch.Native.nn_conv3d(in_channels, out_channels, kernel_size,
      Keyword.get(opts, :stride, 1), Keyword.get(opts, :padding, 0),
      Keyword.get(opts, :dilation, 1), Keyword.get(opts, :groups, 1),
      Keyword.get(opts, :bias, true))
  end

  @doc "Create a 1D transposed convolution layer."
  @spec conv_transpose1d(integer(), integer(), integer(), keyword()) :: Layer.t()
  def conv_transpose1d(in_channels, out_channels, kernel_size, opts \\ []) do
    ExTorch.Native.nn_conv_transpose1d(in_channels, out_channels, kernel_size,
      Keyword.get(opts, :stride, 1), Keyword.get(opts, :padding, 0),
      Keyword.get(opts, :output_padding, 0), Keyword.get(opts, :groups, 1),
      Keyword.get(opts, :bias, true), Keyword.get(opts, :dilation, 1))
  end

  @doc "Create a 2D transposed convolution layer."
  @spec conv_transpose2d(integer(), integer(), integer(), keyword()) :: Layer.t()
  def conv_transpose2d(in_channels, out_channels, kernel_size, opts \\ []) do
    ExTorch.Native.nn_conv_transpose2d(in_channels, out_channels, kernel_size,
      Keyword.get(opts, :stride, 1), Keyword.get(opts, :padding, 0),
      Keyword.get(opts, :output_padding, 0), Keyword.get(opts, :groups, 1),
      Keyword.get(opts, :bias, true), Keyword.get(opts, :dilation, 1))
  end

  # ============================================================================
  # Pooling
  # ============================================================================

  @doc "Create a 1D max pooling layer."
  @spec max_pool1d(integer(), keyword()) :: Layer.t()
  def max_pool1d(kernel_size, opts \\ []) do
    ExTorch.Native.nn_max_pool1d(kernel_size,
      Keyword.get(opts, :stride, kernel_size), Keyword.get(opts, :padding, 0),
      Keyword.get(opts, :dilation, 1), Keyword.get(opts, :ceil_mode, false))
  end

  @doc "Create a 2D max pooling layer."
  @spec max_pool2d(integer(), keyword()) :: Layer.t()
  def max_pool2d(kernel_size, opts \\ []) do
    ExTorch.Native.nn_max_pool2d(kernel_size,
      Keyword.get(opts, :stride, kernel_size), Keyword.get(opts, :padding, 0),
      Keyword.get(opts, :dilation, 1), Keyword.get(opts, :ceil_mode, false))
  end

  @doc "Create a 1D average pooling layer."
  @spec avg_pool1d(integer(), keyword()) :: Layer.t()
  def avg_pool1d(kernel_size, opts \\ []) do
    ExTorch.Native.nn_avg_pool1d(kernel_size,
      Keyword.get(opts, :stride, kernel_size), Keyword.get(opts, :padding, 0),
      Keyword.get(opts, :ceil_mode, false), Keyword.get(opts, :count_include_pad, true))
  end

  @doc "Create a 2D average pooling layer."
  @spec avg_pool2d(integer(), keyword()) :: Layer.t()
  def avg_pool2d(kernel_size, opts \\ []) do
    ExTorch.Native.nn_avg_pool2d(kernel_size,
      Keyword.get(opts, :stride, kernel_size), Keyword.get(opts, :padding, 0),
      Keyword.get(opts, :ceil_mode, false), Keyword.get(opts, :count_include_pad, true))
  end

  @doc "Create a 1D adaptive average pooling layer."
  @spec adaptive_avg_pool1d(integer()) :: Layer.t()
  def adaptive_avg_pool1d(output_size), do: ExTorch.Native.nn_adaptive_avg_pool1d(output_size)

  @doc "Create a 2D adaptive average pooling layer."
  @spec adaptive_avg_pool2d(integer(), integer()) :: Layer.t()
  def adaptive_avg_pool2d(output_h, output_w), do: ExTorch.Native.nn_adaptive_avg_pool2d(output_h, output_w)

  # ============================================================================
  # Additional normalization
  # ============================================================================

  @doc "Create a group normalization layer."
  @spec group_norm(integer(), integer(), keyword()) :: Layer.t()
  def group_norm(num_groups, num_channels, opts \\ []) do
    ExTorch.Native.nn_group_norm(num_groups, num_channels,
      Keyword.get(opts, :eps, 1.0e-5), Keyword.get(opts, :affine, true))
  end

  @doc "Create a 1D instance normalization layer."
  @spec instance_norm1d(integer(), keyword()) :: Layer.t()
  def instance_norm1d(num_features, opts \\ []) do
    ExTorch.Native.nn_instance_norm1d(num_features,
      Keyword.get(opts, :eps, 1.0e-5), Keyword.get(opts, :momentum, 0.1),
      Keyword.get(opts, :affine, false), Keyword.get(opts, :track_running_stats, false))
  end

  @doc "Create a 2D instance normalization layer."
  @spec instance_norm2d(integer(), keyword()) :: Layer.t()
  def instance_norm2d(num_features, opts \\ []) do
    ExTorch.Native.nn_instance_norm2d(num_features,
      Keyword.get(opts, :eps, 1.0e-5), Keyword.get(opts, :momentum, 0.1),
      Keyword.get(opts, :affine, false), Keyword.get(opts, :track_running_stats, false))
  end

  # ============================================================================
  # Recurrent
  # ============================================================================

  @doc "Create an LSTM layer."
  @spec lstm(integer(), integer(), keyword()) :: Layer.t()
  def lstm(input_size, hidden_size, opts \\ []) do
    ExTorch.Native.nn_lstm(input_size, hidden_size,
      Keyword.get(opts, :num_layers, 1), Keyword.get(opts, :bias, true),
      Keyword.get(opts, :batch_first, false), Keyword.get(opts, :dropout, 0.0),
      Keyword.get(opts, :bidirectional, false))
  end

  @doc "Create a GRU layer."
  @spec gru(integer(), integer(), keyword()) :: Layer.t()
  def gru(input_size, hidden_size, opts \\ []) do
    ExTorch.Native.nn_gru(input_size, hidden_size,
      Keyword.get(opts, :num_layers, 1), Keyword.get(opts, :bias, true),
      Keyword.get(opts, :batch_first, false), Keyword.get(opts, :dropout, 0.0),
      Keyword.get(opts, :bidirectional, false))
  end

  # ============================================================================
  # Transformer
  # ============================================================================

  @doc "Create a multi-head attention layer."
  @spec multihead_attention(integer(), integer(), keyword()) :: Layer.t()
  def multihead_attention(embed_dim, num_heads, opts \\ []) do
    ExTorch.Native.nn_multihead_attention(embed_dim, num_heads,
      Keyword.get(opts, :dropout, 0.0), Keyword.get(opts, :bias, true))
  end

  # ============================================================================
  # Utility layers
  # ============================================================================

  @doc "Create a flatten layer."
  @spec flatten(keyword()) :: Layer.t()
  def flatten(opts \\ []) do
    ExTorch.Native.nn_flatten(
      Keyword.get(opts, :start_dim, 1),
      Keyword.get(opts, :end_dim, -1))
  end

  @doc "Create an unflatten layer."
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

  ## Arguments
    - `input` - Input tensor.
    - `layer` - The layer to apply.

  ## Returns
  The output tensor.

  ## Example

      linear = ExTorch.NN.linear(10, 5)
      input = ExTorch.randn({1, 10})
      output = ExTorch.NN.forward(input, linear)
  """
  @spec forward(ExTorch.Tensor.t(), Layer.t()) :: ExTorch.Tensor.t()
  def forward(%ExTorch.Tensor{} = input, %Layer{} = layer) do
    ExTorch.Native.nn_forward(layer, input)
  end

  @doc "Get named parameters of a layer as a list of `{name, tensor}` tuples."
  @spec parameters(Layer.t()) :: [{String.t(), ExTorch.Tensor.t()}]
  def parameters(%Layer{} = layer) do
    ExTorch.Native.nn_parameters(layer)
  end

  @doc "Set a layer to evaluation mode."
  @spec eval(Layer.t()) :: :ok
  def eval(%Layer{} = layer) do
    ExTorch.Native.nn_set_eval(layer)
    :ok
  end

  @doc "Set a layer to training mode."
  @spec train(Layer.t()) :: :ok
  def train(%Layer{} = layer) do
    ExTorch.Native.nn_set_train(layer)
    :ok
  end

  @doc """
  Copy parameter values from a list of `{name, tensor}` tuples into a layer.

  This enables loading pre-trained weights from a JIT model into a
  DSL-created layer. Parameter names must match.
  """
  @spec copy_parameters(Layer.t(), [{String.t(), ExTorch.Tensor.t()}]) :: :ok
  def copy_parameters(%Layer{} = layer, params) do
    ExTorch.Native.nn_copy_parameters(layer, params)
    :ok
  end

  @doc "Move a layer to a different device."
  @spec to(Layer.t(), ExTorch.Device.device()) :: Layer.t()
  def to(%Layer{} = layer, device) do
    ExTorch.Native.nn_to_device(layer, device)
  end
end
