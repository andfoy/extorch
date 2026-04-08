defmodule ExTorch.NN.Linear do
  @moduledoc "Linear (fully connected) layer."
  @behaviour ExTorch.NN.Module

  @impl true
  def create(opts) do
    in_features = Keyword.fetch!(opts, :in_features)
    out_features = Keyword.fetch!(opts, :out_features)
    ExTorch.NN.linear(in_features, out_features, opts)
  end
end

defmodule ExTorch.NN.Conv1d do
  @moduledoc "1D convolution layer."
  @behaviour ExTorch.NN.Module

  @impl true
  def create(opts) do
    in_channels = Keyword.fetch!(opts, :in_channels)
    out_channels = Keyword.fetch!(opts, :out_channels)
    kernel_size = Keyword.fetch!(opts, :kernel_size)
    ExTorch.NN.conv1d(in_channels, out_channels, kernel_size, opts)
  end
end

defmodule ExTorch.NN.Conv2d do
  @moduledoc "2D convolution layer."
  @behaviour ExTorch.NN.Module

  @impl true
  def create(opts) do
    in_channels = Keyword.fetch!(opts, :in_channels)
    out_channels = Keyword.fetch!(opts, :out_channels)
    kernel_size = Keyword.fetch!(opts, :kernel_size)
    ExTorch.NN.conv2d(in_channels, out_channels, kernel_size, opts)
  end
end

defmodule ExTorch.NN.BatchNorm1d do
  @moduledoc "1D batch normalization layer."
  @behaviour ExTorch.NN.Module

  @impl true
  def create(opts) do
    num_features = Keyword.fetch!(opts, :num_features)
    ExTorch.NN.batch_norm1d(num_features, opts)
  end
end

defmodule ExTorch.NN.BatchNorm2d do
  @moduledoc "2D batch normalization layer."
  @behaviour ExTorch.NN.Module

  @impl true
  def create(opts) do
    num_features = Keyword.fetch!(opts, :num_features)
    ExTorch.NN.batch_norm2d(num_features, opts)
  end
end

defmodule ExTorch.NN.LayerNorm do
  @moduledoc "Layer normalization."
  @behaviour ExTorch.NN.Module

  @impl true
  def create(opts) do
    normalized_shape = Keyword.fetch!(opts, :normalized_shape)
    ExTorch.NN.layer_norm(normalized_shape, opts)
  end
end

defmodule ExTorch.NN.Dropout do
  @moduledoc "Dropout layer."
  @behaviour ExTorch.NN.Module

  @impl true
  def create(opts), do: ExTorch.NN.dropout(opts)
end

defmodule ExTorch.NN.Embedding do
  @moduledoc "Embedding layer."
  @behaviour ExTorch.NN.Module

  @impl true
  def create(opts) do
    num_embeddings = Keyword.fetch!(opts, :num_embeddings)
    embedding_dim = Keyword.fetch!(opts, :embedding_dim)
    ExTorch.NN.embedding(num_embeddings, embedding_dim, opts)
  end
end

defmodule ExTorch.NN.ReLU do
  @moduledoc "ReLU activation."
  @behaviour ExTorch.NN.Module

  @impl true
  def create(opts \\ []), do: ExTorch.NN.relu(opts)
end

defmodule ExTorch.NN.GELU do
  @moduledoc "GELU activation."
  @behaviour ExTorch.NN.Module

  @impl true
  def create(_opts \\ []), do: ExTorch.NN.gelu()
end

defmodule ExTorch.NN.Sigmoid do
  @moduledoc "Sigmoid activation."
  @behaviour ExTorch.NN.Module

  @impl true
  def create(_opts \\ []), do: ExTorch.NN.sigmoid()
end

defmodule ExTorch.NN.Tanh do
  @moduledoc "Tanh activation."
  @behaviour ExTorch.NN.Module

  @impl true
  def create(_opts \\ []), do: ExTorch.NN.tanh()
end

defmodule ExTorch.NN.Softmax do
  @moduledoc "Softmax activation."
  @behaviour ExTorch.NN.Module

  @impl true
  def create(opts) do
    dim = Keyword.fetch!(opts, :dim)
    ExTorch.NN.softmax(dim)
  end
end

defmodule ExTorch.NN.LeakyReLU do
  @moduledoc "LeakyReLU activation."
  @behaviour ExTorch.NN.Module
  @impl true
  def create(opts \\ []), do: ExTorch.NN.leaky_relu(opts)
end

defmodule ExTorch.NN.ELU do
  @moduledoc "ELU activation."
  @behaviour ExTorch.NN.Module
  @impl true
  def create(opts \\ []), do: ExTorch.NN.elu(opts)
end

defmodule ExTorch.NN.SiLU do
  @moduledoc "SiLU (Swish) activation."
  @behaviour ExTorch.NN.Module
  @impl true
  def create(_opts \\ []), do: ExTorch.NN.silu()
end

defmodule ExTorch.NN.Mish do
  @moduledoc "Mish activation."
  @behaviour ExTorch.NN.Module
  @impl true
  def create(_opts \\ []), do: ExTorch.NN.mish()
end

defmodule ExTorch.NN.PReLU do
  @moduledoc "PReLU activation."
  @behaviour ExTorch.NN.Module
  @impl true
  def create(opts \\ []), do: ExTorch.NN.prelu(opts)
end

defmodule ExTorch.NN.LogSoftmax do
  @moduledoc "LogSoftmax."
  @behaviour ExTorch.NN.Module
  @impl true
  def create(opts), do: ExTorch.NN.log_softmax(Keyword.fetch!(opts, :dim))
end

defmodule ExTorch.NN.Conv3d do
  @moduledoc "3D convolution layer."
  @behaviour ExTorch.NN.Module
  @impl true
  def create(opts) do
    ExTorch.NN.conv3d(Keyword.fetch!(opts, :in_channels), Keyword.fetch!(opts, :out_channels), Keyword.fetch!(opts, :kernel_size), opts)
  end
end

defmodule ExTorch.NN.ConvTranspose1d do
  @moduledoc "1D transposed convolution."
  @behaviour ExTorch.NN.Module
  @impl true
  def create(opts) do
    ExTorch.NN.conv_transpose1d(Keyword.fetch!(opts, :in_channels), Keyword.fetch!(opts, :out_channels), Keyword.fetch!(opts, :kernel_size), opts)
  end
end

defmodule ExTorch.NN.ConvTranspose2d do
  @moduledoc "2D transposed convolution."
  @behaviour ExTorch.NN.Module
  @impl true
  def create(opts) do
    ExTorch.NN.conv_transpose2d(Keyword.fetch!(opts, :in_channels), Keyword.fetch!(opts, :out_channels), Keyword.fetch!(opts, :kernel_size), opts)
  end
end

defmodule ExTorch.NN.MaxPool1d do
  @moduledoc "1D max pooling."
  @behaviour ExTorch.NN.Module
  @impl true
  def create(opts), do: ExTorch.NN.max_pool1d(Keyword.fetch!(opts, :kernel_size), opts)
end

defmodule ExTorch.NN.MaxPool2d do
  @moduledoc "2D max pooling."
  @behaviour ExTorch.NN.Module
  @impl true
  def create(opts), do: ExTorch.NN.max_pool2d(Keyword.fetch!(opts, :kernel_size), opts)
end

defmodule ExTorch.NN.AvgPool1d do
  @moduledoc "1D average pooling."
  @behaviour ExTorch.NN.Module
  @impl true
  def create(opts), do: ExTorch.NN.avg_pool1d(Keyword.fetch!(opts, :kernel_size), opts)
end

defmodule ExTorch.NN.AvgPool2d do
  @moduledoc "2D average pooling."
  @behaviour ExTorch.NN.Module
  @impl true
  def create(opts), do: ExTorch.NN.avg_pool2d(Keyword.fetch!(opts, :kernel_size), opts)
end

defmodule ExTorch.NN.AdaptiveAvgPool1d do
  @moduledoc "1D adaptive average pooling."
  @behaviour ExTorch.NN.Module
  @impl true
  def create(opts), do: ExTorch.NN.adaptive_avg_pool1d(Keyword.fetch!(opts, :output_size))
end

defmodule ExTorch.NN.AdaptiveAvgPool2d do
  @moduledoc "2D adaptive average pooling."
  @behaviour ExTorch.NN.Module
  @impl true
  def create(opts), do: ExTorch.NN.adaptive_avg_pool2d(Keyword.fetch!(opts, :output_h), Keyword.fetch!(opts, :output_w))
end

defmodule ExTorch.NN.GroupNorm do
  @moduledoc "Group normalization."
  @behaviour ExTorch.NN.Module
  @impl true
  def create(opts), do: ExTorch.NN.group_norm(Keyword.fetch!(opts, :num_groups), Keyword.fetch!(opts, :num_channels), opts)
end

defmodule ExTorch.NN.InstanceNorm1d do
  @moduledoc "1D instance normalization."
  @behaviour ExTorch.NN.Module
  @impl true
  def create(opts), do: ExTorch.NN.instance_norm1d(Keyword.fetch!(opts, :num_features), opts)
end

defmodule ExTorch.NN.InstanceNorm2d do
  @moduledoc "2D instance normalization."
  @behaviour ExTorch.NN.Module
  @impl true
  def create(opts), do: ExTorch.NN.instance_norm2d(Keyword.fetch!(opts, :num_features), opts)
end

defmodule ExTorch.NN.LSTM do
  @moduledoc "LSTM recurrent layer."
  @behaviour ExTorch.NN.Module
  @impl true
  def create(opts), do: ExTorch.NN.lstm(Keyword.fetch!(opts, :input_size), Keyword.fetch!(opts, :hidden_size), opts)
end

defmodule ExTorch.NN.GRU do
  @moduledoc "GRU recurrent layer."
  @behaviour ExTorch.NN.Module
  @impl true
  def create(opts), do: ExTorch.NN.gru(Keyword.fetch!(opts, :input_size), Keyword.fetch!(opts, :hidden_size), opts)
end

defmodule ExTorch.NN.MultiheadAttention do
  @moduledoc "Multi-head attention."
  @behaviour ExTorch.NN.Module
  @impl true
  def create(opts), do: ExTorch.NN.multihead_attention(Keyword.fetch!(opts, :embed_dim), Keyword.fetch!(opts, :num_heads), opts)
end

defmodule ExTorch.NN.Flatten do
  @moduledoc "Flatten layer."
  @behaviour ExTorch.NN.Module
  @impl true
  def create(opts \\ []), do: ExTorch.NN.flatten(opts)
end

defmodule ExTorch.NN.Unflatten do
  @moduledoc "Unflatten layer."
  @behaviour ExTorch.NN.Module
  @impl true
  def create(opts), do: ExTorch.NN.unflatten(Keyword.fetch!(opts, :dim), Keyword.fetch!(opts, :sizes))
end
