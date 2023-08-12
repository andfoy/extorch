defmodule ExTorch.Utils.Indices do
  @spec parse_indices(ExTorch.Index.t()) :: [ExTorch.Index.actual_index()]
  @doc false
  def parse_indices(indices) when is_tuple(indices) do
    parse_indices(Tuple.to_list(indices), [])
  end

  def parse_indices(indices) when is_list(indices) do
    parse_indices(indices, [])
  end

  def parse_indices(x) do
    parse_indices([x], [])
  end

  @spec parse_indices(
          [ExTorch.Index.index()],
          [ExTorch.Index.actual_index()]
        ) :: [ExTorch.Index.actual_index()]
  def parse_indices([], acc) do
    Enum.reverse(acc)
  end

  def parse_indices([index | indices], acc) do
    parse_indices(indices, [parse_index(index) | acc])
  end

  @spec parse_index(ExTorch.Index.index()) :: ExTorch.Index.actual_index()
  def parse_index(index) when is_number(index) do
    index
  end

  def parse_index(index) when is_list(index) do
    ExTorch.tensor(index, dtype: :int64)
  end

  def parse_index(index) when is_tuple(index) do
    ExTorch.tensor(Tuple.to_list(index), dtype: :int64)
  end

  def parse_index(start..stop//step) do
    ExTorch.slice(start, stop, step)
  end

  def parse_index(:"::") do
    ExTorch.slice()
  end

  def parse_index(:ellipsis) do
    :ellipsis
  end

  def parse_index(:...) do
    :ellipsis
  end

  def parse_index(nil) do
    nil
  end

  def parse_index(x) when is_boolean(x) do
    x
  end

  def parse_index(%ExTorch.Tensor{} = tensor) do
    tensor
  end

  def parse_index(%ExTorch.Utils.Indices.Slice{} = slice) do
    slice
  end
end
