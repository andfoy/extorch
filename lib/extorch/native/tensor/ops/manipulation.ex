defmodule ExTorch.Native.Tensor.Ops.Manipulation do
  @moduledoc false

  use ExTorch.Native.BindingDeclaration

  @doc """
  Create a slice to index a tensor.

  ## Arguments
    - `start`: The starting slice value. Default: nil
    - `stop`: The non-inclusive end of the slice. Default: nil
    - `step`: The step between values. Default: nil

  ## Returns
    - `slice`: An `ExTorch.Utils.Indices.Slice` struct that represents the slice.

  ## Notes
  An empty slice will represent the "take-all axis", represented by ":" in
  Python.
  """
  @doc kind: :tensor_indexing
  @spec slice(integer() | nil, integer() | nil, integer() | nil) ::
          ExTorch.Utils.Indices.Slice.t()
  def slice(start \\ nil, stop \\ nil, step \\ nil) do
    bases = [1, 2, 4]

    {mask, [act_step, act_stop, act_start]} =
      bases
      |> Enum.zip([start, stop, step])
      |> Enum.reduce({0, []}, fn {base, value}, {mask, values} ->
        case value do
          nil -> {mask, [0 | values]}
          _ -> {Bitwise.|||(mask, base), [value | values]}
        end
      end)

    %ExTorch.Utils.Indices.Slice{
      start: act_start,
      stop: act_stop,
      step: act_step,
      mask: mask
    }
  end

  defbindings(:tensor_manipulation) do
    @doc """
    Append an empty dimension to a tensor on a given dimension.

    ## Arguments
      - `tensor`: Input tensor (`ExTorch.Tensor`)
      - `dim`: Dimension (`integer()`)
    """
    @spec unsqueeze(
            ExTorch.Tensor.t(),
            integer()
          ) :: ExTorch.Tensor.t()
    defbinding(unsqueeze(tensor, dim))

    end

    defbindings(:tensor_indexing) do
    @doc """
    Index a tensor given a list of integers, ranges, tensors, nil or
    `:ellipsis`.

    ## Arguments
      - `tensor`: Input tensor (`ExTorch.Tensor`)
      - `indices`: Indices to select (`[nil | Range.t() | ExTorch.Tensor.t() | integer() | :ellipsis | ::: | :...]`)
    """
    @spec index(
            ExTorch.Tensor.t(),
            ExTorch.Index.t()
          ) :: ExTorch.Tensor.t()
    defbinding(index(tensor, indices), indices: ExTorch.Utils.Indices.parse_indices(indices))
  end
end
