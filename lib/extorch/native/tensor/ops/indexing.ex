defmodule ExTorch.Native.Tensor.Ops.Indexing do
  @moduledoc false

  use ExTorch.Native.BindingDeclaration

  @doc """
  Create a slice to index a tensor.

  ## Arguments
    - `start`: The starting slice value. Default: nil
    - `stop`: The non-inclusive end of the slice. Default: nil
    - `step`: The step between values. Default: nil

  ## Returns
    - `slice`: An `ExTorch.Index.Slice` struct that represents the slice.

  ## Notes
  An empty slice will represent the "take-all axis", represented by ":" in
  Python.
  """
  @doc kind: :tensor_indexing
  @spec slice(integer() | nil, integer() | nil, integer() | nil) ::
          ExTorch.Index.Slice.t()
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

    %ExTorch.Index.Slice{
      start: act_start,
      stop: act_stop,
      step: act_step,
      mask: mask
    }
  end

  defbindings(:tensor_indexing) do
    @doc """
    Index a tensor given a list of integers, ranges, tensors, nil or
    `:ellipsis`.

    ## Arguments
      - `tensor`: Input tensor (`ExTorch.Tensor`)
      - `indices`: Indices to select (`ExTorch.Index`)

    ## Examples
        iex> a = ExTorch.rand({3, 4, 4})
        #Tensor<
        (1,.,.) =
          0.8974  0.6348  0.4760  0.0726
          0.3809  0.4332  0.9761  0.4656
          0.8544  0.0605  0.1683  0.4142
          0.7736  0.1794  0.2732  0.3165

        (2,.,.) =
          0.1967  0.2013  0.7938  0.8738
          0.0240  0.0098  0.4605  0.3970
          0.9699  0.1057  0.3176  0.2651
          0.7698  0.6383  0.0016  0.7198

        (3,.,.) =
          0.5061  0.0021  0.4804  0.7444
          0.5725  0.2019  0.3524  0.5345
          0.3876  0.3622  0.5318  0.0445
          0.3276  0.2913  0.8069  0.6132
        [ CPUDoubleType{3,4,4} ]
        >

        # Use an integer index
        iex> ExTorch.index(a, 0)
        #Tensor<
        0.8974  0.6348  0.4760  0.0726
        0.3809  0.4332  0.9761  0.4656
        0.8544  0.0605  0.1683  0.4142
        0.7736  0.1794  0.2732  0.3165
        [ CPUDoubleType{4,4} ]
        >

        # Use a slice index
        iex> ExTorch.index(a, 0..2)
        #Tensor<
        (1,.,.) =
          0.8974  0.6348  0.4760  0.0726
          0.3809  0.4332  0.9761  0.4656
          0.8544  0.0605  0.1683  0.4142
          0.7736  0.1794  0.2732  0.3165

        (2,.,.) =
          0.1967  0.2013  0.7938  0.8738
          0.0240  0.0098  0.4605  0.3970
          0.9699  0.1057  0.3176  0.2651
          0.7698  0.6383  0.0016  0.7198
        [ CPUDoubleType{2,4,4} ]
        >

        iex> ExTorch.index(a, ExTorch.slice(0, 1))
        #Tensor<
        (1,.,.) =
          0.8974  0.6348  0.4760  0.0726
          0.3809  0.4332  0.9761  0.4656
          0.8544  0.0605  0.1683  0.4142
          0.7736  0.1794  0.2732  0.3165
        [ CPUDoubleType{1,4,4} ]
        >

        # Index multiple dimensions
        iex> ExTorch.index(a, [:::, ExTorch.slice(0, 2), 0])
        #Tensor<
        0.8974  0.3809
        0.1967  0.0240
        0.5061  0.5725
        [ CPUDoubleType{3,2} ]
        >

    ## Notes
    For more information regarding the kind of accepted indices and their corresponding
    behaviour, please see the `ExTorch.Index` documentation
    """
    @spec index(
            ExTorch.Tensor.t(),
            ExTorch.Index.t()
          ) :: ExTorch.Tensor.t()
    defbinding(index(tensor, indices), indices: ExTorch.Utils.Indices.parse_indices(indices))

    @doc """
    Assign a tensor or value in another tensor given an index or set of indices.

    ## Arguments
    - `tensor` - Input tensor (`ExTorch.Tensor`)
    - `index` - Indices to replace (`ExTorch.Index`)
    - `value` - The value to assign into the tensor (`ExTorch.Tensor` | `number()` | `list()` | `tuple()` | `number()`)
    """
    @spec index_assign(
            ExTorch.Tensor.t(),
            ExTorch.Index.t(),
            ExTorch.Tensor.t() | number() | list() | tuple() | boolean()
          ) :: ExTorch.Tensor.t()
    defbinding(index_assign(tensor, indices, value),
      indices: ExTorch.Utils.Indices.parse_indices(indices),
      value: ExTorch.Utils.to_list_wrapper(value)
    )
  end
end
