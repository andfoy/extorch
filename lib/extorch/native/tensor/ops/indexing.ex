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
        iex> a = ExTorch.arange(3 * 4 * 4) |> ExTorch.reshape({3, 4, 4})
        #Tensor<
        [[[ 0.,  1.,  2.,  3.],
          [ 4.,  5.,  6.,  7.],
          [ 8.,  9., 10., 11.],
          [12., 13., 14., 15.]],

         [[16., 17., 18., 19.],
          [20., 21., 22., 23.],
          [24., 25., 26., 27.],
          [28., 29., 30., 31.]],

         [[32., 33., 34., 35.],
          [36., 37., 38., 39.],
          [40., 41., 42., 43.],
          [44., 45., 46., 47.]]]
        [
          size: {3, 4, 4},
          dtype: :double,
          device: :cpu,
          requires_grad: false
        ]>

        # Use an integer index
        iex> ExTorch.index(a, 0)
        #Tensor<
        [[ 0.,  1.,  2.,  3.],
         [ 4.,  5.,  6.,  7.],
         [ 8.,  9., 10., 11.],
         [12., 13., 14., 15.]]
        [
          size: {4, 4},
          dtype: :double,
          device: :cpu,
          requires_grad: false
        ]>

        # Use a slice index
        iex> ExTorch.index(a, 0..2)
        #Tensor<
        [[[ 0.,  1.,  2.,  3.],
          [ 4.,  5.,  6.,  7.],
          [ 8.,  9., 10., 11.],
          [12., 13., 14., 15.]],

         [[16., 17., 18., 19.],
          [20., 21., 22., 23.],
          [24., 25., 26., 27.],
          [28., 29., 30., 31.]]]
        [
          size: {2, 4, 4},
          dtype: :double,
          device: :cpu,
          requires_grad: false
        ]>

        iex> ExTorch.index(a, ExTorch.slice(0, 1))
        #Tensor<
        [[[ 0.,  1.,  2.,  3.],
          [ 4.,  5.,  6.,  7.],
          [ 8.,  9., 10., 11.],
          [12., 13., 14., 15.]]]
        [
          size: {1, 4, 4},
          dtype: :double,
          device: :cpu,
          requires_grad: false
        ]>

        # Index multiple dimensions
        iex> ExTorch.index(a, [:::, ExTorch.slice(0, 2), 0])
        #Tensor<
        [[ 0.,  4.],
         [16., 20.],
         [32., 36.]]
        [
          size: {3, 2},
          dtype: :double,
          device: :cpu,
          requires_grad: false
        ]>

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
    Assign a value into a tensor given a single or a sequence of indices.

    ## Arguments
    - `tensor` - Input tensor (`ExTorch.Tensor`)
    - `index` - Indices to replace (`ExTorch.Index`)
    - `value` - The value to assign into the tensor (`ExTorch.Tensor` | `number()` | `list()` | `tuple()` | `number()`)

    ## Examples
        # Assign a particular value
        iex> x = ExTorch.zeros({2, 3, 3})
        #Tensor<
        [[[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]],

         [[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]]]
        [
          size: {2, 3, 3},
          dtype: :double,
          device: :cpu,
          requires_grad: false
        ]>

        iex> x = ExTorch.index_put(x, 0, -1)
        #Tensor<
        [[[-1., -1., -1.],
          [-1., -1., -1.],
          [-1., -1., -1.]],

         [[ 0.,  0.,  0.],
          [ 0.,  0.,  0.],
          [ 0.,  0.,  0.]]]
        [
          size: {2, 3, 3},
          dtype: :double,
          device: :cpu,
          requires_grad: false
        ]>

        # Assign a value into an slice
        iex> x = ExTorch.index_put(x, [0, ExTorch.slice(1), ExTorch.slice(1)], 0.3)
        #Tensor<
        [[[-1.0000, -1.0000, -1.0000],
          [-1.0000,  0.3000,  0.3000],
          [-1.0000,  0.3000,  0.3000]],

         [[ 0.0000,  0.0000,  0.0000],
          [ 0.0000,  0.0000,  0.0000],
          [ 0.0000,  0.0000,  0.0000]]]
        [
          size: {2, 3, 3},
          dtype: :double,
          device: :cpu,
          requires_grad: false
        ]>

        # Assign a tensor into an index
        iex> value = ExTorch.eye(3)
        iex> x = ExTorch.index_put(x, 1, value)
        #Tensor<
        [[[-1.0000, -1.0000, -1.0000],
          [-1.0000,  0.3000,  0.3000],
          [-1.0000,  0.3000,  0.3000]],

         [[ 1.0000,  0.0000,  0.0000],
          [ 0.0000,  1.0000,  0.0000],
          [ 0.0000,  0.0000,  1.0000]]]
        [
          size: {2, 3, 3},
          dtype: :double,
          device: :cpu,
          requires_grad: false
        ]>

        # Assign a list of numbers into an index (broadcastable)
        iex> x = ExTorch.index_put(x, [:::, 1], [1, 2, 3])
        #Tensor<
        [[[-1.0000, -1.0000, -1.0000],
          [ 1.0000,  2.0000,  3.0000],
          [-1.0000,  0.3000,  0.3000]],

         [[ 1.0000,  0.0000,  0.0000],
          [ 1.0000,  2.0000,  3.0000],
          [ 0.0000,  0.0000,  1.0000]]]
        [
          size: {2, 3, 3},
          dtype: :double,
          device: :cpu,
          requires_grad: false
        ]>

    """
    @spec index_put(
            ExTorch.Tensor.t(),
            ExTorch.Index.t(),
            ExTorch.Tensor.t() | ExTorch.Complex.t() | number() | list() | tuple() | boolean()
          ) :: ExTorch.Tensor.t()
    defbinding(index_put(tensor, indices, value),
      indices: ExTorch.Utils.Indices.parse_indices(indices),
      value:
        case value do
          %ExTorch.Tensor{} ->
            value

          _ ->
            ExTorch.tensor(value,
              device: ExTorch.Tensor.device(tensor),
              requires_grad: false
            )
        end
    )
  end
end
