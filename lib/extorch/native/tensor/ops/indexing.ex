defmodule ExTorch.Native.Tensor.Ops.Indexing do
  @moduledoc false

  use ExTorch.Native.BindingDeclaration
  import ExTorch.Scalar, only: [is_scalar: 1]

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

    ## Optional arguments
    - `inplace` (`boolean()`) - If `true`, then the values will be replaced on the original `tensor` argument. Else,
    it will return a copy of `tensor` with the values replaced. Default: `false`

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
            ExTorch.Tensor.t() | ExTorch.Scalar.scalar_or_list(),
            boolean()
          ) :: ExTorch.Tensor.t()
    defbinding(index_put(tensor, indices, value, inplace \\ false),
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

    @doc """
    Accumulate the elements of `alpha` times `source` into the `input` tensor by adding
    to the indices in the order given in `index`.

    For example, if `dim == 0`, `index[i] == j`, and `alpha=-1`, then the `i`th row of `source` is
    subtracted from the `j`th row of `input`.

    The `dim`-th dimension of `source` must have the same size as the length of `index`
    (which must be a 1D tensor), and all other dimensions must match `input`, or an error
    will be raised.

    For a 3-D tensor the output is given as:

    ```
    out[index[i], :, :] = input[index[i], :, :] + alpha * src[i, :, :]  # if dim == 0
    out[:, index[i], :] = input[:, index[i], :] + alpha * src[:, i, :]  # if dim == 1
    out[:, :, index[i]] = input[:, :, index[i]] + alpha * src[:, :, i]  # if dim == 2
    ```

    ## Arguments
    - `input` (`ExTorch.Tensor`) - input tensor.
    - `dim` (`integer()`) - dimension along which to index.
    - `index` (`ExTorch.Tensor`) -  indices of `input` to select from, its dtype must be `:long`.
    - `source` (`ExTorch.Tensor`) - the tensor containing values to add.

    ## Optional arguments
    - `alpha` (`ExTorch.Scalar`) - the scalar multiplier for `source`. Default: 1
    - `out` (`ExTorch.Tensor | nil`) - an optional pre-allocated tensor used to
    store the output result. Default: `nil`
    - `inplace` (`boolean`) - if `true`, then the operation will be written to the
    `input` tensor (the `out` argument will be ignored). Else, it returns a new tensor or
    writes to the `out` argument (if not `nil`)

    ## Examples
        iex> x = ExTorch.ones({5, 3})
        iex> t = ExTorch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype: :float)
        iex> index = ExTorch.tensor([0, 4, 2], dtype: :long)
        iex>  ExTorch.index_add(x, 0, index, t)
        #Tensor<
        [[ 2.,  3.,  4.],
         [ 1.,  1.,  1.],
         [ 8.,  9., 10.],
         [ 1.,  1.,  1.],
         [ 5.,  6.,  7.]]
        [size: {5, 3}, dtype: :float, device: :cpu, requires_grad: false]>

    """
    @spec index_add(
            ExTorch.Tensor.t(),
            integer(),
            ExTorch.Tensor.t(),
            ExTorch.Tensor.t(),
            ExTorch.Scalar.t(),
            ExTorch.Tensor.t() | nil,
            boolean()
          ) :: ExTorch.Tensor.t()
    defbinding(index_add(input, dim, index, source, alpha \\ 1, out \\ nil, inplace \\ false))

    @doc """
    Copies the elements of `source` into the `input` tensor by selecting the indices
    in the order given in `index`.

    For example, if `dim == 0` and `index[i] == j`, then the `i`th row of `source` is
    copied to the jth row of `input`.

    The `dim`th dimension of `source` must have the same size as the length of `index`
    (which must be a 1D tensor), and all other dimensions must match `input`, or an error will be raised.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - input tensor.
    - `dim` (`integer()`) - dimension along which to index.
    - `index` (`ExTorch.Tensor`) -  indices of `input` to select from, its dtype must be `:long`.
    - `source` (`ExTorch.Tensor`) - the tensor containing values to copy.

    ## Optional arguments
    - `out` (`ExTorch.Tensor | nil`) - an optional pre-allocated tensor used to
    store the output result. Default: `nil`
    - `inplace` (boolean) - if `true`, the `input` tensor will be modified and be the object of
    the modifications done by this function. Else it will return a new tensor with the changes, or
    it will apply them to `out`. Default: `false`

    ## Notes
    If `index` contains duplicate entries, multiple elements from `source` will be copied to the same
    index of `self`. The result is nondeterministic since it depends on which copy occurs last.

    ## Examples
        iex> x = ExTorch.zeros({5, 3})
        iex> t = ExTorch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype: :float)
        iex> ExTorch.index_copy(input, 0, index, t)
        #Tensor<
        [[1.0000, 2.0000, 3.0000],
         [0.0000, 0.0000, 0.0000],
         [7.0000, 8.0000, 9.0000],
         [0.0000, 0.0000, 0.0000],
         [4.0000, 5.0000, 6.0000]]
        [size: {5, 3}, dtype: :float, device: :cpu, requires_grad: false]>

    """
    @spec index_copy(
            ExTorch.Tensor.t(),
            integer(),
            ExTorch.Tensor.t(),
            ExTorch.Tensor.t(),
            ExTorch.Tensor.t() | nil,
            boolean()
          ) ::
            ExTorch.Tensor.t()
    defbinding(index_copy(input, dim, index, source, out \\ nil, inplace \\ false))

    @doc """
    Accumulate the elements of source into the `self` tensor by accumulating to the indices in
    the order given in `index` using the reduction given by the `reduce` argument.

    For example, if `dim == 0`, `index[i] == j`, `reduce == :prod` and `include_self == true` then
    the `i`th row of `source` is multiplied by the `j`th row of self. If `include_self = true`,
    the values in the `self` tensor are included in the reduction, otherwise, rows in the
    `self` tensor that are accumulated to are treated as if they were filled with the
    reduction identites.

    The `dim`th dimension of `source` must have the same size as the length of `index`
    (which must be a 1D tensor), and all other dimensions must match `self`, or an error
    will be raised.

    For a 3-D tensor with `reduce = :prod` and `include_self = true` the output is given as:

    ```
    self[index[i], :, :] *= src[i, :, :]  # if dim == 0
    self[:, index[i], :] *= src[:, i, :]  # if dim == 1
    self[:, :, index[i]] *= src[:, :, i]  # if dim == 2
    ```

    ## Arguments
    - `self` (`ExTorch.Tensor`) - input tensor.
    - `dim` (`integer()`) - dimension along which to index.
    - `index` (`ExTorch.Tensor`) -  indices of `source` to select from, its dtype must be `:long`.
    - `source` (`ExTorch.Tensor`) - the tensor containing values to copy.
    - `reduce` (`:prod | :mean | :amax | :amin`) - the reduction operation to apply.

    ## Optional arguments
    - `include_self` (`boolean`) - whether the elements from the `self` tensor are included
    in the reduction. Default: `true`
    - `out` (`ExTorch.Tensor | nil`) - an optional pre-allocated tensor used to
    store the output result. Default: `nil`
    - `inplace` (boolean) - if `true`, the `self` tensor will be modified and be the object of
    the modifications done by this function. Else it will return a new tensor with the changes, or
    it will apply them to `out`. Default: `false`

    ## Notes
    * This operation may behave nondeterministically when given tensors on a CUDA device.
    See [Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html) for more information.
    * This function only supports floating point tensors.
    * This function is in beta and may change in the near future.

    ## Examples
        iex> x = ExTorch.full({5, 3}, 2)
        iex> t = ExTorch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype: :float)
        iex> index = ExTorch.tensor([0, 4, 2, 0], dtype: :long)
        #Tensor<
        [0, 4, 2, 0]
        [size: {4}, dtype: :long, device: :cpu, requires_grad: false]>

        iex> ExTorch.index_reduce(x, 0, index, t, :prod)
        #Tensor<
        [[20., 44., 72.],
         [ 2.,  2.,  2.],
         [14., 16., 18.],
         [ 2.,  2.,  2.],
         [ 8., 10., 12.]]
        [size: {5, 3}, dtype: :float, device: :cpu, requires_grad: false]>

        iex> ExTorch.index_reduce(x, 0, index, t, :prod, include_self: false)
        #Tensor<
        [[10., 22., 36.],
         [ 2.,  2.,  2.],
         [ 7.,  8.,  9.],
         [ 2.,  2.,  2.],
         [ 4.,  5.,  6.]]
        [size: {5, 3}, dtype: :float, device: :cpu, requires_grad: false]>

    """
    @spec index_reduce(
            ExTorch.Tensor.t(),
            integer(),
            ExTorch.Tensor.t(),
            ExTorch.Tensor.t(),
            :prod | :mean | :amax | :amin,
            boolean(),
            ExTorch.Tensor.t() | nil,
            boolean()
          ) :: ExTorch.Tensor.t()
    defbinding(
      index_reduce(
        self,
        dim,
        index,
        source,
        reduce,
        include_self \\ true,
        out \\ nil,
        inplace \\ false
      )
    )

    @doc """
    Returns a new tensor which indexes the `input` tensor along dimension `dim`
    using the entries in `index` (whose dtype is `:long`).

    The returned tensor has the same number of dimensions as the original tensor
    (`input`). The `dim`th dimension has the same size as the length of `index`;
    other dimensions have the same size as in the original tensor.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - input tensor.
    - `dim` (`integer()`) - dimension along which to index.
    - `index` (`ExTorch.Tensor`) -  indices of `input` to select from, its dtype must be `:long`.

    ## Optional arguments
    - `out` (`ExTorch.Tensor | nil`) - an optional pre-allocated tensor used to
    store the output result. Default: `nil`

    ## Notes
    * The returned tensor does not use the same storage as the original tensor.
    * If `out` has a different shape than expected, we silently change it to the
    correct shape, reallocating the underlying storage if necessary.

    ## Examples
        iex> x = ExTorch.randn({3, 4})
        #Tensor<
        [[ 2.3564,  1.1268, -0.3407, -0.0561],
         [ 0.6479, -2.3011, -1.6695,  0.5547],
         [ 1.3554,  3.6460,  2.5569, -0.1892]]
        [size: {3, 4}, dtype: :float, device: :cpu, requires_grad: false]>

        iex> indices = ExTorch.tensor([0, 2], dtype: :long)

        iex> ExTorch.index_select(x, 0, indices)
        #Tensor<
        [[ 2.3564,  1.1268, -0.3407, -0.0561],
         [ 1.3554,  3.6460,  2.5569, -0.1892]]
        [size: {2, 4}, dtype: :float, device: :cpu, requires_grad: false]>

        iex> ExTorch.index_select(x, 1, indices)
        #Tensor<
        [[ 2.3564, -0.3407],
         [ 0.6479, -1.6695],
         [ 1.3554,  2.5569]]
        [size: {3, 2}, dtype: :float, device: :cpu, requires_grad: false]>
    """
    @spec index_select(
            ExTorch.Tensor.t(),
            integer(),
            ExTorch.Tensor.t(),
            ExTorch.Tensor.t() | nil
          ) :: ExTorch.Tensor.t()
    defbinding(index_select(input, dim, index, out \\ nil))

    @doc """
    Returns a new 1-D tensor which indexes the `input` tensor according to the
    boolean mask `mask` which has dtype `:bool`.

    The shapes of the `mask` tensor and the `input` tensor don’t need to match,
    but they must be broadcastable.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - input tensor.
    - `mask` (`ExTorch.Tensor`) - the tensor containing the binary mask to index with. It's dtype must be `:bool`

    ## Optional arguments
    - `out` (`ExTorch.Tensor | nil`) - an optional pre-allocated tensor used to
    store the output result. Default: `nil`

    ## Notes
    The returned tensor does **not** use the same storage as the original tensor.

    ## Examples
        iex> x = ExTorch.randn({4, 5})
        #Tensor<
        [[ 1.6055, -0.1662, -0.6764, -0.8615, -2.1960],
         [ 0.8188, -1.1111, -0.2659,  1.4720, -0.0226],
         [-0.7065, -1.0628, -0.7172, -1.0006,  0.3091],
         [-0.8901, -0.6624, -0.4590,  0.0821, -0.9716]]
        [size: {4, 5}, dtype: :float, device: :cpu, requires_grad: false]>
        iex> mask = ExTorch.ge(x, 0)
        #Tensor<
        [[ true, false, false, false, false],
         [ true, false, false,  true, false],
         [false, false, false, false,  true],
         [false, false, false,  true, false]]
        [size: {4, 5}, dtype: :bool, device: :cpu, requires_grad: false]>

        iex> ExTorch.masked_select(x, mask)
        #Tensor<
        [1.6055, 0.8188, 1.4720, 0.3091, 0.0821]
        [size: {5}, dtype: :float, device: :cpu, requires_grad: false]>
    """
    @spec masked_select(ExTorch.Tensor.t(), ExTorch.Tensor.t(), ExTorch.Tensor.t() | nil) ::
            ExTorch.Tensor.t()
    defbinding(masked_select(input, mask, out \\ nil))

    @doc """
    Slices the `input` tensor along the selected dimension at the given `index`.
    This function returns a view of the original tensor with the given dimension removed.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.
    - `dim` (`integer`) - the dimension to slice.
    - `index` (`integer`) - the index to select.

    ## Notes
    `ExTorch.select/3` is equivalent to slicing. For example, `ExTorch.select(0, index)`
    is equivalent to `tensor[index]` and `ExTorch.select(2, index)` is equivalent to
    `tensor[{:::, :::, index}]`.

    ## Examples
        iex> a = ExTorch.arange(2 * 3 * 4) |> ExTorch.reshape({2, 3, 4})
        #Tensor<
        [[[ 0.0000,  1.0000,  2.0000,  3.0000],
          [ 4.0000,  5.0000,  6.0000,  7.0000],
          [ 8.0000,  9.0000, 10.0000, 11.0000]],

         [[12.0000, 13.0000, 14.0000, 15.0000],
          [16.0000, 17.0000, 18.0000, 19.0000],
          [20.0000, 21.0000, 22.0000, 23.0000]]]
        [size: {2, 3, 4}, dtype: :float, device: :cpu, requires_grad: false]>

        iex> ExTorch.select(a, 0, 1)
        #Tensor<
        [[12., 13., 14., 15.],
         [16., 17., 18., 19.],
         [20., 21., 22., 23.]]
        [size: {3, 4}, dtype: :float, device: :cpu, requires_grad: false]>

        iex> ExTorch.select(a, 1, 0)
        #Tensor<
        [[ 0.0000,  1.0000,  2.0000,  3.0000],
         [12.0000, 13.0000, 14.0000, 15.0000]]
        [size: {2, 4}, dtype: :float, device: :cpu, requires_grad: false]>

        iex> ExTorch.select(a, 2, 2)
        #Tensor<
        [[ 2.,  6., 10.],
         [14., 18., 22.]]
        [size: {2, 3}, dtype: :float, device: :cpu, requires_grad: false]>
    """
    @spec select(ExTorch.Tensor.t(), integer(), integer()) :: ExTorch.Tensor.t()
    defbinding(select(input, dim, index))
  end
end
