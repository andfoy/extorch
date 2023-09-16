defmodule ExTorch.Native.Tensor.Ops.Comparison do
  @moduledoc false
  use ExTorch.Native.BindingDeclaration

  defbindings(:tensor_comparison) do
    @doc ~S"""
    This function checks if `input` and `other` satisfy the condition:
    $$|\text{input} - \text{other}| \leq \texttt{atol} + \texttt{rtol} \times |\text{other}|$$
    elementwise, for all elements of `input` and `other`.

    ## Arguments
    - `input` - First tensor to compare (`ExTorch.Tensor`)
    - `other` - Second tensor to compare (`ExTorch.Tensor`)

    ## Optional arguments
    - `rtol` - Relative tolerance (`float`). Default: 1.0e-5
    - `atol` - Absolute tolerance (`float`). Default: 1.0e-8
    - `equal_nan` - If `true`, then two `NaN`s will be considered equal. Default: `false`.

    ## Examples
        iex> ExTorch.allclose(ExTorch.tensor([10000.0, 1.0e-07]), ExTorch.tensor([10000.1, 1.0e-08]))
        false
        iex> ExTorch.allclose(ExTorch.tensor([10000.0, 1.0e-08]), ExTorch.tensor([10000.1, 1.0e-09]))
        true
        iex> ExTorch.allclose(ExTorch.tensor([1.0, :nan]), ExTorch.tensor([1.0, :nan]))
        false
        iex> ExTorch.allclose(ExTorch.tensor([1.0, :nan]), ExTorch.tensor([1.0, :nan]), equal_nan: true)
        true
    """
    @spec allclose(ExTorch.Tensor.t(), ExTorch.Tensor.t(), float(), float(), boolean()) ::
            boolean()
    defbinding(allclose(input, other, rtol \\ 1.0e-5, atol \\ 1.0e-8, equal_nan \\ false))

    @doc """
    Returns the indices that sort a tensor along a given dimension in ascending order by value.

    This is the second value returned by `ExTorch.sort/5`. See its documentation for the exact
    semantics of this method. If `stable` is `true` then the sorting routine becomes stable,
    preserving the order of equivalent elements. If `false`, the relative order of values
    which compare equal is not guaranteed. `true` is slower.

    ## Arguments
    - `input` - Input tensor. (`ExTorch.Tensor`)

    ## Optional arguments
    - `dim` - the dimension to sort along ('integer()'). Default: -1
    - `descending` - controls the sorting order (ascending or descending) (`boolean()`). Default: `false`
    - `stable` - controls the relative order of equivalent elements (`boolean()`). Default: `false`

    ## Examples
        iex> a = ExTorch.randn({4, 4})
        #Tensor<
        [[-1.2732,  0.8419, -0.0140,  0.4717],
         [-1.1627, -0.2813, -0.5655, -0.1348],
         [ 1.5269, -0.2712,  0.5134, -1.5580],
         [ 0.6169, -1.0332,  0.4478, -0.9864]]
        [
          size: {4, 4},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Sort alongside an specific dimension
        iex> ExTorch.argsort(a, dim: 1)
        #Tensor<
        [[0, 2, 3, 1],
         [0, 2, 1, 3],
         [3, 1, 2, 0],
         [1, 3, 2, 0]]
        [
          size: {4, 4},
          dtype: :long,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec argsort(ExTorch.Tensor.t(), integer(), boolean(), boolean()) :: ExTorch.Tensor.t()
    defbinding(argsort(input, dim \\ -1, descending \\ false, stable \\ false))

    @doc """
    Sorts the elements of the input tensor along a given dimension in ascending order by value.

    * If `dim` is not given, the last dimension of the `input` is chosen.
    * If `descending` is `true` then the elements are sorted in descending order by value.
    * If `stable` is `true` then the sorting routine becomes stable, preserving
    the order of equivalent elements.

    A tuple of `{values, indices}` is returned, where the `values` are the sorted values
    and `indices` are the indices of the elements in the original input tensor.

    ## Arguments
    - `input` - Input tensor. (`ExTorch.Tensor`)

    ## Optional arguments
    - `dim` - the dimension to sort along. ('integer()'). Default: -1
    - `descending` - controls the sorting order. (ascending or descending) (`boolean()`). Default: `false`
    - `stable` - controls the relative order of equivalent elements. (`boolean()`). Default: `false`
    - `out` - the output tuple of `{values, indices}` that can be optionally given as
    output buffers. (`{ExTorch.Tensor, ExTorch.Tensor}`). Default: `nil`

    ## Examples
        iex> a = ExTorch.randn({3, 4})
        #Tensor<
        [[ 0.7517,  0.5590, -0.1417, -0.1662],
         [-0.1247,  0.5669,  0.0484,  0.4289],
         [ 0.0876, -0.5951, -1.0296,  0.0093]]
        [
          size: {3, 4},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        # Sort tensor on the last dimension
        iex> {values, indices} = ExTorch.sort(a)
        iex> values
        #Tensor<
        [[-0.1662, -0.1417,  0.5590,  0.7517],
         [-0.1247,  0.0484,  0.4289,  0.5669],
         [-1.0296, -0.5951,  0.0093,  0.0876]]
        [
          size: {3, 4},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>
        iex> indices
        #Tensor<
        [[3, 2, 1, 0],
         [0, 2, 3, 1],
         [2, 1, 3, 0]]
        [
          size: {3, 4},
          dtype: :long,
          device: :cpu,
          requires_grad: false
        ]>

        # Sort tensor on the first dimension while reusing values and indices
        iex> ExTorch.sort(a, 0, out: {values, indices})
        iex> values
        #Tensor<
        [[-0.1247, -0.5951, -1.0296, -0.1662],
         [ 0.0876,  0.5590, -0.1417,  0.0093],
         [ 0.7517,  0.5669,  0.0484,  0.4289]]
        [
          size: {3, 4},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>
        iex> indices
        #Tensor<
        [[1, 2, 2, 0],
         [2, 0, 0, 2],
         [0, 1, 1, 1]]
        [
          size: {3, 4},
          dtype: :long,
          device: :cpu,
          requires_grad: false
        ]>

    """
    @spec sort(
            ExTorch.Tensor.t(),
            integer(),
            boolean(),
            boolean(),
            {ExTorch.Tensor.t(), ExTorch.Tensor.t()} | nil
          ) :: {ExTorch.Tensor.t(), ExTorch.Tensor.t()}
    defbinding(sort(input, dim \\ -1, descending \\ false, stable \\ false, out \\ nil))

    @doc """
    Computes element-wise equality.

    The second argument can be a number or a tensor whose shape is broadcastable with the first argument.
    It will return a boolean tensor of the same shape as `input`, where a `true` entry
    represents a value that is equal on both `input` and `other`, and `false` otherwise.

    ## Arguments
    - `input` - the tensor to compare (`ExTorch.Tensor`).
    - `other` - the tensor or value to compare (`ExTorch.Tensor` or value)

    ## Optional arguments
    - `out` - an optional pre-allocated tensor used to store the comparison result. (`ExTorch.Tensor`)

    ## Examples
        # Compare against an scalar value.
        iex> a = ExTorch.tensor([[1, 2], [3, 4]])
        iex> ExTorch.eq(a, 1)
        #Tensor<
        [[ true, false],
         [false, false]]
        [
          size: {2, 2},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>

        # Compare against a broadcastable value.
        iex> ExTorch.eq(a, [1, 2])
        #Tensor<
        [[ true,  true],
         [false, false]]
        [
          size: {2, 2},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>

        # Compare against another tensor.
        iex> ExTorch.eq(a, ExTorch.tensor([[1, 1], [4, 4]]))
        #Tensor<
        [[ true, false],
         [false,  true]]
        [
          size: {2, 2},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>

    """
    @spec eq(
            ExTorch.Tensor.t(),
            ExTorch.Tensor.t() | ExTorch.Scalar.scalar_or_list(),
            ExTorch.Tensor.t() | nil
          ) :: ExTorch.Tensor.t()
    defbinding(eq(input, other, out \\ nil),
      other:
        case other do
          %ExTorch.Tensor{} ->
            other

          _ ->
            ExTorch.tensor(other,
              device: ExTorch.Tensor.device(input),
              requires_grad: false
            )
        end
    )

    @doc """
    Strict element-wise equality for two tensors.

    This function will return `true` if both inputs have the same size and elements.
    `false`, otherwise.

    ## Arguments
      - `input` - tensor to compare (`ExTorch.Tensor`)
      - `other` - tensor to compare (`ExTorch.Tensor`)

    ## Examples
        iex> ExTorch.equal(ExTorch.tensor([1, 2]), ExTorch.tensor([1, 2]))
        true

        iex> ExTorch.equal(ExTorch.tensor([1, 2]), ExTorch.tensor([1]))
        false
    """
    @spec equal(ExTorch.Tensor.t(), ExTorch.Tensor.t()) :: boolean()
    defbinding(equal(input, other))

    @doc """
    Computes `input >= other` element-wise.

    The second argument can be a number or a tensor whose shape is broadcastable with the first argument.
    It will return a boolean tensor of the same shape as `input`, where a `true` entry
    represents a value that is equal on both `input` and `other`, and `false` otherwise.

    ## Arguments
    - `input` - the tensor to compare (`ExTorch.Tensor`).
    - `other` - the tensor or value to compare (`ExTorch.Tensor` or value)

    ## Optional arguments
    - `out` - an optional pre-allocated tensor used to store the comparison result. (`ExTorch.Tensor`)

    ## Examples
        # Compare against an scalar value.
        iex> a = ExTorch.tensor([[1, 2], [3, 4]])
        iex> ExTorch.ge(a, 2)
        #Tensor<
        [[false,  true],
         [ true,  true]]
        [
          size: {2, 2},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>

        # Compare against a broadcastable value.
        iex> ExTorch.ge(a, [2, 5])
        #Tensor<
        [[false, false],
         [ true, false]]
        [
          size: {2, 2},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>

        # Compare against another tensor.
        iex> ExTorch.ge(a, ExTorch.tensor([[3, 1], [2, 5]]))
        #Tensor<
        [[false,  true],
         [ true, false]]
        [
          size: {2, 2},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec ge(
            ExTorch.Tensor.t(),
            ExTorch.Tensor.t() | ExTorch.Scalar.scalar_or_list(),
            ExTorch.Tensor.t() | nil
          ) :: ExTorch.Tensor.t()
    defbinding(ge(input, other, out \\ nil),
      other:
        case other do
          %ExTorch.Tensor{} ->
            other

          _ ->
            ExTorch.tensor(other,
              device: ExTorch.Tensor.device(input),
              requires_grad: false
            )
        end,
      fn_aliases: [:greater_equal]
    )

    @doc """
    Computes `input > other` element-wise.

    The second argument can be a number or a tensor whose shape is broadcastable with the first argument.
    It will return a boolean tensor of the same shape as `input`, where a `true` entry
    represents a value that is equal on both `input` and `other`, and `false` otherwise.

    ## Arguments
    - `input` - the tensor to compare (`ExTorch.Tensor`).
    - `other` - the tensor or value to compare (`ExTorch.Tensor` or value)

    ## Optional arguments
    - `out` - an optional pre-allocated tensor used to store the comparison result. (`ExTorch.Tensor`)

    ## Examples
        # Compare against an scalar value.
        iex> a = ExTorch.tensor([[1, 2], [3, 4]])
        iex> ExTorch.gt(a, 2)
        #Tensor<
        [[false, false],
         [ true,  true]]
        [
          size: {2, 2},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>

        # Compare against a broadcastable value.
        iex> ExTorch.gt(a, [0, 5])
        #Tensor<
        [[ true, false],
         [ true, false]]
        [
          size: {2, 2},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>

        # Compare against another tensor.
        iex> ExTorch.gt(a, ExTorch.tensor([[0, 1], [2, 5]]))
        #Tensor<
        [[ true,  true],
         [ true, false]]
        [
          size: {2, 2},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec gt(
            ExTorch.Tensor.t(),
            ExTorch.Tensor.t() | ExTorch.Scalar.scalar_or_list(),
            ExTorch.Tensor.t() | nil
          ) :: ExTorch.Tensor.t()
    defbinding(gt(input, other, out \\ nil),
      other:
        case other do
          %ExTorch.Tensor{} ->
            other

          _ ->
            ExTorch.tensor(other,
              device: ExTorch.Tensor.device(input),
              requires_grad: false
            )
        end,
      fn_aliases: [:greater]
    )

    @doc """
    Computes `input <= other` element-wise.

    The second argument can be a number or a tensor whose shape is broadcastable with the first argument.
    It will return a boolean tensor of the same shape as `input`, where a `true` entry
    represents a value that is equal on both `input` and `other`, and `false` otherwise.

    ## Arguments
    - `input` - the tensor to compare (`ExTorch.Tensor`).
    - `other` - the tensor or value to compare (`ExTorch.Tensor` or value)

    ## Optional arguments
    - `out` - an optional pre-allocated tensor used to store the comparison result. (`ExTorch.Tensor`)

    ## Examples
        # Compare against an scalar value.
        iex> a = ExTorch.tensor([[1, 2], [3, 4]])
        iex> ExTorch.le(a, 2)
        #Tensor<
        [[ true,  true],
         [false, false]]
        [
          size: {2, 2},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>

        # Compare against a broadcastable value.
        iex> ExTorch.le(a, [2, 4])
        #Tensor<
        [[ true,  true],
         [false,  true]]
        [
          size: {2, 2},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>

        # Compare against another tensor.
        iex> ExTorch.le(a, ExTorch.tensor([[3, 1], [2, 5]]))
        #Tensor<
        [[ true, false],
         [false,  true]]
        [
          size: {2, 2},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec le(
            ExTorch.Tensor.t(),
            ExTorch.Tensor.t() | ExTorch.Scalar.scalar_or_list(),
            ExTorch.Tensor.t() | nil
          ) :: ExTorch.Tensor.t()
    defbinding(le(input, other, out \\ nil),
      other:
        case other do
          %ExTorch.Tensor{} ->
            other

          _ ->
            ExTorch.tensor(other,
              device: ExTorch.Tensor.device(input),
              requires_grad: false
            )
        end,
      fn_aliases: [:less_equal]
    )

    @doc """
    Computes `input < other` element-wise.

    The second argument can be a number or a tensor whose shape is broadcastable with the first argument.
    It will return a boolean tensor of the same shape as `input`, where a `true` entry
    represents a value that is equal on both `input` and `other`, and `false` otherwise.

    ## Arguments
    - `input` - the tensor to compare (`ExTorch.Tensor`).
    - `other` - the tensor or value to compare (`ExTorch.Tensor` or value)

    ## Optional arguments
    - `out` - an optional pre-allocated tensor used to store the comparison result. (`ExTorch.Tensor`)

    ## Examples
        # Compare against an scalar value.
        iex> a = ExTorch.tensor([[1, 2], [3, 4]])
        iex> ExTorch.lt(a, 2)
        #Tensor<
        [[ true, false],
         [false, false]]
        [
          size: {2, 2},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>

        # Compare against a broadcastable value.
        iex> ExTorch.lt(a, [1, 5])
        #Tensor<
        [[false,  true],
         [false,  true]]
        [
          size: {2, 2},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>

        # Compare against another tensor.
        iex> ExTorch.lt(a, ExTorch.tensor([[0, 1], [2, 5]]))
        #Tensor<
        [[false, false],
         [false,  true]]
        [
          size: {2, 2},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec lt(
            ExTorch.Tensor.t(),
            ExTorch.Tensor.t() | ExTorch.Scalar.scalar_or_list(),
            ExTorch.Tensor.t() | nil
          ) :: ExTorch.Tensor.t()
    defbinding(lt(input, other, out \\ nil),
      other:
        case other do
          %ExTorch.Tensor{} ->
            other

          _ ->
            ExTorch.tensor(other,
              device: ExTorch.Tensor.device(input),
              requires_grad: false
            )
        end,
      fn_aliases: [:less]
    )

    @doc ~S"""
    Returns a new tensor with boolean elements representing if each element of `input` is
    “close” to the corresponding element of `other`. Closeness is defined as:

    $$|\text{input} - \text{other}| \leq \texttt{atol} + \texttt{rtol} \times |\text{other}|$$

    Where `input` and/or `other` are nonfinite they are close if and only if they are
    equal, with `:nan`s being considered equal to each other when `equal_nan` is `true`.

    ## Arguments
    - `input` - First tensor to compare (`ExTorch.Tensor`)
    - `other` - Second tensor to compare (`ExTorch.Tensor`)

    ## Optional arguments
    - `rtol` - Relative tolerance (`float`). Default: 1.0e-5
    - `atol` - Absolute tolerance (`float`). Default: 1.0e-8
    - `equal_nan` - If `true`, then two `NaN`s will be considered equal. Default: `false`.

    ## Examples
        iex> ExTorch.isclose(ExTorch.tensor([10000.0, 1.0e-07]), ExTorch.tensor([10000.1, 1.0e-08]))
        #Tensor<
        [ true, false]
        [
          size: {2},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>

        iex> ExTorch.isclose(ExTorch.tensor([10000.0, 1.0e-08]), ExTorch.tensor([10000.1, 1.0e-09]))
        #Tensor<
        [true, true]
        [
          size: {2},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>

        iex> ExTorch.isclose(ExTorch.tensor([1.0, :nan]), ExTorch.tensor([1.0, :nan]))
        #Tensor<
        [ true, false]
        [
          size: {2},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>

        iex> ExTorch.isclose(ExTorch.tensor([1.0, :nan]), ExTorch.tensor([1.0, :nan]), equal_nan: true)
        #Tensor<
        [true, true]
        [
          size: {2},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec isclose(ExTorch.Tensor.t(), ExTorch.Tensor.t(), float(), float(), boolean()) ::
            ExTorch.Tensor.t()
    defbinding(isclose(input, other, rtol \\ 1.0e-5, atol \\ 1.0e-8, equal_nan \\ false))

    @doc """
    Returns a new tensor with boolean elements representing if each element is finite or not.

    Real values are finite when they are not NaN (`:nan`), negative infinity (`:ninf`), or infinity (`:inf`).
    `ExTorch.Complex` values are finite when both their real and imaginary parts are finite.

    ## Arguments
    - `input` - the input tensor (`ExTorch.Tensor`)

    ## Examples
        iex> input = ExTorch.tensor([1, :inf, 2, :ninf, :nan])
        iex> ExTorch.isfinite(input)
        #Tensor<
        [ true, false,  true, false, false]
        [
          size: {5},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec isfinite(ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
    defbinding(isfinite(input))

    @doc """
    Tests if each element of `elements` is in `test_elements`.
    Returns a boolean tensor of the same shape as `elements` that is `true` for
    elements in `test_elements` and `false` otherwise.

    ## Arguments
    - `elements` - input elements (`ExTorch.Tensor | ExTorch.Scalar`)
    - `test_elements` - values to compare against for each input element. (`ExTorch.Tensor | ExTorch.Scalar`)

    ## Optional arguments
    - `assume_unique` - If `true`, assumes both `elements` and `test_elements` contain unique
    elements, which can speed up the calculation. Default: `false`. (`boolean`)
    - `invert` -  If `true`, inverts the boolean return tensor, resulting in `true` values
    for `elements` not in `test_elements`. Default: `false`. (`boolean`)

    ## Examples
        # Check if any of the values is 2
        iex> x = ExTorch.tensor([[1, 2], [3, 4]])
        iex> ExTorch.isin(x, 2)
        #Tensor<
        [[false,  true],
         [false, false]]
        [
          size: {2, 2},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>

        # Check if any of the values in x is in [1, 3, 5]
        iex> ExTorch.isin(x, [1, 3, 5])
        #Tensor<
        [[ true, false],
         [ true, false]]
        [
          size: {2, 2},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>

        # Invert result
        iex> ExTorch.isin(x, ExTorch.tensor([[1, 3], [5, 4]]), invert: true)
        #Tensor<
        [[false,  true],
         [false, false]]
        [
          size: {2, 2},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>

    """
    @spec isin(
            ExTorch.Tensor.t() | ExTorch.Scalar.scalar_or_list(),
            ExTorch.Tensor.t() | ExTorch.Scalar.scalar_or_list(),
            boolean(),
            boolean()
          ) :: ExTorch.Tensor.t()
    defbinding(isin(elements, test_elements, assume_unique \\ false, invert \\ false),
      elements: ExTorch.Tensor.scalar_to_tensor(elements),
      test_elements:
        ExTorch.Tensor.scalar_to_tensor(test_elements, ExTorch.Tensor.device(elements))
    )

    @doc """
    Computes `input != other` element-wise.

    The second argument can be a number or a tensor whose shape is broadcastable with the first argument.
    It will return a boolean tensor of the same shape as `input`, where a `true` entry
    represents a value that is equal on both `input` and `other`, and `false` otherwise.

    ## Arguments
    - `input` - the tensor to compare (`ExTorch.Tensor`).
    - `other` - the tensor or value to compare (`ExTorch.Tensor` or value)

    ## Optional arguments
    - `out` - an optional pre-allocated tensor used to store the comparison result. (`ExTorch.Tensor`)

    ## Examples
        # Compare against an scalar value.
        iex> a = ExTorch.tensor([[1, 2], [3, 4]])
        iex> ExTorch.ne(a, 2)
        #Tensor<
        [[ true, false],
         [ true,  true]]
        [
          size: {2, 2},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>

        # Compare against a broadcastable value.
        iex> ExTorch.ne(a, [1, 5])
        #Tensor<
        [[false,  true],
         [true,  true]]
        [
          size: {2, 2},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>

        # Compare against another tensor.
        iex> ExTorch.ne(a, ExTorch.tensor([[0, 2], [2, 5]]))
        #Tensor<
        [[true, false],
         [true,  true]]
        [
          size: {2, 2},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec ne(
            ExTorch.Tensor.t(),
            ExTorch.Tensor.t() | ExTorch.Scalar.scalar_or_list(),
            ExTorch.Tensor.t() | nil
          ) :: ExTorch.Tensor.t()
    defbinding(ne(input, other, out \\ nil),
      other:
        case other do
          %ExTorch.Tensor{} ->
            other

          _ ->
            ExTorch.tensor(other,
              device: ExTorch.Tensor.device(input),
              requires_grad: false
            )
        end,
      fn_aliases: [:not_equal]
    )

    @doc """
    Returns a new tensor with boolean elements representing if each element is
    infinity (both positive and negative).

    `ExTorch.Complex` values are infinity when either of their real or imaginary
    parts are infinite.

    ## Arguments
    - `input` - the input tensor (`ExTorch.Tensor`)

    ## Examples
        iex> input = ExTorch.tensor([1, :inf, 2, :ninf, :nan])
        iex> ExTorch.isinf(input)
        #Tensor<
        [ false, true,  false, true, false]
        [
          size: {5},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec isinf(ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
    defbinding(isinf(input))

    @doc """
    Returns a new tensor with boolean elements representing if each element is
    positive infinity.

    `ExTorch.Complex` values are infinity when either of their real or imaginary
    parts are positive infinite.

    ## Arguments
    - `input` - the input tensor (`ExTorch.Tensor`)

    ## Examples
        iex> input = ExTorch.tensor([1, :inf, 2, :ninf, :nan])
        iex> ExTorch.isposinf(input)
        #Tensor<
        [ false, true,  false, false, false]
        [
          size: {5},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec isposinf(ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
    defbinding(isposinf(input))

    @doc """
    Returns a new tensor with boolean elements representing if each element is
    negative infinity.

    `ExTorch.Complex` values are infinity when either of their real or imaginary
    parts are negative infinite.

    ## Arguments
    - `input` - the input tensor (`ExTorch.Tensor`)

    ## Examples
        iex> input = ExTorch.tensor([1, :inf, 2, :ninf, :nan])
        iex> ExTorch.isneginf(input)
        #Tensor<
        [ false, false, false, true, false]
        [
          size: {5},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec isneginf(ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
    defbinding(isneginf(input))

    @doc """
    Returns a new tensor with boolean elements representing if each element is
    `:nan`.

    `ExTorch.Complex` values are infinity when either of their real or imaginary
    parts are `:nan`.

    ## Arguments
    - `input` - the input tensor (`ExTorch.Tensor`)

    ## Examples
        iex> input = ExTorch.tensor([1, :inf, 2, :ninf, :nan])
        iex> ExTorch.isnan(input)
        #Tensor<
        [ false, false, false, false, true]
        [
          size: {5},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec isnan(ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
    defbinding(isnan(input))

    @doc """
    Returns a new tensor with boolean elements representing if each element is
    real valued.

    All real-valued types are considered real. `ExTorch.Complex` values are real when their imaginary
    parts are zero.

    ## Arguments
    - `input` - the input tensor (`ExTorch.Tensor`)

    ## Examples
        iex> input = ExTorch.tensor([1, ExTorch.Complex.complex(-2, 0), ExTorch.Complex.complex(0, 1)])
        iex> ExTorch.isreal(input)
        #Tensor<
        [ true,  true, false]
        [
          size: {3},
          dtype: :bool,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec isreal(ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
    defbinding(isreal(input))

    @doc """
    Returns a tuple `{values, indices}` where `values` is the kth smallest element of
    each row of the `input` tensor in the given dimension `dim`. And `indices` is the
    index location of each element found.

    * If `dim` is not given, the last dimension of the `input` is chosen.
    * If keepdim is `true`, both the `values` and `indices` tensors are the same size as `input`,
    except in the dimension `dim` where they are of size 1. Otherwise, dim is squeezed
    (see `ExTorch.squeeze`), resulting in both the `values` and `indices` tensors having
    1 fewer dimension than the `input` tensor.

    ## Arguments
    - `input` - the input tensor. (`ExTorch.Tensor`)
    - `k` - k for the kth smallest value. (`integer`)

    ## Optional arguments
    - `dim` - the dimension to find the kth smallest value along. Default: -1. (`integer`)
    - `keepdim` - whether the output tensor has `dim` retained or not. Default: `false` (`boolean`)
    - `out` - the output tuple of `{values, indices}` that can be optionally given as
    output buffers. (`{ExTorch.Tensor, ExTorch.Tensor}`). Default: `nil`

    ## Notes
    When `input` is a CUDA tensor and there are multiple valid k th values, this function
    may nondeterministically return `indices` for any of them.

    ## Examples
        # Retrieve the fourth smallest value.
        iex> x = ExTorch.arange(6)
        #Tensor<
        [0.0000, 1.0000, 2.0000, 3.0000, 4.0000, 5.0000]
        [
          size: {6},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>
        iex> {values, indices} = ExTorch.kthvalue(x, 4)
        iex> values
        #Tensor<
        3.
        [size: {}, dtype: :float, device: :cpu, requires_grad: false]>
        iex> indices
        #Tensor<
        3
        [size: {}, dtype: :long, device: :cpu, requires_grad: false]>

        # Retrieve the second smallest value in the first dimension
        iex> x = ExTorch.rand({3, 4})
        #Tensor<
        [[0.7375, 0.2798, 0.7146, 0.0654],
         [0.0163, 0.8829, 0.8946, 0.3852],
         [0.2225, 0.3258, 0.6905, 0.1512]]
        [
          size: {3, 4},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>
        iex> {values, indices} = ExTorch.kthvalue(x, 2, 0, keepdim: true)
        iex> values
        #Tensor<
        [[0.2225, 0.3258, 0.7146, 0.1512]]
        [
          size: {1, 4},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>
        iex(12)> indices
        #Tensor<
        [[2, 2, 0, 2]]
        [
          size: {1, 4},
          dtype: :long,
          device: :cpu,
          requires_grad: false
        ]>

    """
    @spec kthvalue(
            ExTorch.Tensor.t(),
            integer(),
            integer(),
            boolean(),
            {ExTorch.Tensor.t(), ExTorch.Tensor.t()} | nil
          ) :: {ExTorch.Tensor.t(), ExTorch.Tensor.t()}
    defbinding(kthvalue(input, k, dim \\ -1, keepdim \\ false, out \\ nil))
  end
end
